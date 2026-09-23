"""The four core LangGraph nodes powering the multi-agent engine.

Each node is a plain async function `(state, config) -> partial_state_update`,
the standard LangGraph node signature. Nodes are kept framework-light so they
can be unit tested without spinning up a full graph.
"""
from __future__ import annotations

import time
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.state import AgentState, ExecutionError, ToolCallRecord
from app.config import get_settings
from app.mcp.client import MCPToolClient
from app.security.pii_sanitizer import PIISanitizer

_settings = get_settings()
_sanitizer = PIISanitizer()

_AGENT_CARDS = {
    "billing_agent": "Handles invoices, payments, and billing disputes.",
    "refund_agent": "Handles refund requests and eligibility checks.",
    "technical_support_agent": "Handles product defects, bugs, and troubleshooting.",
    "order_agent": "Handles order cancellation, address changes, and shipping.",
}

_ROUTER_SYSTEM_PROMPT = """You are the A2A Supervisor for a customer support engine.
Read the user's message and the Agent Cards below, then output ONLY the single
best agent name to delegate to (one of: {agent_names}), followed by a pipe
character and a short intent label. Example output: "refund_agent|refund_request"

Agent Cards:
{agent_cards}
"""


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=_settings.llm_model, temperature=temperature, api_key=_settings.openai_api_key)


def _record_latency(state: AgentState, node: str, start: float) -> dict[str, float]:
    metrics = dict(state.get("performance_metrics") or {})
    latencies = dict(metrics.get("node_latencies_ms", {}))
    latencies[node] = (time.perf_counter() - start) * 1000
    metrics["node_latencies_ms"] = latencies
    return metrics


# --------------------------------------------------------------------------- #
# Node 1: RouterSupervisorNode
# --------------------------------------------------------------------------- #
async def router_supervisor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    start = time.perf_counter()
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    raw_text = last_human.content if last_human else ""

    sanitized = _sanitizer.sanitize(str(raw_text))

    agent_cards_desc = "\n".join(f"- {name}: {desc}" for name, desc in _AGENT_CARDS.items())
    prompt = _ROUTER_SYSTEM_PROMPT.format(agent_names=", ".join(_AGENT_CARDS), agent_cards=agent_cards_desc)

    llm = _llm()
    response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=sanitized.sanitized)])
    raw = str(response.content).strip()
    agent_name, _, intent = raw.partition("|")
    agent_name = agent_name.strip()
    intent = intent.strip() or "unclassified"

    if agent_name not in _AGENT_CARDS:
        agent_name = "technical_support_agent"  # safe default

    return {
        "active_agent": agent_name,
        "intent": intent,
        "messages": [AIMessage(content=f"[router] delegating to {agent_name} (intent={intent})")],
        "performance_metrics": _record_latency(state, "RouterSupervisorNode", start),
    }


# --------------------------------------------------------------------------- #
# Node 1.5: VisualInspectionNode (CNN-based damage/defect classification)
# --------------------------------------------------------------------------- #
async def visual_inspection_node_factory(vision_service) -> Any:
    """Returns a node closure bound to a DefectClassifierService instance.

    Runs only when the incoming turn has an attached photo (see graph
    routing); classifies it with the CNN and folds the verdict into
    `user_context` so PolicyRAGNode/ActionExecutorNode can condition on it
    (e.g. auto-approve a return only when `requires_manual_review` is False).
    """

    async def visual_inspection_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from app.schemas import ClassifyProductImageInput
        from app.tools.vision_inspection import classify_product_defect_image

        start = time.perf_counter()
        image_b64 = state.get("user_context", {}).get("pending_image_base64")
        order_id = state.get("user_context", {}).get("order_id", "UNKNOWN")

        result = await classify_product_defect_image(
            vision_service,
            ClassifyProductImageInput(
                order_id=order_id, image_base64=image_b64, submitted_by_user_id=state["user_id"],
            ),
        )
        result_dict = result.model_dump()

        summary = (
            f"[visual_inspection] predicted={result.predicted_class} "
            f"confidence={result.confidence:.2f} severity={result.severity_score:.2f} "
            f"manual_review={result.requires_manual_review}"
        )
        return {
            "visual_inspection_result": result_dict,
            "user_context": {**state.get("user_context", {}), "visual_inspection_result": result_dict},
            "messages": [AIMessage(content=summary)],
            "performance_metrics": _record_latency(state, "VisualInspectionNode", start),
        }

    return visual_inspection_node


# --------------------------------------------------------------------------- #
# Node 2: PolicyRAGNode
# --------------------------------------------------------------------------- #
async def policy_rag_node_factory(kb) -> Any:
    """Returns a node closure bound to a HybridKnowledgeBase instance."""

    async def policy_rag_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from app.schemas import KnowledgeBaseQuery

        start = time.perf_counter()
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        query_text = str(last_human.content) if last_human else ""

        vision_result = state.get("visual_inspection_result")
        if vision_result:
            query_text += f" (CNN inspection found: {vision_result['predicted_class']})"

        result = await kb.query(KnowledgeBaseQuery(query=query_text, top_k=5))
        context_block = "\n\n".join(f"[{c.source}] {c.text}" for c in result.chunks)

        return {
            "user_context": {**state.get("user_context", {}), "retrieved_policy_context": context_block},
            "messages": [AIMessage(content=f"[policy_rag] retrieved {len(result.chunks)} chunks "
                                            f"in {result.query_latency_ms:.1f}ms")],
            "performance_metrics": _record_latency(state, "PolicyRAGNode", start),
        }

    return policy_rag_node


# --------------------------------------------------------------------------- #
# Node 3: ActionExecutorNode (MCP tool caller with self-correction retry loop)
# --------------------------------------------------------------------------- #
async def action_executor_node_factory(mcp_client: MCPToolClient) -> Any:
    async def action_executor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        start = time.perf_counter()
        pending = state.get("pending_tool_call")
        if pending is None:
            # Ask the LLM to decide which tool to call given intent + policy context.
            pending = await _plan_tool_call(state)

        attempt = pending.get("attempt", 0) + 1
        tool_name = pending["tool_name"]
        arguments = pending["arguments"]

        try:
            raw_result = await mcp_client.call_tool(tool_name, arguments)
            record: ToolCallRecord = {
                "tool_name": tool_name, "arguments": arguments, "attempt": attempt,
                "status": "success", "result": raw_result, "error": None,
            }
            sanitized_result = _sanitizer.sanitize_payload(raw_result)
            return {
                "tool_call_stack": [record],
                "pending_tool_call": None,
                "retry_count": 0,
                "messages": [AIMessage(content=f"[action_executor] {tool_name} succeeded on attempt {attempt}")],
                "user_context": {**state.get("user_context", {}), "last_tool_result": sanitized_result},
                "performance_metrics": _record_latency(state, "ActionExecutorNode", start),
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any tool failure triggers self-correction
            tb = traceback.format_exc()
            error: ExecutionError = {
                "node": "ActionExecutorNode", "message": str(exc), "traceback": tb, "attempt": attempt,
            }
            failed_record: ToolCallRecord = {
                "tool_name": tool_name, "arguments": arguments, "attempt": attempt,
                "status": "retrying" if attempt < _settings.max_tool_retry_attempts else "failed",
                "result": None, "error": str(exc),
            }

            if attempt < _settings.max_tool_retry_attempts:
                adjusted_args = await _adjust_arguments_from_error(arguments, str(exc))
                new_pending: ToolCallRecord = {**failed_record, "arguments": adjusted_args}
                return {
                    "tool_call_stack": [failed_record],
                    "pending_tool_call": new_pending,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "execution_errors": [error],
                    "messages": [AIMessage(content=f"[action_executor] {tool_name} failed on attempt "
                                                    f"{attempt}, retrying with adjusted args")],
                    "performance_metrics": _record_latency(state, "ActionExecutorNode", start),
                }

            return {
                "tool_call_stack": [failed_record],
                "pending_tool_call": None,
                "execution_errors": [error],
                "messages": [AIMessage(content=f"[action_executor] {tool_name} failed after {attempt} "
                                                f"attempts; degrading gracefully")],
                "performance_metrics": _record_latency(state, "ActionExecutorNode", start),
            }

    return action_executor_node


async def _plan_tool_call(state: AgentState) -> ToolCallRecord:
    """Uses the LLM to select a tool + arguments based on intent and retrieved policy context."""
    llm = _llm()
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    prompt = (
        "Given the user's intent and policy context, output a JSON object with keys "
        "'tool_name' and 'arguments' describing the single next tool call to make.\n"
        f"Intent: {state.get('intent')}\n"
        f"Policy context: {state.get('user_context', {}).get('retrieved_policy_context', '')}\n"
        f"User message: {last_human.content if last_human else ''}\n"
        "Respond with ONLY the JSON object."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    import json
    try:
        parsed = json.loads(str(response.content))
    except json.JSONDecodeError:
        parsed = {"tool_name": "query_knowledge_base", "arguments": {"query": str(last_human.content if last_human else "")}}
    return {
        "tool_name": parsed.get("tool_name", "query_knowledge_base"),
        "arguments": parsed.get("arguments", {}),
        "attempt": 0,
        "status": "pending",
        "result": None,
        "error": None,
    }


async def _adjust_arguments_from_error(arguments: dict, error_message: str) -> dict:
    """Self-correction: ask the LLM to repair the tool arguments given the traceback."""
    llm = _llm()
    prompt = (
        "A tool call failed with this error:\n"
        f"{error_message}\n\n"
        f"Original arguments (JSON): {arguments}\n"
        "Return a corrected JSON arguments object only, fixing whatever caused the failure "
        "(e.g. missing/invalid fields, wrong types). Respond with ONLY the JSON object."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    import json
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return arguments


# --------------------------------------------------------------------------- #
# Node 4: VerificationCriticNode
# --------------------------------------------------------------------------- #
async def verification_critic_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    start = time.perf_counter()
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    last_tool_result = state.get("user_context", {}).get("last_tool_result")

    llm = _llm()
    prompt = (
        "You are a strict verification critic. Given the original user request and the "
        "tool execution result, answer with exactly 'PASS' or 'FAIL' followed by a one-line reason.\n"
        f"Original request: {last_human.content if last_human else ''}\n"
        f"Tool result: {last_tool_result}\n"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    passed = raw.upper().startswith("PASS")

    final_response = None
    if passed:
        final_response = _compose_final_response(state, last_tool_result)

    return {
        "verification_passed": passed,
        "verification_notes": raw,
        "final_response": final_response,
        "messages": [AIMessage(content=f"[verification_critic] {raw}")],
        "performance_metrics": _record_latency(state, "VerificationCriticNode", start),
    }


def _compose_final_response(state: AgentState, tool_result: dict | None) -> str:
    intent = state.get("intent", "your request")
    if tool_result is None:
        return f"I've reviewed your {intent} request; no further action was needed."
    detail = tool_result.get("detail") or tool_result.get("result") or "the action was completed"
    return f"Done — regarding your {intent} request: {detail}"
