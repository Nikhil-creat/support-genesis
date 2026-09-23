"""Assembles the full multi-agent StateGraph.

Flow:
    RouterSupervisorNode
        -> [VisualInspectionNode]  (only when a photo is attached — CNN defect check)
        -> PolicyRAGNode
        -> ActionExecutorNode --(retry loop, self-correction)--> ActionExecutorNode
        -> [HITL interrupt if a high-value/destructive action is pending]
        -> VerificationCriticNode
        -> END  (or loop back to ActionExecutorNode if verification fails, up to a cap)
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    action_executor_node_factory, policy_rag_node_factory, router_supervisor_node,
    verification_critic_node, visual_inspection_node_factory,
)
from app.agents.state import AgentState
from app.config import get_settings
from app.mcp.client import MCPToolClient

_settings = get_settings()

# Tool names whose invocation must pause the graph for human approval before
# the ActionExecutorNode is allowed to actually run them.
_HITL_GATED_TOOLS = {"process_high_value_refund", "execute_order_action"}


def _make_router_edge(vision_enabled: bool):
    def _route_after_router(state: AgentState) -> Literal["visual_inspection", "policy_rag"]:
        """Route through the CNN inspection node only when a photo is attached
        AND the graph was built with a vision service."""
        if vision_enabled and state.get("image_attached"):
            return "visual_inspection"
        return "policy_rag"

    return _route_after_router


def _route_after_executor(state: AgentState) -> Literal["action_executor", "verification_critic"]:
    """Loop back into the executor while a retry is pending; otherwise verify."""
    if state.get("pending_tool_call") is not None and state.get("retry_count", 0) < _settings.max_tool_retry_attempts:
        return "action_executor"
    return "verification_critic"


def _route_after_verification(state: AgentState) -> Literal["action_executor", "__end__"]:
    """If verification fails and we have budget left, re-plan via the executor; else end."""
    if state.get("verification_passed") is False and state.get("retry_count", 0) < _settings.max_tool_retry_attempts:
        return "action_executor"
    return END


def _is_hitl_action_pending(state: AgentState) -> bool:
    pending = state.get("pending_tool_call")
    return bool(pending and pending.get("tool_name") in _HITL_GATED_TOOLS and pending.get("status") == "pending")


async def build_graph(kb, mcp_client: MCPToolClient, checkpointer: AsyncPostgresSaver, vision_service=None):
    """Compiles the StateGraph. `kb` is a HybridKnowledgeBase, `mcp_client` a live MCPToolClient,
    `vision_service` an optional DefectClassifierService (CNN) — omit to disable image inspection."""
    policy_rag_node = await policy_rag_node_factory(kb)
    action_executor_node = await action_executor_node_factory(mcp_client)

    graph = StateGraph(AgentState)

    graph.add_node("router_supervisor", router_supervisor_node)
    graph.add_node("policy_rag", policy_rag_node)
    graph.add_node("action_executor", action_executor_node)
    graph.add_node("verification_critic", verification_critic_node)

    if vision_service is not None:
        graph.add_node("visual_inspection", await visual_inspection_node_factory(vision_service))
        graph.add_edge("visual_inspection", "policy_rag")

    graph.set_entry_point("router_supervisor")

    router_targets = {"policy_rag": "policy_rag"}
    if vision_service is not None:
        router_targets["visual_inspection"] = "visual_inspection"
    graph.add_conditional_edges("router_supervisor", _make_router_edge(vision_service is not None), router_targets)
    graph.add_edge("policy_rag", "action_executor")
    graph.add_conditional_edges(
        "action_executor",
        _route_after_executor,
        {"action_executor": "action_executor", "verification_critic": "verification_critic"},
    )
    graph.add_conditional_edges(
        "verification_critic",
        _route_after_verification,
        {"action_executor": "action_executor", END: END},
    )

    # HITL breakpoint: pause execution before ActionExecutorNode whenever a
    # gated tool call is queued (checked at runtime via `interrupt_before`
    # combined with a state predicate enforced in the API layer's resume
    # logic — LangGraph's static interrupt_before pauses unconditionally
    # before the named node, so the API layer inspects `pending_tool_call`
    # post-interrupt and either resumes immediately (non-gated tool) or
    # holds for an ApprovalDecision (gated tool).
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["action_executor"],
    )
    return compiled


async def get_checkpointer() -> AsyncPostgresSaver:
    saver_cm = AsyncPostgresSaver.from_conn_string(_settings.postgres_checkpoint_dsn)
    saver = await saver_cm.__aenter__()
    await saver.setup()
    return saver


def requires_human_approval(state: AgentState) -> bool:
    """Exposed for the API layer to decide whether to auto-resume or hold for approval."""
    return _is_hitl_action_pending(state)
