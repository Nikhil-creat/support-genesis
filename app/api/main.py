"""Async FastAPI server exposing the autonomous support engine.

Endpoints:
    POST /chat/stream            -> SSE stream of TrajectoryEvents for one turn
    WS   /ws/agent/trajectory    -> WebSocket stream of the same events, bidirectional
    POST /approvals/{id}/decide  -> resolve a pending HITL approval
    GET  /approvals/pending      -> list pending approvals
    GET  /healthz                -> liveness/readiness probe
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph, get_checkpointer, requires_human_approval
from app.agents.proactive_agent import ProactiveSLAAgent, default_notify_sink
from app.branding import branding_payload
from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.mcp.client import MCPToolClient
from app.models import AsyncSessionFactory, ApprovalQueue, AuditTrail, get_session, init_models
from app.schemas import ApprovalDecision, ChatRequest, TrajectoryEvent
from app.security.pii_sanitizer import PIISanitizer
from app.telemetry.tracing import configure_telemetry
from app.tools.knowledge_base import HybridKnowledgeBase
from app.vision.defect_classifier import DefectClassifierService

logger = logging.getLogger("support_engine.api")
settings = get_settings()
sanitizer = PIISanitizer()


async def _fake_embed(text: str) -> np.ndarray:
    """Deterministic placeholder embedding for environments without live API
    access; swap for a real `text-embedding-3-small` call in production."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=384).astype(np.float32)


class AppState:
    graph: object = None
    kb: HybridKnowledgeBase | None = None
    mcp_client: MCPToolClient | None = None
    semantic_cache: SemanticCache | None = None
    checkpointer: object = None
    vision_service: DefectClassifierService | None = None
    proactive_agent: ProactiveSLAAgent | None = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_telemetry()
    await init_models()

    app_state.mcp_client = MCPToolClient(http_endpoints=settings.mcp_server_endpoints)
    await app_state.mcp_client.initialize_all()
    await app_state.mcp_client.discover_tools()

    app_state.kb = HybridKnowledgeBase(embed_fn=_fake_embed)
    app_state.semantic_cache = SemanticCache(embed_fn=_fake_embed)
    app_state.vision_service = DefectClassifierService(weights_path=settings.cnn_weights_path)
    app_state.checkpointer = await get_checkpointer()
    app_state.graph = await build_graph(
        app_state.kb, app_state.mcp_client, app_state.checkpointer, vision_service=app_state.vision_service,
    )

    app_state.proactive_agent = ProactiveSLAAgent(AsyncSessionFactory, default_notify_sink)
    app_state.proactive_agent.start()

    logger.info("support engine started — %s", branding_payload()["credit"])
    yield

    await app_state.proactive_agent.stop()
    await app_state.mcp_client.close()
    await app_state.semantic_cache.close()
    logger.info("support engine shut down")


app = FastAPI(title="Autonomous Customer Support & Action Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("unhandled exception on %s", request.url)
    return StreamingResponse(
        iter([json.dumps({"error": "internal_server_error", "detail": str(exc)})]),
        status_code=500,
        media_type="application/json",
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "service": settings.service_name}


@app.get("/")
async def root() -> dict:
    return branding_payload()


async def _run_turn(chat_request: ChatRequest, session: AsyncSession) -> AsyncIterator[TrajectoryEvent]:
    thread_id = chat_request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    cache_hit = await app_state.semantic_cache.lookup(chat_request.message)
    if cache_hit is not None:
        yield TrajectoryEvent(event_type="cache_hit", payload={"similarity": cache_hit["similarity"]})
        yield TrajectoryEvent(event_type="final_output", payload=cache_hit["trajectory"])
        return

    initial_state = {
        "messages": [HumanMessage(content=chat_request.message)],
        "thread_id": thread_id,
        "user_id": chat_request.user_id,
        "user_context": {"pending_image_base64": chat_request.image_base64} if chat_request.image_base64 else {},
        "active_agent": "",
        "intent": None,
        "image_attached": bool(chat_request.image_base64),
        "visual_inspection_result": None,
        "tool_call_stack": [],
        "pending_tool_call": None,
        "execution_errors": [],
        "retry_count": 0,
        "hitl_required": False,
        "hitl_approval_id": None,
        "hitl_status": "not_required",
        "verification_passed": None,
        "verification_notes": None,
        "performance_metrics": {"node_latencies_ms": {}, "total_tokens": 0, "tool_call_count": 0, "cache_hit": False},
        "final_response": None,
    }

    start = time.perf_counter()
    final_state = None
    async for event in app_state.graph.astream(initial_state, config=config, stream_mode="values"):
        final_state = event
        active_node = event.get("active_agent") or "graph"
        yield TrajectoryEvent(
            event_type="node_end",
            node=active_node,
            payload={"messages_tail": str(event["messages"][-1].content) if event.get("messages") else ""},
            latency_ms=(time.perf_counter() - start) * 1000,
        )

        if requires_human_approval(event):
            yield TrajectoryEvent(
                event_type="hitl_pause",
                node="action_executor",
                payload={"pending_tool_call": event.get("pending_tool_call")},
            )
            session.add(ApprovalQueue(
                thread_id=thread_id,
                node_name="ActionExecutorNode",
                action_summary=str(event.get("pending_tool_call")),
                payload=sanitizer.sanitize_payload(event.get("pending_tool_call") or {}),
                risk_reason="gated tool pending human approval",
                status="pending",
            ))
            await session.commit()
            return  # graph stays interrupted; resumed via /approvals/{id}/decide

    if final_state is not None:
        session.add(AuditTrail(
            user_id=None,
            thread_id=thread_id,
            actor="graph",
            action="complete_turn",
            input_payload=sanitizer.sanitize_payload({"message": chat_request.message}),
            output_payload=sanitizer.sanitize_payload({"final_response": final_state.get("final_response")}),
            pii_redacted=True,
        ))
        await session.commit()

        result_payload = {"final_response": final_state.get("final_response"), "thread_id": thread_id}
        await app_state.semantic_cache.store(chat_request.message, result_payload)
        yield TrajectoryEvent(event_type="final_output", payload=result_payload)


@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest, session: AsyncSession = Depends(get_session)):
    async def event_generator() -> AsyncIterator[str]:
        async for event in _run_turn(chat_request, session):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws/agent/trajectory")
async def ws_agent_trajectory(websocket: WebSocket, session: AsyncSession = Depends(get_session)):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            chat_request = ChatRequest(**raw)
            async for event in _run_turn(chat_request, session):
                await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        logger.info("websocket client disconnected")


@app.get("/approvals/pending")
async def list_pending_approvals(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(ApprovalQueue).where(ApprovalQueue.status == "pending"))
    rows = result.scalars().all()
    return [
        {
            "id": row.id, "thread_id": row.thread_id, "action_summary": row.action_summary,
            "risk_reason": row.risk_reason, "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@app.post("/approvals/{approval_id}/decide")
async def decide_approval(approval_id: str, decision: ApprovalDecision, session: AsyncSession = Depends(get_session)):
    row = await session.get(ApprovalQueue, approval_id)
    if row is None:
        raise HTTPException(status_code=404, detail="approval not found")
    if row.status != "pending":
        raise HTTPException(status_code=409, detail=f"approval already {row.status}")

    row.status = decision.decision
    row.reviewer_id = decision.reviewer_id
    row.reviewer_notes = decision.notes
    await session.commit()

    if decision.decision == "approved":
        config = {"configurable": {"thread_id": row.thread_id}}
        # Resuming with `None` re-enters the graph after the interrupt point,
        # continuing from the persisted checkpoint for this thread.
        async for _ in app_state.graph.astream(None, config=config, stream_mode="values"):
            pass

    return {"approval_id": approval_id, "status": row.status}
