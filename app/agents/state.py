"""Shared graph state for the multi-agent support engine."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


class ToolCallRecord(TypedDict):
    tool_name: str
    arguments: dict[str, Any]
    attempt: int
    status: Literal["pending", "success", "failed", "retrying"]
    result: dict[str, Any] | None
    error: str | None


class ExecutionError(TypedDict):
    node: str
    message: str
    traceback: str
    attempt: int


class PerformanceMetrics(TypedDict):
    node_latencies_ms: dict[str, float]
    total_tokens: int
    tool_call_count: int
    cache_hit: bool


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], operator.add]

    # Session / user context
    thread_id: str
    user_id: str
    user_context: dict[str, Any]

    # Routing
    active_agent: str
    intent: str | None

    # Computer vision (CNN defect inspection)
    image_attached: bool
    visual_inspection_result: dict[str, Any] | None

    # Tooling
    tool_call_stack: Annotated[list[ToolCallRecord], operator.add]
    pending_tool_call: ToolCallRecord | None

    # Errors / self-correction
    execution_errors: Annotated[list[ExecutionError], operator.add]
    retry_count: int

    # HITL
    hitl_required: bool
    hitl_approval_id: str | None
    hitl_status: Literal["not_required", "pending", "approved", "rejected"]

    # Verification
    verification_passed: bool | None
    verification_notes: str | None

    # Observability
    performance_metrics: PerformanceMetrics

    # Terminal output
    final_response: str | None
