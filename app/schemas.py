"""Pydantic v2 contracts shared across tools, API, and agent nodes."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    ESCALATED = "escalated"


# --------------------------------------------------------------------------- #
# Tool I/O contracts
# --------------------------------------------------------------------------- #
class KnowledgeBaseQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=1024)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("sparse_weight", "dense_weight")
    @classmethod
    def weights_bounded(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("weight must be between 0 and 1")
        return v


class KnowledgeChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    dense_score: float
    sparse_score: float
    rerank_score: float | None = None
    final_score: float


class KnowledgeBaseResult(BaseModel):
    chunks: list[KnowledgeChunk]
    query_latency_ms: float


class OrderActionType(str, Enum):
    CANCEL = "cancel"
    MODIFY_ADDRESS = "modify_address"
    EXPEDITE_SHIPPING = "expedite_shipping"
    RESEND = "resend"


class ExecuteOrderActionInput(BaseModel):
    order_id: str = Field(..., min_length=3)
    action: OrderActionType
    reason: str = Field(..., min_length=5, max_length=500)
    requested_by_user_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecuteOrderActionResult(BaseModel):
    order_id: str
    action: OrderActionType
    status: ToolExecutionStatus
    detail: str
    applied_at: datetime = Field(default_factory=datetime.utcnow)


class ProcessHighValueRefundInput(BaseModel):
    order_id: str
    user_id: str
    amount_usd: float = Field(..., gt=0)
    currency: Literal["USD"] = "USD"
    reason: str = Field(..., min_length=5, max_length=500)

    @field_validator("amount_usd")
    @classmethod
    def sane_amount(cls, v: float) -> float:
        if v > 50_000:
            raise ValueError("refund amount exceeds sane processing ceiling")
        return round(v, 2)


class ProcessHighValueRefundResult(BaseModel):
    refund_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    amount_usd: float
    approval_status: ApprovalStatus
    detail: str


# --------------------------------------------------------------------------- #
# Computer vision (CNN defect inspection)
# --------------------------------------------------------------------------- #
class ClassifyProductImageInput(BaseModel):
    order_id: str
    image_base64: str = Field(..., min_length=16)
    submitted_by_user_id: str


class ClassifyProductImageResult(BaseModel):
    order_id: str
    predicted_class: str
    confidence: float
    severity_score: float
    all_scores: dict[str, float]
    requires_manual_review: bool
    inference_latency_ms: float


# --------------------------------------------------------------------------- #
# HITL / Approval queue
# --------------------------------------------------------------------------- #
class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    node_name: str
    action_summary: str
    payload: dict[str, Any]
    risk_reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ApprovalDecision(BaseModel):
    approval_id: str
    decision: Literal["approved", "rejected"]
    reviewer_id: str
    notes: str | None = None


# --------------------------------------------------------------------------- #
# API-facing
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    thread_id: str | None = None
    user_id: str
    message: str = Field(..., min_length=1, max_length=4000)
    image_base64: str | None = Field(
        default=None, description="Optional product/return photo for CNN defect inspection."
    )


class TrajectoryEvent(BaseModel):
    event_type: Literal[
        "node_start", "node_end", "tool_call", "tool_result",
        "agent_thought", "hitl_pause", "error", "final_output", "cache_hit",
    ]
    node: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
