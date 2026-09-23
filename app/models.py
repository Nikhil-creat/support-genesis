"""SQLAlchemy 2.0 async ORM models for the support engine's system-of-record.

Note: LangGraph's own AsyncPostgresSaver manages its checkpoint tables
independently; `AgentCheckpoint` here is an application-level audit mirror
used for dashboards/reporting, not the raw LangGraph checkpoint store.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Index, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


class Base(AsyncAttrs, DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[str]:
    return mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = _uuid_pk()
    external_user_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True)
    tier: Mapped[str] = mapped_column(String(32), default="standard")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="user")
    audit_entries: Mapped[list["AuditTrail"]] = relationship(back_populates="user")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = _uuid_pk()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    order_id: Mapped[str] = mapped_column(String(128), index=True)
    action_type: Mapped[str] = mapped_column(String(64))
    amount_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="transactions")

    __table_args__ = (Index("ix_transactions_order_status", "order_id", "status"),)


class AuditTrail(Base):
    """Immutable append-only log of every agent action for compliance review."""

    __tablename__ = "audit_trail"

    id: Mapped[str] = _uuid_pk()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    actor: Mapped[str] = mapped_column(String(64))  # e.g. "RouterSupervisorNode"
    action: Mapped[str] = mapped_column(String(128))
    input_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    output_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    pii_redacted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="audit_entries")


class AgentCheckpoint(Base):
    """Application-level mirror of graph checkpoints for fast dashboard reads."""

    __tablename__ = "agent_checkpoints"

    id: Mapped[str] = _uuid_pk()
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    active_node: Mapped[str] = mapped_column(String(128))
    state_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    token_usage: Mapped[dict] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_checkpoints_thread_time", "thread_id", "created_at"),)


class ApprovalQueue(Base):
    """HITL approval queue for destructive/high-value tool calls."""

    __tablename__ = "approval_queue"

    id: Mapped[str] = _uuid_pk()
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    node_name: Mapped[str] = mapped_column(String(128))
    action_summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    risk_reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------- #
# Engine / session factory
# --------------------------------------------------------------------------- #
_settings = get_settings()
engine = create_async_engine(_settings.postgres_dsn, pool_size=20, max_overflow=10, pool_pre_ping=True)
AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_models() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionFactory() as session:
        yield session
