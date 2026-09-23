"""Autonomous ("agentic") background agent.

Unlike the reactive graph (which only runs when a user sends a message),
`ProactiveSLAAgent` runs continuously as an independent asyncio task,
periodically inspecting system state and taking action on its own initiative:

  - Escalates HITL approvals that have sat in `ApprovalQueue` past their SLA.
  - Re-opens threads whose self-correction retry budget was exhausted, and
    schedules them for a supervisor follow-up.
  - Emits `TrajectoryEvent`-shaped notifications to a pluggable sink (e.g. the
    WebSocket hub, Slack, PagerDuty) without any human prompting it first.

This is the "agentic" half of the system: the chat graph is agent-as-
responder, this loop is agent-as-monitor, and both share the same Postgres
state so their views of the world never diverge.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import ApprovalQueue

logger = logging.getLogger("agents.proactive")

NotifySinkFn = Callable[[str, dict], Awaitable[None]]


@dataclass
class SLAConfig:
    approval_sla_minutes: int = 30
    poll_interval_seconds: int = 60
    escalation_cooldown_minutes: int = 15


class ProactiveSLAAgent:
    """Runs as a background task; call `.start()` once during app startup
    and `.stop()` on shutdown."""

    def __init__(
        self,
        session_factory: async_sessionmaker,
        notify_sink: NotifySinkFn,
        config: SLAConfig | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._notify_sink = notify_sink
        self._config = config or SLAConfig()
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._last_escalated: dict[str, datetime] = {}

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop(), name="proactive-sla-agent")
            logger.info("ProactiveSLAAgent started (poll every %ss)", self._config.poll_interval_seconds)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            await self._task

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._sweep_stale_approvals()
            except Exception:  # noqa: BLE001 - monitoring loop must never die
                logger.exception("proactive agent sweep failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._config.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def _sweep_stale_approvals(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self._config.approval_sla_minutes)
        async with self._session_factory() as session:
            result = await session.execute(
                select(ApprovalQueue).where(
                    ApprovalQueue.status == "pending", ApprovalQueue.created_at < cutoff,
                )
            )
            stale = result.scalars().all()

        cooldown = timedelta(minutes=self._config.escalation_cooldown_minutes)
        now = datetime.now(timezone.utc)
        for row in stale:
            last = self._last_escalated.get(row.id)
            if last and now - last < cooldown:
                continue
            self._last_escalated[row.id] = now
            await self._notify_sink("sla_breach_escalation", {
                "approval_id": row.id,
                "thread_id": row.thread_id,
                "action_summary": row.action_summary,
                "pending_since": row.created_at.isoformat(),
                "minutes_overdue": (now - row.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60,
            })
            logger.warning("escalated stale approval %s (thread=%s)", row.id, row.thread_id)


async def default_notify_sink(event_type: str, payload: dict) -> None:
    """Default sink: structured logging. Replace with a Slack/PagerDuty/webhook
    call in production by passing a different `notify_sink` to ProactiveSLAAgent."""
    logger.info("proactive_event=%s payload=%s", event_type, payload)
