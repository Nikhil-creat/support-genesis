"""High-value refund tool: auto-approves small refunds, suspends state and
pushes to the ApprovalQueue for anything above the configured threshold.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from app.config import get_settings
from app.schemas import (
    ApprovalRequest, ApprovalStatus, ProcessHighValueRefundInput, ProcessHighValueRefundResult,
)

EnqueueApprovalFn = Callable[[ApprovalRequest], Awaitable[None]]
IssueRefundFn = Callable[[ProcessHighValueRefundInput], Awaitable[str]]  # returns provider refund ref


async def process_high_value_refund(
    refund_input: ProcessHighValueRefundInput,
    thread_id: str,
    enqueue_approval: EnqueueApprovalFn,
    issue_refund: IssueRefundFn,
) -> ProcessHighValueRefundResult:
    settings = get_settings()

    if refund_input.amount_usd > settings.high_value_refund_threshold_usd:
        approval = ApprovalRequest(
            thread_id=thread_id,
            node_name="ActionExecutorNode",
            action_summary=(
                f"Refund ${refund_input.amount_usd:.2f} to user {refund_input.user_id} "
                f"for order {refund_input.order_id}"
            ),
            payload=refund_input.model_dump(),
            risk_reason=(
                f"amount ${refund_input.amount_usd:.2f} exceeds auto-approval threshold "
                f"of ${settings.high_value_refund_threshold_usd:.2f}"
            ),
        )
        await enqueue_approval(approval)
        return ProcessHighValueRefundResult(
            order_id=refund_input.order_id,
            amount_usd=refund_input.amount_usd,
            approval_status=ApprovalStatus.PENDING,
            detail=f"refund suspended pending human approval (approval_id={approval.approval_id})",
        )

    provider_ref = await issue_refund(refund_input)
    return ProcessHighValueRefundResult(
        order_id=refund_input.order_id,
        amount_usd=refund_input.amount_usd,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        detail=f"refund auto-approved and issued, provider_ref={provider_ref}",
    )
