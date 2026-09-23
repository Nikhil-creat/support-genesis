"""Order action execution tool with dynamic, rule-based validation.

Rules are pluggable predicates so new business policy can be registered
without touching the execution path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.schemas import (
    ExecuteOrderActionInput, ExecuteOrderActionResult, OrderActionType, ToolExecutionStatus,
)

RulePredicate = Callable[[ExecuteOrderActionInput, dict], Awaitable[tuple[bool, str]]]


@dataclass
class OrderRuleEngine:
    """Holds ordered validation rules per action type; first failure short-circuits."""

    rules: dict[OrderActionType, list[RulePredicate]] = field(default_factory=dict)

    def register(self, action: OrderActionType, rule: RulePredicate) -> None:
        self.rules.setdefault(action, []).append(rule)

    async def validate(self, action_input: ExecuteOrderActionInput, order_record: dict) -> tuple[bool, str]:
        for rule in self.rules.get(action_input.action, []):
            ok, reason = await rule(action_input, order_record)
            if not ok:
                return False, reason
        return True, "all rules passed"


async def _rule_order_not_already_cancelled(inp: ExecuteOrderActionInput, order: dict) -> tuple[bool, str]:
    if order.get("status") == "cancelled" and inp.action == OrderActionType.CANCEL:
        return False, f"order {inp.order_id} is already cancelled"
    return True, "ok"


async def _rule_order_not_shipped_for_cancel(inp: ExecuteOrderActionInput, order: dict) -> tuple[bool, str]:
    if inp.action == OrderActionType.CANCEL and order.get("status") == "shipped":
        return False, f"order {inp.order_id} has already shipped and cannot be cancelled"
    return True, "ok"


async def _rule_address_change_window(inp: ExecuteOrderActionInput, order: dict) -> tuple[bool, str]:
    if inp.action == OrderActionType.MODIFY_ADDRESS and order.get("status") in {"shipped", "delivered"}:
        return False, f"order {inp.order_id} is past the address-change window ({order.get('status')})"
    return True, "ok"


async def _rule_expedite_requires_in_stock(inp: ExecuteOrderActionInput, order: dict) -> tuple[bool, str]:
    if inp.action == OrderActionType.EXPEDITE_SHIPPING and not order.get("in_stock", True):
        return False, f"order {inp.order_id} contains backordered items; cannot expedite"
    return True, "ok"


def build_default_rule_engine() -> OrderRuleEngine:
    engine = OrderRuleEngine()
    engine.register(OrderActionType.CANCEL, _rule_order_not_already_cancelled)
    engine.register(OrderActionType.CANCEL, _rule_order_not_shipped_for_cancel)
    engine.register(OrderActionType.MODIFY_ADDRESS, _rule_address_change_window)
    engine.register(OrderActionType.EXPEDITE_SHIPPING, _rule_expedite_requires_in_stock)
    return engine


OrderLookupFn = Callable[[str], Awaitable[dict]]
OrderMutateFn = Callable[[str, ExecuteOrderActionInput], Awaitable[None]]


async def execute_order_action(
    action_input: ExecuteOrderActionInput,
    rule_engine: OrderRuleEngine,
    lookup_order: OrderLookupFn,
    mutate_order: OrderMutateFn,
) -> ExecuteOrderActionResult:
    order_record = await lookup_order(action_input.order_id)
    if order_record is None:
        return ExecuteOrderActionResult(
            order_id=action_input.order_id,
            action=action_input.action,
            status=ToolExecutionStatus.FAILED,
            detail=f"order {action_input.order_id} not found",
        )

    valid, reason = await rule_engine.validate(action_input, order_record)
    if not valid:
        return ExecuteOrderActionResult(
            order_id=action_input.order_id,
            action=action_input.action,
            status=ToolExecutionStatus.FAILED,
            detail=reason,
        )

    await mutate_order(action_input.order_id, action_input)
    return ExecuteOrderActionResult(
        order_id=action_input.order_id,
        action=action_input.action,
        status=ToolExecutionStatus.SUCCESS,
        detail=f"{action_input.action.value} applied to order {action_input.order_id}: {reason}",
    )
