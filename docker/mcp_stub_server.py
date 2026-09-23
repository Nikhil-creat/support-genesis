"""Minimal standalone MCP server stub (Streamable HTTP transport).

Exposes the four tools the agent graph expects (`query_knowledge_base`,
`execute_order_action`, `process_high_value_refund`,
`classify_product_defect_image`) with in-memory, deterministic mock
implementations. This lets the full docker-compose stack run end-to-end
without a real backend/CRM integration; swap this container for a real MCP
server (or several) in production by pointing `MCP_SERVER_ENDPOINTS` at it.

Run standalone with: uvicorn docker.mcp_stub_server:app --port 8801
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request

app = FastAPI(title="MCP Tool Server Stub")

_TOOLS = [
    {
        "name": "query_knowledge_base",
        "description": "Hybrid dense+sparse retrieval over the support policy knowledge base.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "execute_order_action",
        "description": "Cancel, modify, expedite, or resend an order.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "action": {"type": "string"}},
            "required": ["order_id", "action"],
        },
    },
    {
        "name": "process_high_value_refund",
        "description": "Issue a refund; amounts over the configured threshold require approval.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount_usd": {"type": "number"}},
            "required": ["order_id", "amount_usd"],
        },
    },
    {
        "name": "classify_product_defect_image",
        "description": "CNN-based product damage/defect classification from a photo.",
        "inputSchema": {"type": "object", "properties": {"image_base64": {"type": "string"}}},
    },
]

_MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1029": {"status": "processing", "in_stock": True},
    "ORD-2044": {"status": "delivered", "in_stock": True},
}


def _handle_tool_call(name: str, arguments: dict) -> dict:
    if name == "query_knowledge_base":
        return {
            "chunks": [
                {"chunk_id": "kb-1", "text": "Orders may be cancelled before they ship.", "source": "policy.md"},
            ],
            "query_latency_ms": 4.2,
        }
    if name == "execute_order_action":
        order = _MOCK_ORDERS.get(arguments.get("order_id"), {"status": "unknown"})
        return {
            "order_id": arguments.get("order_id"),
            "action": arguments.get("action"),
            "status": "success" if order["status"] != "shipped" else "failed",
            "detail": f"mock action applied to order in status={order['status']}",
        }
    if name == "process_high_value_refund":
        amount = float(arguments.get("amount_usd", 0))
        return {
            "refund_id": str(uuid.uuid4()),
            "order_id": arguments.get("order_id"),
            "amount_usd": amount,
            "approval_status": "pending" if amount > 100 else "not_required",
            "detail": "mock refund processed",
        }
    if name == "classify_product_defect_image":
        return {
            "predicted_class": "no_defect",
            "confidence": 0.91,
            "severity_score": 0.0,
            "all_scores": {"no_defect": 0.91, "surface_scratch": 0.05, "crack_or_fracture": 0.01,
                            "discoloration_or_stain": 0.01, "missing_component": 0.01, "packaging_damage": 0.01},
            "requires_manual_review": False,
            "inference_latency_ms": 12.0,
        }
    raise ValueError(f"unknown tool: {name}")


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> dict:
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    req_id = body.get("id")

    if method == "initialize":
        result: dict[str, Any] = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "mcp-stub", "version": "1.0.0"}}
    elif method == "tools/list":
        result = {"tools": _TOOLS}
    elif method == "tools/call":
        try:
            result = _handle_tool_call(params["name"], params.get("arguments", {}))
        except (KeyError, ValueError) as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(exc)}}
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"unknown method {method}"}}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
