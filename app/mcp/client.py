"""Model Context Protocol client.

Supports two transports:
  - Streamable HTTP: JSON-RPC 2.0 over POST, one request per call.
  - stdio: JSON-RPC 2.0 framed over a child process's stdin/stdout, for
    locally-spawned MCP servers.

Both transports expose the same three RPC methods used by MCP tool servers:
`initialize`, `tools/list`, and `tools/call`.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("mcp.client")


class MCPProtocolError(RuntimeError):
    """Raised when the MCP server returns a malformed or error JSON-RPC response."""


class MCPTransportError(RuntimeError):
    """Raised on connection-level failures (process death, HTTP failure, timeout)."""


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_endpoint: str


class _JSONRPCIdCounter:
    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def next(self) -> int:
        return next(self._counter)


class HTTPTransport:
    def __init__(self, endpoint: str, timeout: float = 15.0) -> None:
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(timeout=timeout)

    async def send(self, payload: dict) -> dict:
        try:
            resp = await self._client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"HTTP transport failure to {self.endpoint}: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()


class StdioTransport:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    async def send(self, payload: dict) -> dict:
        async with self._lock:
            await self._ensure_started()
            assert self._process and self._process.stdin and self._process.stdout
            frame = (json.dumps(payload) + "\n").encode("utf-8")
            self._process.stdin.write(frame)
            await self._process.stdin.drain()
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=15.0)
            if not line:
                stderr = await self._process.stderr.read() if self._process.stderr else b""
                raise MCPTransportError(f"stdio MCP server closed pipe: {stderr.decode(errors='ignore')}")
            return json.loads(line.decode("utf-8"))

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()


class MCPToolClient:
    """Discovers and invokes tools exposed by one or more MCP servers."""

    def __init__(self, http_endpoints: list[str] | None = None, stdio_commands: list[list[str]] | None = None) -> None:
        self._id_counter = _JSONRPCIdCounter()
        self._transports: dict[str, HTTPTransport | StdioTransport] = {}
        for ep in http_endpoints or []:
            self._transports[ep] = HTTPTransport(ep)
        for cmd in stdio_commands or []:
            key = " ".join(cmd)
            self._transports[key] = StdioTransport(cmd)
        self._tool_registry: dict[str, MCPTool] = {}

    def _envelope(self, method: str, params: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._id_counter.next(),
            "method": method,
            "params": params,
        }

    @staticmethod
    def _unwrap(response: dict) -> dict:
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(f"MCP error {err.get('code')}: {err.get('message')}")
        if "result" not in response:
            raise MCPProtocolError(f"malformed MCP response, missing 'result': {response}")
        return response["result"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(MCPTransportError),
    )
    async def _call(self, endpoint_key: str, method: str, params: dict) -> dict:
        transport = self._transports[endpoint_key]
        response = await transport.send(self._envelope(method, params))
        return self._unwrap(response)

    async def initialize_all(self) -> None:
        for key in self._transports:
            try:
                await self._call(key, "initialize", {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "autonomous-support-engine", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                })
            except (MCPProtocolError, MCPTransportError) as exc:
                logger.warning("MCP server %s failed to initialize: %s", key, exc)

    async def discover_tools(self) -> list[MCPTool]:
        discovered: list[MCPTool] = []
        for key in self._transports:
            try:
                result = await self._call(key, "tools/list", {})
            except (MCPProtocolError, MCPTransportError) as exc:
                logger.warning("tool discovery failed for %s: %s", key, exc)
                continue
            for raw_tool in result.get("tools", []):
                tool = MCPTool(
                    name=raw_tool["name"],
                    description=raw_tool.get("description", ""),
                    input_schema=raw_tool.get("inputSchema", {}),
                    server_endpoint=key,
                )
                self._tool_registry[tool.name] = tool
                discovered.append(tool)
        return discovered

    def get_tool(self, name: str) -> MCPTool | None:
        return self._tool_registry.get(name)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        tool = self.get_tool(name)
        if tool is None:
            raise MCPProtocolError(f"unknown MCP tool '{name}'; run discover_tools() first")
        result = await self._call(tool.server_endpoint, "tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return result

    async def close(self) -> None:
        await asyncio.gather(*(t.close() for t in self._transports.values()), return_exceptions=True)
