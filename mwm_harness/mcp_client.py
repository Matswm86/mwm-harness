"""MCP client: the servers of ``.mcp.json`` become ``mcp__server__tool`` tools.

Two transports: stdio (newline-delimited JSON-RPC to a child process) and
streamable HTTP (POST, answer as JSON or as a server-sent event stream).

A server process starts on its first tool call, not at session start. The tool
listing the model needs up front comes from a cache file that is keyed on the
server's configuration and on the modification time of the script it runs, so
a session opens without launching six processes. ``refresh`` re-lists.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mwm_harness.config import config_dir, workspace_root
from mwm_harness.tools.base import Tool, ToolContext, ToolResult

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "mwm-harness", "version": "0.1.0"}
DEFAULT_TIMEOUT = 120.0
START_TIMEOUT = 60.0
STDIO_LINE_LIMIT = 64 * 1024 * 1024  # one JSON-RPC message may carry a base64 image


# Names of the harness's own model keys; the Session fills this in. A server keeps the
# rest of the environment (it is a program the person configured and may need its own
# keys), but it has no business holding the keys that pay for the model.
HARNESS_SECRET_ENV: set[str] = {"TYPESAFE_API_KEY"}


def server_env(configured: dict[str, str]) -> dict[str, str]:
    inherited = {k: v for k, v in os.environ.items() if k not in HARNESS_SECRET_ENV}
    return {**inherited, **configured}


class McpError(Exception):
    """A server could not be reached, or answered with a JSON-RPC error."""


@dataclass
class ServerConfig:
    name: str
    transport: str  # stdio | http
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    cwd: Path | None = None

    def fingerprint(self) -> str:
        """Changes when the configuration or a script named in ``args`` changes."""
        stamps = []
        for arg in [self.command, *self.args]:
            path = Path(arg)
            if path.is_absolute() and path.is_file():
                stamps.append(f"{arg}:{path.stat().st_mtime_ns}")
        raw = json.dumps(
            [self.transport, self.command, self.args, sorted(self.env), self.url, stamps]
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def default_mcp_files(cwd: Path) -> list[Path]:
    return [cwd / ".mcp.json", workspace_root() / ".mcp.json", config_dir() / "mcp.json"]


def load_server_configs(paths: list[Path]) -> dict[str, ServerConfig]:
    """Read ``mcpServers`` from each file as Claude Code writes it. The first file wins."""
    configs: dict[str, ServerConfig] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise McpError(f"{path}: {exc}") from exc
        for name, entry in (data.get("mcpServers") or {}).items():
            if name in configs or not isinstance(entry, dict):
                continue
            timeout_ms = entry.get("timeout")
            timeout = float(timeout_ms) / 1000 if timeout_ms else DEFAULT_TIMEOUT
            if entry.get("url"):
                configs[name] = ServerConfig(
                    name,
                    "http",
                    url=str(entry["url"]),
                    headers={str(k): str(v) for k, v in (entry.get("headers") or {}).items()},
                    timeout=timeout,
                )
            elif entry.get("command"):
                configs[name] = ServerConfig(
                    name,
                    "stdio",
                    command=str(entry["command"]),
                    args=[str(a) for a in entry.get("args") or []],
                    env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                    timeout=timeout,
                    cwd=path.parent,
                )
    return configs


# ------------------------------------------------------------------ transports


class StdioConnection:
    def __init__(self, config: ServerConfig, stderr_log: Path | None = None) -> None:
        self.config = config
        self.stderr_log = stderr_log
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._stderr_file: Any = None

    async def start(self) -> None:
        stderr: Any = asyncio.subprocess.DEVNULL
        if self.stderr_log is not None:
            self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = self.stderr_log.open("ab")
            stderr = self._stderr_file
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                env=server_env(self.config.env),
                cwd=self.config.cwd,
                limit=STDIO_LINE_LIMIT,
                start_new_session=True,  # Ctrl-C in the terminal must not kill the server
            )
        except OSError as exc:
            raise McpError(f"cannot start {self.config.command}: {exc}") from exc
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        while True:
            try:
                line = await self._process.stdout.readline()
            except (ValueError, OSError):
                break
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # a server that prints a banner on stdout
            if not isinstance(message, dict):
                continue
            if "method" in message and "id" in message:
                await self._answer_server_request(message)
            elif "id" in message:
                future = self._pending.pop(message["id"], None)
                if future and not future.done():
                    future.set_result(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError(f"{self.config.name}: the server process ended"))
        self._pending.clear()

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        if message["method"] == "ping":
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "not supported by this client"},
            }
        await self._write(reply)

    async def _write(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin or self._process.returncode is not None:
            raise McpError(f"{self.config.name}: the server process is not running")
        self._process.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
        try:
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise McpError(f"{self.config.name}: the server closed its input") from exc

    async def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            message = await asyncio.wait_for(future, timeout)
        except (TimeoutError, asyncio.CancelledError) as exc:
            self._pending.pop(request_id, None)
            reason = "timeout" if isinstance(exc, TimeoutError) else "cancelled by the user"
            with contextlib.suppress(McpError):
                await self.notify(
                    "notifications/cancelled", {"requestId": request_id, "reason": reason}
                )
            if isinstance(exc, TimeoutError):
                raise McpError(
                    f"{self.config.name}: no answer to {method} in {timeout:.0f} s"
                ) from exc
            raise
        return unwrap(self.config.name, message)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        await self._write(message)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def close(self) -> None:
        process = self._process
        if process and process.returncode is None:
            if process.stdin:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        if self._stderr_file:
            self._stderr_file.close()
        self._process = None


class HttpConnection:
    def __init__(self, config: ServerConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._client = httpx.AsyncClient(transport=transport, headers=config.headers)
        self._session_id = ""
        self._protocol = ""
        self._next_id = 0
        self.alive = False

    async def start(self) -> None:
        self.alive = True

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol:
            headers["MCP-Protocol-Version"] = self._protocol
        return headers

    async def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        self._next_id += 1
        request_id = self._next_id
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        name = self.config.name
        try:
            async with self._client.stream(
                "POST", self.config.url, json=body, headers=self._headers(), timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode(errors="replace")[:300]
                    raise McpError(f"{name}: HTTP {response.status_code} on {method}: {text}")
                if response.headers.get("mcp-session-id"):
                    self._session_id = response.headers["mcp-session-id"]
                if "text/event-stream" in response.headers.get("content-type", ""):
                    message = await self._read_events(response, request_id)
                else:
                    message = json.loads(await response.aread())
        except httpx.TimeoutException as exc:
            raise McpError(f"{name}: no answer to {method} in {timeout:.0f} s") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise McpError(f"{name}: {type(exc).__name__}: {exc}") from exc
        result = unwrap(name, message)
        if method == "initialize" and isinstance(result, dict):
            self._protocol = str(result.get("protocolVersion") or "")
        return result

    async def _read_events(self, response: httpx.Response, request_id: int) -> dict[str, Any]:
        data: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
                continue
            if line.strip() or not data:
                continue
            payload, data = "\n".join(data), []
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        raise McpError(f"{self.config.name}: the event stream ended without an answer")

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            body["params"] = params
        try:
            await self._client.post(self.config.url, json=body, headers=self._headers(), timeout=30)
        except httpx.HTTPError as exc:
            raise McpError(f"{self.config.name}: {type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        self.alive = False
        await self._client.aclose()


def unwrap(server: str, message: dict[str, Any]) -> Any:
    if "error" in message:
        error = message["error"] or {}
        raise McpError(f"{server}: {error.get('message', 'error')} (code {error.get('code')})")
    return message.get("result")


# ---------------------------------------------------------------------- server


class McpServer:
    """One configured server. Connects and initializes on first use."""

    def __init__(
        self,
        config: ServerConfig,
        log_dir: Path | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.log_dir = log_dir
        self._http_transport = http_transport
        self._connection: StdioConnection | HttpConnection | None = None
        self._lock = asyncio.Lock()
        self.server_info: dict[str, Any] = {}
        self.instructions = ""
        self.error = ""

    @property
    def running(self) -> bool:
        return self._connection is not None and self._connection.alive

    async def _connect(self) -> StdioConnection | HttpConnection:
        async with self._lock:
            if self._connection is not None and self._connection.alive:
                return self._connection
            if self._connection is not None:
                await self._connection.close()
            connection: StdioConnection | HttpConnection
            if self.config.transport == "http":
                connection = HttpConnection(self.config, self._http_transport)
            else:
                log = self.log_dir / f"mcp-{self.config.name}.log" if self.log_dir else None
                connection = StdioConnection(self.config, log)
            try:
                await connection.start()
                result = await connection.request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": CLIENT_INFO,
                    },
                    START_TIMEOUT,
                )
                await connection.notify("notifications/initialized")
            except (McpError, asyncio.CancelledError) as exc:
                await connection.close()
                self.error = str(exc) or type(exc).__name__
                raise
            self.server_info = (result or {}).get("serverInfo") or {}
            self.instructions = str((result or {}).get("instructions") or "")
            self.error = ""
            self._connection = connection
            return connection

    async def list_tools(self) -> list[dict[str, Any]]:
        connection = await self._connect()
        tools: list[dict[str, Any]] = []
        cursor = None
        for _ in range(50):
            params = {"cursor": cursor} if cursor else {}
            result = await connection.request("tools/list", params, START_TIMEOUT) or {}
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        connection = await self._connect()
        result = await connection.request(
            "tools/call", {"name": tool, "arguments": arguments}, self.config.timeout
        )
        return result or {}

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


# --------------------------------------------------------------------- manager


class McpManager:
    def __init__(
        self,
        configs: dict[str, ServerConfig],
        log_dir: Path | None = None,
        cache_dir: Path | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.servers = {
            name: McpServer(config, log_dir, http_transport) for name, config in configs.items()
        }
        self.cache_dir = cache_dir or config_dir() / "cache" / "mcp"
        self.listings: dict[str, list[dict[str, Any]]] = {}
        self.from_cache: set[str] = set()

    def _cache_file(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def _read_cache(self, name: str) -> list[dict[str, Any]] | None:
        server = self.servers[name]
        try:
            data = json.loads(self._cache_file(name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("fingerprint") != server.config.fingerprint():
            return None
        server.instructions = server.instructions or str(data.get("instructions") or "")
        return data.get("tools")

    def _write_cache(self, name: str, tools: list[dict[str, Any]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        server = self.servers[name]
        data = {
            "fingerprint": server.config.fingerprint(),
            "instructions": server.instructions,
            "tools": tools,
        }
        self._cache_file(name).write_text(json.dumps(data), encoding="utf-8")

    async def _listing(self, name: str, refresh: bool) -> None:
        cached = None if refresh else self._read_cache(name)
        if cached is not None:
            self.listings[name] = cached
            self.from_cache.add(name)
            return
        server = self.servers[name]
        try:
            tools = await server.list_tools()
        except McpError as exc:
            server.error = str(exc)
            self.listings[name] = []
            return
        self.listings[name] = tools
        self.from_cache.discard(name)
        self._write_cache(name, tools)

    async def discover(self, refresh: bool = False) -> dict[str, Tool]:
        """Tool objects for every server that answered, keyed by ``mcp__server__tool``."""
        await asyncio.gather(*(self._listing(name, refresh) for name in self.servers))
        tools: dict[str, Tool] = {}
        for name, listing in self.listings.items():
            for entry in listing:
                tool = McpTool(self, name, entry)
                tools[tool.name] = tool
        return tools

    def status(self) -> list[str]:
        rows = []
        for name, server in self.servers.items():
            count = len(self.listings.get(name, []))
            if server.error:
                state = f"FAILED: {server.error}"
            elif server.running:
                state = "running"
            else:
                state = "not started (starts on first call)"
            source = ", listing from cache" if name in self.from_cache else ""
            rows.append(
                f"  {name:<20} {server.config.transport:<5} {count:>3} tools  {state}{source}"
            )
        return rows

    async def close(self) -> None:
        await asyncio.gather(*(server.close() for server in self.servers.values()))


# ------------------------------------------------------------------------ tool


def tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


class McpTool(Tool):
    def __init__(self, manager: McpManager, server: str, entry: dict[str, Any]) -> None:
        self.manager = manager
        self.server = server
        self.remote_name = str(entry.get("name") or "")
        self.name = tool_name(server, self.remote_name)  # type: ignore[misc]
        self.description = str(entry.get("description") or "")[:4000]  # type: ignore[misc]
        schema = entry.get("inputSchema")
        self.input_schema = (  # type: ignore[misc]
            schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
        )
        annotations = entry.get("annotations") or {}
        self.read_only = bool(annotations.get("readOnlyHint"))  # type: ignore[misc]

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await self.manager.servers[self.server].call_tool(self.remote_name, tool_input)
        except McpError as exc:
            return ToolResult(str(exc), True)
        return ToolResult(ctx.cap(render_result(result, ctx.scratch)), bool(result.get("isError")))


def render_result(result: dict[str, Any], scratch: Path) -> str:
    """Flatten MCP content blocks to text. Binary blocks are saved and named, not inlined."""
    parts: list[str] = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind in ("image", "audio"):
            parts.append(_save_binary(block, scratch))
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(str(resource.get("text") or f"[resource {resource.get('uri', '')}]"))
        elif kind == "resource_link":
            parts.append(f"[resource link {block.get('uri', '')}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False, indent=2))
    return "\n".join(parts) if parts else "(empty result)"


def _save_binary(block: dict[str, Any], scratch: Path) -> str:
    mime = str(block.get("mimeType") or "application/octet-stream")
    try:
        raw = base64.b64decode(block.get("data") or "")
    except ValueError:
        return f"[{block.get('type')} {mime}: undecodable data]"
    scratch.mkdir(parents=True, exist_ok=True)
    suffix = mime.rsplit("/", 1)[-1].split("+")[0][:8] or "bin"
    path = scratch / f"mcp-{uuid.uuid4().hex[:8]}.{suffix}"
    path.write_bytes(raw)
    return f"[{block.get('type')} {mime}, {len(raw):,} bytes, saved to {path}]"
