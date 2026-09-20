"""Browser panel front-end: one page, one websocket, the same ``Session`` as the terminal.

Like the terminal REPL it holds no agent logic: it forwards bus events to the
page and calls ``Session`` methods for what the page sends back.

Who may connect: the server binds to 127.0.0.1 only; every request must name
this host in its ``Host`` header (blocks DNS rebinding), a websocket with an
``Origin`` header must come from this page, and the websocket needs the token
that is generated for each launch and printed once in the terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hmac
import secrets
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from mwm_harness import events as ev
from mwm_harness.config import ModelSpec
from mwm_harness.loop import Session
from mwm_harness.preview import OutsideProject, list_dir, preview_change, read_file
from mwm_harness.repl.terminal import run_async_command, run_command
from mwm_harness.web.vendor import monaco_dir, monaco_ready

STATIC = Path(__file__).parent / "static"
POLICY_VIOLATION = 1008
CDN = "https://cdn.jsdelivr.net"  # the code viewer (Monaco) loads from here; see index.html
QUIET_EVENTS = (ev.HookContext,)  # large and only useful in the transcript


class HostGuard:
    """Refuse any request whose Host header is not this server's own address."""

    def __init__(self, app: ASGIApp, allowed_hosts: set[str]) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            host = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
            if host not in self.allowed_hosts:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": POLICY_VIOLATION})
                else:
                    await PlainTextResponse("unknown host", 421)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class CapturedOutput:
    """Stands in for the terminal printer so slash commands can answer into the page."""

    def __init__(self) -> None:
        self.rows: list[str] = []

    def line(self, text: str = "") -> None:
        self.rows.append(text)


class Panel:
    """The state shared by all open pages of one session."""

    def __init__(self, session: Session, models: dict[str, ModelSpec], token: str) -> None:
        self.session = session
        self.models = models
        self.token = token
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.pending: dict[str, tuple[dict[str, Any], asyncio.Future[str]]] = {}
        self.turn: asyncio.Task[Any] | None = None
        session.bus.subscribe(self.on_event)
        session.approver = self

    # ----------------------------------------------------------- core -> page

    def on_event(self, event: ev.Event) -> None:
        if isinstance(event, QUIET_EVENTS):
            return
        payload = {"type": type(event).__name__, **dataclasses.asdict(event)}
        if isinstance(event, ev.UsageUpdated):
            payload.update(self.usage())
        self.broadcast(payload)

    def broadcast(self, payload: dict[str, Any]) -> None:
        for queue in self.clients:
            queue.put_nowait(payload)

    def usage(self) -> dict[str, Any]:
        meter = self.session.meter
        return {
            "prompt_tokens": meter.prompt_tokens,
            "completion_tokens": meter.completion_tokens,
            "cached_tokens": meter.cached_tokens,
            "context_window": meter.context_window,
            "soft_budget": meter.soft_budget,
            "context_used": meter.used,
            "context_fraction": meter.fraction,
            "total_prompt": meter.total_prompt,
            "total_completion": meter.total_completion,
            "requests": meter.requests,
        }

    def state(self) -> dict[str, Any]:
        session = self.session
        return {
            "type": "State",
            "model": session.model.id,
            "models": list(self.models),
            "mode": session.permissions.mode,
            "cwd": str(session.cwd),
            "session_id": session.transcript.session_id,
            "sandbox": session.tool_ctx.sandbox.enabled,
            "busy": self.busy,
            "todos": list(session.tool_ctx.todos),
            "plan": session.plan,
            "touched": list(session.touched),
            "usage": self.usage(),
            "history": [
                {"role": m.role, "blocks": m.blocks()} for m in session.messages if not m.is_meta
            ],
            "approvals": [request for request, _ in self.pending.values()],
        }

    # ----------------------------------------------------------- approvals

    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool:
        """The ``Approver`` protocol: show a dialog in every open page and wait."""
        request_id = uuid.uuid4().hex[:12]
        request = {
            "type": "ApprovalRequest",
            "id": request_id,
            "tool": tool_name,
            "input": tool_input,
            "reason": reason,
        }
        change = preview_change(tool_name, tool_input, self.session.tool_ctx)
        if change is not None:
            request["diff"] = {**dataclasses.asdict(change), "unified": change.unified()}
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (request, future)
        self.broadcast(request)
        try:
            answer = await future
        finally:
            self.pending.pop(request_id, None)
            self.broadcast({"type": "ApprovalResolved", "id": request_id})
        if answer == "always":
            self.session.permissions.session_allow.add(tool_name)
        return answer in ("yes", "always")

    # ----------------------------------------------------------- page -> core

    @property
    def busy(self) -> bool:
        return self.turn is not None and not self.turn.done()

    async def handle(self, message: dict[str, Any], reply: asyncio.Queue[dict[str, Any]]) -> None:
        kind = message.get("type")
        if kind == "prompt":
            text = str(message.get("text") or "").strip()
            if text.startswith("/"):
                text = await self.command(text, reply)
            if text:
                self.start_turn(text, reply)
        elif kind == "cancel":
            self.session.cancel()
        elif kind == "approval":
            entry = self.pending.get(str(message.get("id")))
            answer = str(message.get("answer"))
            if entry and not entry[1].done() and answer in ("yes", "always", "no"):
                entry[1].set_result(answer)
        elif kind == "state":
            reply.put_nowait(self.state())
        elif kind in ("tree", "open"):
            relative = str(message.get("path") or "")
            try:
                if kind == "tree":
                    entries = list_dir(self.session.cwd, relative)
                    reply.put_nowait({"type": "Tree", "path": relative, "entries": entries})
                else:
                    reply.put_nowait(
                        {"type": "FileContent", **read_file(self.session.cwd, relative)}
                    )
            except OutsideProject:
                reply.put_nowait(
                    {"type": "FileContent", "path": relative, "error": "outside the project"}
                )

    def start_turn(self, text: str, reply: asyncio.Queue[dict[str, Any]]) -> None:
        if self.busy:
            reply.put_nowait(
                {"type": "Notice", "level": "warn", "text": "a turn is running; cancel it first"}
            )
            return
        self.turn = asyncio.create_task(self.session.send(text))

    async def command(self, text: str, reply: asyncio.Queue[dict[str, Any]]) -> str:
        """Run a slash command. Returns the prompt it stands for, or an empty string."""
        words = text.split()
        if words[0] in ("/quit", "/exit"):
            reply.put_nowait(
                {"type": "CommandOutput", "command": text, "text": "close the tab; Ctrl-C ends mwm"}
            )
            return ""
        if words[0] == "/open" and len(words) > 1:
            await self.handle({"type": "open", "path": text.split(None, 1)[1]}, reply)
            return ""
        out = CapturedOutput()
        if self.busy and words[0] == "/compact":
            out.line("a turn is running; /compact works between turns")
        elif await run_async_command(self.session, text, out):
            self.broadcast(self.state())
        if out.rows:
            reply.put_nowait(
                {"type": "CommandOutput", "command": text, "text": "\n".join(out.rows)}
            )
            return ""
        outcome = run_command(self.session, self.models, text, out)  # type: ignore[arg-type]
        if out.rows:
            reply.put_nowait(
                {"type": "CommandOutput", "command": text, "text": "\n".join(out.rows)}
            )
        if words[0] == "/clear":
            self.broadcast(self.state())
        return outcome if isinstance(outcome, str) else ""

    async def shutdown(self) -> None:
        self.session.cancel()
        for _, future in list(self.pending.values()):
            if not future.done():
                future.set_result("no")
        if self.turn is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.turn


def create_app(
    session: Session,
    models: dict[str, ModelSpec],
    token: str,
    port: int,
    manage_session: bool = True,
) -> FastAPI:
    panel = Panel(session, models, token)
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_session:
            await session.start()
        yield
        await panel.shutdown()
        if manage_session:
            await session.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.panel = panel

    # With the code viewer fetched once (mwm --vendor-monaco) the page needs no other
    # host at all, so the CDN leaves the policy; the page tries /vendor first either way.
    vendored = monaco_ready()
    cdn = "" if vendored else f" {CDN}"
    csp = (
        f"default-src 'self'; script-src 'self' 'unsafe-inline'{cdn}; "
        f"style-src 'self' 'unsafe-inline'{cdn}; font-src 'self'{cdn} data:; "
        f"worker-src blob:; connect-src 'self'{cdn} "
        f"ws://127.0.0.1:{port} ws://localhost:{port}; frame-ancestors 'none'"
    )
    if vendored:
        app.mount("/vendor/monaco", StaticFiles(directory=monaco_dir()), name="monaco")

    @app.get("/")
    async def index() -> Response:
        # The page holds no data; everything arrives over the token-checked websocket.
        return FileResponse(
            STATIC / "index.html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": csp,
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        offered = websocket.query_params.get("token", "")
        if (origin is not None and origin not in origins) or not hmac.compare_digest(
            offered.encode(), token.encode()
        ):
            await websocket.close(POLICY_VIOLATION)
            return
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        queue.put_nowait(panel.state())
        panel.clients.add(queue)

        async def pump() -> None:
            while True:
                await websocket.send_json(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            while True:
                message = await websocket.receive_json()
                if isinstance(message, dict):
                    await panel.handle(message, queue)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            panel.clients.discard(queue)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender

    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    app.add_middleware(HostGuard, allowed_hosts=hosts)
    return app


def new_token() -> str:
    return secrets.token_urlsafe(24)


def app_window_command(url: str) -> list[str] | None:
    """A browser command that shows ``url`` as its own window, without tabs or an address bar."""
    for name in ("chromium", "chromium-browser", "google-chrome", "brave-browser"):
        if shutil.which(name):
            return [name, f"--app={url}"]
    if shutil.which("flatpak"):
        listed = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        for app_id in ("com.google.Chrome", "org.chromium.Chromium", "com.brave.Browser"):
            if app_id in listed:
                return ["flatpak", "run", app_id, f"--app={url}"]
    if shutil.which("xdg-open"):
        return ["xdg-open", url]
    return None


async def serve(
    session: Session, models: dict[str, ModelSpec], port: int, open_window: bool = False
) -> None:
    import uvicorn

    token = new_token()
    app = create_app(session, models, token, port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets")
    url = f"http://127.0.0.1:{port}/#token={token}"
    print(f"MWM Harness panel: {url}")
    print("The address holds this launch's access token. Ctrl-C ends the session.")
    server = uvicorn.Server(config)
    if open_window:
        # Only on request (mwm --open, the desktop launcher): the person asked for a window.
        command = app_window_command(url)

        async def show() -> None:
            while not server.started:
                await asyncio.sleep(0.05)
            if command is None:
                print("no browser found to open; use the address above")
                return
            subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        opener = asyncio.create_task(show())
        await server.serve()
        opener.cancel()
        return
    await server.serve()
