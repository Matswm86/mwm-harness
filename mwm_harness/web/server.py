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
import base64
import binascii
import contextlib
import dataclasses
import hmac
import itertools
import secrets
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from mwm_harness import events as ev
from mwm_harness.config import ModelSpec, config_dir
from mwm_harness.loop import Session
from mwm_harness.mcp_client import McpError, McpManager
from mwm_harness.memory_panel import MemorySearch, OutsideMemory, list_handoffs, read_memory
from mwm_harness.mods import Mod, ModError, ModSet, fill, load_mods, tab_values, why_not
from mwm_harness.preview import OutsideProject, list_dir, preview_change, read_file
from mwm_harness.repl.terminal import run_async_command, run_command
from mwm_harness.voice import Transcriber
from mwm_harness.web.market import MarketFeed
from mwm_harness.web.vendor import design_dir, design_ready, monaco_dir, monaco_ready

STATIC = Path(__file__).parent / "static"
POLICY_VIOLATION = 1008
MAX_AUDIO_B64 = 8 * 1024 * 1024 * 4 // 3  # an 8 MB clip, base64-encoded
CDN = "https://cdn.jsdelivr.net"  # code viewer, 3D library and fonts load from here; see index.html
QUIET_EVENTS = (ev.HookContext,)  # large and only useful in the transcript
LIVE_EVENTS = (
    ev.TextDelta,
    ev.ReasoningDelta,
    ev.ToolArgsDelta,
    ev.ToolStarted,
    ev.ToolFinished,
    ev.Notice,
    ev.HookBlocked,
    ev.SubagentStarted,
    ev.SubagentFinished,
)
MAX_LIVE_EVENTS = 2000
SIDE_COMMANDS = ("/check", "/runit", "/trace")  # slash commands that run as background tasks
CLOSED_TABS_KEPT = 10


def mcp_servers(manager: McpManager | None) -> list[dict[str, Any]]:
    """One row per configured MCP server, for the panel's rail and scene."""
    if manager is None:
        return []
    rows = []
    for name, server in manager.servers.items():
        state = "failed" if server.error else "running" if server.running else "idle"
        rows.append(
            {
                "name": name,
                "transport": server.config.transport,
                "tools": len(manager.listings.get(name, [])),
                "state": state,
                "error": server.error,
            }
        )
    return rows


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


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "(no output)")[:300]


class WorktreeError(Exception):
    pass


async def _git(cwd: Path, *args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "git", "-C", str(cwd), *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = await process.communicate()
    if process.returncode != 0:
        raise WorktreeError(err.decode(errors="replace").strip() or f"git {args[0]} failed")
    return out.decode().strip()


async def make_worktree(cwd: Path, root: Path, number: int) -> tuple[Path, str]:
    """A new git worktree of ``cwd``'s repository on its own branch; returns (folder, branch)."""
    top = Path(await _git(cwd, "rev-parse", "--show-toplevel"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    branch = f"mwm/tab{number}-{stamp}"
    target = root / f"{top.name}-tab{number}-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    await _git(top, "worktree", "add", "-b", branch, str(target), "HEAD")
    return target / cwd.relative_to(top), branch


class Tab:
    """One session in the panel: its own transcript, model, mode, tasks and approvals."""

    def __init__(self, panel: Panel, tab_id: str, session: Session, label: str) -> None:
        self.panel = panel
        self.id = tab_id
        self.session = session
        self.label = label
        self.pending: dict[str, tuple[dict[str, Any], asyncio.Future[str]]] = {}
        self.turn: asyncio.Task[Any] | None = None
        self.ended = False  # TurnEnded seen while the turn's task is still finishing
        # The running turn as the page saw it, so a page that switches here mid-turn can replay it.
        self.live: list[dict[str, Any]] = []
        self.turn_base = 0  # messages that existed before the running turn began
        self.worktree = ""  # set when the tab works in its own git worktree
        self.written: set[Path] = set()  # files this tab's Write/Edit calls changed
        self._writing: dict[str, str] = {}
        self.checks: dict[str, Any] | None = None  # the last ChecksFinished, for State
        self.runit: dict[str, Any] | None = None  # the last RunItFinished, for State
        self._side_tasks: set[asyncio.Task[None]] = set()
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
        self.remember(event, payload)
        if isinstance(event, ev.ChecksFinished):
            self.checks = payload
        elif isinstance(event, ev.RunItFinished):
            self.runit = payload
        if isinstance(event, ev.ToolStarted) and event.name in ("Write", "Edit", "MultiEdit"):
            self._writing[event.tool_use_id] = str(event.input.get("file_path") or "")
        elif isinstance(event, ev.ToolFinished) and event.tool_use_id in self._writing:
            path = self._writing.pop(event.tool_use_id)
            if path and not event.is_error:
                self.written.add(self.session.tool_ctx.resolve(path).resolve())
        if isinstance(event, ev.TurnStarted | ev.TurnEnded):
            self.ended = isinstance(event, ev.TurnEnded)
            self.panel.broadcast(self.panel.tabs_payload())

    def broadcast(self, payload: dict[str, Any]) -> None:
        self.panel.broadcast({**payload, "tab": self.id})

    def remember(self, event: ev.Event, payload: dict[str, Any]) -> None:
        if isinstance(event, ev.TurnStarted):
            self.turn_base = len(self.session.messages)
            self.live = [payload]
        elif isinstance(event, ev.TurnEnded):
            self.live = []
        elif self.live and isinstance(event, LIVE_EVENTS):
            last = self.live[-1]
            if (
                isinstance(event, ev.TextDelta | ev.ReasoningDelta)
                and last["type"] == payload["type"]
            ):
                last["text"] += payload["text"]  # one entry per streamed block, not per chunk
            elif (
                isinstance(event, ev.ToolArgsDelta)
                and last["type"] == payload["type"]
                and last["index"] == payload["index"]
            ):
                last["fragment"] += payload["fragment"]  # one entry per call being written
                last["call_id"] = last["call_id"] or payload["call_id"]
                last["name"] = last["name"] or payload["name"]
            elif len(self.live) < MAX_LIVE_EVENTS:
                self.live.append(payload)
        if isinstance(event, ev.Compacted):
            self.turn_base = min(self.turn_base, len(self.session.messages))

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
            "tab": self.id,
            "label": self.label,
            "model": session.model.id,
            "models": list(self.panel.models),
            "mode": session.permissions.mode,
            "cwd": str(session.cwd),
            "session_id": session.transcript.session_id,
            "sandbox": session.tool_ctx.sandbox.enabled,
            "busy": self.busy,
            "todos": list(session.tool_ctx.todos),
            "plan": session.plan,
            "touched": list(session.touched),
            "usage": self.usage(),
            # While a turn runs, its own messages come as ``live`` events instead.
            "history": [
                {"role": m.role, "blocks": m.blocks(), "ts": m.ts}
                for m in (session.messages[: self.turn_base] if self.live else session.messages)
                if not m.is_meta
            ],
            "live": list(self.live),
            "worktree": self.worktree,
            "mods": self.mods_payload(""),
            "approvals": [request for request, _ in self.pending.values()],
            "mcp": mcp_servers(session.mcp or self.panel.mcp),
            "checks": self.checks,
            "runit": self.runit,
        }

    def stop(self) -> bool:
        """Emergency stop for this tab: cancel the turn (its shell tree dies with it),
        refuse every open approval, cancel test runs. True when something was running."""
        hit = bool(self.pending) or any(not t.done() for t in self._side_tasks)
        for _, future in list(self.pending.values()):
            if not future.done():
                future.set_result("no")
        for task in list(self._side_tasks):
            task.cancel()
        if self.session.cancel():
            hit = True
        elif self.turn is not None and not self.turn.done():
            self.turn.cancel()
            hit = True
        return hit

    # ----------------------------------------------------------- approvals

    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool:
        """The ``Approver`` protocol: show a dialog in every open page and wait."""
        request_id = uuid.uuid4().hex[:12]
        request = {
            "type": "ApprovalRequest",
            "tab": self.id,
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
        self.panel.broadcast(self.panel.tabs_payload())
        try:
            answer = await future
        finally:
            self.pending.pop(request_id, None)
            self.broadcast({"type": "ApprovalResolved", "id": request_id})
            self.panel.broadcast(self.panel.tabs_payload())
        if answer == "always":
            self.session.permissions.session_allow.add(tool_name)
        return answer in ("yes", "always")

    # ----------------------------------------------------------- page -> core

    @property
    def busy(self) -> bool:
        return self.turn is not None and not self.turn.done() and not self.ended

    async def handle(self, message: dict[str, Any], reply: asyncio.Queue[dict[str, Any]]) -> None:
        kind = message.get("type")
        if kind == "prompt":
            text = str(message.get("text") or "").strip()
            if text.startswith("/"):
                text = await self.command(text, reply)
            if text:
                self.start_turn(text, reply)
        elif kind == "cancel":
            # A turn still queued for the local model has no session turn yet: drop the task.
            if not self.session.cancel() and self.turn is not None and not self.turn.done():
                self.turn.cancel()
        elif kind == "approval":
            entry = self.pending.get(str(message.get("id")))
            answer = str(message.get("answer"))
            if entry and not entry[1].done() and answer in ("yes", "always", "no"):
                entry[1].set_result(answer)
        elif kind == "state":
            reply.put_nowait(self.state())
        elif kind == "mods":
            reply.put_nowait(
                {
                    "type": "Mods",
                    "tab": self.id,
                    "mods": self.mods_payload(str(message.get("prompt") or "")),
                }
            )
        elif kind == "mod":
            await self.run_mod(
                str(message.get("key") or ""), str(message.get("prompt") or ""), reply
            )
        elif kind in ("tree", "open"):
            relative = str(message.get("path") or "")
            try:
                if kind == "tree":
                    entries = list_dir(self.session.cwd, relative)
                    reply.put_nowait(
                        {"type": "Tree", "tab": self.id, "path": relative, "entries": entries}
                    )
                else:
                    reply.put_nowait(
                        {
                            "type": "FileContent",
                            "tab": self.id,
                            **read_file(self.session.cwd, relative),
                        }
                    )
            except OutsideProject:
                reply.put_nowait(
                    {
                        "type": "FileContent",
                        "tab": self.id,
                        "path": relative,
                        "error": "outside the project",
                    }
                )

    # ----------------------------------------------------------------- mods

    def mod_values(self, prompt: str) -> dict[str, str]:
        return tab_values(self.panel.mods(), self.session.cwd, self.label, self.written, prompt)

    def mods_payload(self, prompt: str) -> list[dict[str, Any]]:
        try:
            mods = self.panel.mods()
        except ModError as exc:
            return [
                {
                    "key": "",
                    "label": "mods.toml",
                    "kind": "error",
                    "note": str(exc),
                    "why": str(exc),
                }
            ]
        values = self.mod_values(prompt)
        return [
            {
                "key": m.key,
                "label": m.label,
                "kind": m.kind,
                "note": m.note,
                "ask": m.ask,
                "confirm": m.confirm,
                "why": why_not(m, {**values, "prompt": "?"}),  # the prompt comes with the key press
            }
            for m in mods.mods
        ]

    async def run_mod(self, key: str, prompt: str, reply: asyncio.Queue[dict[str, Any]]) -> None:
        def refuse(text: str) -> None:
            reply.put_nowait({"type": "Notice", "tab": self.id, "level": "warn", "text": text})

        try:
            mod = self.panel.mods().get(key)
        except ModError as exc:
            refuse(str(exc))
            return
        if mod is None:
            refuse(f"no mod on {key}; Edit mods shows mods.toml")
            return
        values = self.mod_values(prompt)
        problem = why_not(mod, values)
        if problem:
            refuse(f"{mod.key} {mod.label}: {problem}")
            return
        if mod.kind == "prompt":
            self.start_turn(fill(mod.text, values, shell=False), reply)
            return
        if mod.kind == "command":
            text = mod.builtin or f"/{mod.file} {fill(mod.args, values, shell=False)}".strip()
            await self.handle({"type": "prompt", "text": text}, reply)
            return
        if self.busy:
            refuse("a turn is running; the mod waits for it to end")
            return
        self.ended = False
        self.turn = asyncio.create_task(self._mod_turn(mod, values))

    async def _mod_turn(self, mod: Mod, values: dict[str, str]) -> None:
        """A mod runs as a turn of its own: the page shows it live and keeps a result card."""
        session = self.session
        head = f"[{mod.key} {mod.label}]"
        lock = self.panel.mod_locks.setdefault(mod.lock, asyncio.Lock()) if mod.lock else None
        session.bus.emit(ev.TurnStarted(head))
        if lock is not None and lock.locked():
            text = f"SKIPPED: another tab holds the {mod.lock} lock; nothing was {mod.lock}ed here."
            session.bus.emit(ev.TextDelta(text))
            session.bus.emit(ev.TurnEnded("done", text))
            return
        try:
            async with lock or contextlib.nullcontext():
                if mod.kind == "shell":
                    command = fill(mod.run, values, shell=True)
                    result = await session.run_tool_call(
                        "Bash", {"command": command, "network": mod.network}, mod.confirm
                    )
                    text = f"{head} {'failed' if result.is_error else 'done'}: {first_line(result.content)}"
                    session.bus.emit(ev.TextDelta(text))
                    session.bus.emit(ev.TurnEnded("error" if result.is_error else "done", text))
                else:
                    subject = await self._mod_subject(mod)
                    if not subject:
                        text = f"{head}: nothing to check ({mod.on} is empty)"
                        session.bus.emit(ev.TextDelta(text))
                        session.bus.emit(ev.TurnEnded("done", text))
                        return
                    brief = fill(mod.instruction, values, shell=False) or "Review this."
                    result = await session.run_agent(
                        mod.agent, f"{brief}\n\n<input>\n{subject}\n</input>"
                    )
                    session.bus.emit(ev.TextDelta(result.content))
                    session.bus.emit(
                        ev.TurnEnded("error" if result.is_error else "done", result.content)
                    )
        except KeyError:
            text = f"{head}: no agent named {mod.agent}"
            session.bus.emit(ev.Notice("error", text))
            session.bus.emit(ev.TurnEnded("error", text))
        except asyncio.CancelledError:
            session.bus.emit(ev.TurnEnded("cancelled", f"{head} stopped"))

    async def _mod_subject(self, mod: Mod) -> str:
        if mod.on == "uncommitted_diff":
            process = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(self.session.cwd),
                "diff",
                "HEAD",
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            out, _ = await process.communicate()
            return out.decode(errors="replace")[:60_000]
        for message in reversed(self.session.messages):
            if message.role == "assistant" and not message.is_meta:
                text = "\n".join(b["text"] for b in message.blocks() if b["type"] == "text").strip()
                if text:
                    return text[:60_000]
        return ""

    def start_turn(self, text: str, reply: asyncio.Queue[dict[str, Any]]) -> None:
        if self.busy:
            reply.put_nowait(
                {
                    "type": "Notice",
                    "tab": self.id,
                    "level": "warn",
                    "text": "a turn is running; cancel it first",
                }
            )
            return
        self.ended = False
        self.turn = asyncio.create_task(self.run_turn(text))

    async def run_turn(self, text: str) -> None:
        """Run one turn; tabs on the same local model take turns, since it answers one at a time."""
        lock = self.panel.model_lock(self.session.model)
        if lock is None:
            await self.session.send(text)
            return
        if lock.locked():
            holder = self.panel.lock_holders.get(self.session.model.id)
            self.broadcast(
                {
                    "type": "Notice",
                    "level": "info",
                    "text": f"waiting for the local model {self.session.model.id}: "
                    f"tab {holder.label if holder else '?'} is using it",
                }
            )
        async with lock:
            self.panel.lock_holders[self.session.model.id] = self
            try:
                await self.session.send(text)
            finally:
                self.panel.lock_holders.pop(self.session.model.id, None)

    async def command(self, text: str, reply: asyncio.Queue[dict[str, Any]]) -> str:
        """Run a slash command. Returns the prompt it stands for, or an empty string."""
        words = text.split()

        def answer(body: str) -> None:
            reply.put_nowait(
                {"type": "CommandOutput", "tab": self.id, "command": text, "text": body}
            )

        if words[0] in ("/quit", "/exit"):
            answer("close the tab; Ctrl-C ends mwm")
            return ""
        if words[0] == "/open" and len(words) > 1:
            await self.handle({"type": "open", "path": text.split(None, 1)[1]}, reply)
            return ""
        out = CapturedOutput()
        if words[:2] == ["/stop", "all"]:
            answer(self.panel.stop_all())
            return ""
        if words[0] in SIDE_COMMANDS:
            # Test runs take minutes: run them beside the websocket reader, report by event.
            async def side() -> None:
                await run_async_command(self.session, text, out)
                if out.rows:
                    answer("\n".join(out.rows))

            task = asyncio.create_task(side())
            self._side_tasks.add(task)
            task.add_done_callback(self._side_tasks.discard)
            return ""
        if self.busy and words[0] in ("/compact", "/undo"):
            out.line(f"a turn is running; {words[0]} works between turns")
        elif await run_async_command(self.session, text, out):
            self.broadcast(self.state())
            if out.rows:
                answer("\n".join(out.rows))
            return ""  # handled, even when it printed nothing (its result came as events)
        if out.rows:
            answer("\n".join(out.rows))
            return ""
        outcome = run_command(self.session, self.panel.models, text, out)  # type: ignore[arg-type]
        if out.rows:
            answer("\n".join(out.rows))
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


SessionFactory = Callable[[Path, ModelSpec], Session]


class Panel:
    """All open pages and all tabs of one launch.

    Every message to the page carries the ``tab`` it belongs to; a message from
    the page without one goes to the first tab. Tabs share the MCP server
    processes of the first session; closing a tab never stops them.
    """

    def stop_all(self) -> str:
        """Stop every tab at once; returns the line shown to the person."""
        stopped = [tab.label for tab in self.tabs.values() if tab.stop()]
        text = f"EMERGENCY STOP: stopped {len(stopped)} of {len(self.tabs)} tab(s)" + (
            f" ({', '.join(stopped)})" if stopped else ": nothing was running"
        )
        for tab in self.tabs.values():
            tab.broadcast({"type": "Notice", "level": "error", "text": text, "ts": ev.now_stamp()})
        return text

    def __init__(
        self,
        session: Session,
        models: dict[str, ModelSpec],
        token: str,
        factory: SessionFactory | None = None,
        max_tabs: int = 6,
    ) -> None:
        self.models = models
        self.token = token
        self.max_tabs = max_tabs
        self.mcp = session.mcp
        self.market = MarketFeed(session.settings.chart_env_file)
        settings = session.settings
        self.voice = Transcriber(
            settings.voice_model, settings.voice_device, settings.voice_language
        )
        self.voice_warming: asyncio.Task[None] | None = None
        self.memory = MemorySearch(session.cwd)
        self.factory = factory or session.sibling
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.tabs: dict[str, Tab] = {}
        self.closed: list[dict[str, Any]] = []  # newest last: what Ctrl+Shift+T reopens
        self.worktree_root = Path(session.settings.worktree_dir).expanduser()
        self.model_locks: dict[str, asyncio.Lock] = {}
        self.mod_locks: dict[str, asyncio.Lock] = {}
        self._mods: tuple[float, ModSet] | None = None
        self.lock_holders: dict[str, Tab] = {}
        self._numbers = itertools.count(1)
        self.add_tab(session)

    @property
    def first(self) -> Tab:
        return next(iter(self.tabs.values()))

    @property
    def session(self) -> Session:
        return self.first.session

    def add_tab(self, session: Session, label: str = "") -> Tab:
        number = next(self._numbers)
        tab = Tab(self, f"t{number}", session, label or f"{session.cwd.name} {number}")
        self.tabs[tab.id] = tab
        return tab

    def mods(self) -> ModSet:
        """mods.toml, read again whenever the file changes on disk."""
        path = config_dir() / "mods.toml"
        stamp = path.stat().st_mtime if path.is_file() else 0.0
        if self._mods is None or self._mods[0] != stamp:
            self._mods = (stamp, load_mods(path))
        return self._mods[1]

    def model_lock(self, model: ModelSpec) -> asyncio.Lock | None:
        """One lock per model served from this machine; hosted models need none."""
        host = urlsplit(model.base_url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            return None
        return self.model_locks.setdefault(model.id, asyncio.Lock())

    def tabs_payload(self) -> dict[str, Any]:
        return {
            "type": "Tabs",
            "max": self.max_tabs,
            "tabs": [
                {
                    "id": tab.id,
                    "label": tab.label,
                    "model": tab.session.model.id,
                    "busy": tab.busy,
                    "waiting": bool(tab.pending),
                    "cwd": str(tab.session.cwd),
                    "worktree": tab.worktree,
                }
                for tab in self.tabs.values()
            ],
            "closed": [t["label"] for t in self.closed],
        }

    def broadcast(self, payload: dict[str, Any]) -> None:
        for queue in self.clients:
            queue.put_nowait(payload)

    def start_turn(self, text: str, reply: asyncio.Queue[dict[str, Any]]) -> None:
        self.first.start_turn(text, reply)

    async def handle(self, message: dict[str, Any], reply: asyncio.Queue[dict[str, Any]]) -> None:
        kind = message.get("type")
        if kind == "stop_all":
            reply.put_nowait({"type": "StopAll", "text": self.stop_all()})
            return
        if kind == "tab_new":
            await self.new_tab(message, reply)
        elif kind == "tab_close":
            await self.close_tab(str(message.get("tab")), reply)
        elif kind == "tab_rename":
            tab = self.tabs.get(str(message.get("tab")))
            label = str(message.get("label") or "").strip()[:40]
            if tab and label:
                tab.label = label
                self.broadcast(self.tabs_payload())
        elif kind == "prompt" and str(message.get("text") or "").split()[:1] == ["/tab"]:
            await self.tab_command(message, reply)
        elif kind == "mods_file":
            try:
                path = self.mods().path
                body = path.read_text(encoding="utf-8") if path else ""
            except ModError as exc:
                path, body = config_dir() / "mods.toml", str(exc)
            reply.put_nowait(
                {
                    "type": "CommandOutput",
                    "command": f"mods: {path}",
                    "text": f"edit {config_dir() / 'mods.toml'} (read again on change)\n\n{body}",
                }
            )
        elif kind == "tab_reopen":
            await self.reopen_tab(reply)
        elif kind == "tabs":
            reply.put_nowait(self.tabs_payload())
        elif kind == "voice_warm":
            self.voice_warming = asyncio.create_task(self.voice.warm())
        elif kind == "transcribe":
            reply.put_nowait(await self.transcribe(message))
        elif kind in ("handoffs", "memory_search", "memory_open"):
            reply.put_nowait(await self.memory_reply(kind, message))
        elif kind == "bars":
            minutes = message.get("minutes")
            reply.put_nowait(
                await self.market.bars(
                    str(message.get("symbol") or "MNQ"), minutes if isinstance(minutes, int) else 5
                )
            )
        else:
            tab = self.tabs.get(str(message.get("tab"))) or self.first
            await tab.handle(message, reply)

    async def new_tab(self, message: dict[str, Any], reply: asyncio.Queue[dict[str, Any]]) -> None:
        def refuse(text: str) -> None:
            reply.put_nowait({"type": "Notice", "level": "warn", "text": text})

        if len(self.tabs) >= self.max_tabs:
            refuse(f"{self.max_tabs} tabs are open, the most this panel allows (max_tabs)")
            return
        model = self.models.get(str(message.get("model") or ""), self.first.session.model)
        folder = str(message.get("cwd") or "").strip()
        cwd = Path(folder).expanduser() if folder else self.first.session.cwd
        if not cwd.is_dir():
            refuse(f"no such folder: {cwd}")
            return
        cwd, worktree = cwd.resolve(), ""
        if message.get("isolated"):
            try:
                cwd, worktree = await make_worktree(cwd, self.worktree_root, next(self._numbers))
            except WorktreeError as exc:
                refuse(f"no isolated tab: {exc}")
                return
        tab = await self.open_session(cwd, model, str(message.get("label") or ""))
        tab.worktree = worktree
        if worktree:
            tab.broadcast(
                {
                    "type": "Notice",
                    "level": "info",
                    "text": f"isolated tab: git worktree {cwd} on branch {worktree}; "
                    "it stays on disk after the tab closes (merge or remove it yourself)",
                }
            )
        elif any(t.session.cwd == tab.session.cwd for t in self.tabs.values() if t is not tab):
            tab.broadcast(
                {
                    "type": "Notice",
                    "level": "warn",
                    "text": "another tab works in this folder; edits can collide",
                }
            )
        self.broadcast(tab.state())
        self.broadcast({**self.tabs_payload(), "active": tab.id})

    async def open_session(
        self, cwd: Path, model: ModelSpec, label: str = "", resume_from: Path | None = None
    ) -> Tab:
        if resume_from is None:
            session = self.factory(cwd, model)
        else:
            session = self.factory(cwd, model, resume_from=resume_from)  # type: ignore[call-arg]
        await session.start()
        return self.add_tab(session, label)

    async def tab_command(
        self, message: dict[str, Any], reply: asyncio.Queue[dict[str, Any]]
    ) -> None:
        """``/tab``, ``/tab new [DIR]``, ``/tab N``, ``/tab close``, ``/tab reopen`` typed or spoken."""
        text = str(message.get("text") or "")
        words = text.split()
        here = self.tabs.get(str(message.get("tab"))) or self.first
        action = words[1] if len(words) > 1 else "list"
        if action == "new":
            folder = " ".join(words[2:])
            await self.new_tab({"cwd": folder, "model": here.session.model.id}, reply)
        elif action == "close":
            await self.close_tab(here.id, reply)
        elif action == "reopen":
            await self.reopen_tab(reply)
        elif action.isdigit() and 1 <= int(action) <= len(self.tabs):
            reply.put_nowait({**self.tabs_payload(), "active": list(self.tabs)[int(action) - 1]})
        else:
            rows = [
                f"{'*' if t is here else ' '} {n}  {t.label}  {t.session.cwd}  {t.session.model.id}"
                for n, t in enumerate(self.tabs.values(), 1)
            ]
            reply.put_nowait(
                {"type": "CommandOutput", "tab": here.id, "command": text, "text": "\n".join(rows)}
            )

    async def reopen_tab(self, reply: asyncio.Queue[dict[str, Any]]) -> None:
        """Ctrl+Shift+T: the newest closed tab comes back with its transcript."""
        if not self.closed:
            reply.put_nowait({"type": "Notice", "level": "info", "text": "no closed tab to reopen"})
            return
        if len(self.tabs) >= self.max_tabs:
            reply.put_nowait(
                {
                    "type": "Notice",
                    "level": "warn",
                    "text": f"{self.max_tabs} tabs are open already",
                }
            )
            return
        entry = self.closed.pop()
        model = self.models.get(entry["model"], self.first.session.model)
        transcript = Path(entry["transcript"])
        tab = await self.open_session(
            Path(entry["cwd"]), model, entry["label"], transcript if transcript.is_file() else None
        )
        tab.worktree = entry["worktree"]
        self.broadcast(tab.state())
        self.broadcast({**self.tabs_payload(), "active": tab.id})

    async def close_tab(self, tab_id: str, reply: asyncio.Queue[dict[str, Any]]) -> None:
        tab = self.tabs.get(tab_id)
        if tab is None:
            return
        if len(self.tabs) == 1:
            reply.put_nowait({"type": "Notice", "level": "warn", "text": "the last tab stays open"})
            return
        del self.tabs[tab_id]
        await tab.shutdown()
        tab.session.mcp = None  # shared with the other tabs; the panel stops it at the end
        await tab.session.close()
        self.closed.append(
            {
                "label": tab.label,
                "cwd": str(tab.session.cwd),
                "model": tab.session.model.id,
                "transcript": str(tab.session.transcript.path),
                "worktree": tab.worktree,
            }
        )
        del self.closed[:-CLOSED_TABS_KEPT]
        self.broadcast({"type": "TabClosed", "tab": tab_id})
        self.broadcast(self.tabs_payload())

    def snapshot(self) -> list[dict[str, Any]]:
        """What a page that just connected needs: every tab's state, then the tab list."""
        return [tab.state() for tab in self.tabs.values()] + [self.tabs_payload()]

    async def shutdown(self, close_sessions: bool) -> None:
        for tab in list(self.tabs.values()):
            await tab.shutdown()
            if close_sessions:
                tab.session.mcp = None
                await tab.session.close()
        if close_sessions and self.mcp is not None:
            await self.mcp.close()
        await self.market.close()
        await self.voice.close()
        await self.memory.close()

    async def memory_reply(self, kind: str, message: dict[str, Any]) -> dict[str, Any]:
        """Handoffs rail: newest handoffs, archive search, and read-only memory files."""
        if kind == "handoffs":
            try:
                return {"type": "Handoffs", "items": list_handoffs()}
            except OSError as exc:
                return {"type": "Handoffs", "items": [], "error": str(exc)}
        if kind == "memory_open":
            relative = str(message.get("path") or "")
            try:
                return {"type": "FileContent", **read_memory(relative)}
            except (OutsideMemory, OSError):
                return {"type": "FileContent", "path": relative, "error": "outside memory/"}
        query = str(message.get("query") or "").strip()[:300]
        if not query:
            return {"type": "MemoryHits", "query": "", "hits": []}
        try:
            hits = await self.memory.search(query)
        except (McpError, OSError, ValueError) as exc:
            return {
                "type": "MemoryHits",
                "query": query,
                "hits": [],
                "error": str(exc) or type(exc).__name__,
            }
        return {"type": "MemoryHits", "query": query, "hits": hits}

    async def transcribe(self, message: dict[str, Any]) -> dict[str, Any]:
        audio = str(message.get("audio") or "")
        if not audio or len(audio) > MAX_AUDIO_B64:
            return {"type": "Transcript", "text": "", "error": "no audio, or a clip over 8 MB"}
        try:
            blob = base64.b64decode(audio, validate=True)
        except binascii.Error:
            return {"type": "Transcript", "text": "", "error": "the clip was not valid base64"}
        return {"type": "Transcript", **dataclasses.asdict(await self.voice.transcribe(blob))}


def create_app(
    session: Session,
    models: dict[str, ModelSpec],
    token: str,
    port: int,
    manage_session: bool = True,
    factory: SessionFactory | None = None,
) -> FastAPI:
    panel = Panel(session, models, token, factory, session.settings.max_tabs)
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_session:
            await session.start()
        yield
        await panel.shutdown(close_sessions=manage_session)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.panel = panel

    # With the code viewer, 3D library and fonts fetched once (mwm --vendor-monaco) the page
    # needs no other host at all, so the CDN leaves the policy; the page tries /vendor first.
    vendored = monaco_ready() and design_ready()
    cdn = "" if vendored else f" {CDN}"
    csp = (
        f"default-src 'self'; script-src 'self' 'unsafe-inline'{cdn}; "
        f"style-src 'self' 'unsafe-inline'{cdn}; font-src 'self'{cdn} data:; "
        f"worker-src blob:; connect-src 'self'{cdn} "
        f"ws://127.0.0.1:{port} ws://localhost:{port}; frame-ancestors 'none'"
    )
    if monaco_ready():
        app.mount("/vendor/monaco", StaticFiles(directory=monaco_dir()), name="monaco")
    if design_ready():
        app.mount("/vendor/design", StaticFiles(directory=design_dir()), name="design")

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
        for payload in panel.snapshot():
            queue.put_nowait(payload)
        panel.clients.add(queue)

        async def pump() -> None:
            while True:
                payload = await queue.get()
                payload.setdefault("ts", ev.now_stamp())  # panel-made replies carry a time too
                await websocket.send_json(payload)

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
