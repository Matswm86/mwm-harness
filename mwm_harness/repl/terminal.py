"""Plain terminal front-end: works over SSH, in tests and when the browser panel is down.

It only listens to the event bus and calls ``Session`` methods; it holds no
agent logic. Ctrl-C during a turn cancels the turn; ``/quit`` or Ctrl-D leaves.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from mwm_harness import events as ev
from mwm_harness.config import ModelSpec
from mwm_harness.loop import Session
from mwm_harness.permissions import MODES
from mwm_harness.transcript import list_sessions

DIM, RED, YELLOW, CYAN, RESET = "\033[2m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"

HELP = """\
/help                 this list
/model [id]           show or switch the model
/models               list configured models
/context  /usage      context meter and session token totals
/mode [name]          show or set the permission mode (default, acceptEdits, bypassPermissions)
/permissions          mode, sandbox state and the hard deny list
/tasks                the current task list
/hooks                registered hooks per event
/memory               the rule and memory files in the system prompt
/resume               list earlier sessions for this directory (start with: mwm --resume ID)
/clear                forget the conversation (the transcript file is kept)
/quit                 leave"""


class Printer:
    """Turns bus events into terminal output."""

    def __init__(self, color: bool) -> None:
        self.color = color
        self._mid_line = False

    def paint(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def line(self, text: str = "") -> None:
        if self._mid_line:
            print()
            self._mid_line = False
        print(text)

    def __call__(self, event: ev.Event) -> None:
        if isinstance(event, ev.TextDelta):
            print(event.text, end="", flush=True)
            self._mid_line = not event.text.endswith("\n")
        elif isinstance(event, ev.ToolStarted):
            summary = json.dumps(event.input, ensure_ascii=False)
            self.line(self.paint(CYAN, f"> {event.name} {summary[:300]}"))
        elif isinstance(event, ev.ToolFinished):
            first = event.content.strip().splitlines()[:6]
            code = RED if event.is_error else DIM
            for row in first:
                self.line(self.paint(code, f"  {row[:200]}"))
        elif isinstance(event, ev.HookBlocked):
            self.line(self.paint(YELLOW, f"[{event.event} hook blocked] {event.reason[:600]}"))
        elif isinstance(event, ev.Notice):
            code = RED if event.level == "error" else YELLOW
            self.line(self.paint(code, f"[{event.level}] {event.text}"))
        elif isinstance(event, ev.TodosUpdated):
            self.line(format_todos(event.todos))
        elif isinstance(event, ev.UsageUpdated):
            self.line(
                self.paint(
                    DIM,
                    f"[ctx {event.context_fraction:.1%} | in {event.prompt_tokens:,} "
                    f"out {event.completion_tokens:,} cached {event.cached_tokens:,}]",
                )
            )
        elif isinstance(event, ev.TurnEnded) and event.reason != "done":
            self.line(self.paint(YELLOW, f"[turn ended: {event.reason}]"))


def format_todos(todos: list[dict[str, Any]]) -> str:
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    rows = [
        f"  {marks.get(item.get('status', ''), '[?]')} {item.get('content', '')}" for item in todos
    ]
    return "\n".join(rows) if rows else "  (no tasks)"


class TerminalApprover:
    def __init__(self, session_getter: Any) -> None:
        self._session = session_getter

    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool:
        print(f"\n{YELLOW}Approve {tool_name}?{RESET} ({reason})")
        print(json.dumps(tool_input, indent=2, ensure_ascii=False)[:2000])
        answer = await asyncio.to_thread(input, "[y] yes  [a] always this session  [n] no > ")
        answer = answer.strip().lower()
        if answer == "a":
            self._session().permissions.session_allow.add(tool_name)
            return True
        return answer in ("y", "yes")


def run_command(session: Session, models: dict[str, ModelSpec], text: str, out: Printer) -> bool:
    """Handle one slash command. Returns False when the REPL should exit."""
    name, _, argument = text.partition(" ")
    argument = argument.strip()
    if name in ("/quit", "/exit"):
        return False
    if name == "/help":
        out.line(HELP)
    elif name == "/models":
        for model_id, spec in models.items():
            mark = "*" if model_id == session.model.id else " "
            out.line(f" {mark} {model_id}  ({spec.family}, window {spec.context_window:,})")
    elif name == "/model":
        if not argument:
            out.line(session.model.id)
        elif argument in models:
            session.set_model(models[argument])
            out.line(f"model is now {argument}")
        else:
            out.line(f"unknown model {argument}; see /models")
    elif name in ("/context", "/usage"):
        out.line(session.meter.line())
    elif name == "/mode":
        if argument in MODES:
            session.set_mode(argument)
        elif argument:
            out.line(f"unknown mode; choose one of {', '.join(MODES)}")
        out.line(f"permission mode: {session.permissions.mode}")
    elif name == "/permissions":
        state = "on (bubblewrap)" if session.tool_ctx.sandbox.enabled else "OFF"
        out.line(f"mode: {session.permissions.mode} | shell sandbox: {state}")
        for rule in session.permissions.rules:
            out.line(f"  deny {rule.id}: {rule.why}")
    elif name == "/tasks":
        out.line(format_todos(session.tool_ctx.todos))
    elif name == "/hooks":
        for registration in session.hooks.registrations:
            matcher = registration.matcher or "*"
            out.line(f"  {registration.event:<17} {matcher[:30]:<30} {registration.command[-70:]}")
        out.line(f"{len(session.hooks.registrations)} hook registrations")
    elif name == "/memory":
        out.line(f"system prompt: {len(session.system_prompt):,} characters")
        for row in session.system_prompt.splitlines():
            if row.startswith(("# Rules from", "# Memory index")):
                out.line(f"  {row[2:]}")
    elif name == "/resume":
        for path in list_sessions(session.cwd)[:15]:
            out.line(f"  {path.stem}  ({path.stat().st_size:,} bytes)")
    elif name == "/clear":
        session.messages.clear()
        session.transcript.note("cleared", "conversation cleared by the user")
        out.line("conversation cleared")
    else:
        out.line(f"unknown command {name}; try /help")
    return True


async def repl(session: Session, models: dict[str, ModelSpec]) -> None:
    out = Printer(color=sys.stdout.isatty())
    session.bus.subscribe(out)
    await session.start()
    sandbox = "on" if session.tool_ctx.sandbox.enabled else "OFF"
    out.line(
        f"MWM Harness | {session.model.id} | mode {session.permissions.mode} | "
        f"sandbox {sandbox} | {len(session.hooks.registrations)} hooks | /help"
    )
    loop = asyncio.get_running_loop()

    def on_interrupt() -> None:
        if not session.cancel():
            out.line("(no turn is running; /quit or Ctrl-D leaves)")

    loop.add_signal_handler(signal.SIGINT, on_interrupt)
    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if not text:
                continue
            if text.startswith("/"):
                if not run_command(session, models, text, out):
                    break
                continue
            await session.send(text)
            out.line()
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await session.close()


def resolve_session(cwd: Path, token: str) -> Path | None:
    """Find an earlier transcript by id prefix; ``last`` means the newest one."""
    sessions = list_sessions(cwd)
    if token == "last":
        return sessions[0] if sessions else None
    return next((p for p in sessions if p.stem.startswith(token)), None)
