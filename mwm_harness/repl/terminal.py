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
from mwm_harness.jev import format_report, report
from mwm_harness.loop import Session
from mwm_harness.permissions import MODES
from mwm_harness.preview import OutsideProject, preview_change, read_file
from mwm_harness.transcript import list_sessions

DIM, RED, YELLOW, CYAN, RESET = "\033[2m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"

HELP = """\
/help                 this list
/model [id]           show or switch the model
/models               list configured models
/context  /usage      context meter and session token totals
/mode [name]          show or set the permission mode (default, acceptEdits, bypassPermissions, plan)
/plan [off]           plan mode: reading tools only until you approve a plan; shows the last plan
/permissions          mode, sandbox state and the hard deny list
/tasks                the current task list
/hooks                registered hooks per event
/memory               the rule and memory files in the system prompt
/mcp [refresh]        MCP servers, their state and tool counts; refresh re-lists the tools
/skills               skills the model can load; /NAME [arguments] runs one
/commands             command files; /NAME [arguments] runs one
/files                files the tools have read or written this session
/open PATH            show a project file with line numbers
/agents               subagents the model can start with the Task tool
/jev                  Jev's decision log: calls, known outcomes, hit rate per domain
/compact [focus]      replace the history with a summary written by the model
/init                 have the model write AGENTS.md (project rules) for this directory
/resume               list earlier sessions for this directory (start with: mwm --resume ID)
/clear                forget the conversation (the transcript file is kept)
/quit                 leave"""


INIT_PROMPT = """\
Study this project (README, build and test configuration, source layout, existing rule files) \
and write AGENTS.md in the project root: what the project is, how to build, test and lint it \
(exact commands), the code conventions that are visible in the source, and anything a newcomer \
would get wrong. Under 60 lines, facts only, nothing generic. If AGENTS.md exists, improve it."""


async def run_async_command(session: Session, text: str, out: Any) -> bool:
    """Slash commands that have to await something. Returns True when ``text`` was one."""
    name, _, argument = text.partition(" ")
    if name == "/compact":
        before = len(session.messages)
        summary = await session.compact("manual", argument.strip())
        if summary:
            out.line(f"compacted {before} messages into a summary of {len(summary):,} characters")
        elif not before:
            out.line("nothing to compact")
        return True
    if name == "/mcp" and "refresh" in argument.split():
        await session.connect_mcp(refresh=True)
    return False


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
        change = preview_change(tool_name, tool_input, self._session().tool_ctx)
        if change is not None:
            print(change.unified()[:6000] or "(no change)")
        else:
            print(json.dumps(tool_input, indent=2, ensure_ascii=False)[:2000])
        answer = await asyncio.to_thread(input, "[y] yes  [a] always this session  [n] no > ")
        answer = answer.strip().lower()
        if answer == "a":
            self._session().permissions.session_allow.add(tool_name)
            return True
        return answer in ("y", "yes")


def run_command(
    session: Session, models: dict[str, ModelSpec], text: str, out: Printer
) -> bool | str:
    """Handle one slash command.

    Returns False when the REPL should exit, True when the command is done, and
    a string when the command stands for a prompt that goes to the model.
    """
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
    elif name == "/plan":
        if argument == "off":
            session.set_mode("default")
        elif session.permissions.mode != "plan":
            session.set_mode("plan")
        out.line(f"permission mode: {session.permissions.mode}")
        if session.plan:
            out.line(session.plan)
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
    elif name == "/agents":
        for agent in session.agents.values():
            tools = ", ".join(agent.tools) if agent.tools else "all tools"
            out.line(f"  {agent.name:<32} model {agent.model or 'inherit':<8} {tools[:70]}")
        out.line(f"{len(session.agents)} agents")
    elif name == "/init":
        return INIT_PROMPT + (f"\n\nExtra instructions: {argument}" if argument else "")
    elif name == "/jev":
        out.line(format_report(report()))
    elif name == "/files":
        for path in session.touched:
            out.line(f"  {path}")
        out.line(f"{len(session.touched)} files touched this session")
    elif name == "/open":
        try:
            shown = read_file(session.cwd, argument)
        except OutsideProject:
            shown = {"error": "outside the project directory"}
        if "error" in shown:
            out.line(f"cannot open {argument or '(no path)'}: {shown['error']}")
        else:
            rows = shown["content"].splitlines()
            for number, row in enumerate(rows[:400], 1):
                out.line(f"{number:>5}  {row}")
            if len(rows) > 400:
                out.line(f"[{len(rows) - 400} more lines]")
    elif name == "/mcp":
        if session.mcp is None:
            out.line("MCP is off (mcp_enabled = false, --no-mcp, or no .mcp.json found)")
        else:
            for row in session.mcp.status():
                out.line(row)
    elif name == "/skills":
        for skill in session.skills.values():
            out.line(f"  {skill.name:<28} {skill.description[:90]}")
        out.line(f"{len(session.skills)} skills")
    elif name == "/commands":
        for command in session.commands.values():
            out.line(f"  /{command.name:<27} {command.description[:90]}")
        out.line(f"{len(session.commands)} command files")
    elif name[1:] in session.commands:
        return session.commands[name[1:]].render(argument)
    elif name[1:] in session.skills:
        skill = session.skills[name[1:]]
        tail = f" Arguments: {argument}" if argument else ""
        return f"Load the skill {skill.name} with the Skill tool and follow it.{tail}"
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
        f"sandbox {sandbox} | {len(session.hooks.registrations)} hooks | "
        f"{sum(1 for name in session.tools if name.startswith('mcp__'))} MCP tools | "
        f"{len(session.skills)} skills | /help"
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
                if await run_async_command(session, text, out):
                    continue
                outcome = run_command(session, models, text, out)
                if outcome is False:
                    break
                if outcome is True:
                    continue
                text = outcome
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
