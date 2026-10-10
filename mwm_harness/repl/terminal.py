"""Plain terminal front-end: works over SSH, in tests and when the browser panel is down.

It only listens to the event bus and calls ``Session`` methods; it holds no
agent logic. Ctrl-C during a turn cancels the turn; ``/quit`` or Ctrl-D leaves.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mwm_harness import events as ev
from mwm_harness import postmortem
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
/think [on|off]       show or hide the model's thinking (dimmed)
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
/postmortem [draft] [ID]  which harness layer failed this session (or session ID); draft = model writes one fix as a draft
/check                compile, lint and type-check the changed files (also runs after each editing turn)
/runit [pytest args]  run the tests under coverage: which changed lines did a test execute
/trace TEST           run one test (pytest node id) and list the calls it made, with argument values
/compact [focus]      replace the history with a summary written by the model
/init                 have the model write AGENTS.md (project rules) for this directory
/resume               list earlier sessions for this directory (start with: mwm --resume ID)
/tab [new [DIR]|N|close [N]]  sessions in this terminal: list, open one (same model), switch, close
/clear                forget the conversation (the transcript file is kept)
/quit                 leave"""


def run_postmortem(session: Session, argument: str, out: Printer) -> bool | str:
    """Write the layer report for this session or an earlier one; ``draft`` asks the model for one fix."""
    words = argument.split()
    draft = "draft" in words
    wanted = next((w for w in words if w != "draft"), "")
    if wanted:
        matches = [p for p in list_sessions(session.cwd) if p.stem.startswith(wanted)]
        if len(matches) != 1:
            out.line(f"{len(matches)} sessions match {wanted!r}; give a longer ID (see /resume)")
            return True
        path = matches[0]
    else:
        path = session.transcript.path
    if not path.is_file():
        out.line("this session has no transcript yet")
        return True
    report = postmortem.analyze(path)
    target = postmortem.write(report)
    ranked = report.ranked()
    out.line(
        f"postmortem: {report.turns} turns, {report.tool_calls} tool calls, "
        f"{sum(f.count for f in ranked)} failures in {len(ranked)} classes -> {target}"
    )
    for finding in ranked[:5]:
        out.line(f"  {finding.layer:<13} {finding.count:>3}x  {finding.kind}", stamp=False)
    if not draft:
        return True
    if not ranked:
        out.line("nothing failed, so there is no fix to draft")
        return True
    return postmortem.DRAFT_PROMPT.format(
        report=target, draft=target.with_name(f"{target.stem}.draft.md")
    )


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
    if name == "/check":
        files = await session.changed_files()
        if not await session.run_checks(files):
            out.line("nothing to check: no changed file matches a check")
        return True
    if name == "/runit":
        await session.run_it(shlex.split(argument))
        return True
    if name == "/trace":
        if not argument.strip():
            out.line("usage: /trace <pytest node id>, e.g. /trace tests/test_a.py::test_b")
        else:
            await session.trace(argument.strip())
        return True
    if name == "/mcp" and "refresh" in argument.split():
        await session.connect_mcp(refresh=True)
    return False


class Printer:
    """Turns bus events into terminal output."""

    def __init__(self, color: bool) -> None:
        self.color = color
        self.show_thinking = True
        self._mid_line = False
        self._thinking = False
        self._answering = False  # inside a stamped answer block
        self._tool_started: dict[str, str] = {}  # tool_use_id -> ts of its ToolStarted

    def paint(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def stamp(self, event: ev.Event | None = None) -> str:
        """Local wall-clock time of the event (now when there is none), as a dim prefix."""
        try:
            if event is None:
                raise ValueError
            when = datetime.fromisoformat(event.ts.replace("Z", "+00:00")).astimezone()
        except ValueError:
            when = datetime.now().astimezone()
        return self.paint(DIM, when.strftime("%H:%M:%S")) + " "

    def line(self, text: str = "", event: ev.Event | None = None, stamp: bool = True) -> None:
        self._end_thinking()
        self._answering = False
        if self._mid_line:
            print()
            self._mid_line = False
        print((self.stamp(event) if stamp and text else "") + text)

    def _end_thinking(self) -> None:
        """Close a thinking stream: reset the colour and end its line."""
        if not self._thinking:
            return
        self._thinking = False
        if self._mid_line:
            print()
        print(self.paint(DIM, "[end thinking]"))
        self._mid_line = False

    def __call__(self, event: ev.Event) -> None:
        if isinstance(event, ev.ReasoningDelta):
            if not self.show_thinking:
                return
            if not self._thinking:
                if self._mid_line:
                    print()
                print(self.stamp(event) + self.paint(DIM, "[thinking]"))
                self._thinking = True
            print(self.paint(DIM, event.text), end="", flush=True)
            self._mid_line = not event.text.endswith("\n")
        elif isinstance(event, ev.TextDelta):
            self._end_thinking()
            if not self._answering:
                if self._mid_line:
                    print()
                print(self.stamp(event), end="")
                self._answering = True
            print(event.text, end="", flush=True)
            self._mid_line = not event.text.endswith("\n")
        elif isinstance(event, ev.ToolStarted):
            self._tool_started[event.tool_use_id] = event.ts
            summary = json.dumps(event.input, ensure_ascii=False)
            self.line(self.paint(CYAN, f"> {event.name} {summary[:300]}"), event)
        elif isinstance(event, ev.ToolFinished):
            first = event.content.strip().splitlines()[:6] or ["(empty)"]
            code = RED if event.is_error else DIM
            took = postmortem.seconds_between(
                self._tool_started.pop(event.tool_use_id, ""), event.ts
            )
            if took is not None:
                first[0] = f"{first[0][:180]}  ({took:.1f}s)"
            for index, row in enumerate(first):
                self.line(self.paint(code, f"  {row[:200]}"), event, stamp=index == 0)
        elif isinstance(event, ev.HookBlocked):
            self.line(
                self.paint(YELLOW, f"[{event.event} hook blocked] {event.reason[:600]}"), event
            )
        elif isinstance(event, ev.Notice):
            code = RED if event.level == "error" else YELLOW
            self.line(self.paint(code, f"[{event.level}] {event.text}"), event)
        elif isinstance(event, ev.TodosUpdated):
            self.line(format_todos(event.todos), event)
        elif isinstance(event, ev.UsageUpdated):
            self.line(
                self.paint(
                    DIM,
                    f"[ctx {event.context_fraction:.1%} | in {event.prompt_tokens:,} "
                    f"out {event.completion_tokens:,} cached {event.cached_tokens:,}]",
                ),
                event,
            )
        elif isinstance(event, ev.ChecksFinished):
            for result in event.results:
                code = {"pass": DIM, "fail": RED}.get(result["status"], YELLOW)
                self.line(self.paint(code, f"[check] {check_line(result)}"), event)
                for finding in result["findings"][:5]:
                    where = f"{finding['file']}:{finding['line']}"
                    self.line(self.paint(code, f"  {where} {finding['message'][:160]}"), event)
        elif isinstance(event, ev.RunItFinished):
            for row in runit_lines(event.result):
                self.line(
                    self.paint(RED if "never run" in row or "FAIL" in row else DIM, row), event
                )
        elif isinstance(event, ev.TraceFinished):
            for row in trace_lines(event.result):
                self.line(self.paint(DIM, row), event)
        elif isinstance(event, ev.TurnEnded) and event.reason != "done":
            self.line(self.paint(YELLOW, f"[turn ended: {event.reason}]"), event)


def runit_lines(result: dict[str, Any]) -> list[str]:
    if result["status"] != "ran":
        label = "NOT RUN" if result["status"] == "not_run" else "TIMEOUT"
        return [f"[run-it] {label}: {result['reason']}"]
    tests = "tests pass" if result["tests_passed"] else "TESTS FAIL"
    rows = [f"[run-it] {tests} ({result['summary']})"]
    for path, states in result["files"].items():
        missed = [s["line"] for s in states if s["state"] == "miss"]
        hit = len(states) - len(missed)
        rows.append(
            f"  {path}: {hit} changed line(s) run" + (f", never run: {missed}" if missed else "")
        )
    return rows + ["  " + r for r in result["tail"].splitlines()]


def trace_lines(result: dict[str, Any]) -> list[str]:
    if result["status"] != "ran":
        return [f"[trace] {result['test']}: {result['status'].upper()}: {result['reason']}"]
    rows = [f"[trace] {result['test']}: {'passed' if result['passed'] else 'FAILED'}"]
    for call in result["calls"][:60]:
        args = ", ".join(f"{k}={v}" for k, v in call["args"].items())
        rows.append(
            f"  {'  ' * call['depth']}{call['function']}({args})  {call['file']}:{call['line']}"
        )
    if result["skipped"]:
        rows.append(f"  … {result['skipped']} deeper or later call(s) not shown")
    return rows


def check_line(result: dict[str, Any]) -> str:
    status = {"pass": "pass", "fail": "FAIL", "not_run": "NOT RUN", "timeout": "TIMEOUT"}
    text = f"{result['check']}: {status.get(result['status'], result['status'])}"
    if result["status"] == "fail":
        text += f", {len(result['findings'])} finding(s)"
    return text + (f": {result['reason']}" if result["reason"] else "")


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
    elif name == "/think":
        if argument in ("on", "off"):
            out.show_thinking = argument == "on"
        elif argument:
            out.line("usage: /think [on|off]")
        out.line(f"thinking display: {'on' if out.show_thinking else 'off'}")
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
    elif name == "/postmortem":
        return run_postmortem(session, argument, out)
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


class Tabs:
    """The terminal's sessions; one turn runs at a time, in the session on screen."""

    def __init__(self, first: Session, printer: Any) -> None:
        self.sessions = [first]
        self.current = first
        self.printer = printer

    def listing(self) -> list[str]:
        return [
            f"{'*' if s is self.current else ' '} {n}  {s.cwd}  {s.model.id}"
            for n, s in enumerate(self.sessions, 1)
        ]

    async def command(self, text: str, out: Any) -> bool:
        """Handle ``/tab ...``. Returns False when ``text`` is not a tab command."""
        words = text.split()
        if words[0] != "/tab":
            return False
        action = words[1] if len(words) > 1 else "list"
        if action == "new":
            folder = Path(" ".join(words[2:])).expanduser() if len(words) > 2 else self.current.cwd
            if not folder.is_dir():
                out.line(f"no such folder: {folder}")
                return True
            session = self.current.sibling(folder.resolve(), self.current.model)
            session.bus.subscribe(self.printer)
            await session.start()
            self.sessions.append(session)
            self.current = session
            out.line(f"tab {len(self.sessions)}: {session.cwd}")
        elif action == "close":
            number = int(words[2]) if len(words) > 2 and words[2].isdigit() else None
            target = (
                self.sessions[number - 1]
                if number and number <= len(self.sessions)
                else self.current
            )
            if len(self.sessions) == 1:
                out.line("the last tab stays open; /quit leaves")
                return True
            self.sessions.remove(target)
            await target.close()
            if target is self.current:
                self.current = self.sessions[-1]
            out.line(f"closed; on tab {self.sessions.index(self.current) + 1}: {self.current.cwd}")
        elif action.isdigit():
            number = int(action)
            if not 1 <= number <= len(self.sessions):
                out.line(f"no tab {number}; /tab lists them")
                return True
            self.current = self.sessions[number - 1]
            out.line(f"tab {number}: {self.current.cwd} ({len(self.current.messages)} messages)")
        else:
            for row in self.listing():
                out.line(row)
        return True

    async def close(self) -> None:
        for session in self.sessions:
            await session.close()


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
    tabs = Tabs(session, out)
    loop = asyncio.get_running_loop()

    def on_interrupt() -> None:
        if not tabs.current.cancel():
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
            current = tabs.current
            if text.startswith("/"):
                if await tabs.command(text, out):
                    continue
                if await run_async_command(current, text, out):
                    continue
                outcome = run_command(current, models, text, out)
                if outcome is False:
                    break
                if outcome is True:
                    continue
                text = outcome
            await current.send(text)
            out.line()
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await tabs.close()


def resolve_session(cwd: Path, token: str) -> Path | None:
    """Find an earlier transcript by id prefix; ``last`` means the newest one."""
    sessions = list_sessions(cwd)
    if token == "last":
        return sessions[0] if sessions else None
    return next((p for p in sessions if p.stem.startswith(token)), None)
