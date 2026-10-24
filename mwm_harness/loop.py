"""The agent session: one turn state machine that every front-end drives.

A turn: run the prompt hooks, stream the model, run its tool calls behind the
permission checks and tool hooks, stream again, and when the model stops, run
the Stop hooks. A Stop hook may block; its reason goes back to the model and
the turn continues with ``stop_hook_active`` set, up to ``max_stop_blocks``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from mwm_harness import codemap
from mwm_harness import events as ev
from mwm_harness.agents import Agent, TaskTool, agent_roots, load_agents, resolve_model
from mwm_harness.assurance import Card as AssuranceCard
from mwm_harness.assurance import build_card
from mwm_harness.bugflag import FlagReport, failed_node_ids, flag_files, n_plus_one
from mwm_harness.checks import CheckConfigError, CheckResult, as_dicts, repo_root, run_checks
from mwm_harness.config import (
    EFFORTS,
    ModelSpec,
    Settings,
    config_dir,
    is_subscription_endpoint,
    workspace_root,
)
from mwm_harness.context import ContextMeter, build_system_prompt
from mwm_harness.costwatch import TurnCostWatch, log_spike
from mwm_harness.diagnose import (
    DiagnoseBudget,
    Diagnosis,
    check_failure_text,
    format_changes,
    is_test_command,
    is_test_failure,
    run_diagnosis,
    run_failure_text,
)
from mwm_harness.hooks import HookEngine, load_registrations
from mwm_harness.mcp_client import HARNESS_SECRET_ENV, McpManager
from mwm_harness.messages import (
    Block,
    Message,
    assistant_from_turn,
    text_block,
    tool_result_block,
)
from mwm_harness.observation_pack import pack_observations
from mwm_harness.permissions import Permissions, load_extra_deny
from mwm_harness.providers import Provider, ProviderError
from mwm_harness.redact import WITHHELD, Redactor, describe
from mwm_harness.review import (
    SECURITY_EXTRA,
    Comment,
    Review,
    attach_radius,
    call_model,
    collect_changes,
    comment_text,
    exclude_from_git,
    lock_held,
    remember_comment,
    review_changes,
    skipped_names,
)
from mwm_harness.runit import (
    RunItResult,
    TraceResult,
    changed_coverage,
    git_changed,
    is_test_file,
    rerun_nodes,
    trace_test,
)
from mwm_harness.runit import as_dict as runit_dict
from mwm_harness.sandbox import Sandbox
from mwm_harness.skills import (
    Command,
    Skill,
    SkillTool,
    command_roots,
    load_commands,
    load_skills,
    skill_roots,
    skills_prompt,
)
from mwm_harness.spec import (
    DOC_NAME,
    DRAFT_TIMEOUT,
    SPEC_DIR,
    STEPS,
    Spec,
    SpecReport,
    Step,
    changed_criteria,
    criterion_coverage,
    draft_prompt,
    existing_slugs,
    is_slug,
    load_spec,
    slug_from_idea,
    spec_dir,
    strip_fences,
    threat_surface,
    validate_design,
    validate_requirements,
    validate_tasks,
)
from mwm_harness.spec import (
    system_prompt as spec_system_prompt,
)
from mwm_harness.streaming import StreamAssembler
from mwm_harness.tools import Tool, ToolContext, ToolResult, default_tools
from mwm_harness.tools.search import ToolSearch
from mwm_harness.transcript import Transcript, load_messages, usage_to_anthropic
from mwm_harness.undo import UndoLog, UndoReport

TOOL_CRASHES = (OSError, ValueError, RuntimeError, KeyError, TypeError, UnicodeError)


def default_hook_settings(cwd: Path) -> list[Path]:
    """The settings files whose hooks apply.

    ``<workspace>/harness/hooks.json`` is the merged canonical file (built
    2026-10-20 from the user- and workspace-level ``.claude/settings.json``) and
    replaces both when present, so no registration loads twice.
    ``config_dir()/hooks.json`` is a machine-level fallback used only when the
    workspace file is missing. With neither merged file present, the legacy
    ``.claude`` pair is read. A project's own ``.claude/settings.json`` still
    applies whenever the cwd is a subproject rather than the workspace root.
    """
    canonical = workspace_root() / "harness" / "hooks.json"
    fallback = config_dir() / "hooks.json"
    workspace_settings = workspace_root() / ".claude" / "settings.json"
    project = cwd / ".claude" / "settings.json"
    if canonical.is_file():
        candidates = [canonical]
    elif fallback.is_file():
        candidates = [fallback]
    else:
        candidates = [Path.home() / ".claude" / "settings.json", workspace_settings]
    if project.resolve() != workspace_settings.resolve():
        candidates.append(project)
    unique: list[Path] = []
    for path in candidates:
        if path.resolve() not in [p.resolve() for p in unique]:
            unique.append(path)
    return unique


COMPACT_PROMPT = """\
Write a summary of this conversation that will REPLACE it: after this message the history above \
is gone and only your summary remains. Include, in this order:
1. What the user asked for, in their words where the wording matters, and any standing instructions.
2. What was done: files read, created and changed (full paths), commands run and their outcomes.
3. Decisions made and why; approaches that failed and why.
4. Facts learned that are needed later (names, numbers, error messages, IDs).
5. The task list with each item's state.
6. The exact next step.
Do not call tools. Answer with the summary only."""

SUBAGENT_HOOK_EVENTS = ("PreToolUse", "PostToolUse", "PermissionRequest")
FORWARDED = (ev.Notice, ev.HookBlocked, ev.FilesTouched)  # subagent events the person should see

PLAN_MODE_NOTE = (
    "Plan mode is on. Research with the reading tools only; do not edit files or run commands "
    "that change anything. When the plan is complete, call ExitPlanMode with it."
)


def reminder(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>"


# How many changed-on-disk files one reminder names. A mass checkout can move hundreds of files,
# and a reminder longer than the request it protects is worse than no reminder.
MAX_STALE_LISTED = 20


# Tool names the fix-cycle brake treats as changing code and as looking at something.
EDIT_TOOLS = frozenset({"Edit", "Write", "NotebookEdit"})
WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})  # file_path tools the check runner follows
LOOK_TOOLS = frozenset({"Read", "Grep", "Glob", "WebFetch", "WebSearch", "Task"})


class Session:
    def __init__(
        self,
        cwd: Path,
        model: ModelSpec,
        provider: Provider,
        settings: Settings | None = None,
        approver: ev.Approver | None = None,
        bus: ev.EventBus | None = None,
        hook_settings: list[Path] | None = None,
        resume_from: Path | None = None,
        sessions_dir: Path | None = None,
        tools: dict[str, Tool] | None = None,
        system_prompt: str | None = None,
        mcp: McpManager | None = None,
        skills: dict[str, Skill] | None = None,
        commands: dict[str, Command] | None = None,
        agents: dict[str, Agent] | None = None,
        models: dict[str, ModelSpec] | None = None,
    ) -> None:
        self.cwd = cwd.resolve()
        self.model = model
        self.provider = provider
        self.settings = settings or Settings()
        self.effort = self.settings.reasoning_effort
        self.approver = approver or ev.DenyAll()
        self.bus = bus or ev.EventBus()
        self.redactor = (
            Redactor(self.cwd, self.settings.redact_extra_patterns)
            if self.settings.redact
            else None
        )
        self.tools = tools if tools is not None else default_tools(self.settings.web_allow_private)
        self.mcp = mcp
        self.skills = (
            skills
            if skills is not None
            else load_skills(
                skill_roots(self.cwd, self.settings.skill_dirs), self.settings.skill_exclude
            )
        )
        self.commands = commands if commands is not None else load_commands(command_roots(self.cwd))
        if self.skills and tools is None:
            self.tools["Skill"] = SkillTool(self.skills)
        self.models = models or {model.id: model}
        self.agents = agents if agents is not None else load_agents(agent_roots(self.cwd))
        if self.agents and tools is None:
            self.tools["Task"] = TaskTool(self.agents, self.run_agent)
        self._sessions_dir = sessions_dir
        self.transcript = Transcript.create(self.cwd, sessions_dir)
        self.resumed = resume_from is not None
        self.messages: list[Message] = load_messages(resume_from) if resume_from else []
        if resume_from:
            self.transcript.note("resumed", f"continues {resume_from.name}")
            for message in self.messages:
                self.transcript.append(message)
        self.meter = ContextMeter(model.context_window, model.soft_budget)
        self.cost_watch = TurnCostWatch()
        self._turn_completion_start = 0
        HARNESS_SECRET_ENV.update(spec.key_env for spec in self.models.values())
        scratch = self.transcript.path.with_suffix("") / "scratch"
        self.permissions = Permissions(
            self.settings.permission_mode,
            self.cwd,
            load_extra_deny(config_dir() / "deny.toml"),
            tuple(self.settings.mcp_allow),
            read_roots=(
                workspace_root(),
                scratch,
                *(Path(p).expanduser() for p in self.settings.extra_writable),
                *(Path(p).expanduser() for p in self.settings.extra_readable),
            ),
        )
        writable = [
            self.cwd,
            scratch,
            *(Path(p).expanduser() for p in self.settings.extra_writable),
        ]
        scratch.mkdir(parents=True, exist_ok=True)
        self.tool_ctx = ToolContext(
            cwd=self.cwd,
            scratch=scratch,
            sandbox=Sandbox(
                self.settings.sandbox,
                writable,
                secret_env=frozenset(spec.key_env for spec in self.models.values()),
                env_keep=frozenset(self.settings.bash_env_keep),
                private_readable=frozenset(self.settings.sandbox_private_readable),
                tasks_max=self.settings.sandbox_tasks_max,
                memory_max_mb=self.settings.sandbox_memory_max_mb,
                file_max_mb=self.settings.sandbox_file_max_mb,
                tmpfs_max_mb=self.settings.sandbox_tmpfs_max_mb,
            ),
            output_cap=self.settings.tool_output_cap,
            session_write_cap=self.settings.session_write_cap_mb * 1024 * 1024,
        )
        self._hook_settings_arg = hook_settings
        if hook_settings is None:
            configured = [Path(p).expanduser() for p in self.settings.hook_settings]
            hook_settings = configured or default_hook_settings(self.cwd)
        self.hooks = HookEngine(
            load_registrations(hook_settings, self.settings.hook_timeout),
            self.transcript.session_id,
            self.transcript.path,
            self.cwd,
        )
        self.hooks.permission_mode = self.permissions.mode
        self.plan = ""
        self.touched: list[str] = []
        self.written: list[Path] = []  # files this turn wrote, for the check runner
        self.undo_log = UndoLog()  # what each recent turn's writes replaced, for /undo
        self.review: Review | None = None  # the Review Desk's last pass, comments included
        self.diagnosis: Diagnosis | None = None  # the last failure diagnosis, if any failed
        self.diagnose_budget = DiagnoseBudget(limit=0)  # set per turn in _turn
        self._test_failure = ""  # the newest test-shaped failure seen in a tool result this turn
        # Item 30: the failing node ids from the last test-shaped failure, kept between turns
        # until a prove-the-fix rerun passes them; the flag report and rerun of the last turn.
        self._pending_fix_nodes: list[str] = []
        self.last_flags = FlagReport()  # the method bug_flags() writes its newest report here
        self.proof_of_fix: dict[str, Any] | None = None
        self.last_checks: list[CheckResult] = []  # newest run_checks() results, for /assurance
        # Item 22: evidence for the assurance card and the card itself, newest turn.
        self._saw_test_run = False
        self._saw_test_pass = False
        self._last_runit: dict[str, Any] | None = None
        self.assurance: AssuranceCard | None = None
        # Item 23: the spec this session wrote or loaded, and the test coverage read for its
        # criteria. The card checks coverage fresh each turn; this copy is what the last /spec
        # or /spec status saw, so the report can quote a number it actually measured.
        self.spec: Spec | None = None
        self.spec_coverage: dict[str, list[str]] = {}
        self.tool_ctx.assurance_handler = self.assurance_report
        self._mode_before_plan = "default"
        self.tool_ctx.plan_handler = self._handle_plan
        self._system_prompt = system_prompt
        self._code_map: str | None = None  # this turn's map, for a model with a small window
        self.loaded_tools: set[str] = set()
        self.tool_search = ToolSearch(self._deferred_tools, self.loaded_tools)
        self._turn_task: asyncio.Task[ev.TurnEnded] | None = None

    # ------------------------------------------------------------------ setup

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            parts = [build_system_prompt(self.cwd, workspace_root(), self.model)]
            parts.append(skills_prompt(self.skills) if "Skill" in self.tools else "")
            parts.append(self._mcp_instructions())
            parts.append(self._code_map_prompt())
            self._system_prompt = "\n\n".join(part for part in parts if part)
        return self._system_prompt

    def _code_map_prompt(self) -> str:
        """A one-line-per-function map of the repo, for a model whose window cannot hold files.

        The map itself is built off the event loop by :meth:`_load_code_map`; ``codemap`` stops
        at ``MAX_FILES`` files and cuts its text at ``MAX_TEXT_CHARS``, so a small-window model
        gets a few thousand tokens of orientation instead of reading whole files.
        """
        if not self._wants_code_map():
            return ""
        return f"# Code map\n\n{self._code_map}" if self._code_map else ""

    def _wants_code_map(self) -> bool:
        limit = self.settings.code_map_below
        return limit > 0 and self.model.context_window < limit

    async def _load_code_map(self) -> None:
        """Build this turn's map in a worker: it walks every ``.py`` file under the repo root.

        Rebuilt every turn rather than cached, because a map that still shows the line ranges
        from before this session's edits would send a small-window model to the wrong place. The
        prompt is only rebuilt when the map text actually moved, so prompt caching still holds.
        """
        if not self._wants_code_map():
            return
        try:
            mapped = await asyncio.to_thread(codemap.text, repo_root(self.cwd))
        except Exception as exc:  # noqa: BLE001 - no map means a shorter prompt, not a dead session
            self.bus.emit(
                ev.Notice("warn", f"no code map in the prompt: {type(exc).__name__}: {exc}"[:200])
            )
            return
        if mapped != self._code_map:
            self._code_map = mapped
            self._system_prompt = None

    def _defers_mcp(self) -> bool:
        limit = self.settings.defer_mcp_tools_below
        return limit > 0 and self.model.context_window < limit

    def _deferred_tools(self) -> dict[str, Tool]:
        """MCP tools whose definitions stay out of the request until ToolSearch loads them."""
        if not self._defers_mcp():
            return {}
        return {
            n: t
            for n, t in self.tools.items()
            if n.startswith("mcp__") and n not in self.loaded_tools
        }

    def _request_specs(self) -> list[dict[str, Any]]:
        deferred = self._deferred_tools()
        specs = [tool.spec() for name, tool in self.tools.items() if name not in deferred]
        if deferred or self.loaded_tools:
            specs.append(self.tool_search.spec())
        return specs

    def _mcp_instructions(self) -> str:
        if self.mcp is None:
            return ""
        rows = [
            f"## {name}\n{server.instructions}"
            for name, server in self.mcp.servers.items()
            if server.instructions
        ]
        return "# MCP server instructions\n\n" + "\n\n".join(rows) if rows else ""

    async def connect_mcp(self, refresh: bool = False) -> None:
        """List the MCP tools (from the cache when it is current) and register them."""
        if self.mcp is None:
            return
        for name in [n for n in self.tools if n.startswith("mcp__")]:
            del self.tools[name]
        self.tools.update(await self.mcp.discover(refresh))
        for name, server in self.mcp.servers.items():
            if server.error:
                self.bus.emit(ev.Notice("warn", f"MCP server {name} failed: {server.error}"))

    def set_model(self, model: ModelSpec) -> None:
        self.model = model
        self.meter.context_window = model.context_window
        self.meter.soft_budget = model.soft_budget
        self._system_prompt = None
        self.bus.emit(ev.ModelChanged(model.id))

    def set_mode(self, mode: str) -> None:
        if mode == "plan" and self.permissions.mode != "plan":
            self._mode_before_plan = self.permissions.mode
        self.permissions.set_mode(mode)
        self.hooks.permission_mode = mode
        self.bus.emit(ev.ModeChanged(mode))

    def set_effort(self, effort: str) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"unknown effort {effort!r}; choose one of {', '.join(EFFORTS)}")
        self.effort = effort
        self.bus.emit(ev.EffortChanged(effort))

    async def _handle_plan(self, plan: str) -> bool:
        """Show the plan, ask the person, and leave plan mode on a yes."""
        self.plan = plan
        self.bus.emit(ev.PlanProposed(plan))
        if self.permissions.mode != "plan":
            return True  # nothing to unlock; the plan is shown and work goes on
        approved = await self.approver.ask("ExitPlanMode", {"plan": plan}, "approve this plan")
        self.bus.emit(ev.PlanResolved(approved))
        if approved:
            self.set_mode(self._mode_before_plan)
        return approved

    async def start(self) -> None:
        if self.tool_ctx.sandbox.warning:
            self.bus.emit(ev.Notice("warn", self.tool_ctx.sandbox.warning))
        await self.connect_mcp()
        outcome = await self.hooks.run(
            "SessionStart", source="resume" if self.resumed else "startup"
        )
        self._report(outcome, "SessionStart")
        if outcome.context:
            self._add(
                Message("user", reminder(self._clean("\n\n".join(outcome.context))), is_meta=True)
            )

    async def close(self, reason: str = "exit") -> None:
        outcome = await self.hooks.run("SessionEnd", reason=reason)
        self._report(outcome, "SessionEnd")
        await self.hooks.drain()
        if self.mcp is not None:
            await self.mcp.close()

    # ------------------------------------------------------------- compaction

    async def compact(self, trigger: str = "manual", instructions: str = "") -> str:
        """Replace the history with a summary written by the model. Returns the summary."""
        if not self.messages:
            return ""
        ask = COMPACT_PROMPT + (f"\n\nExtra instructions: {instructions}" if instructions else "")
        assembler = StreamAssembler()
        try:
            request = [*self.messages, Message("user", ask)]
            async for chunk in self.provider.stream(
                self.model, self.system_prompt, request, [], self.effort
            ):
                assembler.feed(chunk)
        except ProviderError as exc:
            self.bus.emit(ev.Notice("error", f"compaction failed, history kept: {exc}"))
            return ""
        turn = assembler.finish()
        summary = turn.text.strip()
        if not summary:
            self.bus.emit(ev.Notice("error", "compaction failed, history kept: empty summary"))
            return ""
        self.meter.record(turn.usage)
        before = len(self.messages)
        self.transcript.note("compact_boundary", f"{before} messages summarised", trigger=trigger)
        self.messages = []
        self._add(
            Message(
                "user",
                "This session continues from an earlier conversation that was summarised to "
                f"save context. The summary:\n\n{summary}",
            ),
            isCompactSummary=True,
        )
        self.meter.prompt_tokens = self.meter.completion_tokens = self.meter.cached_tokens = 0
        self.meter.floor_pending = True
        self._emit_usage()
        self.bus.emit(ev.Compacted(trigger, summary, before))
        outcome = await self.hooks.run("PostCompact", trigger=trigger, compact_summary=summary)
        self._report(outcome, "PostCompact")
        if outcome.context:
            self._add(
                Message("user", reminder(self._clean("\n\n".join(outcome.context))), is_meta=True)
            )
        return summary

    # -------------------------------------------------------------- subagents

    def sibling(self, cwd: Path, model: ModelSpec, resume_from: Path | None = None) -> Session:
        """A new session that runs like this one: same provider, settings, hooks and MCP pool."""
        same_folder = cwd.resolve() == self.cwd
        sibling = Session(
            cwd=cwd,
            model=model,
            provider=self.provider,
            settings=self.settings,
            hook_settings=self._hook_settings_arg,
            resume_from=resume_from,
            sessions_dir=self._sessions_dir,
            mcp=self.mcp,
            skills=self.skills if same_folder else None,
            commands=self.commands if same_folder else None,
            agents=self.agents if same_folder else None,
            models=self.models,
        )
        sibling.effort = self.effort
        return sibling

    async def run_tool_call(
        self, name: str, tool_input: dict[str, Any], force_ask: bool = False
    ) -> ToolResult:
        """Run one tool call that no model asked for (a mod), with the usual checks and events."""
        call_id = f"mod_{uuid.uuid4().hex[:10]}"
        self.bus.emit(ev.ToolStarted(call_id, name, tool_input))
        call: Block = {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
        result = await self._run_one_tool(call, None, force_ask)
        self.bus.emit(ev.ToolFinished(call_id, name, result.content, result.is_error))
        return result

    async def run_agent(self, name: str, prompt: str) -> ToolResult:
        """Run one subagent to its final answer. Cancelling the parent turn cancels it too."""
        agent = self.agents[name]
        model, note = resolve_model(agent, self.model, self.models, self.settings.agent_models)
        if note:
            self.bus.emit(ev.Notice("warn", f"{note}; it runs on {model.id}"))
        tools = {n: tool for n, tool in self.tools.items() if agent.allows(n)}
        child_bus = ev.EventBus()
        counted: list[str] = []
        child_bus.subscribe(
            lambda e: counted.append(e.name) if isinstance(e, ev.ToolStarted) else None
        )
        child_bus.subscribe(lambda e: self.bus.emit(e) if isinstance(e, FORWARDED) else None)
        child = Session(
            cwd=self.cwd,
            model=model,
            provider=self.provider,
            settings=self.settings,
            approver=self.approver,
            bus=child_bus,
            hook_settings=self._hook_settings_arg,
            sessions_dir=self._sessions_dir,
            tools=tools,
            system_prompt=f"{agent.prompt}\n\n{self._environment_note(model)}",
            skills=self.skills,
            commands={},
            agents={},
            models=self.models,
        )
        # A subagent is not a session of its own: only the tool hooks apply to it.
        child.hooks.registrations = [
            r for r in child.hooks.registrations if r.event in SUBAGENT_HOOK_EVENTS
        ]
        child.permissions.session_allow = self.permissions.session_allow
        child.tool_ctx.read_files = self.tool_ctx.read_files
        # The changed-on-disk stamps are copied, not shared: the child inherits what this session
        # last saw, so it warns about a file that moved after the parent read it, but the child's
        # own reads cannot make the parent's warning go away.
        child.tool_ctx.stamps = dict(self.tool_ctx.stamps)
        # The session disk cap spans subagents too: the child starts at the parent's count
        # (its cap does not reset with a Task call) and the parent adopts the child's total
        # below, so writes across a whole turn family charge one budget.
        child.tool_ctx.session_written = self.tool_ctx.session_written
        child.transcript.note("subagent", f"{name} started by session {self.transcript.session_id}")
        self.bus.emit(ev.SubagentStarted(name, model.id, prompt))
        try:
            ended = await child.send(prompt)
        except asyncio.CancelledError:
            self.bus.emit(ev.SubagentFinished(name, "cancelled", len(counted)))
            raise
        finally:
            self.meter.total_prompt += child.meter.total_prompt
            self.meter.total_completion += child.meter.total_completion
            self.meter.requests += child.meter.requests
            self.tool_ctx.session_written = max(
                self.tool_ctx.session_written, child.tool_ctx.session_written
            )
            await child.hooks.drain()
        self.bus.emit(ev.SubagentFinished(name, ended.reason, len(counted)))
        if ended.reason == "cancelled":
            # Stopping the parent turn cancels the task it awaits, and the child turn absorbs
            # that cancel to clean up. Pass it on, or the parent would go on as if nothing happened.
            raise asyncio.CancelledError
        if ended.reason != "done":
            return ToolResult(
                f"agent {name} ended with {ended.reason}: {ended.text}"[:20_000], True
            )
        return ToolResult(self.tool_ctx.cap(ended.text or "(the agent returned no text)"))

    def _environment_note(self, model: ModelSpec) -> str:
        return (
            f"# Environment\n- Working directory: {self.cwd}\n- Model: {model.id}\n"
            "- You are a subagent: your final message is the only thing the caller receives."
        )

    # ------------------------------------------------------------------- turn

    async def send(self, prompt: str) -> ev.TurnEnded:
        """Run one turn as a cancellable task. ``cancel()`` stops it from another task."""
        self._turn_task = asyncio.create_task(self._turn(prompt))
        try:
            return await self._turn_task
        except asyncio.CancelledError:
            # Cancelled while a hook was running: the turn ends, the session lives on.
            if self._turn_task.cancelled():
                return self._end("cancelled")
            # The caller itself was cancelled (a subagent whose parent turn was stopped):
            # stop the inner turn too, and wait until it has killed its running tool.
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
            raise
        finally:
            self._turn_task = None

    def cancel(self) -> bool:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            return True
        return False

    async def _turn(self, prompt: str) -> ev.TurnEnded:
        self.bus.emit(ev.TurnStarted(prompt))
        await self._load_code_map()
        self._turn_completion_start = self.meter.total_completion
        self.undo_log.start(prompt)
        outcome = await self.hooks.run("UserPromptSubmit", prompt=prompt)
        self._report(outcome, "UserPromptSubmit")
        if outcome.blocked:
            self.bus.emit(ev.HookBlocked("UserPromptSubmit", outcome.reason))
            return self._end("error", f"prompt blocked by a hook: {outcome.reason}")
        if outcome.context:
            self._add(
                Message("user", reminder(self._clean("\n\n".join(outcome.context))), is_meta=True)
            )
        if self.permissions.mode == "plan":
            self._add(Message("user", reminder(PLAN_MODE_NOTE), is_meta=True))
        self._add(Message("user", prompt))

        stop_blocks = 0
        requests = 0
        self._last_call: tuple[str, str] | None = None
        self._identical_calls = 0
        self._stuck = False
        self._blind_fixes = 0
        self._edited_since_run = False
        self.written: list[Path] = []
        self._test_failure = ""
        self._fix_edited = False
        self.proof_of_fix = None
        # The assurance card describes THIS turn: a test run in an earlier turn is not evidence
        # that nobody checked today's edit, so the run/pass witnesses start empty each turn.
        self._saw_test_run = False
        self._saw_test_pass = False
        # The diagnosis of the turn before this one arrives as one meta message, and it is spent
        # here: repeating it on every request would keep a stale failure in every later prompt.
        self.diagnose_budget.limit = self.settings.diagnose_rounds
        self._suspected: list[str] = []
        self._radius_noted = False
        self._diagnosis_root: Path | None = None
        if self.diagnosis is not None and not self.diagnosis.sent:
            note = self.diagnosis.note()
            self.diagnosis.sent = True
            if note:
                self._add(Message("user", reminder(self._clean(note)), is_meta=True))
                self._suspected = list(self.diagnosis.suspected_files)
                self._diagnosis_root = Path(self.diagnosis.root)
        while True:
            cap = self.settings.max_requests_per_turn
            if cap and requests >= cap:
                note = f"turn ended: {requests} model requests without a final answer"
                self.bus.emit(ev.Notice("error", note + " (max_requests_per_turn)"))
                return self._end("error", note)
            requests += 1
            if self.meter.needs_compaction and self.settings.auto_compact:
                self.bus.emit(
                    ev.Notice("info", f"soft budget reached, compacting: {self.meter.line()}")
                )
                if await self.compact("auto"):
                    self._add(
                        Message(
                            "user",
                            reminder("Continue the task from the exact next step."),
                            is_meta=True,
                        )
                    )
                else:
                    self.meter.prompt_tokens = 0  # do not retry on every request of this turn
            assembler = StreamAssembler()
            try:
                await self._stream(assembler)
            except asyncio.CancelledError:
                _uncancel()
                partial = assembler.finish()
                if partial.text:
                    self._add(
                        Message("assistant", [text_block(partial.text)]), stop_reason="cancelled"
                    )
                return self._end("cancelled", partial.text)
            except ProviderError as exc:
                self.bus.emit(ev.Notice("error", str(exc)))
                return self._end("error", str(exc))

            turn = assembler.finish()
            floor_was_pending = self.meter.floor_pending
            if self.meter.record(turn.usage):
                self._emit_usage()
                if floor_was_pending and self.meter.floor_over_budget:
                    self.bus.emit(
                        ev.Notice(
                            "warn",
                            f"right after compaction the prompt is {self.meter.floor:,} tokens, "
                            f"over the soft budget of {self.meter.soft_budget:,}: the system "
                            "prompt and tool list alone do not fit. Raise soft_budget or trim "
                            "the rules; compaction now waits for half a budget of growth",
                        )
                    )
            else:
                self.bus.emit(
                    ev.Notice("warn", "the endpoint sent no usage numbers for this answer")
                )
            message, argument_errors = assistant_from_turn(turn)
            if not message.blocks():
                self.bus.emit(ev.Notice("warn", "the model returned an empty answer"))
                return self._end("error", "empty answer")
            message.extra = {"model": self.model.id, "usage": usage_to_anthropic(turn.usage)}
            calls = message.tool_uses()
            self._add(message, stop_reason="tool_use" if calls else "end_turn")

            if calls:
                cancelled = await self._run_tools(calls, argument_errors)
                if cancelled:
                    return self._end("cancelled", message.text())
                if self._stuck:
                    note = f"turn ended: the same tool call {self._identical_calls} times in a row"
                    self.bus.emit(ev.Notice("error", note + " (max_identical_calls)"))
                    return self._end("error", note)
                continue

            outcome = await self.hooks.run("Stop", stop_hook_active=stop_blocks > 0)
            self._report(outcome, "Stop")
            if not outcome.blocked:
                if self.meter.over_budget:
                    self.bus.emit(
                        ev.Notice(
                            "warn",
                            f"over the soft context budget: {self.meter.line()}"
                            + ("; the next request compacts" if self.settings.auto_compact else ""),
                        )
                    )
                results = await self.run_checks()
                await self._diagnose_turn(results)
                await self._review_turn()
                await self._post_turn_looks(results)
                return self._end("done", message.text())
            stop_blocks += 1
            self.bus.emit(ev.HookBlocked("Stop", outcome.reason))
            if stop_blocks >= self.settings.max_stop_blocks:
                self.bus.emit(
                    ev.Notice(
                        "warn",
                        f"Stop hooks blocked {stop_blocks} times; the last answer stands unfixed",
                    )
                )
                await self.run_checks()
                await self._review_turn()
                await self._post_turn_looks([])
                return self._end("stop_blocks_exhausted", message.text())
            self._add(Message("user", f"Stop hook feedback:\n{outcome.reason}", is_meta=True))

    def _with_disk_note(self, messages: list[Message]) -> list[Message]:
        """Add the changed-on-disk reminder to the tail of this request, and to nothing else.

        The reminder goes into a copy of the message list: it is never appended to ``self.messages``
        and never written to the transcript, so it cannot pollute the cached prefix, a later turn or
        a ``/resume``. A session's own write re-stamps the file, so the writing session never reminds
        itself; a file drops off the list once this session reads it again.
        """
        stale = self.tool_ctx.stale_files()
        if not stale:
            return messages
        listed = "\n".join(f"- {path}" for path in stale[:MAX_STALE_LISTED])
        if len(stale) > MAX_STALE_LISTED:
            listed += f"\n- and {len(stale) - MAX_STALE_LISTED} more"
        self.bus.emit(ev.Notice("info", f"{len(stale)} file(s) changed on disk since last read"))
        note = reminder(
            "these files changed on disk since you last read them:\n"
            f"{listed}\n"
            "Read them again before editing them; the text you have is out of date."
        )
        return [*messages, Message("user", note, is_meta=True)]

    async def _stream(self, assembler: StreamAssembler) -> None:
        specs = self._request_specs()
        messages = self.messages
        if self.settings.observation_pack_limit > 0:
            messages, errors = pack_observations(
                messages, self.tool_ctx.scratch, self.settings.observation_pack_limit
            )
            for error in errors:
                self.bus.emit(
                    ev.Notice("warn", f"observation archive failed, sent in full: {error}")
                )
        messages = self._with_disk_note(messages)
        stream = self.provider.stream(self.model, self.system_prompt, messages, specs, self.effort)
        async for chunk in stream:
            assembler.feed(chunk)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self.bus.emit(ev.TextDelta(delta["content"]))
                # Alibaba and DeepSeek send reasoning_content, Ollama and OpenRouter send reasoning.
                thinking = delta.get("reasoning_content") or delta.get("reasoning")
                if thinking:
                    self.bus.emit(ev.ReasoningDelta(thinking))
                for fragment in delta.get("tool_calls") or []:
                    arguments = (fragment.get("function") or {}).get("arguments")
                    if not arguments:
                        continue
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    index = fragment.get("index", 0)
                    call_id, name = assembler.call_head(index)
                    self.bus.emit(ev.ToolArgsDelta(index, call_id, name, arguments))

    # ------------------------------------------------------------------ tools

    async def _run_tools(self, calls: list[Block], argument_errors: dict[str, str]) -> bool:
        """Run the calls in order. Returns True when the turn was cancelled meanwhile."""
        results: list[Block] = []
        cancelled = False
        for call in calls:
            if cancelled:
                results.append(
                    tool_result_block(call["id"], "Interrupted before this tool ran.", True)
                )
                continue
            self.bus.emit(ev.ToolStarted(call["id"], call["name"], call["input"]))
            self._note_outside_radius(call)
            if self._repeats(call):
                self._stuck = True
                result = ToolResult(
                    f"Not run: this exact call was made {self._identical_calls} times in a "
                    "row. The turn ends here.",
                    True,
                )
                self.bus.emit(
                    ev.ToolFinished(call["id"], call["name"], result.content, result.is_error)
                )
                results.append(
                    tool_result_block(call["id"], result.content, result.is_error, result.images)
                )
                continue
            if self._blind_fix_blocked(call):
                note = (
                    f"{self._blind_fixes} edit-and-run cycles in a row failed with no Read, "
                    "Grep or Glob in between"
                )
                self.bus.emit(ev.Notice("warn", f"edit refused: {note} (max_blind_fix_cycles)"))
                result = ToolResult(
                    f"Not run: {note}. Read the failing code, the error location or the "
                    "environment (config, installed packages, env vars) before editing again.",
                    True,
                )
            else:
                try:
                    result = await self._run_one_tool(call, argument_errors.get(call["id"]))
                except asyncio.CancelledError:
                    _uncancel()
                    cancelled = True
                    result = ToolResult("Interrupted by the user; the process was killed.", True)
                self._track_fix_cycle(call, result)
            self.bus.emit(
                ev.ToolFinished(call["id"], call["name"], result.content, result.is_error)
            )
            results.append(
                tool_result_block(call["id"], result.content, result.is_error, result.images)
            )
        self._add(Message("user", results))
        return cancelled

    def _note_outside_radius(self, call: Block) -> None:
        """Tell the model when it edits a file the last diagnosis did not suspect. Advice only.

        The note is the diagnosis doing its job on the turn it was sent into; one line per turn
        is enough, and a block would make advice a gate, which item 31 explicitly is not.
        """
        if not self._suspected or self._radius_noted or call["name"] not in EDIT_TOOLS:
            return
        path = self.tool_ctx.resolve(str(call["input"].get("file_path") or ""))
        root = self._diagnosis_root or self.cwd
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            return
        if relative in self._suspected:
            return
        self._radius_noted = True
        self.bus.emit(
            ev.Notice(
                "info",
                f"editing {relative}, which the last diagnosis did not suspect "
                f"(it named {', '.join(self._suspected)}). Advice only: carry on if you "
                "have read the reason",
            )
        )

    def _repeats(self, call: Block) -> bool:
        """Count consecutive identical calls; True once the streak hits the cap."""
        key = (call["name"], json.dumps(call["input"], sort_keys=True, default=str))
        self._identical_calls = self._identical_calls + 1 if key == self._last_call else 1
        self._last_call = key
        cap = self.settings.max_identical_calls
        return bool(cap) and self._identical_calls >= cap

    def _blind_fix_blocked(self, call: Block) -> bool:
        """True when an edit arrives after too many failed edit-then-run cycles."""
        cap = self.settings.max_blind_fix_cycles
        return bool(cap) and call["name"] in EDIT_TOOLS and self._blind_fixes >= cap

    def _track_fix_cycle(self, call: Block, result: ToolResult) -> None:
        """Count edit-then-Bash cycles whose run failed; a look or a passing run resets.

        A failing Bash run that reads like a test also keeps its output for the diagnosis at the
        end of the turn; a test command that passes clears it, so a failure the model already
        fixed is not diagnosed after the fact. The failing node ids are kept between turns for
        the prove-the-fix rerun (item 30 (c)), and a passing test run is recorded as evidence
        for the assurance card's correct row (item 22).
        """
        name = call["name"]
        if name in LOOK_TOOLS:
            self._blind_fixes = 0
        elif name in EDIT_TOOLS and not result.is_error:
            self._edited_since_run = True
            if self._pending_fix_nodes:
                self._fix_edited = True
        elif name == "Bash":
            command = str(call["input"].get("command") or "")
            if not result.is_error:
                self._blind_fixes = 0
                if is_test_command(command):
                    self._test_failure = ""
                    self._saw_test_run = True
                    self._saw_test_pass = True
                    if self._pending_fix_nodes:
                        self._pending_fix_nodes = []
                        self.proof_of_fix = {
                            "status": "ran",
                            "proved": True,
                            "node_ids": [],
                            "passed": [],
                            "still_failing": [],
                            "missing": [],
                            "summary": "the model reran the tests itself",
                            "seconds": 0.0,
                            "tail": "",
                            "by": "model",
                        }
            else:
                if self._edited_since_run:
                    self._blind_fixes += 1
                if is_test_failure(result.content):
                    self._test_failure = result.content
                    self._saw_test_run = True
                    self._saw_test_pass = False
                    nodes = failed_node_ids(result.content)
                    if nodes:
                        self._pending_fix_nodes = nodes
                        self.bus.emit(
                            ev.Notice(
                                "info",
                                f"{len(nodes)} test(s) failed; after your next edit the harness "
                                "reruns exactly these to prove the fix (item 30)",
                            )
                        )
            self._edited_since_run = False

    async def _run_one_tool(
        self, call: Block, argument_error: str | None, force_ask: bool = False
    ) -> ToolResult:
        """Run one call; its output (hook feedback included) leaves here with secrets stripped."""
        result = await self._run_one_tool_raw(call, argument_error, force_ask)
        if self.redactor is None:
            return result
        try:
            text, counts = self.redactor(result.content)
        except Exception as exc:  # noqa: BLE001 fail closed: no raw output when the redactor breaks
            self.bus.emit(
                ev.Notice("error", f"redaction failed ({type(exc).__name__}): tool output withheld")
            )
            return ToolResult(WITHHELD, True)
        if counts:
            self.bus.emit(ev.Notice("info", f"{call['name']}: {describe(counts)}"))
        return ToolResult(text, result.is_error, result.images)

    def _clean(self, text: str) -> str:
        """Hook notes that go to the model: same redaction, same fail-closed rule."""
        if self.redactor is None:
            return text
        try:
            out, counts = self.redactor(text)
        except Exception as exc:  # noqa: BLE001 fail closed
            self.bus.emit(
                ev.Notice("error", f"redaction failed ({type(exc).__name__}): hook note withheld")
            )
            return "[redaction failed: hook note withheld]"
        if counts:
            self.bus.emit(ev.Notice("info", f"hook note: {describe(counts)}"))
        return out

    async def _run_one_tool_raw(
        self, call: Block, argument_error: str | None, force_ask: bool = False
    ) -> ToolResult:
        name, tool_input = call["name"], dict(call["input"])
        if argument_error:
            return ToolResult(f"The call was not run: {argument_error}", True)
        tool = self.tool_search if name == ToolSearch.name else self.tools.get(name)
        if tool is None:
            return ToolResult(f"unknown tool {name}; available: {', '.join(self.tools)}", True)
        problem = tool.check(tool_input)
        if problem:
            return ToolResult(f"The call was not run: {problem}", True)

        pre = await self.hooks.run("PreToolUse", tool_name=name, tool_input=tool_input)
        self._report(pre, "PreToolUse")
        if pre.blocked or pre.permission == "deny":
            reason = pre.reason or pre.permission_reason or "no reason given"
            self.bus.emit(ev.HookBlocked("PreToolUse", reason))
            return ToolResult(f"Blocked by a PreToolUse hook: {reason}", True)
        if pre.updated_input is not None:
            tool_input = pre.updated_input

        # The deny list sees the final input, so a hook rewrite cannot smuggle anything past it.
        decision = self.permissions.decide(name, tool_input, tool.read_only)
        if decision.verdict == "deny":
            return ToolResult(decision.reason, True)
        # Off by default: the shell-rewrite hook answers "allow" for every command it
        # rewrites, destructive ones included, so its "allow" must not replace the person.
        if (
            decision.verdict == "ask"
            and pre.permission == "allow"
            and self.settings.hooks_may_approve
        ):
            decision.verdict = "allow"
        if decision.verdict == "allow" and pre.permission == "ask":
            decision.verdict, decision.reason = "ask", pre.permission_reason or "a hook asked"
        if decision.verdict == "allow" and force_ask:
            decision.verdict, decision.reason = "ask", "this action always asks first"
        if name == "Bash" and tool_input.get("network") and decision.verdict != "deny":
            policy = self.settings.shell_network
            if policy == "deny":
                return ToolResult(
                    "network access for shell commands is switched off (shell_network = "
                    '"deny" in settings.toml); do this without the network',
                    True,
                )
            if policy == "ask" and self.permissions.mode != "bypassPermissions":
                # "ask" means ask a person. bypassPermissions is the mode where the
                # person said up front "run it all, I am not watching", so it cannot
                # also be the mode that stops. A headless job's approver is DenyAll,
                # where this ask was an instant refusal that killed every git push.
                # shell_network = "deny" above still refuses, and so does the deny list.
                decision.verdict, decision.reason = "ask", "this command asks for network access"
        if decision.verdict == "ask":
            doomed = tool.refusal(tool_input, self.tool_ctx)
            if doomed:  # seen live: 15 approval prompts for an Edit that could not apply
                return ToolResult(doomed, True)
        if decision.verdict == "ask" and not await self._approve(name, tool_input, decision.reason):
            return ToolResult("The user declined this action. Ask before trying another way.", True)

        if name in WRITE_TOOLS:
            self.undo_log.before_write(
                self.tool_ctx.resolve(str(tool_input.get("file_path") or ""))
            )
        elif name == "Bash":
            self.undo_log.note_bash()
        try:
            result = await tool.run(tool_input, self.tool_ctx)
        except TOOL_CRASHES as exc:
            result = ToolResult(f"{name} crashed: {type(exc).__name__}: {exc}", True)
        if name == "TodoWrite" and not result.is_error:
            self.bus.emit(ev.TodosUpdated(list(self.tool_ctx.todos)))
        if name in ("Read", "Write", "Edit") and not result.is_error:
            self._touch(str(tool_input.get("file_path") or ""))
        if name in WRITE_TOOLS and not result.is_error:
            path = self.tool_ctx.resolve(str(tool_input.get("file_path") or ""))
            if path not in self.written:
                self.written.append(path)

        post = await self.hooks.run(
            "PostToolUse",
            tool_name=name,
            tool_input=tool_input,
            tool_response={"content": result.content, "is_error": result.is_error},
        )
        self._report(post, "PostToolUse")
        feedback = [post.reason] if post.blocked else []
        feedback += post.context
        if feedback:
            if post.blocked:
                self.bus.emit(ev.HookBlocked("PostToolUse", post.reason))
            joined = "\n\n".join(f for f in feedback if f)
            note = reminder("PostToolUse hook feedback:\n" + joined)
            result = ToolResult(f"{result.content}\n\n{note}", result.is_error, result.images)
        return result

    async def run_checks(self, files: list[Path] | None = None) -> list[CheckResult]:
        """Compile/lint/type-check ``files`` (default: what this turn wrote) and report it."""
        files = self.written if files is None else files
        if not files or (files is self.written and not self.settings.checks_after_turn):
            return []
        root = await asyncio.to_thread(repo_root, self.cwd)
        self.bus.emit(ev.ChecksStarted([str(p) for p in files]))
        try:
            results = await run_checks(root, files)
        except CheckConfigError as exc:
            self.bus.emit(ev.Notice("error", f"checks not run: {exc}"))
            return []
        self.bus.emit(ev.ChecksFinished(str(root), as_dicts(results)))
        self.last_checks = results
        return results

    def undo(self) -> UndoReport:
        """Put back the files the newest file-changing turn wrote (between turns only)."""
        report = self.undo_log.undo(self.cwd)
        if report.restored or report.deleted:
            self.bus.emit(ev.FilesTouched(list(self.touched)))
        return report

    def review_spec(self) -> tuple[ModelSpec, str]:
        """The Review Desk's model: ``review_model`` as a registry id or an ``agent_models`` key."""
        wanted = (self.settings.review_model or "").strip()
        if not wanted:
            return self.model, ""
        target = self.settings.agent_models.get(wanted, wanted)
        if target in self.models:
            return self.models[target], ""
        return (
            self.model,
            f"the reviewer model {wanted!r} is not configured; "
            f"this session's model ({self.model.id}) reviews instead",
        )

    def diagnose_spec(self) -> tuple[ModelSpec | None, str]:
        """The diagnosis critic: ``diagnose_model``, else the Review Desk's model.

        No name at all means this session's model reads the failure, as the Review Desk does. A
        name that resolves to nothing does NOT fall back here the way a review does: the
        deterministic half needs no model, so an unreachable critic costs a line in the report
        instead of a request spent on the lead model.
        """
        wanted = (self.settings.diagnose_model or "").strip() or (
            self.settings.review_model or ""
        ).strip()
        if not wanted:
            return self.model, ""
        target = self.settings.agent_models.get(wanted, wanted)
        if target in self.models:
            return self.models[target], ""
        return (
            None,
            f"no diagnosis critic: {wanted!r} is not a configured model id or agent tier, "
            "so only where the failure points is known",
        )

    async def _diagnose_turn(self, results: list[CheckResult], force: bool = False) -> None:
        """Item 31: after an editing turn whose checks or test run failed, diagnose the failure.

        A diagnosis is advice about a run that already failed, so nothing here may break the
        turn: a cancel or a crash becomes a notice, and the turn ends as it would have.
        ``force`` is the panel's ``/runit`` path: a person asked for that run, so the location
        ships even when this turn wrote no file.
        """
        if not self.settings.diagnose_after_fail or not (self.written or force):
            return
        checks_text = check_failure_text(as_dicts(results))
        source, failure = ("check", checks_text) if checks_text else ("", "")
        if not failure and self._test_failure:
            source, failure = "test run", self._test_failure
        if not failure:
            self.diagnose_budget.used = 0
            return
        if not self.diagnose_budget.take():
            last = self.diagnosis
            said = (
                f" The last diagnosis said: {last.root_cause or last.reason}"
                if last and (last.root_cause or last.reason)
                else ""
            )
            self.bus.emit(
                ev.Notice(
                    "warn",
                    f"{self.diagnose_budget.limit} diagnoses already spent on this failure, "
                    f"so the harness stops here and asks you: it needs a decision, not another "
                    f"read.{said}",
                )
            )
            return
        try:
            await self.diagnose(failure, source)
        except asyncio.CancelledError:
            _uncancel()
            self.bus.emit(ev.Notice("info", "diagnosis cancelled: the failure was not read"))
        except Exception as exc:  # noqa: BLE001 - a broken diagnosis is reported, never swallowed
            self.bus.emit(
                ev.Notice("error", f"diagnosis not run ({type(exc).__name__}: {exc})"[:300])
            )

    async def diagnose(self, failure: str, source: str = "manual") -> Diagnosis:
        """Diagnose one failing run over this turn's diff. It advises; it never edits."""
        root = await asyncio.to_thread(repo_root, self.cwd)
        paths = [path for path in self.written if path.exists()]
        spec, note = self.diagnose_spec()
        if note:
            self.bus.emit(ev.Notice("warn", note))
        caller = None
        if spec is not None:
            if isinstance(self.approver, ev.DenyAll) and is_subscription_endpoint(spec.base_url):
                self.bus.emit(
                    ev.Notice(
                        "warn",
                        f"no critic call for this diagnosis: {spec.id} sits on a subscription plan "
                        "endpoint and a headless run may not call it. Where the failure points "
                        "below comes from the failure text alone",
                    )
                )
            else:
                caller = self._critic_caller(spec)
        before = self.undo_log.turns[-1].before if self.undo_log.turns else {}
        changes = await asyncio.to_thread(collect_changes, root, paths, before)
        for change in changes:
            change.diff = self._clean(change.diff)
        changes = [change for change in changes if not change.diff.startswith("[redaction failed")]
        round_no = self.diagnose_budget.used
        critic = spec.id if caller is not None else "no critic"
        self.bus.emit(ev.DiagnoseStarted(source, critic, round_no))
        diagnosis = await run_diagnosis(
            root,
            source,
            self._clean(failure),
            self.undo_log.turns[-1].prompt if self.undo_log.turns else "",
            format_changes(changes),
            [change.path for change in changes],
            caller,
            critic,
            round_no=round_no,
        )
        self.diagnosis = diagnosis
        self.bus.emit(ev.DiagnoseFinished(str(root), diagnosis.as_dict()))
        return diagnosis

    async def diagnose_command(self, argument: str = "") -> Diagnosis:
        """``/diagnose``: the last failure this session saw, or the text the person pasted.

        Without an argument and with nothing to diagnose it says so instead of calling a model
        with an empty failure, which would answer about nothing.
        """
        text = argument.strip() or self._test_failure
        source = "manual" if argument.strip() else ("test run" if text else "")
        if not text:
            root = await asyncio.to_thread(repo_root, self.cwd)
            diagnosis = Diagnosis(
                root=str(root),
                model="",
                status="skipped",
                source="manual",
                reason=(
                    "no failing run to read: give /diagnose the output, or run the tests first"
                ),
            )
            self.bus.emit(ev.DiagnoseFinished(str(root), diagnosis.as_dict()))
            return diagnosis
        self.diagnose_budget.take()
        return await self.diagnose(text, source)

    def spec_model(self) -> tuple[ModelSpec, str]:
        """The model that drafts a spec: this session's model unless ``spec_model`` names one.

        An unresolvable name falls back to the session's own model with a warning, which is the
        opposite of the diagnosis critic. Drafting requirements is the lead model's own job, so
        spending its tokens on it is not the silent surprise an unasked-for critic call is.
        """
        wanted = (self.settings.spec_model or "").strip()
        if not wanted:
            return self.model, ""
        target = self.settings.agent_models.get(wanted, wanted)
        if target in self.models:
            return self.models[target], ""
        return (
            self.model,
            f"spec_model {wanted!r} is not a configured model id or an agent tier, "
            f"so {self.model.id} drafts the spec instead",
        )

    def _repo_note(self) -> str:
        """The top level of this repo, so a design names real paths instead of invented ones."""
        try:
            entries = sorted(
                p.name + ("/" if p.is_dir() else "")
                for p in self.cwd.iterdir()
                if not p.name.startswith(".")
            )
        except OSError:
            return ""
        return ", ".join(entries[:60])

    async def spec_command(self, argument: str) -> SpecReport:
        """``/spec``: write a spec, redo one step of one, or read what is already on disk.

        ``/spec <idea>`` walks requirements, design and tasks in order, each step waiting for the
        person. ``/spec <slug> <step>`` redrafts that one step from the documents on disk.
        ``/spec <slug>`` loads that spec, puts its tasks on the rail and reports its coverage.
        With no argument it reports the spec this session holds, or lists what is on disk.
        """
        root = await asyncio.to_thread(repo_root, self.cwd)
        arg = argument.strip()
        tokens = arg.split()
        if len(tokens) == 2 and is_slug(tokens[0]) and tokens[1] in STEPS:
            if not spec_dir(root, tokens[0]).is_dir():
                return self._spec_stop(
                    root,
                    tokens[0],
                    f"no {SPEC_DIR}/{tokens[0]} folder is on disk, so there is no {tokens[1]} "
                    "step to redo: /spec <idea> writes the first one",
                )
            return await self._spec_walk(root, tokens[0], "", [tokens[1]])
        elif len(tokens) == 1 and is_slug(tokens[0]) and spec_dir(root, tokens[0]).is_dir():
            return await self._spec_status(root, tokens[0], load_tasks=True)
        if not arg:
            slug = self.spec.slug if self.spec else ""
            return await self._spec_status(root, slug, load_tasks=False)
        return await self._spec_walk(root, slug_from_idea(arg), arg, list(STEPS))

    def _spec_stop(self, root: Path, slug: str, reason: str) -> SpecReport:
        report = SpecReport(slug=slug, dir=f"{SPEC_DIR}/{slug}", status="skipped", reason=reason)
        self.bus.emit(ev.SpecFinished(str(root), report.as_dict()))
        return report

    async def _spec_status(self, root: Path, slug: str, load_tasks: bool) -> SpecReport:
        """Read one spec off disk and report its criteria against the repo's tests."""
        if not slug:
            slugs = existing_slugs(root)
            return self._spec_report(
                root,
                SpecReport(
                    slug="",
                    dir=f"{SPEC_DIR}/",
                    status="skipped",
                    reason=(
                        "on disk: " + ", ".join(slugs) + " (open one with /spec <slug>)"
                        if slugs
                        else "no spec is loaded this session and none is on disk: "
                        "/spec <idea> writes one"
                    ),
                ),
            )
        if not is_slug(slug):
            return self._spec_report(
                root,
                SpecReport(
                    slug=slug,
                    dir="",
                    status="skipped",
                    reason=(
                        f"{slug!r} is not a spec folder name: lowercase letters, digits and "
                        "hyphens, at most 48 characters, no path"
                    ),
                ),
            )
        loaded = await asyncio.to_thread(load_spec, root, slug)
        if loaded is None:
            return self._spec_report(
                root,
                SpecReport(
                    slug=slug,
                    dir=f"{SPEC_DIR}/{slug}",
                    status="skipped",
                    reason="no readable document in that folder",
                ),
            )
        self.spec = loaded
        report = SpecReport(
            slug=slug,
            dir=loaded.directory,
            status="loaded",
            steps=[
                Step(step, f"{loaded.directory}/{DOC_NAME[step]}", "on-disk")
                for step in STEPS
                if DOC_NAME[step] in loaded.files
            ],
            criteria=loaded.criteria,
            tasks=loaded.tasks,
            coverage=await asyncio.to_thread(criterion_coverage, root, loaded.criterion_ids()),
            surface=threat_surface(loaded.text()),
            orphans=loaded.orphan_criteria(),
            refuted=[t.id for t in loaded.tasks_without_criteria()],
            stale_tasks=[t.id for t in loaded.uncovered_tasks()],
        )
        self.spec_coverage = report.coverage
        if load_tasks and loaded.tasks:
            report.tasks_loaded = self._spec_load_tasks(loaded)
        return self._spec_report(root, report)

    def _spec_report(self, root: Path, report: SpecReport) -> SpecReport:
        self.bus.emit(ev.SpecFinished(str(root), report.as_dict()))
        return report

    def _spec_load_tasks(self, loaded: Spec) -> int:
        """Item 23: the spec's tasks become this session's task list.

        The rail is replaced, not merged. Once a spec exists its tasks are the plan, and a
        half-finished model list sitting beside it would be two answers to "what is next".
        """
        self.tool_ctx.todos[:] = [
            {
                "content": task.label(),
                "status": "completed" if task.done else "pending",
                "activeForm": task.text[:80],
            }
            for task in loaded.tasks
        ]
        self.bus.emit(ev.TodosUpdated(list(self.tool_ctx.todos)))
        return len(loaded.tasks)

    async def _spec_write(self, path: Path, text: str) -> bool:
        """Write one document through the same rails as a model Write, always asking first.

        ``force_ask`` is the gate the plan asks for: a step needs a person even in acceptEdits,
        and a headless run has none, which is why that walk refuses before any draft call.
        """
        if path.exists():
            await self.run_tool_call("Read", {"file_path": str(path)})
        result = await self.run_tool_call(
            "Write", {"file_path": str(path), "content": text}, force_ask=True
        )
        if result.is_error:
            self.bus.emit(ev.Notice("warn", f"spec file not written: {result.content}"[:300]))
            return False
        return True

    async def _spec_step(
        self,
        root: Path,
        slug: str,
        idea: str,
        step: str,
        caller: Any,
        target: ModelSpec,
    ) -> tuple[bool, Step]:
        """Draft one step, check its shape, and write it only if the person approves.

        Returns whether the walk may continue, with the record of what happened.
        """
        rel = f"{SPEC_DIR}/{slug}/{DOC_NAME[step]}"
        current = await asyncio.to_thread(load_spec, root, slug)
        if step != "requirements" and (current is None or not current.requirements):
            return False, Step(
                step,
                rel,
                "skipped",
                "requirements.md is not on disk yet, so there is nothing to design or task "
                "out of: /spec <slug> requirements first",
            )
        if step == "requirements":
            prior = idea or (
                f"REVISE the existing requirements below. Keep every AC id that still applies "
                f"and keep its number: a test quotes the id, so a renumbered criterion looks "
                f"like an untested one.\n\n{current.requirements if current else ''}"
            )
        elif step == "design":
            prior = current.requirements if current else ""
        else:
            prior = current.text() if current else ""
        self.bus.emit(ev.SpecStarted(slug, step, target.id))
        prompt = draft_prompt(step, slug, idea=prior, repo_note=self._repo_note(), prior=prior)
        try:
            draft = await asyncio.wait_for(caller(spec_system_prompt(), prompt), DRAFT_TIMEOUT)
        except TimeoutError:
            return False, Step(
                step, rel, "failed", f"the draft gave nothing in {DRAFT_TIMEOUT:.0f} s"
            )
        except Exception as exc:  # noqa: BLE001 - a dead model costs the step, not the session
            return False, Step(
                step,
                rel,
                "failed",
                f"the draft call could not be made: {type(exc).__name__}: {exc}"[:300],
            )
        body = self._clean(strip_fences(draft)) + "\n"
        if step == "requirements":
            errors = validate_requirements(body)
        elif step == "design":
            errors = validate_design(body, threat_surface(prior))
        else:
            errors = validate_tasks(body, current.criteria if current else [])
        if errors:
            return False, Step(
                step,
                rel,
                "invalid",
                "the draft does not match the contract, so nothing was written",
                errors,
            )
        if not await self._spec_write(root / rel, body):
            return False, Step(
                step, rel, "refused", "you did not approve this step, so the walk stops here"
            )
        return True, Step(step, rel, "written")

    async def _spec_walk(self, root: Path, slug: str, idea: str, steps: list[str]) -> SpecReport:
        """Run the walk: one draft, one shape check and one approval per step, in order."""
        started = time.monotonic()
        if self.permissions.mode == "plan":
            return self._spec_stop(
                root,
                slug,
                "plan mode is on, and a spec is files: leave plan mode (/plan off) first",
            )
        if isinstance(self.approver, ev.DenyAll):
            return self._spec_stop(
                root,
                slug,
                "this run has no person to approve each step, so it writes nothing and spends "
                "no draft call: run /spec where you can answer the approval card",
            )
        target, note = self.spec_model()
        if note:
            self.bus.emit(ev.Notice("warn", note))
        caller = self._critic_caller(target)
        before = await asyncio.to_thread(load_spec, root, slug)
        # The OLD criteria's coverage, read before anything is rewritten: a criterion whose text
        # moved is owed a rerun of exactly the tests that quoted it while it said the old thing.
        old_coverage = (
            await asyncio.to_thread(criterion_coverage, root, before.criterion_ids())
            if before is not None and "requirements" in steps
            else {}
        )
        report = SpecReport(slug=slug, dir=f"{SPEC_DIR}/{slug}", status="written", model=target.id)
        for step in steps:
            go_on, record = await self._spec_step(root, slug, idea, step, caller, target)
            report.steps.append(record)
            if not go_on:
                break
        after = await asyncio.to_thread(load_spec, root, slug)
        landed = [record for record in report.steps if record.status == "written"]
        last = report.steps[-1] if report.steps else None
        if not landed and last is not None:
            report.status = last.status
            report.reason = last.reason or (last.errors[0] if last.errors else "")
        elif last is not None and last.status != "written":
            report.status = "partial"
            report.reason = (
                f"{last.step} {last.status}: " + "; ".join(last.errors[:3])
                if last.errors
                else f"{last.step} {last.status}: {last.reason}"
            )
        if after is not None:
            self.spec = after
            report.criteria = after.criteria
            report.tasks = after.tasks
            report.coverage = await asyncio.to_thread(
                criterion_coverage, root, after.criterion_ids()
            )
            self.spec_coverage = report.coverage
            report.surface = threat_surface(after.text())
            report.orphans = after.orphan_criteria()
            report.refuted = [task.id for task in after.tasks_without_criteria()]
            report.stale_tasks = [task.id for task in after.uncovered_tasks()]
            if any(record.step == "tasks" for record in landed) and after.tasks:
                report.tasks_loaded = self._spec_load_tasks(after)
            if before is not None and old_coverage:
                moved = changed_criteria(before.criteria, after.criteria)
                if moved:
                    owed = [
                        node
                        for cid in moved
                        for node in old_coverage.get(cid, [])
                        if node not in self._pending_fix_nodes
                    ]
                    self._pending_fix_nodes.extend(owed)
                    report.reflagged = moved
                    self.bus.emit(
                        ev.Notice(
                            "info",
                            f"criteria changed ({', '.join(moved)}): the "
                            f"{len(owed)} test(s) that quoted them are owed a rerun, queued "
                            "for prove-the-fix",
                        )
                    )
        report.seconds = time.monotonic() - started
        return self._spec_report(root, report)

    def _critic_caller(self, spec: ModelSpec):
        async def caller(system: str, user: str) -> str:
            return await call_model(self.provider, spec, system, user)

        return caller

    async def _review_turn(self) -> None:
        """The automatic Review Desk pass: after a turn that wrote files, when the setting is on.

        A review is advice about work the turn already finished, so nothing it does may break
        that turn: a cancel or a crash becomes a notice and the turn ends normally.
        """
        if not self.written or not self.settings.review_after_turn:
            return
        try:
            await self.run_review()
        except asyncio.CancelledError:
            _uncancel()
            self.bus.emit(
                ev.Notice("info", "review cancelled: the files changed but were not reviewed")
            )
        except Exception as exc:  # noqa: BLE001 - a broken review is reported, never swallowed
            self.bus.emit(ev.Notice("error", f"review not run ({type(exc).__name__}: {exc})"[:300]))

    async def run_review(self, files: list[Path] | None = None) -> Review:
        """The Review Desk: a second reader over this turn's diff. It advises; it never edits."""
        paths = list(files) if files is not None else await self.changed_files()
        paths = [path for path in paths if path.exists()]
        root = await asyncio.to_thread(repo_root, self.cwd)
        spec, note = self.review_spec()
        if note:
            self.bus.emit(ev.Notice("warn", note))

        def skip(reason: str) -> Review:
            review = Review(str(root), spec.id, "skipped", reason)
            self.review = review
            self.bus.emit(ev.ReviewFinished(str(root), review.as_dict()))
            return review

        if not paths:
            return skip("no changed file to review")
        if isinstance(self.approver, ev.DenyAll) and is_subscription_endpoint(spec.base_url):
            return skip(
                f"a headless run may not call {spec.id}: it sits on a subscription plan endpoint. "
                "Set review_model to a pay-as-you-go or local id, or review from an interactive session"
            )
        held = lock_held(self.settings.review_lock_dir)
        if held:
            return skip(held)
        before = self.undo_log.turns[-1].before if self.undo_log.turns else {}
        changes = await asyncio.to_thread(collect_changes, root, paths, before)
        for change in changes:
            change.diff = self._clean(change.diff)
        changes = [change for change in changes if not change.diff.startswith("[redaction failed")]
        if not changes:
            return skip(
                "no diff to read: the changed files show no difference, or redaction withheld them"
            )
        warnings: list[str] = []
        await asyncio.to_thread(attach_radius, root, changes, warnings.append)
        for warning in warnings:
            self.bus.emit(ev.Notice("warn", warning))
        self.bus.emit(ev.ReviewStarted([change.path for change in changes], spec.id))

        async def caller(system: str, user: str) -> str:
            return await call_model(self.provider, spec, system, user)

        review = await review_changes(
            root,
            self.undo_log.turns[-1].prompt if self.undo_log.turns else "",
            changes,
            caller,
            spec.id,
            extra=SECURITY_EXTRA if any(change.sensitive for change in changes) else "",
            left_out=skipped_names(root, paths),
        )
        self.review = review
        self.bus.emit(ev.ReviewFinished(str(root), review.as_dict()))
        return review

    def review_comment(self, number: int) -> Comment:
        """One comment from the last review, counted from 1 the way the desk lists them."""
        comments = self.review.comments if self.review else []
        if not comments:
            raise ValueError("the last review left no comment; run /review first")
        if not 1 <= number <= len(comments):
            raise ValueError(f"no comment {number}: the last review has {len(comments)}")
        return comments[number - 1]

    def comment_prompt(self, number: int, note: str = "") -> str:
        """The user text one reviewer comment becomes. The front-end sends it as the next turn.

        Nothing is queued here: a comment is advice until the person decides to send it, and
        this is that decision.
        """
        comment = self.review_comment(number)
        self.bus.emit(ev.ReviewComment(comment.file, comment.line, comment.text))
        return comment_text(comment, note)

    def remember(self, number: int) -> Path:
        """Keep one comment in ``.mwm/review-learnings.md``, so every later review is told it."""
        comment = self.review_comment(number)
        root = repo_root(self.cwd)
        path = remember_comment(root, comment)
        kept_out = exclude_from_git(root)
        if kept_out:
            self.bus.emit(ev.Notice("warn", kept_out))
        self.bus.emit(ev.ReviewComment(comment.file, comment.line, comment.text, remember=True))
        return path

    async def changed_files(self) -> list[Path]:
        """What the last editing turn wrote, else what git reports changed or new."""
        if self.written:
            return list(self.written)
        root = await asyncio.to_thread(repo_root, self.cwd)
        return await asyncio.to_thread(git_changed, root)

    async def run_it(self, pytest_args: list[str] | None = None) -> RunItResult:
        """Run the tests under coverage and report which changed lines they executed."""
        root = await asyncio.to_thread(repo_root, self.cwd)
        files = [p for p in await self.changed_files() if not is_test_file(p)]
        self.bus.emit(ev.Notice("info", f"run-it: running the tests for {len(files)} file(s)"))
        result = await changed_coverage(root, files, pytest_args)
        data = runit_dict(result)
        self._last_runit = data
        self.bus.emit(ev.RunItFinished(str(root), data))
        if result.status == "ran" and not result.tests_passed:
            self._saw_test_run = True
            self._saw_test_pass = False
            self._test_failure = run_failure_text(data)
            self._pending_fix_nodes = failed_node_ids(result.tail)
            await self._diagnose_turn([], force=True)
        elif result.status == "ran":
            self._saw_test_run = True
            self._saw_test_pass = result.tests_passed
            self._test_failure = ""
            self._pending_fix_nodes = []
        return result

    async def bug_flags(self, files: list[Path] | None = None) -> FlagReport:
        """Item 30 (a): the harness's own AST looks over the files this turn changed."""
        paths = list(files) if files is not None else list(self.written)
        report = FlagReport()
        py = [p for p in paths if p.suffix == ".py"]
        if not py or not self.settings.bug_flags:
            self.last_flags = report
            return report
        root = await asyncio.to_thread(repo_root, self.cwd)
        report = await asyncio.to_thread(flag_files, root, py)
        self.last_flags = report
        self.bus.emit(
            ev.BugFlagsFinished(
                str(root),
                {"flags": report.as_dicts(), "counts": report.counts(), "files": report.files},
            )
        )
        for flag in report.flags:
            self.bus.emit(ev.Notice("info", f"bug flag {flag.label()}"))
        return report

    async def prove_fix(self) -> dict[str, Any] | None:
        """Item 30 (c): after an edit that followed a failing test, rerun exactly those tests."""
        if (
            not self.settings.prove_fix_after_turn
            or not self._pending_fix_nodes
            or not self._fix_edited
        ):
            return None
        root = await asyncio.to_thread(repo_root, self.cwd)
        self.bus.emit(
            ev.Notice(
                "info",
                f"prove-the-fix: rerunning {len(self._pending_fix_nodes)} failed test(s) alone",
            )
        )
        try:
            proof = await rerun_nodes(root, self._pending_fix_nodes)
        except Exception as exc:  # noqa: BLE001 - a broken rerun is reported, never swallowed
            self.bus.emit(
                ev.Notice("error", f"prove-the-fix not run ({type(exc).__name__}: {exc})"[:300])
            )
            return None
        self.proof_of_fix = proof
        self.bus.emit(ev.FixProved(str(root), proof))
        if proof.get("proved"):
            self._pending_fix_nodes = []
            self._test_failure = ""
            self._saw_test_pass = True
            self.bus.emit(
                ev.Notice("info", f"prove-the-fix: PASS, {len(proof['passed'])} test(s) pass alone")
            )
        else:
            bad = proof.get("still_failing") or proof.get("missing") or proof["node_ids"]
            self.bus.emit(ev.Notice("error", "prove-the-fix: still failing: " + ", ".join(bad[:6])))
        return proof

    async def assurance_card(self, checks: list[CheckResult]) -> AssuranceCard | None:
        """Item 22: fill the six quality rows from this session's evidence. No tests, no model."""
        if not self.settings.assurance_after_turn or not self.written:
            return None
        spec_data: dict[str, Any] | None = None
        if self.spec is not None:
            # Measured fresh, not remembered: a test deleted during this turn must show up as a
            # gap in the very next card. That is plan item 23's closing test.
            ids = self.spec.criterion_ids()
            if ids:
                scan_root = await asyncio.to_thread(repo_root, self.cwd)
                self.spec_coverage = await asyncio.to_thread(criterion_coverage, scan_root, ids)
            spec_data = {
                "slug": self.spec.slug,
                "dir": self.spec.directory,
                "criteria": ids,
                "coverage": self.spec_coverage,
            }
        try:
            card = await build_card(
                self.cwd,
                list(self.written),
                checks=checks,
                runit=self._last_runit,
                flags=self.last_flags,
                fix=self.proof_of_fix,
                saw_test_run=self._saw_test_run,
                saw_test_pass=self._saw_test_pass,
                scans=self.settings.assurance_scans,
                spec=spec_data,
            )
        except Exception as exc:  # noqa: BLE001 - a broken card is a notice, never a silent pass
            self.bus.emit(
                ev.Notice("error", f"assurance card not built ({type(exc).__name__}: {exc})"[:300])
            )
            return None
        self.assurance = card
        self.bus.emit(ev.AssuranceFinished(card.root, card.as_dict()))
        if not card.done_allowed:
            self.bus.emit(
                ev.Notice("warn", card.gate_line() + ": fix it or fill the model-written rows")
            )
        return card

    async def assurance_report(self, rows: list[dict[str, Any]]) -> str:
        """The AssuranceReport tool's sink: the model fills only its three rows."""
        if self.assurance is None:
            raise ValueError(
                "no assurance card is built yet: it lands at the end of an editing turn"
            )
        for entry in rows:
            if not isinstance(entry, dict):
                raise ValueError("each row must be {quality, status, reason}")
            self.assurance.fill(
                str(entry.get("quality", "")),
                str(entry.get("status", "")),
                str(entry.get("reason", "")),
            )
        self.bus.emit(ev.AssuranceFinished(self.assurance.root, self.assurance.as_dict()))
        return (
            f"card updated (rows you wrote are labelled model-written): "
            f"{self.assurance.gate_line()}"
        )

    async def _post_turn_looks(self, checks: list[CheckResult]) -> None:
        """The turn's closing passes: bug flags (30 a), prove-the-fix (30 c), card (22)."""
        await self.bug_flags()
        await self.prove_fix()
        await self.assurance_card(checks)

    async def trace(self, test: str) -> TraceResult:
        root = await asyncio.to_thread(repo_root, self.cwd)
        result = await trace_test(root, test)
        data = runit_dict(result)
        repeats = n_plus_one(data)
        data["n_plus_one"] = [flag.as_dict() for flag in repeats]
        for flag in repeats:
            self.bus.emit(ev.Notice("info", f"bug flag {flag.label()}"))
        self.bus.emit(ev.TraceFinished(str(root), data))
        return result

    async def _approve(self, name: str, tool_input: dict[str, Any], reason: str) -> bool:
        request = await self.hooks.run("PermissionRequest", tool_name=name, tool_input=tool_input)
        self._report(request, "PermissionRequest")
        if request.blocked:
            return False
        if request.request_allowed:
            return True
        return await self.approver.ask(name, tool_input, reason)

    # ---------------------------------------------------------------- helpers

    def _add(self, message: Message, **fields: Any) -> None:
        self.messages.append(message)
        if fields.get("stop_reason"):
            message.extra = {**message.extra, "stop_reason": fields.pop("stop_reason")}
        self.transcript.append(message, **fields)

    def _touch(self, file_path: str) -> None:
        path = self.tool_ctx.resolve(file_path)
        try:
            shown = str(path.relative_to(self.cwd))
        except ValueError:
            shown = str(path)
        if shown in self.touched:
            self.touched.remove(shown)
        self.touched.append(shown)
        self.bus.emit(ev.FilesTouched(list(self.touched)))

    def _end(self, reason: str, text: str = "") -> ev.TurnEnded:
        self.undo_log.end()
        self._watch_turn_cost()
        ended = ev.TurnEnded(reason, text)
        self.bus.emit(ended)
        return ended

    def _watch_turn_cost(self) -> None:
        """Flag a turn that billed >3x the session's median completion tokens."""
        turn_tokens = self.meter.total_completion - self._turn_completion_start
        spike = self.cost_watch.record(max(turn_tokens, 0))
        if spike is None:
            return
        ratio = spike["ratio"]
        self.bus.emit(
            ev.Notice(
                "warn",
                f"cost spike: this turn used {turn_tokens:,} completion tokens, "
                f"{ratio}x the session median of {int(spike['median_tokens']):,}; "
                "check for a looping tool call or re-reads before continuing",
            )
        )
        log_spike(self.transcript.session_id, self.model.id, spike)

    def _emit_usage(self) -> None:
        meter = self.meter
        self.bus.emit(
            ev.UsageUpdated(
                meter.prompt_tokens,
                meter.completion_tokens,
                meter.cached_tokens,
                meter.context_window,
                meter.soft_budget,
            )
        )

    def _report(self, outcome: Any, event: str) -> None:
        for notice in outcome.notices:
            self.bus.emit(ev.Notice("warn", f"[{event}] {notice}"))
        for text in outcome.context:
            self.bus.emit(ev.HookContext(event, text))


def _uncancel() -> None:
    task = asyncio.current_task()
    if task is not None:
        task.uncancel()
