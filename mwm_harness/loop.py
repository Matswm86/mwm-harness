"""The agent session: one turn state machine that every front-end drives.

A turn: run the prompt hooks, stream the model, run its tool calls behind the
permission checks and tool hooks, stream again, and when the model stops, run
the Stop hooks. A Stop hook may block; its reason goes back to the model and
the turn continues with ``stop_hook_active`` set, up to ``max_stop_blocks``.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from mwm_harness import events as ev
from mwm_harness.agents import Agent, TaskTool, agent_roots, load_agents, resolve_model
from mwm_harness.config import ModelSpec, Settings, config_dir, workspace_root
from mwm_harness.context import ContextMeter, build_system_prompt
from mwm_harness.hooks import HookEngine, load_registrations
from mwm_harness.mcp_client import McpManager
from mwm_harness.messages import (
    Block,
    Message,
    assistant_from_turn,
    text_block,
    tool_result_block,
)
from mwm_harness.permissions import Permissions, load_extra_deny
from mwm_harness.providers import Provider, ProviderError
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
from mwm_harness.streaming import StreamAssembler
from mwm_harness.tools import Tool, ToolContext, ToolResult, default_tools
from mwm_harness.transcript import Transcript, load_messages, usage_to_anthropic

TOOL_CRASHES = (OSError, ValueError, RuntimeError, KeyError, TypeError, UnicodeError)


def default_hook_settings(cwd: Path) -> list[Path]:
    """The settings files whose hooks apply: the user's, the workspace's, the project's."""
    candidates = [
        Path.home() / ".claude" / "settings.json",
        workspace_root() / ".claude" / "settings.json",
        cwd / ".claude" / "settings.json",
        config_dir() / "hooks.json",
    ]
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
        self.approver = approver or ev.DenyAll()
        self.bus = bus or ev.EventBus()
        self.tools = tools if tools is not None else default_tools(self.settings.web_allow_private)
        self.mcp = mcp
        self.skills = (
            skills
            if skills is not None
            else load_skills(skill_roots(self.cwd, self.settings.skill_dirs))
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
        self.permissions = Permissions(
            self.settings.permission_mode,
            self.cwd,
            load_extra_deny(config_dir() / "deny.toml"),
            tuple(self.settings.mcp_allow),
        )
        scratch = self.transcript.path.with_suffix("") / "scratch"
        writable = [
            self.cwd,
            scratch,
            *(Path(p).expanduser() for p in self.settings.extra_writable),
        ]
        scratch.mkdir(parents=True, exist_ok=True)
        self.tool_ctx = ToolContext(
            cwd=self.cwd,
            scratch=scratch,
            sandbox=Sandbox(self.settings.sandbox, writable),
            output_cap=self.settings.tool_output_cap,
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
        self._mode_before_plan = "default"
        self.tool_ctx.plan_handler = self._handle_plan
        self._system_prompt = system_prompt
        self._turn_task: asyncio.Task[ev.TurnEnded] | None = None

    # ------------------------------------------------------------------ setup

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            parts = [build_system_prompt(self.cwd, workspace_root(), self.model)]
            parts.append(skills_prompt(self.skills) if "Skill" in self.tools else "")
            parts.append(self._mcp_instructions())
            self._system_prompt = "\n\n".join(part for part in parts if part)
        return self._system_prompt

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
        await self.connect_mcp()
        outcome = await self.hooks.run(
            "SessionStart", source="resume" if self.resumed else "startup"
        )
        self._report(outcome, "SessionStart")
        if outcome.context:
            self._add(Message("user", reminder("\n\n".join(outcome.context)), is_meta=True))

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
            async for chunk in self.provider.stream(self.model, self.system_prompt, request, []):
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
            self._add(Message("user", reminder("\n\n".join(outcome.context)), is_meta=True))
        return summary

    # -------------------------------------------------------------- subagents

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
        outcome = await self.hooks.run("UserPromptSubmit", prompt=prompt)
        self._report(outcome, "UserPromptSubmit")
        if outcome.blocked:
            self.bus.emit(ev.HookBlocked("UserPromptSubmit", outcome.reason))
            return self._end("error", f"prompt blocked by a hook: {outcome.reason}")
        if outcome.context:
            self._add(Message("user", reminder("\n\n".join(outcome.context)), is_meta=True))
        if self.permissions.mode == "plan":
            self._add(Message("user", reminder(PLAN_MODE_NOTE), is_meta=True))
        self._add(Message("user", prompt))

        stop_blocks = 0
        requests = 0
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
                return self._end("stop_blocks_exhausted", message.text())
            self._add(Message("user", f"Stop hook feedback:\n{outcome.reason}", is_meta=True))

    async def _stream(self, assembler: StreamAssembler) -> None:
        specs = [tool.spec() for tool in self.tools.values()]
        stream = self.provider.stream(self.model, self.system_prompt, self.messages, specs)
        async for chunk in stream:
            assembler.feed(chunk)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self.bus.emit(ev.TextDelta(delta["content"]))
                if delta.get("reasoning_content"):
                    self.bus.emit(ev.ReasoningDelta(delta["reasoning_content"]))

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
            try:
                result = await self._run_one_tool(call, argument_errors.get(call["id"]))
            except asyncio.CancelledError:
                _uncancel()
                cancelled = True
                result = ToolResult("Interrupted by the user; the process was killed.", True)
            self.bus.emit(
                ev.ToolFinished(call["id"], call["name"], result.content, result.is_error)
            )
            results.append(tool_result_block(call["id"], result.content, result.is_error))
        self._add(Message("user", results))
        return cancelled

    async def _run_one_tool(self, call: Block, argument_error: str | None) -> ToolResult:
        name, tool_input = call["name"], dict(call["input"])
        if argument_error:
            return ToolResult(f"The call was not run: {argument_error}", True)
        tool = self.tools.get(name)
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
        if decision.verdict == "ask":
            doomed = tool.refusal(tool_input, self.tool_ctx)
            if doomed:  # seen live: 15 approval prompts for an Edit that could not apply
                return ToolResult(doomed, True)
        if decision.verdict == "ask" and not await self._approve(name, tool_input, decision.reason):
            return ToolResult("The user declined this action. Ask before trying another way.", True)

        try:
            result = await tool.run(tool_input, self.tool_ctx)
        except TOOL_CRASHES as exc:
            result = ToolResult(f"{name} crashed: {type(exc).__name__}: {exc}", True)
        if name == "TodoWrite" and not result.is_error:
            self.bus.emit(ev.TodosUpdated(list(self.tool_ctx.todos)))
        if name in ("Read", "Write", "Edit") and not result.is_error:
            self._touch(str(tool_input.get("file_path") or ""))

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
            result = ToolResult(f"{result.content}\n\n{note}", result.is_error)
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
        ended = ev.TurnEnded(reason, text)
        self.bus.emit(ended)
        return ended

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
