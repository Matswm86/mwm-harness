"""Hooks engine: runs the hook commands registered in Claude-Code-style settings files.

The workspace already has its rule gates written as hook scripts. This engine
reads the same ``settings.json`` files, sends each script the same JSON payload
on stdin and honours the same answers, so the scripts run unchanged.

Contract per hook process:
    exit 0   stdout is JSON (fields below) or, for UserPromptSubmit and
             SessionStart, plain text that becomes model context
    exit 2   block; stderr is the reason shown to the model
    other    a non-blocking error, reported as a notice

JSON fields honoured: ``decision: "block"`` + ``reason``, ``continue: false`` +
``stopReason``, ``systemMessage``, and under ``hookSpecificOutput``:
``additionalContext``, ``permissionDecision`` (allow | deny | ask) +
``permissionDecisionReason``, ``updatedInput``, ``decision.behavior`` (the
PermissionRequest answer).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Stop",
    "PostCompact",
    "SessionEnd",
)
CONTEXT_EVENTS = ("SessionStart", "UserPromptSubmit")


@dataclass(frozen=True)
class HookRegistration:
    event: str
    matcher: str
    command: str
    timeout: float
    run_async: bool
    source: str

    def matches(self, tool_name: str | None) -> bool:
        if not self.matcher or self.matcher == "*" or tool_name is None:
            return True
        try:
            return re.fullmatch(self.matcher, tool_name) is not None
        except re.error:
            return self.matcher == tool_name


@dataclass
class HookOutcome:
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    updated_input: dict[str, Any] | None = None
    permission: str | None = None  # allow | deny | ask, from PreToolUse
    permission_reason: str = ""
    request_allowed: bool = False  # PermissionRequest answered "allow"
    stop_session: bool = False

    @property
    def reason(self) -> str:
        return "\n\n".join(r for r in self.reasons if r)


def load_registrations(paths: list[Path], default_timeout: float = 60.0) -> list[HookRegistration]:
    registrations: list[HookRegistration] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A broken settings file must not pass for "no hooks": say so loudly.
            raise ValueError(f"hook settings file is not valid JSON: {path}") from None
        for event, groups in (data.get("hooks") or {}).items():
            if event not in EVENTS:
                continue
            for group in groups or []:
                for hook in group.get("hooks") or []:
                    if hook.get("type", "command") != "command" or not hook.get("command"):
                        continue
                    registrations.append(
                        HookRegistration(
                            event=event,
                            matcher=group.get("matcher") or "",
                            command=hook["command"],
                            timeout=float(hook.get("timeout") or default_timeout),
                            run_async=bool(hook.get("async")),
                            source=str(path),
                        )
                    )
    return registrations


class HookEngine:
    def __init__(
        self,
        registrations: list[HookRegistration],
        session_id: str,
        transcript_path: Path,
        cwd: Path,
    ) -> None:
        self.registrations = registrations
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.cwd = cwd
        self.permission_mode = "default"
        self._background: set[asyncio.Task[Any]] = set()

    def for_event(self, event: str, tool_name: str | None = None) -> list[HookRegistration]:
        return [r for r in self.registrations if r.event == event and r.matches(tool_name)]

    async def run(self, event: str, **fields: Any) -> HookOutcome:
        """Run every hook registered for ``event`` in parallel and merge the answers."""
        tool_name = fields.get("tool_name")
        payload = {
            "session_id": self.session_id,
            "transcript_path": str(self.transcript_path),
            "cwd": str(self.cwd),
            "permission_mode": self.permission_mode,
            "hook_event_name": event,
            **fields,
        }
        outcome = HookOutcome()
        waiting = []
        for registration in self.for_event(event, tool_name):
            job = self._run_one(registration, payload)
            if registration.run_async:
                task = asyncio.create_task(job)
                self._background.add(task)
                task.add_done_callback(self._background.discard)
            else:
                waiting.append(job)
        for result in await asyncio.gather(*waiting):
            _merge(outcome, event, *result)
        return outcome

    async def drain(self, timeout: float = 5.0) -> None:
        """Give fire-and-forget hooks a moment to finish before the process exits."""
        if self._background:
            await asyncio.wait(self._background, timeout=timeout)

    async def _run_one(
        self, registration: HookRegistration, payload: dict[str, Any]
    ) -> tuple[HookRegistration, int | None, str, str]:
        env = {**os.environ, "CLAUDE_PROJECT_DIR": str(self.cwd), "MWM_HARNESS": "1"}
        try:
            process = await asyncio.create_subprocess_shell(
                registration.command,
                cwd=str(self.cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return registration, None, "", f"could not start: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload).encode()), registration.timeout
            )
        except TimeoutError:
            _kill(process)
            return registration, None, "", f"timed out after {registration.timeout:.0f} s"
        except asyncio.CancelledError:
            _kill(process)
            raise
        return (
            registration,
            process.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )


def _kill(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def _merge(
    outcome: HookOutcome,
    event: str,
    registration: HookRegistration,
    code: int | None,
    stdout: str,
    stderr: str,
) -> None:
    label = registration.command.split("/")[-1][:60]
    if code == 2:
        outcome.blocked = True
        outcome.reasons.append(stderr.strip() or f"{label} blocked without a reason")
        return
    if code != 0:
        # A hook that fails must be visible: a silent failure reads as "gate passed".
        outcome.notices.append(f"hook {label} failed (exit {code}): {stderr.strip()[:300]}")
        return
    text = stdout.strip()
    if not text:
        return
    data = _parse_json(text)
    if data is None:
        if event in CONTEXT_EVENTS:
            outcome.context.append(text)
        elif text.startswith("{"):
            # It tried to answer in JSON and failed: a gate that meant to block did not.
            outcome.notices.append(
                f"hook {label} exited 0 but its output is not valid JSON, so it decided "
                f"nothing: {text[:200]!r}"
            )
        return
    if data.get("systemMessage"):
        outcome.notices.append(str(data["systemMessage"]))
    if data.get("continue") is False:
        outcome.stop_session = True
        outcome.notices.append(str(data.get("stopReason") or f"{label} stopped the session"))
    if data.get("decision") == "block":
        outcome.blocked = True
        outcome.reasons.append(str(data.get("reason") or f"{label} blocked without a reason"))
    specific = data.get("hookSpecificOutput") or {}
    if specific.get("additionalContext"):
        outcome.context.append(str(specific["additionalContext"]))
    decision = specific.get("permissionDecision")
    if decision in ("allow", "deny", "ask"):
        # deny beats ask beats allow when several hooks answer.
        rank = {"allow": 0, "ask": 1, "deny": 2}
        if outcome.permission is None or rank[decision] > rank[outcome.permission]:
            outcome.permission = decision
            outcome.permission_reason = str(specific.get("permissionDecisionReason") or "")
    if isinstance(specific.get("updatedInput"), dict):
        outcome.updated_input = specific["updatedInput"]
    behavior = (specific.get("decision") or {}).get("behavior")
    if behavior == "allow":
        outcome.request_allowed = True
    elif behavior == "deny":
        outcome.blocked = True
        outcome.reasons.append(str((specific.get("decision") or {}).get("message") or "denied"))


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
