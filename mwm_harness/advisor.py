"""Advisor on a timer (plan item 24): a second model reads the working session, off by default.

Every N minutes or every N tool calls, at a tool-call boundary, one short note arrives: what
is missing, what to verify next. The note is a labelled meta message the working model reads
before its next request. It can never block, approve or stop a tool: the "no" stays with
permissions, tests and the assurance card. MercuryCLI's author reported that an advisor
holding a veto caused loops; advice-only arriving at the next tool call did not.

Nothing here edits a file or touches ``Permissions``. A pass that cannot run says why: a
skip, a timeout or an unparseable answer lands in ``Advice.reason``, never as quiet success.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# The digest is the advisor's whole view. Cut at the tail (newest first) costs the oldest
# lines first, and a cut always says so: an advisor reading half a session must not present
# its note as if it had read all of it.
MAX_DIGEST_CHARS = 6_000
MAX_TASK_CHARS = 500
MAX_NOTE_CHARS = 400
ADVISOR_TIMEOUT = 120.0
TAIL_MESSAGES = 12
MAX_BLOCK_CHARS = 220

SYSTEM = """\
You are the advisor of a working agent session. You wrote nothing and decide nothing: you \
cannot block, approve or stop a tool, and no permission changes because of what you say. \
Read the session digest and answer with one short note: what is missing, or what is worth \
verifying next. If the session is on track and a note would only be noise, answer with an \
empty note instead. You see a digest, not the full transcript: never name a file, a test or \
a number the digest does not contain. Plain English, no hedges. Answer with one JSON object \
and nothing else."""

PROMPT = """\
TASK THE CURRENT TURN WAS ASKED TO DO
{task}

SESSION DIGEST (harness plumbing left out, newest exchange last)
{digest}

Answer with one JSON object: {{"note": "one or two sentences: what is missing or what to \
verify next"}}, or {{"note": ""}} when nothing is worth a note."""


@dataclass
class Advice:
    """One advisor pass. A skip or a failure names itself; silence is never an implied pass."""

    model: str
    status: str  # advised | quiet | skipped | failed
    note: str = ""
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    minutes: int = 0
    calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "note": self.note,
            "reason": self.reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 1),
            "minutes": self.minutes,
            "calls": self.calls,
        }


def digest(
    task: str,
    todos: list[dict[str, Any]],
    touched: list[str],
    exchanges: list[str],
) -> str:
    """The bounded view one advisor pass reads. A cut says so inside the text, never silently.

    ``exchanges`` is the newest-first list of one-line renders of the last messages, built by
    the session (it holds the redaction and the block shapes).
    """
    done = sum(1 for todo in todos if todo.get("status") == "completed")
    lines = [f"task: {task[:MAX_TASK_CHARS]}"]
    if todos:
        active = "; ".join(
            str(todo.get("activeForm") or todo.get("content") or "")
            for todo in todos
            if todo.get("status") == "in_progress"
        )
        state = f"task list: {done} of {len(todos)} done"
        lines.append(state + (f"; active: {active}" if active else ""))
    else:
        lines.append("task list: none written this session")
    lines.append("files touched: " + (", ".join(touched[:20]) if touched else "none"))
    body = "\n".join(exchanges) if exchanges else "(no exchanges yet beyond the task above)"
    head = "\n".join(lines)
    budget = MAX_DIGEST_CHARS - len(head) - 2
    cut = len(body) > budget
    if cut:
        # Keep the newest lines: an advisor told the oldest exchanges were dropped can still
        # say what the end of the session is missing. The reverse would be a silent lie.
        kept: list[str] = []
        used = 0
        for line in reversed(exchanges):
            if used + len(line) + 1 > budget:
                break
            kept.insert(0, line)
            used += len(line) + 1
        body = "\n".join(kept) if kept else "(every exchange was longer than the digest budget)"
        body += f"\n(older exchanges cut: the newest are kept, the digest hit {MAX_DIGEST_CHARS:,} characters)"
    return f"{head}\n{body}"


async def advice_call(
    provider: Any, spec: Any, system: str, user: str
) -> tuple[str, dict[str, Any]]:
    """One no-tools call that keeps the usage block, because advisor cost must be countable.

    Mirrors ``review.call_model``; that one throws the usage away, this one may not: the plan
    asks for the advisor's cost shown in tokens, and an uncounted background call is exactly
    the surprise cost a timer should not be able to hide.
    """
    from mwm_harness.messages import Message
    from mwm_harness.streaming import StreamAssembler

    assembler = StreamAssembler()
    async for chunk in provider.stream(spec, system, [Message("user", user)], []):
        assembler.feed(chunk)
    turn = assembler.finish()
    return turn.text.strip(), turn.usage or {}


def parse_note(answer: str) -> tuple[str, str]:
    """(note, error) from the advisor's JSON reply. Never raises; error explains."""
    start = answer.find("{")
    end = answer.rfind("}")
    if start < 0 or end <= start:
        return "", f"the advisor answered without a JSON object: {answer[:120]!r}"
    try:
        data = json.loads(answer[start : end + 1])
    except json.JSONDecodeError as exc:
        return "", f"the advisor answered invalid JSON ({exc.msg}): {answer[:120]!r}"
    if not isinstance(data, dict) or "note" not in data:
        return "", 'the advisor answer has no "note" key'
    note = data["note"]
    if not isinstance(note, str):
        return "", "the advisor note is not a string"
    return note.strip(), ""


Caller = Callable[[str, str], Awaitable[tuple[str, dict[str, Any]]]]


async def run_advice(
    digest_text: str,
    task: str,
    caller: Caller,
    model_id: str,
    minutes: int,
    calls_since: int,
    timeout: float = ADVISOR_TIMEOUT,
) -> Advice:
    """Ask the advisor once. Never raises: ``status`` says advised, quiet, skipped or failed."""
    started = time.monotonic()
    prompt = PROMPT.format(
        task=task or "(the turn carries no prompt)",
        digest=digest_text,
    )
    try:
        answer, usage = await asyncio.wait_for(caller(SYSTEM, prompt), timeout)
    except TimeoutError:
        return Advice(
            model=model_id,
            status="failed",
            reason=f"the advisor model gave no answer in {timeout:.0f} s",
            minutes=minutes,
            calls=calls_since,
            seconds=time.monotonic() - started,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any provider failure is a failed note, not a crash
        return Advice(
            model=model_id,
            status="failed",
            reason=f"the advisor call broke ({type(exc).__name__}: {exc})"[:300],
            minutes=minutes,
            calls=calls_since,
            seconds=time.monotonic() - started,
        )
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    note, error = parse_note(answer)
    if error:
        return Advice(
            model=model_id,
            status="failed",
            reason=error,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            minutes=minutes,
            calls=calls_since,
            seconds=time.monotonic() - started,
        )
    if not note:
        return Advice(
            model=model_id,
            status="quiet",
            reason="the advisor read the session and found nothing worth a note",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            minutes=minutes,
            calls=calls_since,
            seconds=time.monotonic() - started,
        )
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS] + "…"
    return Advice(
        model=model_id,
        status="advised",
        note=note,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        minutes=minutes,
        calls=calls_since,
        seconds=time.monotonic() - started,
    )
