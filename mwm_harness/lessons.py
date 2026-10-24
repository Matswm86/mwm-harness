"""Session-end lessons (plan item 5): one cheap model reads the finished session and
writes candidate lines for a person to accept or delete.

Before the cutover Claude wrote feedback memories by hand. After it nobody does, so the
capture has to live in the harness. It captures candidates and nothing more: this module
never edits ``QWEN.md``, never writes under ``memory/`` and never touches a skill, the same
approval-first rule brain ingestion follows. A line becomes a rule only when a person says so.

The shape was measured, not assumed, on 2026-10-24 against this box's own 122 sessions:

* A whole transcript does not fit. Counting every block, the largest is about 335,000 tokens
  and 12 of 122 pass 131,072, which is qwen3.8-27b's window. ``tool_result`` and ``thinking``
  are 85% of the corpus while the prose a lesson needs is 5.5%, so the raw file is the wrong
  input even before the limit bites.
* The transport limit is tighter than the window. Groq refuses a 27b request over 7,000 INPUT
  tokens per minute (HTTP 413 on ITPM, and the gpt-oss-20b fallback 413s at 8,000 TPM). These
  digests tokenize at 2.86 characters per token, measured from a provider rejection that
  counted 24,000 characters as 8,403 tokens, so the budget is characters divided by that, not
  the usual four.
* The lessons themselves were worth reading. 27b named the real defect in a probe
  (``backup.sh`` hardcoding a ``/tmp`` lock that the sandbox mounts read-only) and quoted the
  two failing lines, then over-claimed one consequence the digest did not prove. That is why
  the output is a candidate file, never a rule.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mwm_harness import redact
from mwm_harness.messages import Message

# Every MATS and FAILED line is kept ahead of anything else, so the cap can only ever drop
# the agent's own prose and routine tool calls. An over-claim in a lesson is survivable; an
# invisible failure is not.
# 12,000 characters measured 4,487 INPUT tokens on a live 27b call (spike/real_lessons_check.py,
# 2026-10-24, a 1.5 MB transcript cut to the cap): 2.65 chars per token, 64% of Groq's 7,000
# input-tokens-per-minute ceiling. chars/4 would have promised 3,000, and a 24,000-char digest
# measured 8,403 and was refused with HTTP 413, so this budget is the provider's own count.
MAX_DIGEST_CHARS = 12_000
MAX_LESSONS = 3
MIN_TOOL_CALLS = 8
LESSONS_TIMEOUT = 90.0
MAX_LINE_CHARS = 600
LESSONS_FILE = "data/lessons/LESSONS.md"

# An email address is identity, not evidence, and this digest leaves the machine for a hosted
# endpoint. redact.py masks credential shapes but not addresses, so the lessons digest adds it.
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

SYSTEM = """\
You review ONE finished agent session and propose candidate lessons for its human operator. \
You decide nothing and change nothing: a person reads every line you write and deletes most \
of them. Rules:
- At most 3 lessons. Fewer is better. If nothing is worth keeping, answer with an empty list.
- Each lesson names one concrete thing that went wrong or worked: a command, a tool, a file, \
a gate, a setting. Anything that would be true of every session is worthless.
- Only claim what the digest shows. It holds every human message and every failed tool result, \
but the agent's own prose and its routine tool calls may be trimmed. Never name a file, a \
number or a consequence the digest does not contain.
- Write plain English a stranger understands. No hedges, no advice about general practice.
Answer with one JSON object and nothing else."""

PROMPT = """\
SESSION DIGEST (harness plumbing left out; every human message and every failure is in here, \
the middle may be trimmed)
{digest}

Answer with one JSON object: {{"lessons": [{{"lesson": "one line, one concrete thing", \
"evidence": "quote or name the digest line it rests on"}}]}}, at most 3 of them, or \
{{"lessons": []}} when nothing is worth a line."""


@dataclass
class Lessons:
    """One session-end pass. A skip or a failure names itself; silence is never a pass."""

    model: str
    status: str  # recorded | quiet | skipped | failed
    entries: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    path: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "entries": self.entries,
            "reason": self.reason,
            "path": self.path,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 1),
            "calls": self.calls,
        }


def tool_call_count(messages: list[Message]) -> int:
    return sum(len(message.tool_uses()) for message in messages if message.role == "assistant")


def _line(text: str, cap: int) -> str:
    body = EMAIL.sub("[redacted:email]", text or "")
    body = redact.redact_text(body)
    body = " ".join(body.split())
    return body if len(body) <= cap else body[:cap] + " ..."


def digest(messages: list[Message]) -> tuple[str, int]:
    """The bounded view one lessons pass reads, plus how many lines the cap dropped.

    Reads the transcript through ``load_messages()``, which skips the ``system`` request and
    request-blob lines item 4 added, so a lessons call never sees the audit rows. Human turns
    and failed tool results are kept whole first: they are the two classes a lesson comes
    from. Meta messages stay out, they are harness plumbing.
    """
    must: list[str] = []
    soft: list[str] = []
    for message in messages:
        if message.is_meta:
            continue
        if message.role == "user":
            text = message.text()
            if text and not text.lstrip().startswith("<system-reminder>"):
                must.append(f"[person] {_line(text, 600)}")
            for block in message.blocks():
                if block.get("type") == "tool_result" and block.get("is_error"):
                    must.append(f"[FAILED] {_line(_result_text(block), 700)}")
        else:
            text = message.text()
            if text:
                soft.append(f"[model] {_line(text, 400)}")
            for call in message.tool_uses():
                arguments = json.dumps(call["input"], default=str, sort_keys=True)
                soft.append(f"[calls {call['name']}] {_line(arguments, 160)}")
    head = must + soft
    used = 0
    kept: list[str] = []
    for line in head:
        if used + len(line) + 1 > MAX_DIGEST_CHARS:
            break
        kept.append(line)
        used += len(line) + 1
    dropped = len(head) - len(kept)
    if dropped:
        kept.append(
            f"(cut: {dropped} of {len(head)} lines left out, the digest hit "
            f"{MAX_DIGEST_CHARS:,} characters)"
        )
    return "\n".join(kept), dropped


def _result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def parse_lessons(answer: str) -> tuple[list[dict[str, str]], str]:
    """(entries, error) from the model's JSON reply. Never raises; error explains."""
    start = answer.find("{")
    end = answer.rfind("}")
    if start < 0 or end <= start:
        return [], f"no JSON object in the answer: {answer[:120]!r}"
    try:
        data = json.loads(answer[start : end + 1])
    except json.JSONDecodeError as exc:
        return [], f"invalid JSON from the lessons model ({exc.msg}): {answer[:120]!r}"
    if not isinstance(data, dict) or "lessons" not in data:
        return [], 'the answer has no "lessons" key'
    raw = data["lessons"]
    if not isinstance(raw, list):
        return [], 'the "lessons" value is not a list'
    entries: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            evidence = ""
        elif isinstance(item, dict):
            text = str(item.get("lesson") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
        else:
            return [], f"a lesson is neither a string nor an object: {item!r}"
        if not text:
            continue
        entries.append({"lesson": text[:MAX_LINE_CHARS], "evidence": evidence[:MAX_LINE_CHARS]})
    return entries[:MAX_LESSONS], ""


def lessons_path(path: str | Path | None = None) -> Path:
    """Where the candidate queue lives, resolved against the isolated workspace root."""
    from mwm_harness.config import workspace_root

    if path is None:
        return workspace_root() / LESSONS_FILE
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else workspace_root() / candidate


SESSION_ID = re.compile(r"session=([^\s)]+)")


def seen_session_ids(path: Path) -> set[str]:
    """Session ids already recorded, so a second pass on one transcript writes nothing."""
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    found: set[str] = set()
    for line in lines:
        found.update(SESSION_ID.findall(line))
    return found


def append_entries(path: Path, session_id: str, model: str, entries: list[dict[str, str]]) -> Path:
    """Append the candidate lines, newest last. The file is a queue, not a rule book."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(
            "# Candidate lessons\n\n"
            "One line per candidate lesson, newest last. Nothing here is a rule: a person "
            "accepts or deletes each line, and accepting one means editing QWEN.md or memory "
            "by hand. The harness that writes this file never edits those.\n\n",
            encoding="utf-8",
        )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    block = [
        f"- {stamp} {entry['lesson']}"
        + (f" [{entry['evidence']}]" if entry.get("evidence") else "")
        + f" (session={session_id} model={model})"
        for entry in entries
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block) + "\n")
    return path


Caller = Callable[[str, str], Awaitable[tuple[str, dict[str, Any]]]]


async def call_model(provider: Any, spec: Any, system: str, user: str) -> tuple[str, dict]:
    """One no-tools call, keeping the usage block so the pass is countable like the advisor's."""
    from mwm_harness.streaming import StreamAssembler

    assembler = StreamAssembler()
    async for chunk in provider.stream(spec, system, [Message("user", user)], []):
        assembler.feed(chunk)
    turn = assembler.finish()
    return turn.text.strip(), turn.usage or {}


async def run_lessons(
    messages: list[Message],
    session_id: str,
    caller: Caller,
    model_id: str,
    path: Path,
    min_calls: int = MIN_TOOL_CALLS,
    timeout: float = LESSONS_TIMEOUT,
) -> Lessons:
    """Ask once at session end. Never raises: the status says recorded, quiet, skipped, failed."""
    started = time.monotonic()
    calls = tool_call_count(messages)
    if calls < min_calls:
        return Lessons(
            model=model_id,
            status="skipped",
            reason=f"{calls} tool calls, under the {min_calls} floor",
            path=str(path),
            calls=calls,
            seconds=time.monotonic() - started,
        )
    if session_id in seen_session_ids(path):
        return Lessons(
            model=model_id,
            status="skipped",
            reason=f"session {session_id} is already recorded in this file",
            path=str(path),
            calls=calls,
            seconds=time.monotonic() - started,
        )
    text, _dropped = digest(messages)
    if not text.strip():
        return Lessons(
            model=model_id,
            status="skipped",
            reason="the transcript has nothing a lessons pass could read",
            path=str(path),
            calls=calls,
            seconds=time.monotonic() - started,
        )
    prompt = PROMPT.format(digest=text)
    try:
        answer, usage = await asyncio.wait_for(caller(SYSTEM, prompt), timeout)
    except TimeoutError:
        return Lessons(
            model=model_id,
            status="failed",
            reason=f"the lessons model gave no answer in {timeout:.0f} s",
            path=str(path),
            calls=calls,
            seconds=time.monotonic() - started,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any provider failure is a status, never a crash
        return Lessons(
            model=model_id,
            status="failed",
            reason=f"the lessons call broke ({type(exc).__name__}: {exc})"[:300],
            path=str(path),
            calls=calls,
            seconds=time.monotonic() - started,
        )
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    entries, error = parse_lessons(answer)
    if error:
        return Lessons(
            model=model_id,
            status="failed",
            reason=error,
            path=str(path),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            calls=calls,
            seconds=time.monotonic() - started,
        )
    if not entries:
        return Lessons(
            model=model_id,
            status="quiet",
            reason="the lessons pass read the session and kept nothing",
            path=str(path),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            calls=calls,
            seconds=time.monotonic() - started,
        )
    written = append_entries(path, session_id, model_id, entries)
    return Lessons(
        model=model_id,
        status="recorded",
        entries=entries,
        reason=f"{len(entries)} candidate lesson(s) written for review",
        path=str(written),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        calls=calls,
        seconds=time.monotonic() - started,
    )


__all__ = [
    "EMAIL",
    "LESSONS_FILE",
    "MAX_DIGEST_CHARS",
    "MAX_LESSONS",
    "MIN_TOOL_CALLS",
    "Lessons",
    "append_entries",
    "call_model",
    "digest",
    "lessons_path",
    "parse_lessons",
    "run_lessons",
    "seen_session_ids",
    "tool_call_count",
]
