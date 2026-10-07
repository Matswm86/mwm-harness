"""Postmortem: read one session transcript and say which harness layer failed.

Every failure the transcript records (tool errors, hook and permission blocks,
Stop-gate feedback, cancelled turns, compactions, slow tools) is sorted into the
harness layer it points at, and each layer gets the kind of fix that keeps the
same failure from coming back: a test, a hook, a skill, an instruction, a
permission rule. The report is a draft. Nothing here edits hooks, settings,
tests or skills; ``/postmortem draft`` hands the report to the model, and every
file it then writes goes through the normal approval.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mwm_harness.transcript import sessions_root

SLOW_TOOL_SECONDS = 30.0
MAX_EXAMPLES = 3

# Harness layer -> the fix that turns a failure of that layer into infrastructure.
FIXES = {
    "tools": "a stricter tool description or argument check, plus a regression test",
    "permissions": "a permission rule (allow or deny) or one CLAUDE.md line, so the call is not tried",
    "orchestration": "an instruction or skill step, so the model stops reaching for what the hook blocks",
    "environment": "a skill step that checks the command, path or package before running it",
    "verification": "a skill step that runs the check the Stop gate asked for before answering",
    "context": "a retrieval rule or a smaller first read, so the window lasts the task",
    "loop": "a skill or instruction for the case the model kept retrying",
    "person": "a question to the person: why did they stop the turn",
    "observability": "a timeout or a faster tool for the slowest calls",
}


@dataclass
class Finding:
    layer: str
    kind: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, example: str) -> None:
        self.count += 1
        text = " ".join(example.split())[:200]
        if text and len(self.examples) < MAX_EXAMPLES and text not in self.examples:
            self.examples.append(text)


@dataclass
class SlowCall:
    name: str
    seconds: float
    summary: str


@dataclass
class Postmortem:
    session_id: str
    path: Path
    started: str = ""
    ended: str = ""
    turns: int = 0
    tool_calls: int = 0
    findings: dict[tuple[str, str], Finding] = field(default_factory=dict)
    slow: list[SlowCall] = field(default_factory=list)

    def note(self, layer: str, kind: str, example: str = "") -> None:
        key = (layer, kind)
        if key not in self.findings:
            self.findings[key] = Finding(layer, kind)
        self.findings[key].add(example)

    def ranked(self) -> list[Finding]:
        return sorted(self.findings.values(), key=lambda f: (-f.count, f.layer, f.kind))


def classify_error(tool: str, content: str) -> tuple[str, str]:
    """(layer, kind) for one failed tool result, from the texts the loop writes."""
    if content.startswith(("The call was not run:", "unknown tool")):
        return "tools", "bad arguments or unknown tool"
    if " crashed: " in content:
        return "tools", f"{tool} crashed"
    if content.startswith("Blocked by a PreToolUse hook"):
        return "orchestration", "PreToolUse hook blocked a call"
    if content.startswith("The user declined"):
        return "permissions", "person declined a call"
    if content.startswith("Not run: this exact call was made"):
        return "loop", "identical call repeated until the cap"
    if content.startswith("Not run:") and "edit-and-run cycles" in content:
        return "loop", "edit-and-run without reading, until the cap"
    if content.startswith(("Interrupted", "Interrupted before")):
        return "person", "person interrupted a tool"
    if "denied" in content.lower() or "deny list" in content.lower():
        return "permissions", "deny list refused a call"
    if tool == "Bash":
        return "environment", "shell command failed"
    return "tools", f"{tool} returned an error"


def seconds_between(start: str, end: str) -> float | None:
    """Seconds from one ISO stamp to another; None when either is missing or unreadable."""
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (b - a).total_seconds()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return json.dumps(content, ensure_ascii=False)


def analyze(path: Path) -> Postmortem:
    """Walk the transcript once. Torn or unknown lines are skipped, as on resume."""
    report = Postmortem(path.stem, path)
    pending: dict[str, tuple[str, str, dict[str, Any]]] = {}  # tool id -> (name, ts, input)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = str(entry.get("timestamp") or "")
        if ts:
            report.started = report.started or ts
            report.ended = ts
        if entry.get("type") == "system":
            if entry.get("subtype") == "compact_boundary":
                report.note("context", "history compacted", str(entry.get("content", "")))
            continue
        body = entry.get("message") or {}
        content = body.get("content")
        if entry.get("type") == "assistant":
            if body.get("stop_reason") == "cancelled":
                report.note("person", "person cancelled the turn", _text(content))
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use":
                    report.tool_calls += 1
                    pending[block.get("id", "")] = (
                        block.get("name", "?"),
                        ts,
                        block.get("input") or {},
                    )
            continue
        if entry.get("type") != "user":
            continue
        if isinstance(content, str):
            if content.startswith("Stop hook feedback:"):
                reason = content.split("\n", 1)[-1]
                report.note("verification", "Stop gate sent the answer back", reason)
            elif not entry.get("isMeta"):
                report.turns += 1
            continue
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "text" and not entry.get("isMeta"):
                report.turns += 1
            if block.get("type") != "tool_result":
                continue
            name, started, tool_input = pending.pop(block.get("tool_use_id", ""), ("?", "", {}))
            result = _text(block.get("content"))
            if block.get("is_error"):
                report.note(*classify_error(name, result), f"{name}: {result}")
            took = seconds_between(started, ts) if started and ts else None
            if took is not None and took >= SLOW_TOOL_SECONDS:
                summary = json.dumps(tool_input, ensure_ascii=False)[:120]
                report.slow.append(SlowCall(name, took, summary))
    report.slow.sort(key=lambda s: -s.seconds)
    if report.slow:
        for call in report.slow:
            report.note(
                "observability",
                f"tool call over {SLOW_TOOL_SECONDS:.0f}s",
                f"{call.name} {call.seconds:.0f}s",
            )
    return report


def render(report: Postmortem) -> str:
    span = (
        seconds_between(report.started, report.ended) if report.started and report.ended else None
    )
    lines = [
        f"# Postmortem: session {report.session_id[:8]}",
        "",
        f"Transcript: {report.path}",
        f"Span: {report.started or '?'} to {report.ended or '?'}"
        + (f" ({span / 60:.0f} min)" if span is not None else ""),
        f"Turns: {report.turns} · tool calls: {report.tool_calls} · "
        f"failure classes: {len(report.findings)}",
        "",
    ]
    ranked = report.ranked()
    if not ranked:
        lines.append(
            "No failures recorded: no tool errors, blocks, Stop-gate returns, cancels or slow calls."
        )
        return "\n".join(lines) + "\n"
    lines += ["## What failed, by harness layer", ""]
    by_layer = Counter()
    for finding in ranked:
        by_layer[finding.layer] += finding.count
        lines.append(f"- **{finding.layer}** · {finding.kind}: {finding.count}x")
        lines += [f"  - {example}" for example in finding.examples]
    lines += ["", "## Proposed fixes (drafts; nothing was changed)", ""]
    for layer, count in by_layer.most_common():
        lines.append(f"- **{layer}** ({count} failures): {FIXES[layer]}")
    if report.slow:
        lines += ["", "## Slowest tool calls", ""]
        lines += [f"- {s.name} {s.seconds:.1f}s: {s.summary}" for s in report.slow[:5]]
    return "\n".join(lines) + "\n"


def postmortems_root() -> Path:
    return sessions_root().parent / "postmortems"


def write(report: Postmortem, root: Path | None = None) -> Path:
    folder = root or postmortems_root()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{report.session_id}.md"
    target.write_text(render(report), encoding="utf-8")
    return target


DRAFT_PROMPT = """\
Read the postmortem report at {report}. Pick the ONE failure class with the
highest count whose fix is a hook, test, skill or instruction, and write a
draft of that one fix to {draft}. The draft names the file the fix would go in,
shows the exact text or code, and says which failure in the report it stops.
Do not edit any hook, setting, test, skill or CLAUDE.md file: the person reads
the draft and applies it.
"""
