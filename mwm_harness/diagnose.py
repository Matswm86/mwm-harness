"""Failure diagnosis (plan item 31, the HERMES bug-diagnosis step only).

When the check chain or a test run fails after a turn that wrote files, one call answers
"where is this failure coming from": first deterministically, from traceback frames and the
check findings inside the repo (newest frame first, no model needed), then the critic model
fills the root cause and the guidance from the failure text plus that turn's diff.

The answer is advice handed to the next request, never a fix: nothing here edits a file, and
a diagnosis that cannot run says so instead of staying silent. A diagnosis whose note has been
sent once is dropped, so an old failure cannot keep riding on later requests.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Caller = Callable[[str, str], Awaitable[str]]  # (system, user) -> answer text

MAX_FAILURE_CHARS = 8_000  # failure text kept in one prompt
MAX_FILES = 8  # suspected files in one diagnosis
TAIL_LINES = 60  # output lines kept from a failing command

SOURCES = ("check", "test run", "manual")

TB_HEADER = re.compile(r"^(Traceback \(most recent call last\)|The above exception was)")
TB_FRAME = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+?)\s*$')
PYTEST_SHORT = re.compile(
    r"^(?P<file>[\w./\\-]+\.py):(?P<line>\d+): "
    r"(?P<rest>(?P<func>test_\w+)|E\s+(?P<err>\S.*))$"
)
SUMMARY_LINE = re.compile(r"^(?:FAILED |ERROR )[\w./\\-]+\.py")
CHECK_HEAD = re.compile(
    r"^(?P<check>\S[^:]*): (?P<label>FAIL|NOT RUN|TIMEOUT)(?::|,) ?(?P<rest>.*)$"
)
# One finding under a failing check row, and the shape pytest prints for a failed assertion:
# ``bad.py:1 SyntaxError: ...``, ``a.py:12:3: E   assert 1 == 2``, ``t.py:42: in test_x``.
# The name must hold a dot or a slash: ``ERROR: 404 not found`` is not a file position.
CHECK_FINDING = re.compile(
    r"^(?P<file>[\w./\\-]*[./][\w./\\-]*):(?P<line>\d+)(?::\d+)?:?\s+(?P<msg>\S.*)$"
)

SYSTEM = """\
You are diagnosing one failing run in a repository. You did not write the code and you fix \
nothing: you name where the failure comes from and what is worth checking first. Read only the \
failure text and the diff you are given. Never name a file, a line or a function that appears \
in neither. Plain English, no hedges. Answer with one JSON object and nothing else."""

PROMPT = """\
A run failed after the agent's last turn. Diagnose it.

WHERE THE FAILURE IS, EXTRACTED WITHOUT A MODEL
{frames}

TASK THE TURN WAS DOING
{task}

FAILURE TEXT ({source})
```
{failure}
```

THE TURN'S DIFF
{changes}

Answer with exactly this JSON shape:
{{
  "root_cause": "one or two sentences: the defect that makes this fail, or what you cannot \
tell from the evidence",
  "suspected_files": ["<path exactly as it appears above>", ...],
  "guidance": ["<first thing to check or change, one line>", "<second>", "<third>"]
}}

Rules:
- suspected_files: only files that appear in the traceback, the failure text or the diff, \
newest point of failure first, at most {max_files}. An empty list is a fine answer when none \
of them looks implicated.
- guidance: at most three ordered steps, each a short sentence, never a paragraph.
- Never restate the failure text as the root cause; say what is wrong in the code.
- If the evidence only shows an environment problem (a missing tool, a missing key, a \
forbidden operation), say that and leave suspected_files empty.{extra}"""


class DiagnoseError(Exception):
    pass


@dataclass
class Frame:
    """One repo-internal position a failure points at, newest first in ``Traceback.frames``."""

    file: str  # relative to the repo root
    line: int  # 0 when the failure names a file but no line (a FAILED summary row)
    func: str
    kind: str  # traceback | pytest | location (a file:line from a check or a test report)

    @property
    def place(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file

    def label(self) -> str:
        where = self.place
        return f"{where} in {self.func} ({self.kind})" if self.func else f"{where} ({self.kind})"


@dataclass
class Traceback:
    frames: list[Frame] = field(default_factory=list)
    error: str = ""  # e.g. "TypeError: unsupported operand"
    detail: str = ""  # the failure text kept for the prompt

    @property
    def found(self) -> bool:
        return bool(self.frames)

    def suspected(self, limit: int = MAX_FILES) -> list[str]:
        seen: list[str] = []
        for frame in self.frames:
            if frame.file not in seen:
                seen.append(frame.file)
        return seen[:limit]


@dataclass
class Diagnosis:
    root: str
    model: str
    status: str  # diagnosed (something to say) | skipped (nothing inside the repo to point at)
    source: str = ""
    reason: str = ""
    failure: str = ""
    suspected_files: list[str] = field(default_factory=list)
    root_cause: str = ""
    guidance: list[str] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    critic: bool = False  # the root cause and guidance came from a model
    extra_files: int = 0  # named by the model, not present in the evidence, so dropped
    round: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def note(self) -> str:
        """The text handed to the next request. Empty when there is nothing to say."""
        if self.status != "diagnosed" or self.sent:
            return ""
        lines = [f"FAILURE DIAGNOSIS ({self.source}, read by {self.model})"]
        if self.root_cause:
            lines.append(f"Root cause: {self.root_cause}")
        if self.suspected_files:
            lines.append("Suspected files: " + ", ".join(self.suspected_files))
        if self.frames and not self.suspected_files:
            lines.append(
                "Where it points in this repo: " + ", ".join(f.place for f in self.frames[:3])
            )
        if self.guidance:
            lines.append("Check first:")
            lines += [f"{n}. {step}" for n, step in enumerate(self.guidance, start=1)]
        if not self.critic:
            missed = f" ({self.reason})" if self.reason else ""
            lines.append(
                f"No critic model answered{missed}, so this is only where the failure points, "
                "not why."
            )
        lines.append("This is advice: nothing was changed. Open the files before editing them.")
        return "\n".join(lines)

    sent: bool = False


@dataclass
class DiagnoseBudget:
    """How many diagnoses one task may ask for before the harness stops and asks the person."""

    limit: int = 3
    used: int = 0

    def take(self) -> bool:
        """True when a round is left, counting it. 0 as the limit means no cap."""
        if self.limit and self.used >= self.limit:
            return False
        self.used += 1
        return True


# --------------------------------------------------- deterministic extraction


def _rel(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def _internal(path_text: str, root: Path) -> str | None:
    """A frame's path as a repo-relative path, or None when it is outside the repo."""
    if "<" in path_text:
        # ``<string>``, ``<stdin>``, ``<frozen importlib._bootstrap>``: positions with no file.
        return None
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists():
        # A deleted or renamed file still names the place the run reached.
        inside = _rel(root / path_text if not Path(path_text).is_absolute() else candidate, root)
        return inside if inside is not None and not path_text.startswith("/") else None
    return _rel(candidate, root)


def _tail_lines(text: str, count: int = TAIL_LINES) -> str:
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    return "\n".join(lines[-count:])


def extract_failure(text: str, root: Path) -> Traceback:
    """Repo-internal frames, the error line and the kept failure text. Never raises."""
    frames: list[Frame] = []
    error = ""
    body = text or ""
    lines = body.splitlines()
    # A traceback block ends at the exception line; its last frame is the point of failure.
    index = 0
    while index < len(lines):
        if TB_HEADER.match(lines[index]):
            block = []
            walk = index + 1
            while walk < len(lines):
                frame = TB_FRAME.match(lines[walk])
                if frame:
                    block.append(frame)
                    walk += 1
                    # Skip the source line Python prints under each frame.
                    while walk < len(lines) and not TB_FRAME.match(lines[walk]):
                        if not lines[walk].startswith("    "):
                            break
                        walk += 1
                    continue
                if lines[walk].startswith("    "):
                    walk += 1
                    continue
                break
            tail = lines[walk] if walk < len(lines) else ""
            if re.match(r"^\s*\S+(Error|Exception|Exit|Interrupt|Warning)\b", tail):
                error = tail.strip()
            for match in reversed(block):  # newest frame first
                inside = _internal(match["file"], root)
                if inside:
                    frames.append(
                        Frame(inside, int(match["line"]), match["func"].strip(), "traceback")
                    )
            index = walk
            continue
        short = PYTEST_SHORT.match(lines[index].strip())
        if short and (short["func"] or short["err"]):
            inside = _internal(short["file"], root)
            message = (short["err"] or "").strip()
            if inside:
                frames.append(
                    Frame(inside, int(short["line"]), (short["func"] or "").strip(), "pytest")
                )
            if message.startswith(("E ", "assert", "Error")) or message:
                error = error or message
            index += 1
            continue
        if SUMMARY_LINE.match(lines[index].strip()):
            head = lines[index].strip()
            path_text = head.split(":", 1)[0].removeprefix("FAILED ").removeprefix("ERROR ")
            inside = _internal(path_text, root)
            reason = head.split(" - ", 1)[1] if " - " in head else ""
            if inside:
                frames.append(Frame(inside, 0, "", "pytest"))
            error = error or reason
            index += 1
            continue
        check = CHECK_HEAD.match(lines[index].strip())
        if check:
            where = check["rest"].strip()
            hit = re.match(r"^(?P<file>[\w./-]+):(?P<line>\d+)", where)
            if hit:
                inside = _internal(hit["file"], root)
                if inside:
                    frames.append(Frame(inside, int(hit["line"]), "", "location"))
            index += 1
            continue
        finding = CHECK_FINDING.match(lines[index].strip())
        if finding:
            inside = _internal(finding["file"], root)
            message = finding["msg"].strip()
            func = message.removeprefix("in ").strip() if message.startswith("in ") else ""
            if inside:
                frames.append(Frame(inside, int(finding["line"]), func, "location"))
            error = error or message
            index += 1
            continue
        index += 1
    detail = _tail_lines(body)
    if len(detail) > MAX_FAILURE_CHARS:
        detail = detail[-MAX_FAILURE_CHARS:]
    return Traceback(frames=dedupe(frames), error=error, detail=detail)


def dedupe(frames: list[Frame]) -> list[Frame]:
    """Drop repeat frames of the same position; the traceback's own order is kept."""
    out: list[Frame] = []
    seen: set[tuple[str, int, str]] = set()
    for frame in frames:
        key = (frame.file, frame.line, frame.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(frame)
    return out


TEST_COMMAND = re.compile(
    r"\b(pytest|py\.test|unittest|nosetests|go test|cargo test|npm test|pnpm test|yarn test|"
    r"make test|mvn test|gradle test|rspec|jest|vitest|playwright test)\b"
)


def is_test_command(text: str) -> bool:
    """True when a shell command reads like a test run, so a passing one clears an earlier failure."""
    return bool(TEST_COMMAND.search(text or ""))


def is_test_failure(text: str) -> bool:
    """True when a command's output reads like a test run that did not pass."""
    body = text or ""
    if TB_HEADER.search(body):
        return True
    if re.search(r"^(?:FAILED |ERROR )[\w./\\-]+\.py", body, re.MULTILINE):
        return True
    if re.search(r"^\s*=+ .*(?:failed|error).* =+ .*", body, re.MULTILINE | re.IGNORECASE):
        return True
    return bool(re.search(r"^Ran \d+ test", body, re.MULTILINE)) and "FAILED" in body


def failing_checks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The check rows that failed; NOT RUN and TIMEOUT rows are reported, not diagnosed."""
    return [row for row in results if row.get("status") == "fail"]


def check_failure_text(results: list[dict[str, Any]]) -> str:
    """The failing check rows with their findings, the shape ``checks.CheckResult.line`` prints."""
    lines: list[str] = []
    for row in failing_checks(results):
        lines.append(f"{row['check']}: FAIL, {len(row.get('findings') or [])} finding(s)")
        for finding in (row.get("findings") or [])[:20]:
            where = f"{finding['file']}:{finding['line']}"
            lines.append(f"{where} {finding['message']}")
    return "\n".join(lines)


def run_failure_text(result: dict[str, Any]) -> str:
    """The summary line and pytest tail from a ``runit.RunItResult`` that failed."""
    parts = [str(result.get("summary") or "")]
    if result.get("tail"):
        parts.append(str(result["tail"]))
    return "\n".join(part for part in parts if part.strip())


# -------------------------------------------------------------- the prompt


def frame_lines(trace: Traceback) -> str:
    """The extracted positions, as the critic sees them under a heading."""
    rows = [frame.label() for frame in trace.frames[:MAX_FILES]]
    head = [f"Error: {trace.error}"] if trace.error else []
    return "\n".join(head + rows) if (head or rows) else "none found in this repo"


def build_prompt(
    source: str,
    task: str,
    trace: Traceback,
    changes_text: str,
    max_files: int = MAX_FILES,
    extra: str = "",
) -> str:
    return PROMPT.format(
        frames=frame_lines(trace),
        task=(task.strip() or "(the task this turn was doing is not known)"),
        source=source,
        failure=trace.detail or "(no output was captured)",
        changes=changes_text or "(no diff: nothing this turn wrote can be shown)",
        max_files=max_files,
        extra=extra,
    )


def _entries(value: Any, split_lines: bool = False) -> list[str]:
    """A JSON list as strings, or one string as its lines: a model answers either shape."""
    if isinstance(value, str):
        parts = value.splitlines() if split_lines else [value]
    else:
        parts = [str(entry) for entry in (value or [])]
    return [part.strip() for part in parts if str(part or "").strip()]


def parse_reply(
    text: str, trace: Traceback, known_files: list[str]
) -> tuple[str, list[str], list[str], int]:
    """Root cause, suspected files (only ones the evidence holds), guidance, dropped count."""
    data = _json_object(text)
    if data is None:
        raise DiagnoseError("the diagnosis answer held no JSON object")
    evidence = set(known_files) | set(trace.suspected())
    root_cause = str(data.get("root_cause") or "").strip()
    suspected: list[str] = []
    dropped = 0
    for path in _entries(data.get("suspected_files")):
        if path not in evidence:
            dropped += 1
            continue
        if path not in suspected:
            suspected.append(path)
        if len(suspected) >= MAX_FILES:
            break
    guidance = [
        step.replace("\n", " ") for step in _entries(data.get("guidance"), split_lines=True)[:3]
    ]
    return root_cause, suspected, guidance, dropped


def _json_object(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in ``text``, or None. Braces inside strings are skipped."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escape = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return data if isinstance(data, dict) else None
        start = text.find("{", start + 1)
    return None


def format_changes(changes: list[Any]) -> str:
    """The turn's diffs as prompt text (``review.FileChange`` objects, duck-typed here)."""
    blocks = []
    for change in changes:
        head = f"### {change.path}" + (" (new file)" if getattr(change, "new", False) else "")
        blocks.append(f"{head}\n```diff\n{change.diff}\n```")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------- run


def deterministic(
    root: Path, source: str, trace: Traceback, model_id: str, round_no: int = 0
) -> Diagnosis:
    """The no-model half: where the failure points inside this repo."""
    return Diagnosis(
        root=str(root),
        model=model_id,
        status="diagnosed" if trace.found else "skipped",
        source=source,
        reason="" if trace.found else "no failure inside this repo to point at",
        failure=trace.detail,
        suspected_files=trace.suspected(),
        frames=trace.frames,
        critic=False,
        round=round_no,
    )


async def run_diagnosis(
    root: Path,
    source: str,
    failure_text: str,
    task: str,
    changes_text: str,
    known_files: list[str],
    caller: Caller | None,
    model_id: str,
    round_no: int = 0,
    timeout: float = 180.0,
    extra: str = "",
) -> Diagnosis:
    """Diagnose one failure.

    The deterministic half works alone: a critic that is down, times out or answers with nothing
    readable costs the "why", never the "where".
    """
    trace = extract_failure(failure_text, root)
    base = deterministic(root, source, trace, model_id, round_no)
    if caller is None:
        missed = "no critic model is configured: the location comes from the failure text alone"
        base.reason = f"{base.reason}; {missed}" if base.reason else missed
        return base
    prompt = build_prompt(source, task, trace, changes_text, extra=extra)
    started = time.monotonic()

    def critic_could_not_read(why: str) -> Diagnosis:
        # The location extracted without a model is still shipped: a critic that is down costs
        # the "why", not the "where". Status says what the diagnosis as a whole could do.
        base.reason = f"{base.reason}; {why}" if base.reason else why
        base.seconds = time.monotonic() - started
        return base

    try:
        answer = await asyncio.wait_for(caller(SYSTEM, prompt), timeout)
    except TimeoutError:
        return critic_could_not_read(f"the critic model gave no answer in {timeout:.0f} s")
    except Exception as exc:  # noqa: BLE001 - any provider failure costs the why, not the where
        return critic_could_not_read(f"the critic model could not be reached: {exc}"[:300])
    try:
        root_cause, suspected, guidance, dropped = parse_reply(answer, trace, known_files)
    except DiagnoseError as exc:
        return critic_could_not_read(f"{exc}; the raw answer starts: {answer[:200]!r}")
    base.status = "diagnosed"
    base.reason = ""
    base.root_cause = root_cause
    base.suspected_files = suspected or trace.suspected()
    base.guidance = guidance
    base.extra_files = dropped
    base.critic = True
    base.model = model_id
    base.seconds = time.monotonic() - started
    return base
