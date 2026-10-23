"""Review Desk: a second reader for the files a turn changed.

One model call at the end of a turn that wrote files, on the reviewer model, over
that turn's diff. It answers with a short summary, one plain-English walkthrough
line per file, and inline comments carrying a severity. Nothing here is applied:
a comment becomes the person's next turn when they send it, and a comment marked
"remember" is appended to the repo's ``.mwm/review-learnings.md``, which every
later review reads.

The diff comes from the undo log's before-bytes when it has them (exact, and it
works outside a git repo), else from ``git diff``, else from the file's own text
shown as new. A review that cannot run says so: a skip is never reported as a
review, and a reply that will not parse is a failure with the raw text kept.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mwm_harness.messages import Message
from mwm_harness.streaming import StreamAssembler

Caller = Callable[[str, str], Awaitable[str]]  # (system, user) -> answer text

LEARNINGS_FILE = Path(".mwm") / "review-learnings.md"
LEARNINGS_CAP = 60  # lines kept, newest wins
MAX_FILES = 12  # files in one review; the rest are named, not shown
MAX_DIFF_LINES = 240  # diff lines per file
MAX_DIFF_CHARS = 60_000  # total diff text in one prompt
GIT_TIMEOUT = 15
SEVERITIES = ("critical", "suggestion", "note")
# Files whose change also gets the security reviewer's reading (auth, secrets, LLM surface).
SENSITIVE = re.compile(
    r"(auth|secret|token|credential|permission|sandbox|redact|provider|hooks?|"
    r"mcp_client|loop|approv|deny)",
    re.IGNORECASE,
)

SYSTEM = """\
You are a second reader reviewing one turn of an agent's work. You did not write it and you \
never claim credit for it. Read only the diff you are given: do not assume what the rest of \
the file says, and do not invent a line, a number or a reason. Plain English a stranger \
understands, no internal labels, no hedges. Answer with one JSON object and nothing else."""

PROMPT = """\
Review this turn's changes.

{learnings}
TASK
{task}

CHANGED FILES
{changes}

Answer with exactly this JSON shape:
{{
  "summary": "one or two sentences: what this turn did and the number that matters",
  "files": [{{"path": "<path as given above>", "walkthrough": "one line: what changed here and why it matters"}}],
  "comments": [{{"file": "<path as given above>", "line": <line number in the NEW file>, \
"severity": "critical|suggestion|note", "text": "the problem in one sentence", \
"fix": "the fix in one sentence, or empty"}}]
}}

Rules for comments:
- Only lines the diff adds or changes. "line" is the number in the new file, from the @@ header.
- "critical" only for a defect that breaks a run, loses data, weakens a safety check or publishes \
a secret. Style and taste are "suggestion". Context the person should have is "note".
- Fewer, better: at most 3 comments per file, and none when the change is sound. An empty list is \
a fine answer.
- Never repeat a check tool's finding (lint, type-check); the harness already reports those.{extra}"""

SECURITY_EXTRA = """

Also read this change as a security reviewer: a secret that reaches a log, a transcript, a \
prompt or a network body; a permission, deny rule or sandbox boundary that got wider; a path \
that trusts input from outside the box; a destructive operation reachable without a person \
answering. Report what you find as "critical" comments, and say what an attacker or a stray \
tool call would actually get. If you find nothing, add no comment: do not pad."""


class ReviewError(Exception):
    pass


@dataclass
class FileChange:
    path: str  # relative to the repo root
    diff: str  # unified diff, capped
    added: int = 0
    removed: int = 0
    new: bool = False
    truncated: bool = False
    walkthrough: str = ""
    callers: list[str] = field(default_factory=list)  # item 32: "loop._run_tools" style
    unresolved: list[str] = field(default_factory=list)  # calls only resolvable at runtime

    @property
    def sensitive(self) -> bool:
        return bool(SENSITIVE.search(self.path))


@dataclass
class Comment:
    file: str
    line: int
    severity: str  # critical | suggestion | note
    text: str
    fix: str = ""
    author: str = "reviewer"  # reviewer | mats


@dataclass
class Review:
    root: str
    model: str
    status: str  # reviewed | skipped | failed
    reason: str = ""
    summary: str = ""
    files: list[FileChange] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    learnings: int = 0
    dropped: int = 0  # comments the reviewer put on a file or line outside the diff
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------- diffs


def _decode(data: bytes | None) -> list[str]:
    if data is None:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def _unified(path: str, old: list[str], new: list[str]) -> tuple[str, bool]:
    """The diff body under a ---/+++ pair, cut to ``MAX_DIFF_LINES`` lines, and whether it was cut."""
    lines = list(
        difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3)
    )
    if not lines:
        return "", False
    body = lines[2:]  # drop the ---/+++ pair; the paths are written back below, once
    truncated = len(body) > MAX_DIFF_LINES
    kept = body[:MAX_DIFF_LINES]
    head = f"--- a/{path}\n+++ b/{path}\n"
    if truncated:
        kept.append(f"@@ {len(body) - MAX_DIFF_LINES} more line(s) not shown @@")
    return head + "\n".join(kept), truncated


def _counts(diff: str) -> tuple[int, int]:
    """Added and removed lines, ignoring the ---/+++ header pair."""
    body = [line for line in diff.splitlines() if not line.startswith(("--- ", "+++ "))]
    return (
        sum(1 for line in body if line.startswith("+")),
        sum(1 for line in body if line.startswith("-")),
    )


def git_diff(root: Path, path: Path) -> tuple[str, bool, bool]:
    """``git diff`` for one file: the diff, whether it was cut, and whether the file is new."""
    relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    def run(*args: str) -> str:
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return out.stdout if out.returncode == 0 else ""

    diff = run("diff", "--no-color", "--unified=3", "--", relative)
    if diff.strip():
        lines = diff.splitlines()
        truncated = len(lines) > MAX_DIFF_LINES + 2
        return "\n".join(lines[:MAX_DIFF_LINES]), truncated, False
    if run("ls-files", "--error-unmatch", "--", relative).strip():
        return "", False, False  # tracked and unmodified: nothing to show
    if not path.is_file():
        return "", False, False
    diff, truncated = _unified(relative, [], _decode(path.read_bytes()))
    return diff, truncated, True


def collect_changes(
    root: Path, files: list[Path], before: dict[Path, bytes | None] | None = None
) -> list[FileChange]:
    """One ``FileChange`` per changed file, newest state against its pre-turn state."""
    before = before or {}
    changes: list[FileChange] = []
    budget = MAX_DIFF_CHARS
    for path in files[:MAX_FILES]:
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        known = path in before or path.resolve() in before
        if known:
            old_bytes = before[path] if path in before else before[path.resolve()]
            new = _decode(path.read_bytes()) if path.is_file() else []
            diff, truncated = _unified(relative, _decode(old_bytes), new)
            is_new = old_bytes is None
        else:
            diff, truncated, is_new = git_diff(root, path)
        if not diff:
            continue
        if budget - len(diff) < 0:
            changes.append(FileChange(relative, "@@ left out: the review prompt was full @@", 0, 0))
            break
        budget -= len(diff)
        added, removed = _counts(diff)
        changes.append(FileChange(relative, diff, added, removed, is_new, truncated))
    return changes


def skipped_names(root: Path, files: list[Path]) -> list[str]:
    """Files beyond ``MAX_FILES``, named so the person sees the review was partial."""
    out = []
    for path in files[MAX_FILES:]:
        try:
            out.append(str(path.relative_to(root)))
        except ValueError:
            out.append(str(path))
    return out


# --------------------------------------------------------------- learnings


def learnings_path(root: Path) -> Path:
    return root / LEARNINGS_FILE


def read_learnings(root: Path, cap: int = LEARNINGS_CAP) -> list[str]:
    """The standing notes from earlier reviews, newest last, at most ``cap`` of them."""
    path = learnings_path(root)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    kept = [line.strip() for line in lines if line.strip().startswith("- ")]
    return kept[-cap:]


def remember(root: Path, text: str, where: str = "", cap: int = LEARNINGS_CAP) -> Path:
    """Append one learning line and keep the file at ``cap`` lines, newest winning."""
    line = text.strip().replace("\n", " ")
    if not line:
        raise ReviewError("an empty note cannot be remembered")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = f"- {stamp} {line}" + (f" ({where})" if where else "")
    path = learnings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    kept = [item for item in old if item.strip().startswith("- ")][-(cap - 1) :]
    head = [item for item in old if not item.strip().startswith("- ")]
    body = head or ["# Review learnings", "", "Standing notes for the next review. Newest wins."]
    path.write_text("\n".join([*body, *kept, entry]) + "\n", encoding="utf-8")
    return path


def remember_comment(root: Path, comment: Comment, cap: int = LEARNINGS_CAP) -> Path:
    """Keep a reviewer comment as a standing note for every later review of this repo."""
    return remember(root, comment.text, f"{comment.file}:{comment.line}", cap)


def _git_dir(root: Path) -> Path | None:
    """This clone's ``.git`` folder, following the pointer file a worktree carries."""
    entry = root / ".git"
    if entry.is_dir():
        return entry
    if not entry.is_file():
        return None
    try:
        text = entry.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    target = Path(text.removeprefix("gitdir:").strip())
    return target if target.is_absolute() else (root / target).resolve()


def exclude_from_git(root: Path) -> str:
    """Keep ``.mwm/review-learnings.md`` out of this clone's index. "" when it is out.

    The notes name real defects in the repo, and a public repo publishes anything committed,
    so the plan puts them in ``.git/info/exclude`` (local to this clone, never committed)
    instead of ``.gitignore``. A non-empty return is the reason it could not be done, and the
    caller says so out loud: an untracked file in a public repo is one bulk add from being
    published, which already happened once in this workspace.
    """
    git = _git_dir(root)
    if git is None:
        return ""  # no clone here, so nothing can publish the file
    exclude = git / "info" / "exclude"
    try:
        listed = exclude.read_text(encoding="utf-8").splitlines() if exclude.is_file() else []
        if any(line.strip() == str(LEARNINGS_FILE) for line in listed):
            return ""
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as handle:
            if listed and listed[-1].strip():
                handle.write("\n")
            handle.write(f"{LEARNINGS_FILE}\n")
    except OSError as exc:
        return (
            f"{exclude} could not be written ({exc}), so {LEARNINGS_FILE} is untracked "
            "in this repo: do not add everything with one command here"
        )
    return ""


# ------------------------------------------------- a comment as the next turn


def comment_text(comment: Comment, note: str = "") -> str:
    """The user text a Review Desk comment becomes when the person sends it.

    It quotes file:line so the agent opens the exact line, and names who wrote what: the
    reviewer's finding is advice, the person's own note below it is the decision.
    """
    lines = [
        f"Review comment on {comment.file}:{comment.line} "
        f"({comment.severity}, written by the {comment.author} model):",
        comment.text,
    ]
    if comment.fix:
        lines.append(f"Its proposed fix: {comment.fix}")
    if note:
        lines.append(f"My decision: {note}")
    return "\n".join(lines)


# ------------------------------------------------------------------ prompt


def build_prompt(
    task: str,
    changes: list[FileChange],
    learnings: list[str],
    extra: str = "",
    left_out: list[str] | None = None,
) -> str:
    blocks = []
    for change in changes:
        head = f"### {change.path}" + (" (new file)" if change.new else "")
        if change.callers:
            head += f"\ncalled by {len(change.callers)}: {', '.join(change.callers[:6])}"
        if change.unresolved:
            head += f"\ncalls resolved only at runtime: {', '.join(change.unresolved[:6])}"
        blocks.append(f"{head}\n```diff\n{change.diff}\n```")
    if left_out:
        blocks.append("### not reviewed (over the file cap)\n" + "\n".join(left_out))
    notes = ""
    if learnings:
        notes = (
            "STANDING NOTES FROM EARLIER REVIEWS BY THE PERSON (apply them, do not repeat them "
            "back as comments)\n" + "\n".join(learnings) + "\n"
        )
    return PROMPT.format(
        learnings=notes,
        task=(task.strip() or "(the person's prompt for this turn is not known)"),
        changes="\n\n".join(blocks),
        extra=extra,
    )


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


def parse_reply(
    text: str, changes: list[FileChange]
) -> tuple[str, dict[str, str], list[Comment], int]:
    """Summary, walkthrough per path, comments on lines the diff really touches, drop count."""
    data = _json_object(text)
    if data is None:
        raise ReviewError("the reviewer's answer held no JSON object")
    known = {change.path: change for change in changes}
    added_lines = {path: _added_line_numbers(change.diff) for path, change in known.items()}
    summary = str(data.get("summary") or "").strip()
    walkthrough: dict[str, str] = {}
    for entry in data.get("files") or []:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        line = str(entry.get("walkthrough") or "").strip()
        if path in known and line:
            walkthrough[path] = line
    comments: list[Comment] = []
    dropped = 0
    for entry in data.get("comments") or []:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        path = str(entry.get("file") or entry.get("path") or "").strip()
        body = str(entry.get("text") or "").strip()
        try:
            line = int(entry.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        severity = str(entry.get("severity") or "note").strip().lower()
        if severity not in SEVERITIES:
            severity = "note"
        if path not in known or not body:
            dropped += 1
            continue
        # A comment on a line the diff never touched is dropped: the reviewer is guessing.
        if line > 0 and added_lines[path] and line not in added_lines[path]:
            dropped += 1
            continue
        comments.append(
            Comment(
                path,
                line,
                severity,
                body,
                str(entry.get("fix") or "").strip(),
            )
        )
    return summary, walkthrough, comments, dropped


def _added_line_numbers(diff: str) -> set[int]:
    """The new-file line numbers a diff adds or replaces, read from its @@ headers."""
    out: set[int] = set()
    number = 0
    for line in diff.splitlines():
        head = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)", line)
        if head:
            number = int(head.group(1))
            continue
        if line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith("+"):
            out.add(number)
            number += 1
        elif line.startswith("-"):
            continue
        elif number:
            number += 1
    return out


# -------------------------------------------------------------------- run


async def call_model(provider: Any, spec: Any, system: str, user: str) -> str:
    """One model call with no tools, the shape every reviewer request uses."""
    assembler = StreamAssembler()
    async for chunk in provider.stream(spec, system, [Message("user", user)], []):
        assembler.feed(chunk)
    return assembler.finish().text.strip()


async def review_changes(
    root: Path,
    task: str,
    changes: list[FileChange],
    caller: Caller,
    model_id: str,
    learnings: list[str] | None = None,
    extra: str = "",
    left_out: list[str] | None = None,
    timeout: float = 300.0,
) -> Review:
    """Ask the reviewer model about ``changes``. Never raises for a bad answer: status says."""
    learnings = learnings if learnings is not None else read_learnings(root)
    started = time.monotonic()
    if not changes:
        return Review(
            str(root), model_id, "skipped", "nothing in this turn's files had a diff to read"
        )
    prompt = build_prompt(task, changes, learnings, extra, left_out)
    try:
        answer = await asyncio.wait_for(caller(SYSTEM, prompt), timeout)
    except TimeoutError:
        return Review(
            str(root),
            model_id,
            "failed",
            f"the reviewer model gave no answer in {timeout:.0f} s",
            learnings=len(learnings),
            seconds=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure is a failed review, not a crash
        return Review(
            str(root),
            model_id,
            "failed",
            f"the reviewer model could not be reached: {exc}"[:300],
            learnings=len(learnings),
            seconds=time.monotonic() - started,
        )
    try:
        summary, walkthrough, comments, dropped = parse_reply(answer, changes)
    except ReviewError as exc:
        return Review(
            str(root),
            model_id,
            "failed",
            f"{exc}; the raw answer starts: {answer[:200]!r}",
            learnings=len(learnings),
            seconds=time.monotonic() - started,
        )
    for change in changes:
        change.walkthrough = walkthrough.get(change.path, "")
    return Review(
        str(root),
        model_id,
        "reviewed",
        "",
        summary,
        changes,
        comments,
        len(learnings),
        dropped,
        time.monotonic() - started,
    )


def lock_held(lock_dir: str | Path) -> str:
    """Why a review is skipped because another reviewer already runs, else an empty string.

    ``lock_dir`` holds PID-stamped lock files named ``.active-<pid>`` (the MWM workspace's
    autoreview lock). A lock whose process is gone is stale and does not skip anything.
    """
    if not lock_dir:
        return ""
    folder = Path(str(lock_dir)).expanduser()
    if not folder.is_dir():
        return ""
    for lock in sorted(folder.glob(".active-*")):
        pid = lock.name.removeprefix(".active-")
        if not pid.isdigit():
            continue
        if Path(f"/proc/{pid}").is_dir():
            return f"a code reviewer already runs (PID {pid}); this review is skipped, not passed"
    return ""
