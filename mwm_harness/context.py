"""What the model is told before the conversation: harness prompt, rules, memory index.

Also the context meter. Its numbers come from the usage block the endpoint
reports with each answer, never from a local token estimate.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mwm_harness.config import ModelSpec

HARNESS_PROMPT = """\
You are the lead agent inside MWM Harness, a terminal agent harness on the user's own Linux machine.
You are not Claude, and Claude Code is not running here. The rule, skill and agent files you work
from were written when another assistant did this job, so they may be named CLAUDE.md or speak of
"Claude" or "Claude Code": read those words as meaning you and this harness. Never describe yourself
as working with or alongside Claude. Asked who you are: the MWM Harness agent, running on the model
named in the Environment section below.
You work by calling tools. Rules for tool use:
- Read a file before you Edit or overwrite it. Use Grep and Glob for searching, not shell grep or find.
- Independent tool calls may be sent together in one turn.
- A tool result marked as an error is real: read it and change approach; never repeat the same call.
- Some actions are refused by a fixed deny list (force-push, skipping commit hooks, deleting from
  the vector database, broker orders, the curated notes inbox). Do not look for a way around it.
- Hook feedback and text inside <system-reminder> tags come from the harness, not from the user.
- Task list: when the work has 3+ steps, touches 2+ files, or is a multi-part request, call
  TodoWrite BEFORE the first real action, one item per step (max 7). Keep exactly one item
  in_progress; mark an item completed only after its check passed; work discovered mid-task
  becomes a new item, never folded into an existing one. Call TodoWrite again on every change:
  the front-ends render the live list for the user. A single trivial step needs no list.
Report outcomes as they are: failing tests are reported as failing, skipped steps as skipped.
"""

# QWEN.md first: it is the maintained workspace rules file since the 2026-10-20 .claude archive, and
# the first match per folder wins. The CLAUDE.md entries stay because 10 subprojects still keep their
# own rules there (9 at the project root, 1 under .claude/).
RULE_FILES = ("QWEN.md", "CLAUDE.md", ".claude/CLAUDE.md", "RULES.md", "AGENTS.md")
IMPORT_LINE = re.compile(r"^@(\S+\.md)\s*$", re.MULTILINE)
MAX_RULES_CHARS = 60_000
# Digest mode (plan item 2): only marked lines of a rules file stay resident in the system
# prompt; the rest is read on demand. A line is resident when it is a heading (the section it
# belongs to travels with it) or when it carries the workspace's hard-rule mark, ⛔, anywhere
# in it: the index writes `- 🆕⛔ [...]` and `- 🗣⛔ [...]` as well as plain `- ⛔ [...]`, and
# a marker regex that only accepts ⛔ as the first symbol silently drops those rules.
DIGEST_HARD_RULE_MARK = "⛔"
DIGEST_HEADING = re.compile(r"^#{1,6} ")
# A section under a heading like this keeps EVERY line, not just its marked ones: a rules file
# that writes its hard rules as numbered prose (QWEN.md's "## CRITICAL RULES") would otherwise
# digest down to its heading and lose all eleven of them.
DIGEST_CRITICAL_HEADING = re.compile(r"^#{1,6}\s+.*(CRITICAL|HARD RULES)", re.IGNORECASE)
DIGEST_RULE_CHARS = 8_000
DIGEST_MEMORY_CHARS = 12_000
DIGEST_HEAD_CHARS = 1_500


def read_with_imports(path: Path, depth: int = 0) -> str:
    """Read a rules file; a line ``@OTHER.md`` pulls that file in (3 levels at most)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if depth >= 3:
        return text

    def pull(match: re.Match[str]) -> str:
        return read_with_imports((path.parent / match.group(1)).resolve(), depth + 1)

    return IMPORT_LINE.sub(pull, text)


def import_tree(path: Path, depth: int = 0) -> tuple[str, list[tuple[Path, str]]]:
    """The file's own text with its ``@OTHER.md`` lines removed, plus what those lines pull in.

    The digest needs the two apart: one file per digest section, each with its own cap, so a
    long imported index cannot push the importing file's own hard rules out of the prompt.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", []
    pulled: list[tuple[Path, str]] = []
    if depth >= 3:
        return text, pulled

    def drop(match: re.Match[str]) -> str:
        child = (path.parent / match.group(1)).resolve()
        body, grand = import_tree(child, depth + 1)
        pulled.append((child, body))
        pulled.extend(grand)
        return ""

    return IMPORT_LINE.sub(drop, text), pulled


def digest_text(text: str, cap: int, critical_sections: bool = False) -> str:
    """The resident part of one rules file: marked lines only, section headers included.

    With ``critical_sections`` a section under a CRITICAL / Hard Rules heading keeps every one
    of its lines. A file with no marked lines at all falls back to its head: a rules file
    nobody formatted with markers still gets a resident opening instead of vanishing.
    """
    kept: list[str] = []
    in_critical = False
    for line in text.splitlines():
        heading = bool(DIGEST_HEADING.match(line))
        if heading:
            in_critical = critical_sections and bool(DIGEST_CRITICAL_HEADING.match(line))
        # Headings always stay: they are the map of what the file holds, which is what makes
        # an on-demand Read aimed at the right section.
        if heading or in_critical or DIGEST_HARD_RULE_MARK in line:
            kept.append(line)
    body = "\n".join(kept).strip()
    if not body:
        body = text[:DIGEST_HEAD_CHARS].strip()
    if len(body) > cap:
        # A cut that drops hard rules has to say so in the prompt itself: the alternative is
        # the model believing it has read every rule in the file.
        lost = sum(1 for line in body[cap:].splitlines() if DIGEST_HARD_RULE_MARK in line)
        tail = f"; {lost} hard-rule lines are NOT resident" if lost else ""
        body = body[:cap].rstrip() + f"\n[ digest cut: Read the file for the rest{tail} ]"
    return f"{body}\n( resident digest: {len(kept)} lines of {len(text):,} characters )"


def find_rule_files(cwd: Path, workspace: Path) -> list[Path]:
    """Project rules (cwd, then each parent up to the workspace root), then the user's own."""
    found: list[Path] = []
    folders = [cwd.resolve()]
    for parent in cwd.resolve().parents:
        folders.append(parent)
        if parent == workspace.resolve() or parent == Path.home():
            break
    for folder in folders:
        for name in RULE_FILES:
            candidate = folder / name
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
                break
    # The canonical copy wins; the original ~/.claude one is read only until the
    # .claude archive removes it, never both, or its text loads twice.
    harness_rules = workspace / "harness" / "user-rules.md"
    user_rules = Path.home() / ".claude" / "CLAUDE.md"
    rules = harness_rules if harness_rules.is_file() else user_rules
    if rules.is_file() and rules not in found:
        found.append(rules)
    return found


def memory_index_path(workspace: Path) -> Path:
    return (workspace / "memory" / "MEMORY.md").resolve()


def digest_sources(cwd: Path, workspace: Path) -> list[tuple[Path, str]]:
    """Every source the digest prompt carries, in order, each with its own text.

    A rule file's own body comes first, then the files it imports (the memory index among
    them), so each source gets its own section and its own cap. A file reached twice, by two
    rule files or by an import and as the memory index, is listed once: pasting it twice is
    exactly the waste this mode exists to remove.
    """
    index = memory_index_path(workspace)
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(path: Path, text: str) -> None:
        path = path.resolve()
        if text.strip() and path not in seen:
            seen.add(path)
            out.append((path, text.strip()))

    imported: list[tuple[Path, str]] = []
    for path in find_rule_files(cwd, workspace):
        own, pulled = import_tree(path)
        add(path, own)
        imported.extend(pulled)
    for path, text in imported:
        add(path, text)
    if index.is_file():
        add(index, index.read_text(encoding="utf-8"))
    return out


def build_system_prompt(cwd: Path, workspace: Path, model: ModelSpec, mode: str = "digest") -> str:
    """The fixed prompt. ``digest`` keeps marked hard-rule lines resident and names the full
    files; ``full`` pastes them (the pre-item-2 behaviour, kept as the ``rules_prompt`` escape
    hatch). Volatile environment fields (cwd, date, git branch, mode) are NOT in here in digest
    mode: they arrive as a reminder in the first user turn and are resent only when they move,
    so this text stays byte-stable across turns and the hosted prompt cache holds."""
    parts = [HARNESS_PROMPT]
    files = find_rule_files(cwd, workspace)
    if mode == "full":
        for path in files:
            text = read_with_imports(path).strip()
            if text:
                parts.append(f"# Rules from {path}\n\n{text[:MAX_RULES_CHARS]}")
        memory_index = workspace / "memory" / "MEMORY.md"
        if memory_index.is_file():
            parts.append(
                f"# Memory index ({memory_index})\n"
                "Each line points at a memory file; Read the file before relying on a line.\n\n"
                + memory_index.read_text(encoding="utf-8")[:MAX_RULES_CHARS]
            )
        parts.append(
            "# Environment\n"
            f"- Working directory: {cwd}\n"
            f"- Platform: {platform.system()} {platform.release()}\n"
            f"- Today's date: {date.today().isoformat()}\n"
            f"- Model: {model.id}"
        )
        return "\n\n".join(parts)

    index = memory_index_path(workspace)
    for path, text in digest_sources(cwd, workspace):
        is_index = path == index
        cap = DIGEST_MEMORY_CHARS if is_index else DIGEST_RULE_CHARS
        label = "Memory index" if is_index else "Rules"
        note = (
            "\nEach line points at a memory file; Read the file before relying on a line."
            if is_index
            else ""
        )
        parts.append(
            f"# {label} from {path} (digest: marked lines only; Read the file for the rest)"
            f"{note}\n\n"
            # The index's own convention already marks its hard rules (⛔), and keeping its four
            # Hard Rules sections whole would cost ~3,500 more characters for the same lines.
            f"{digest_text(text, cap, critical_sections=not is_index)}"
        )
    parts.append(
        "# Environment\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        f"- Model: {model.id}\n"
        "Working directory, today's date, the git branch and the permission mode arrive in an "
        "environment reminder at the start of the conversation, resent whenever one changes."
    )
    return "\n\n".join(parts)


ENV_FIELDS = ("Working directory", "Today's date", "Git branch", "Permission mode")


def git_branch(cwd: Path) -> str:
    """The checked-out branch, or "" when this is not a git repo (or git is missing)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    branch = out.stdout.strip()
    return branch if out.returncode == 0 and branch != "HEAD" else ""


class EnvironmentTracker:
    """The volatile environment fields, as a reminder in the user turn.

    Plan item 2: the system prompt has to stay byte-stable across turns or the hosted prompt
    cache misses on every request, so cwd, date, branch and permission mode do not live there.
    The first note names all four; later notes name only the fields that moved, and a turn
    where nothing moved sends no message at all.
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self._sent: dict[str, str] = {}

    def fields(self, mode: str) -> dict[str, str]:
        values = {
            "Working directory": str(self.cwd),
            "Today's date": date.today().isoformat(),
            "Git branch": git_branch(self.cwd),
            "Permission mode": mode,
        }
        return {name: values[name] for name in ENV_FIELDS}

    def note(self, mode: str) -> str:
        """The reminder text for this turn, or "" when nothing changed since the last one."""
        now = self.fields(mode)
        changed = {k: v for k, v in now.items() if self._sent.get(k) != v}
        if not changed:
            return ""
        heading = "# Environment" if not self._sent else "# Environment changed"
        self._sent = now
        lines = "\n".join(f"- {name}: {value or '(none)'}" for name, value in changed.items())
        return f"{heading}\n{lines}"


@dataclass
class ContextMeter:
    context_window: int
    soft_budget: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_prompt: int = 0
    total_completion: int = 0
    requests: int = 0
    # Prompt size of the first request after a compaction: what the system
    # prompt, the tool list and the summary cost with no history at all.
    floor: int = 0
    floor_pending: bool = False

    def record(self, usage: dict[str, Any] | None) -> bool:
        """Store one answer's usage. Returns False when the endpoint sent none."""
        if not usage:
            return False
        self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.completion_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens = int(details.get("cached_tokens") or 0)
        self.total_prompt += self.prompt_tokens
        self.total_completion += self.completion_tokens
        self.requests += 1
        if self.floor_pending:
            self.floor = self.prompt_tokens
            self.floor_pending = False
        return True

    @property
    def used(self) -> int:
        """Tokens the next request starts with: the last prompt plus the last answer."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def fraction(self) -> float:
        return self.used / self.context_window if self.context_window else 0.0

    @property
    def over_budget(self) -> bool:
        return self.used >= self.soft_budget

    @property
    def floor_over_budget(self) -> bool:
        return self.floor >= self.soft_budget > 0

    @property
    def needs_compaction(self) -> bool:
        """Over budget AND a compaction can still win something back.

        When the fixed part of the prompt alone is over the budget, compacting
        on every request only burns requests (seen live on a 16k local model:
        12,845 fixed tokens against a 12,000 budget compacted after every tool
        call). After a compaction the history has to grow by half a budget
        before the next one.
        """
        if not self.over_budget:
            return False
        return self.floor == 0 or self.used >= self.floor + self.soft_budget // 2

    def line(self) -> str:
        return (
            f"context {self.used:,} of {self.context_window:,} tokens ({self.fraction:.1%}), "
            f"soft budget {self.soft_budget:,}, cached {self.cached_tokens:,}; "
            f"session total {self.total_prompt:,} in / {self.total_completion:,} out "
            f"over {self.requests} requests"
        )
