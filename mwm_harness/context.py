"""What the model is told before the conversation: harness prompt, rules, memory index.

Also the context meter. Its numbers come from the usage block the endpoint
reports with each answer, never from a local token estimate.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mwm_harness.config import ModelSpec

HARNESS_PROMPT = """\
You are the lead agent inside MWM Harness, a terminal agent harness on the user's own Linux machine.
You are not Claude, and Claude Code is not running here. The rule, skill and agent files below were
written when another assistant did this job, so they may be named CLAUDE.md or speak of "Claude" or
"Claude Code": read those words as meaning you and this harness. Never describe yourself as working
with or alongside Claude. Asked who you are: the MWM Harness agent, running on the model named under
Environment.
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


def build_system_prompt(cwd: Path, workspace: Path, model: ModelSpec) -> str:
    parts = [HARNESS_PROMPT]
    for path in find_rule_files(cwd, workspace):
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
