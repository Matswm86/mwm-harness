"""Permission modes plus a hard deny list that no mode and no hook can lift.

Modes (names match Claude Code's, because hooks receive ``permission_mode``):
    default            read-only tools run, everything else asks
    acceptEdits        Write/Edit inside the project also run; Bash still asks
    bypassPermissions  everything runs, except the deny list

The deny list holds actions that have cost real data or money before. It is
checked on the final tool input, after any hook has rewritten it.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("default", "acceptEdits", "bypassPermissions")
READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "TodoWrite"}
EDIT_TOOLS = {"Write", "Edit"}
QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")


@dataclass(frozen=True)
class DenyRule:
    id: str
    why: str
    bash: str = ""  # regex searched in a Bash command
    path: str = ""  # regex searched in a Write/Edit file path
    tool: str = ""  # regex matched against the whole tool name
    strip_quotes: bool = False  # match the command with quoted text removed (commit messages)


BUILTIN_DENY = (
    DenyRule(
        "git-force-push",
        "force-push rewrites shared history",
        bash=r"\bgit\b[^|;&]*\bpush\b[^|;&]*(\s--force\b|\s--force-with-lease\b|\s-f\b|\s\+\S)",
    ),
    DenyRule(
        "no-verify",
        "commits and pushes go through the pre-commit hooks",
        bash=r"\bgit\b[^|;&]*--no-verify",
    ),
    DenyRule(
        "no-verify-short",
        "commits go through the pre-commit hooks (-n is --no-verify)",
        bash=r"\bgit\b[^|;&]*\bcommit\b[^|;&]*\s-[a-mo-zA-Z]*n[a-zA-Z]*\b",
        strip_quotes=True,
    ),
    DenyRule(
        "rm-root-or-home",
        "recursive delete of the filesystem root or the home directory",
        bash=r"\brm\s+(-\w+\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(/|~|\$HOME|/home/?\w*)/?(\s|$)",
    ),
    DenyRule(
        "vector-collection-delete",
        "nothing is ever deleted from a vector-brain collection",
        bash=r"(-X\s*DELETE|--request\s+DELETE)[^|;&]*/collections|/points/delete|delete_collection",
    ),
    DenyRule(
        "vector-brain-delete-tool",
        "nothing is ever deleted from a vector-brain collection",
        tool=r"mcp__.*vector.*__delete_.*",
    ),
    DenyRule(
        "broker-mutation",
        "the harness never places, changes or cancels broker orders",
        bash=r"topstepx?[^|;&]*/api/(Order/(place|cancel|modify)|Position/(close|partialClose))",
    ),
    DenyRule(
        "broker-mutation-tool",
        "the harness never places, changes or cancels broker orders",
        tool=r"mcp__.*topstep.*__(place|cancel|modify|close|flatten).*",
    ),
    DenyRule(
        "curated-inbox",
        "the notes inbox is curated by hand",
        bash=r"\b(mv|rm|rmdir|rename|truncate)\b[^|;&]*notes/inbox",
        path=r"(^|/)notes/inbox(/|$)",
    ),
)


@dataclass
class Decision:
    verdict: str  # allow | ask | deny
    reason: str = ""


def load_extra_deny(path: Path) -> tuple[DenyRule, ...]:
    """Extra rules from ``deny.toml``: ``[[rule]]`` tables with the DenyRule fields."""
    if not path.is_file():
        return ()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return tuple(DenyRule(**entry) for entry in data.get("rule", []))


class Permissions:
    def __init__(self, mode: str, project: Path, extra_deny: tuple[DenyRule, ...] = ()) -> None:
        self.set_mode(mode)
        self.project = project.resolve()
        self.rules = BUILTIN_DENY + extra_deny
        self._compiled = {
            rule.id: {
                kind: re.compile(pattern, re.IGNORECASE)
                for kind, pattern in (("bash", rule.bash), ("path", rule.path), ("tool", rule.tool))
                if pattern
            }
            for rule in self.rules
        }
        self.session_allow: set[str] = set()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"permission mode must be one of {MODES}, got {mode!r}")
        self.mode = mode

    def denied(self, tool_name: str, tool_input: dict[str, Any]) -> DenyRule | None:
        command = str(tool_input.get("command") or "") if tool_name == "Bash" else ""
        file_path = str(tool_input.get("file_path") or "") if tool_name in EDIT_TOOLS else ""
        for rule in self.rules:
            compiled = self._compiled[rule.id]
            if "tool" in compiled and compiled["tool"].fullmatch(tool_name):
                return rule
            if command and "bash" in compiled:
                subject = QUOTED.sub("''", command) if rule.strip_quotes else command
                if compiled["bash"].search(subject):
                    return rule
            if file_path and "path" in compiled and compiled["path"].search(file_path):
                return rule
        return None

    def decide(self, tool_name: str, tool_input: dict[str, Any], read_only: bool) -> Decision:
        rule = self.denied(tool_name, tool_input)
        if rule:
            return Decision("deny", f"Denied by the hard deny list ({rule.id}): {rule.why}.")
        if read_only or tool_name in READ_ONLY_TOOLS or tool_name in self.session_allow:
            return Decision("allow")
        if self.mode == "bypassPermissions":
            return Decision("allow")
        if self.mode == "acceptEdits" and tool_name in EDIT_TOOLS:
            if self._inside_project(str(tool_input.get("file_path") or "")):
                return Decision("allow")
            return Decision("ask", "the file is outside the project directory")
        return Decision("ask", f"{tool_name} changes state")

    def _inside_project(self, file_path: str) -> bool:
        if not file_path:
            return False
        try:
            (self.project / Path(file_path).expanduser()).resolve().relative_to(self.project)
        except ValueError:
            return False
        return True
