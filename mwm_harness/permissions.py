"""Permission modes plus a hard deny list that no mode and no hook can lift.

Modes (names match Claude Code's, because hooks receive ``permission_mode``):
    default            read-only tools run, everything else asks
    acceptEdits        Write/Edit inside the project also run; Bash still asks
    bypassPermissions  everything runs, except the deny list
    plan               only reading tools run, until the person approves a plan

The deny list holds actions that have cost real data or money before. It is
checked on the final tool input, after any hook has rewritten it.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

MODES = ("default", "acceptEdits", "bypassPermissions", "plan")
READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "TodoWrite"}
EDIT_TOOLS = {"Write", "Edit"}
READ_PATH_KEYS = {"Read": "file_path", "Glob": "path", "Grep": "path"}
QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")


@dataclass(frozen=True)
class DenyRule:
    id: str
    why: str
    bash: str = ""  # regex searched in a Bash command
    path: str = ""  # regex searched in a Write/Edit file path
    tool: str = ""  # regex matched against the whole tool name
    read: str = ""  # regex searched in the resolved path a Read/Glob/Grep call names
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
        bash=r"\brm\s+(-{1,2}[\w-]+\s+)*(-\w*[rR]\w*|--recursive)\s+(-{1,2}[\w-]+\s+)*"
        r"(/|~|\$HOME|/home/?\w*)/?(\s|$|['\"])",
    ),
    DenyRule(
        "find-delete-root-or-home",
        "find -delete over the filesystem root or the home directory",
        bash=r"\bfind\s+(/|~|\$HOME|/home/?\w*)/?\s[^|;&]*(-delete\b|-exec\s+rm\b)",
    ),
    DenyRule(
        "git-hooks-off",
        "commits go through the pre-commit hooks (core.hooksPath switches them off)",
        bash=r"\bgit\b[^|;&]*core\.hookspath|\bGIT_CONFIG[\w]*=[^\s]*hookspath",
    ),
    DenyRule(
        "decode-into-shell",
        "a command that is decoded and piped into a shell hides from every rule here",
        bash=r"\b(base64\s+(-d|--decode)|xxd\s+-r|openssl\s+enc\s+-d)\b[^;&]*\|\s*(ba|z|da)?sh\b",
    ),
    DenyRule(
        "secret-files",
        "key material and credential stores are never read into a model's context",
        read=r"/\.ssh/|/\.gnupg/|/\.aws/|/\.config/mwm-harness/secrets\.env$|/\.qwen/settings\.json$"
        r"|/\.config/gh/hosts\.yml$|/\.netrc$|/\.git-credentials$",
        bash=r"(/|~|\$HOME)[^\s|;&]*(\.ssh/id_|secrets\.env|\.git-credentials|\.netrc)|/proc/\w+/environ",
    ),
    DenyRule(
        "vector-collection-delete",
        "nothing is ever deleted from a vector-brain collection",
        bash=r"(-X\s*DELETE|--request\s+DELETE)[^|;&]*/collections|/points/delete|delete_collection",
    ),
    DenyRule(
        "vector-brain-delete-tool",
        "nothing is ever deleted from a vector-brain collection",
        # Keyed on what the tool does, not on the server's name: renaming the server
        # entry in .mcp.json must not lift the rule. Other delete tools still ask.
        tool=r"mcp__.+__(delete|drop|wipe|purge)_(source|collection|point|vector|index)s?\b.*",
    ),
    DenyRule(
        "broker-mutation",
        "the harness never places, changes or cancels broker orders",
        bash=r"topstepx?[^|;&]*/api/(Order/(place|cancel|modify)|Position/(close|partialClose))",
    ),
    DenyRule(
        "broker-mutation-tool",
        "the harness never places, changes or cancels broker orders",
        tool=r"mcp__.+__(place|cancel|modify|close|flatten)_?(order|position|bracket|all).*",
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
    def __init__(
        self,
        mode: str,
        project: Path,
        extra_deny: tuple[DenyRule, ...] = (),
        allow_patterns: tuple[str, ...] = (),
        read_roots: tuple[Path, ...] = (),
    ) -> None:
        self.set_mode(mode)
        self.allow_patterns = allow_patterns
        self.project = project.resolve()
        self.read_roots = (self.project, *(root.resolve() for root in read_roots))
        self.rules = BUILTIN_DENY + extra_deny
        self._compiled = {
            rule.id: {
                kind: re.compile(pattern, re.IGNORECASE)
                for kind, pattern in (
                    ("bash", rule.bash),
                    ("path", rule.path),
                    ("tool", rule.tool),
                    ("read", rule.read),
                )
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
        read_path = str(self._read_target(tool_name, tool_input) or "")
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
            if read_path and "read" in compiled and compiled["read"].search(read_path):
                return rule
            # An edit of a secret file is as bad as a read of it.
            if file_path and "read" in compiled:
                resolved = str((self.project / Path(file_path).expanduser()).resolve())
                if compiled["read"].search(resolved):
                    return rule
        return None

    def _read_target(self, tool_name: str, tool_input: dict[str, Any]) -> Path | None:
        """The resolved file or folder a reading tool is pointed at; None = the project."""
        if tool_name not in READ_PATH_KEYS:
            return None
        raw = str(tool_input.get(READ_PATH_KEYS[tool_name]) or "")
        if not raw:
            return None
        return (self.project / Path(raw).expanduser()).resolve()

    def _readable(self, target: Path) -> bool:
        return any(target == root or root in target.parents for root in self.read_roots)

    def decide(self, tool_name: str, tool_input: dict[str, Any], read_only: bool) -> Decision:
        rule = self.denied(tool_name, tool_input)
        if rule:
            return Decision("deny", f"Denied by the hard deny list ({rule.id}): {rule.why}.")
        if read_only or tool_name in READ_ONLY_TOOLS:
            # Reading is free inside the project and the declared read roots. Anywhere
            # else on the disk the person is asked: a hostile page or tool result can
            # steer a model into reading private files and sending them out.
            target = self._read_target(tool_name, tool_input)
            if target is not None and not self._readable(target):
                if self.mode == "bypassPermissions" or tool_name in self.session_allow:
                    return Decision("allow")
                return Decision("ask", f"{target} is outside the project and its read roots")
            return Decision("allow")
        if any(fnmatchcase(tool_name, pattern) for pattern in self.allow_patterns):
            return Decision("allow")
        if self.mode == "plan":
            return Decision(
                "deny",
                "Plan mode is on: only reading tools run. Finish the research, then call "
                "ExitPlanMode with the plan and wait for the person's answer.",
            )
        if tool_name in self.session_allow:
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
