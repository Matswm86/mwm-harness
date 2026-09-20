"""Skills (``<folder>/SKILL.md``) and command files (``commands/<name>.md``).

Both use the layout Claude Code and Qwen Code share: a ``---`` frontmatter block
with ``name`` and ``description``, then instructions. The system prompt lists
every skill by name and description; the body enters the conversation only when
the model calls the ``Skill`` tool or the user types ``/name``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mwm_harness.config import workspace_root
from mwm_harness.tools.base import Tool, ToolContext, ToolResult

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
MAX_DESCRIPTION = 600


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Top-level ``key: value`` pairs only; folded and quoted values are flattened."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    key = ""
    for line in match.group(1).splitlines():
        pair = re.match(r"^([A-Za-z][\w-]*):\s*(.*)$", line)
        if pair:
            key, value = pair.group(1), pair.group(2).strip()
            fields[key] = "" if value in (">", "|", ">-", "|-") else value.strip("'\"")
        elif key and line.startswith((" ", "\t")):
            fields[key] = f"{fields[key]} {line.strip()}".strip()
    return fields, text[match.end() :]


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        return split_frontmatter(self.path.read_text(encoding="utf-8"))[1].strip()


def skill_roots(cwd: Path, extra: list[str]) -> list[tuple[str, Path]]:
    """(prefix, folder) pairs. An extra entry is ``folder`` or ``prefix=folder``."""
    roots = [
        ("", cwd / ".claude" / "skills"),
        ("", workspace_root() / ".claude" / "skills"),
        ("", Path.home() / ".claude" / "skills"),
    ]
    for entry in extra:
        prefix, _, folder = entry.rpartition("=")
        roots.append((prefix, Path(folder).expanduser()))
    return roots


def load_skills(roots: list[tuple[str, Path]]) -> dict[str, Skill]:
    """The first root that defines a name wins, so a project skill shadows a user skill."""
    skills: dict[str, Skill] = {}
    for prefix, root in roots:
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            path = folder / "SKILL.md"
            if not path.is_file():
                continue
            try:
                fields, _ = split_frontmatter(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            name = fields.get("name") or folder.name
            name = f"{prefix}:{name}" if prefix else name
            skills.setdefault(name, Skill(name, fields.get("description", ""), path))
    return skills


def skills_prompt(skills: dict[str, Skill]) -> str:
    if not skills:
        return ""
    rows = [f"- {s.name}: {s.description[:MAX_DESCRIPTION]}" for s in skills.values()]
    return (
        "# Skills\n"
        "A skill is a set of instructions for one kind of task. When a task matches a skill "
        "below, call the Skill tool with its name first and follow what it returns.\n\n"
        + "\n".join(rows)
    )


class SkillTool(Tool):
    name = "Skill"
    description = (
        "Load a skill's instructions by name. The available names are listed under Skills."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "args": {"type": "string", "description": "Optional arguments for the skill"},
        },
        "required": ["skill"],
    }
    read_only = True

    def __init__(self, skills: dict[str, Skill]) -> None:
        self.skills = skills

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = str(tool_input["skill"]).lstrip("/")
        skill = self.skills.get(name)
        if skill is None:
            return ToolResult(f"unknown skill {name}; available: {', '.join(self.skills)}", True)
        try:
            body = skill.body()
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(f"cannot read {skill.path}: {exc}", True)
        args = str(tool_input.get("args") or "")
        head = f"Skill folder: {skill.path.parent}\n"
        if args:
            head += f"Arguments: {args}\n"
        return ToolResult(ctx.cap(f"{head}\n{body}"))


# -------------------------------------------------------------- command files


@dataclass
class Command:
    name: str
    description: str
    path: Path

    def render(self, arguments: str) -> str:
        """The prompt sent to the model: ``$ARGUMENTS`` and ``$1``..``$9`` are filled in."""
        body = split_frontmatter(self.path.read_text(encoding="utf-8"))[1].strip()
        words = arguments.split()
        used = "$ARGUMENTS" in body or re.search(r"\$[1-9]", body) is not None
        body = body.replace("$ARGUMENTS", arguments)
        body = re.sub(
            r"\$([1-9])", lambda m: words[int(m[1]) - 1] if int(m[1]) <= len(words) else "", body
        )
        if arguments and not used:
            body += f"\n\nArguments: {arguments}"
        return body


def command_roots(cwd: Path) -> list[Path]:
    return [
        cwd / ".claude" / "commands",
        workspace_root() / ".claude" / "commands",
        Path.home() / ".claude" / "commands",
    ]


def load_commands(roots: list[Path]) -> dict[str, Command]:
    commands: dict[str, Command] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(root).with_suffix("")
            name = ":".join(relative.parts)
            try:
                fields, body = split_frontmatter(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            first_line = next(
                (row.strip("# ").strip() for row in body.splitlines() if row.strip()), ""
            )
            commands.setdefault(name, Command(name, fields.get("description") or first_line, path))
    return commands
