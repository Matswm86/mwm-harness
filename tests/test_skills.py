"""Skills, command files, and their slash commands in the terminal front-end."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mwm_harness.repl.terminal import Printer, run_command
from mwm_harness.sandbox import Sandbox
from mwm_harness.skills import (
    SkillTool,
    load_commands,
    load_skills,
    skills_prompt,
    split_frontmatter,
)
from mwm_harness.tools.base import ToolContext


def run(coro):
    return asyncio.run(coro)


def write_skill(root: Path, folder: str, text: str) -> None:
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(text, encoding="utf-8")


def test_frontmatter_handles_folded_values_and_missing_blocks():
    fields, body = split_frontmatter(
        "---\nname: x\ndescription: >\n  first line\n  second line\n---\n# Body\n"
    )
    assert fields == {"name": "x", "description": "first line second line"}
    assert body == "# Body\n"
    assert split_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_first_root_wins_and_prefix_is_applied(tmp_path):
    project, user, plugin = tmp_path / "p", tmp_path / "u", tmp_path / "g"
    write_skill(project, "tidy", "---\nname: tidy\ndescription: project version\n---\nP body")
    write_skill(user, "tidy", "---\nname: tidy\ndescription: user version\n---\nU body")
    write_skill(user, "unnamed", "just instructions")
    write_skill(plugin, "audit", "---\nname: audit\ndescription: d\n---\nA")
    skills = load_skills([("", project), ("", user), ("core", plugin), ("", tmp_path / "none")])
    assert sorted(skills) == ["core:audit", "tidy", "unnamed"]
    assert skills["tidy"].description == "project version"
    assert "- tidy: project version" in skills_prompt(skills)


def test_skill_tool_returns_the_body_and_rejects_unknown_names(tmp_path):
    write_skill(tmp_path, "tidy", "---\nname: tidy\ndescription: d\n---\nStep one.")
    tool = SkillTool(load_skills([("", tmp_path)]))
    ctx = ToolContext(cwd=tmp_path, scratch=tmp_path / "s", sandbox=Sandbox("off", []))
    result = run(tool.run({"skill": "tidy", "args": "now"}, ctx))
    assert "Step one." in result.content and "Arguments: now" in result.content
    assert f"Skill folder: {tmp_path / 'tidy'}" in result.content
    assert run(tool.run({"skill": "ghost"}, ctx)).is_error


def test_command_files_fill_in_arguments(tmp_path):
    (tmp_path / "git").mkdir()
    (tmp_path / "watch.md").write_text(
        "---\ndescription: Watch a video\n---\nIngest $ARGUMENTS, first word $1, third [$3]."
    )
    (tmp_path / "git" / "sync.md").write_text("# Sync the repo\nPull then push.")
    commands = load_commands([tmp_path])
    assert sorted(commands) == ["git:sync", "watch"]
    assert commands["watch"].description == "Watch a video"
    assert commands["git:sync"].description == "Sync the repo"
    assert (
        commands["watch"].render("abc --force") == "Ingest abc --force, first word abc, third []."
    )
    assert commands["git:sync"].render("main").endswith("Arguments: main")


def test_slash_command_for_a_command_file_becomes_a_prompt(make_session, tmp_path, capsys):
    session, _, _ = make_session([])
    (tmp_path / "watch.md").write_text("Ingest $ARGUMENTS.")
    write_skill(tmp_path / "skills", "tidy", "---\nname: tidy\ndescription: d\n---\nbody")
    session.commands = load_commands([tmp_path])
    session.skills = load_skills([("", tmp_path / "skills")])
    out = Printer(color=False)
    assert run_command(session, {}, "/watch abc", out) == "Ingest abc."
    assert "Load the skill tidy" in run_command(session, {}, "/tidy x", out)
    assert run_command(session, {}, "/skills", out) is True
    assert run_command(session, {}, "/commands", out) is True
    assert run_command(session, {}, "/mcp", out) is True
    printed = capsys.readouterr().out
    assert "1 skills" in printed and "/watch" in printed and "MCP is off" in printed
