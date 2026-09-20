"""Deny list, permission modes, the shell sandbox and the file tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mwm_harness.permissions import Permissions
from mwm_harness.sandbox import Sandbox, bwrap_works
from mwm_harness.tools import ToolContext, default_tools

DENIED_COMMANDS = [
    "git push --force origin main",
    "git push -f",
    "cd repo && git push origin +main",
    "git push --force-with-lease",
    "git commit --no-verify -m wip",
    "git commit -n -m wip",
    "git commit -anm wip",
    "rm -rf /",
    "rm -rf ~",
    "rm -fr $HOME",
    "sudo rm -rf /home/someone",
    "curl -X DELETE http://127.0.0.1:6333/collections/brain",
    "curl -s -XDELETE localhost:6333/collections/brain",
    "curl -X POST localhost:6333/collections/brain/points/delete -d '{}'",
    "python -c 'client.delete_collection(\"brain\")'",
    "mv notes/inbox/a.md notes/archive/",
    "rm notes/inbox/old.md",
    "curl https://api.topstepx.com/api/Order/place -d '{}'",
]
ALLOWED_COMMANDS = [
    "git push origin main",
    "git commit -m 'fix: handle -n flag in parser'",
    'git commit -m "docs: explain -n and --dry-run"',
    "git commit --amend -m x",
    "rm -rf build",
    "rm -rf ~/project/build",
    "rm -rf /home/someone/project/dist",
    "curl http://127.0.0.1:6333/collections",
    "ls notes/inbox",
    "git log -n 5",
]


@pytest.fixture
def permissions(tmp_path: Path) -> Permissions:
    return Permissions("bypassPermissions", tmp_path)


@pytest.mark.parametrize("command", DENIED_COMMANDS)
def test_deny_list_refuses(permissions: Permissions, command: str) -> None:
    assert permissions.decide("Bash", {"command": command}, False).verdict == "deny", command


@pytest.mark.parametrize("command", ALLOWED_COMMANDS)
def test_deny_list_leaves_normal_commands_alone(permissions: Permissions, command: str) -> None:
    assert permissions.decide("Bash", {"command": command}, False).verdict == "allow", command


def test_deny_list_covers_paths_and_tool_names(permissions: Permissions) -> None:
    inbox = {"file_path": "/data/notes/inbox/new.md", "content": "x"}
    assert permissions.decide("Write", inbox, False).verdict == "deny"
    assert permissions.decide("mcp__mwm-vector-brain__delete_source", {}, False).verdict == "deny"
    assert permissions.decide("mcp__mwm-vector-brain__search_books", {}, False).verdict == "allow"


def test_modes(tmp_path: Path) -> None:
    inside = {"file_path": "src/a.py", "content": ""}
    outside = {"file_path": "/etc/hosts", "content": ""}
    default = Permissions("default", tmp_path)
    assert default.decide("Read", {"file_path": "a"}, True).verdict == "allow"
    assert default.decide("Write", inside, False).verdict == "ask"
    edits = Permissions("acceptEdits", tmp_path)
    assert edits.decide("Write", inside, False).verdict == "allow"
    assert edits.decide("Write", outside, False).verdict == "ask"
    assert (
        edits.decide("Write", {"file_path": "../escape.py", "content": ""}, False).verdict == "ask"
    )
    assert edits.decide("Bash", {"command": "ls"}, False).verdict == "ask"
    with pytest.raises(ValueError, match="permission mode"):
        Permissions("yolo", tmp_path)


def context(tmp_path: Path, sandbox: str = "off") -> ToolContext:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    scratch = tmp_path / "scratch"
    return ToolContext(project, scratch, Sandbox(sandbox, [project, scratch]), output_cap=200)


def call(tool: str, tool_input: dict, ctx: ToolContext):
    return asyncio.run(default_tools()[tool].run(tool_input, ctx))


def test_edit_requires_a_read_and_a_unique_match(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    target = ctx.cwd / "a.txt"
    target.write_text("one\ntwo\none\n")
    edit = {"file_path": "a.txt", "old_string": "two", "new_string": "2"}
    assert call("Edit", edit, ctx).is_error  # not read yet
    assert "     2\ttwo" in call("Read", {"file_path": "a.txt"}, ctx).content
    assert not call("Edit", edit, ctx).is_error
    twice = {"file_path": "a.txt", "old_string": "one", "new_string": "1"}
    assert "occurs 2 times" in call("Edit", twice, ctx).content
    assert not call("Edit", {**twice, "replace_all": True}, ctx).is_error
    assert target.read_text() == "1\n2\n1\n"


def test_write_refuses_to_clobber_an_unread_file(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    (ctx.cwd / "keep.txt").write_text("precious")
    result = call("Write", {"file_path": "keep.txt", "content": "gone"}, ctx)
    assert result.is_error
    assert (ctx.cwd / "keep.txt").read_text() == "precious"


def test_grep_and_glob(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    (ctx.cwd / "pkg").mkdir()
    (ctx.cwd / "pkg" / "a.py").write_text("import os\nVALUE = 1\n")
    (ctx.cwd / "pkg" / "b.txt").write_text("VALUE = 2\n")
    (ctx.cwd / ".git").mkdir()
    (ctx.cwd / ".git" / "c.py").write_text("VALUE = 3\n")
    found = call("Grep", {"pattern": "VALUE", "glob": "*.py", "output_mode": "content"}, ctx)
    assert found.content.endswith("a.py:2:VALUE = 1")
    assert call("Grep", {"pattern": "("}, ctx).is_error
    listed = call("Glob", {"pattern": "**/*.py"}, ctx).content
    assert "a.py" in listed and ".git" not in listed


def test_long_output_is_capped_and_spilled(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    result = call("Bash", {"command": "seq 1 500"}, ctx)
    assert "output cut at 200" in result.content
    spill = Path(result.content.rsplit("saved to ", 1)[1].rstrip("]"))
    assert spill.read_text().splitlines()[-1] == "500"


def test_bash_timeout_returns_partial_output(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    result = call("Bash", {"command": "echo started; sleep 30", "timeout": 500}, ctx)
    assert result.is_error
    assert "started" in result.content and "killed after 500 ms" in result.content


@pytest.mark.skipif(not bwrap_works(), reason="bubblewrap cannot run here")
def test_sandbox_blocks_writes_outside_the_project(tmp_path: Path) -> None:
    ctx = context(tmp_path, sandbox="bwrap")
    outside = tmp_path / "outside.txt"
    result = call("Bash", {"command": f"echo x > {outside}; echo y > inside.txt"}, ctx)
    assert "Read-only file system" in result.content
    assert not outside.exists()
    assert (ctx.cwd / "inside.txt").read_text() == "y\n"
