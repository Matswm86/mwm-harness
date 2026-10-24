"""Plan item 36: resource caps for sandboxed shell commands and session writes.

An unattended --yolo run has no human to notice a job filling the disk or forking
without end. These tests cover each cap at three levels: the argv or prefix it builds,
the note it reports when hit, and the real kernel behaviour under a small cap.
Real-behaviour tests skip where their mechanism is unavailable, never pass silently.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from mwm_harness.sandbox import Sandbox, bwrap_works, user_scope_works
from mwm_harness.tools.base import ToolContext
from mwm_harness.tools.files import Edit, Write

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not bwrap_works(), reason="bubblewrap does not run here"
)
needs_scope = pytest.mark.skipif(
    shutil.which("systemd-run") is None or not user_scope_works(),
    reason="no user systemd scope here, so process and memory caps cannot be armed",
)


# --- what the command line carries -----------------------------------------------------


def test_tmpfs_size_sits_before_the_tmpfs_argument(tmp_path):
    sandbox = Sandbox("auto", [tmp_path], tmpfs_max_mb=64)
    if not sandbox.enabled:
        pytest.skip("bubblewrap does not run here")
    argv = sandbox.argv("true")
    at = argv.index("/tmp")
    assert argv[at - 5 : at + 1] == [
        "--size",
        str(64 * 1024 * 1024),
        "--perms",
        "1777",
        "--tmpfs",
        "/tmp",
    ]


def test_zero_tmpfs_sends_no_size_flag(tmp_path):
    sandbox = Sandbox("auto", [tmp_path], tmpfs_max_mb=0)
    if not sandbox.enabled:
        pytest.skip("bubblewrap does not run here")
    assert "--size" not in sandbox.argv("true")


def test_scope_prefix_carries_both_properties_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: True)
    sandbox = Sandbox("bwrap", [tmp_path], tasks_max=512, memory_max_mb=1024)
    prefix = sandbox.scope_prefix()
    assert prefix[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "--expand-environment=no" in prefix
    assert "-p" in prefix and "TasksMax=512" in prefix and "MemoryMax=1024M" in prefix
    assert "OOMPolicy=continue" in prefix  # a memcg kill must not take the wrapper down
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: False)
    assert Sandbox("bwrap", [tmp_path]).scope_prefix() == []
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: True)
    assert Sandbox("bwrap", [tmp_path], tasks_max=0, memory_max_mb=0).scope_prefix() == []


def test_a_missing_scope_is_said_out_loud(tmp_path, monkeypatch):
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: False)
    sandbox = Sandbox("bwrap", [tmp_path], tasks_max=512, memory_max_mb=1024)
    assert "NOT in force" in sandbox.warning
    assert "TasksMax=512" in sandbox.warning and "MemoryMax=1024 MB" in sandbox.warning
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: True)
    assert Sandbox("bwrap", [tmp_path]).warning == ""


# --- the note a capped run reports ------------------------------------------------------


@pytest.fixture
def noted(tmp_path, monkeypatch) -> Sandbox:
    """A Sandbox with enabled and scoped forced, so the notes are tested as a pure function."""
    monkeypatch.setattr("mwm_harness.sandbox.bwrap_works", lambda: True)
    monkeypatch.setattr("mwm_harness.sandbox.user_scope_works", lambda: True)
    return Sandbox("bwrap", [tmp_path])


def test_cap_notes_name_their_own_cap(noted):
    killed = [n for n in noted.cap_notes(137, "boom\nKilled\n", False) if "memory cap" in n]
    assert killed and "16384 MB" in killed[0]
    forks = noted.cap_notes(0, "bash: fork: retry: Resource temporarily unavailable", False)
    assert any("process cap of 512" in n for n in forks)
    sizes = noted.cap_notes(153, "", False)
    assert any("size cap of 2048 MB" in n for n in sizes)  # 153 == 128 + SIGXFSZ
    space = noted.cap_notes(1, "dd: failed: No space left on device", False)
    assert any("sandbox_tmpfs_max_mb" in n for n in space)


def test_cap_notes_stay_quiet_for_a_normal_exit(noted):
    assert noted.cap_notes(0, "all good", False) == []
    assert noted.cap_notes(None, "whatever", timed_out=True) == []


def test_uncapped_layers_never_note_themselves(tmp_path, noted):
    plain = Sandbox("off", [])
    assert plain.cap_notes(137, "Killed", False) == []  # no sandbox, no caps were armed
    noted.tmpfs_max_mb = 0
    noted.file_max_mb = 0
    assert noted.cap_notes(1, "No space left on device", False) == []
    assert noted.cap_notes(153, "", False) == []


# --- real behaviour under small caps -----------------------------------------------------


@needs_bwrap
def test_tmp_writes_hit_the_tmpfs_cap(tmp_path):
    sandbox = Sandbox("bwrap", [tmp_path], tmpfs_max_mb=1, tasks_max=0, memory_max_mb=0)
    run = asyncio.run(sandbox.run("dd if=/dev/zero of=/tmp/fill bs=1M count=8", tmp_path, 30))
    assert run.exit_code != 0
    assert "No space left on device" in run.output
    assert "sandbox_tmpfs_max_mb" in run.output  # named cap, not a bare crash


@needs_bwrap
def test_a_single_big_file_hits_the_size_cap(tmp_path):
    sandbox = Sandbox(
        "bwrap", [tmp_path], file_max_mb=1, tmpfs_max_mb=0, tasks_max=0, memory_max_mb=0
    )
    target = tmp_path / "big.bin"
    run = asyncio.run(sandbox.run(f"dd if=/dev/zero of={target} bs=1M count=8", tmp_path, 30))
    assert run.exit_code != 0
    assert "sandbox_file_max_mb" in run.output
    assert target.exists() and target.stat().st_size <= 1024 * 1024  # truncated, not beyond


@needs_scope
def test_an_unbounded_allocation_is_stopped_and_named(tmp_path):
    sandbox = Sandbox(
        "bwrap", [tmp_path], memory_max_mb=256, tasks_max=0, file_max_mb=0, tmpfs_max_mb=0
    )
    # A scope's default OOM policy stops the whole unit when the kernel memcg-kills one
    # process: the wrapper died with the allocation (probed: rc -15, empty output). The
    # scope prefix therefore adds OOMPolicy=continue, so the shell survives to report rc.
    alloc = (
        "python3 -c 'b = []\nwhile True: b.append(bytearray(32 * 1024 * 1024))'\n"
        'rc=$?; echo "python exited $rc"; exit $rc'
    )
    run = asyncio.run(sandbox.run(alloc, tmp_path, 60))
    assert run.exit_code != 0
    assert "sandbox_memory_max_mb" in run.output


@needs_scope
def test_forks_beyond_the_process_cap_are_refused_and_named(tmp_path):
    sandbox = Sandbox(
        "bwrap", [tmp_path], tasks_max=32, memory_max_mb=0, file_max_mb=0, tmpfs_max_mb=0
    )
    bomb = "for i in $(seq 1 200); do sleep 2 & done; true"
    run = asyncio.run(sandbox.run(bomb, tmp_path, 30))
    assert "Resource temporarily unavailable" in run.output
    assert "sandbox_tasks_max" in run.output


@needs_bwrap
def test_a_normal_task_feels_nothing(tmp_path):
    sandbox = Sandbox("bwrap", [tmp_path])  # every default cap in force
    probe = tmp_path / "task.py"
    probe.write_text("import os\nprint(os.getpid())\n", encoding="utf-8")
    run = asyncio.run(sandbox.run(f"python3 {probe} && echo done", tmp_path, 30))
    assert run.exit_code == 0 and "done" in run.output
    for phrase in ("sandbox_", "cap of", "NOT in force"):
        assert phrase not in run.output


# --- the session-write cap on the file tools ---------------------------------------------


def made_ctx(tmp_path: Path, cap_bytes: int) -> ToolContext:
    return ToolContext(
        cwd=tmp_path,
        scratch=tmp_path / "scratch",
        sandbox=Sandbox("off", []),
        session_write_cap=cap_bytes,
    )


def test_session_write_cap_refuses_before_touching_disk(tmp_path):
    ctx = made_ctx(tmp_path, cap_bytes=1_000)
    first = asyncio.run(Write().run({"file_path": "a.txt", "content": "x" * 600}, ctx))
    assert not first.is_error and ctx.session_written == 600
    second = Write().refusal({"file_path": "b.txt", "content": "y" * 600}, ctx)
    assert second is not None
    assert "session_write_cap_mb" in second and "runaway-write guard" in second
    assert not (tmp_path / "b.txt").exists()  # refused before the approval card, nothing on disk
    result = asyncio.run(Write().run({"file_path": "b.txt", "content": "y" * 600}, ctx))
    assert result.is_error and ctx.session_written == 600  # a refused write charges nothing


def test_edit_is_charged_and_capped_too(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("old\n", encoding="utf-8")
    ctx = made_ctx(tmp_path, cap_bytes=10_000)
    ctx.read_files.add(target)
    ok = asyncio.run(
        Edit().run({"file_path": "f.txt", "old_string": "old", "new_string": "n" * 500}, ctx)
    )
    assert not ok.is_error and ctx.session_written > 0  # the whole rewritten file is charged
    over = Edit().refusal(
        {"file_path": "f.txt", "old_string": "n" * 500, "new_string": "z" * 20_000}, ctx
    )
    assert over is not None and "runaway-write guard" in over
    assert ctx.session_written == target.stat().st_size  # the refused edit wrote nothing


def test_no_cap_means_no_refusal(tmp_path):
    ctx = made_ctx(tmp_path, cap_bytes=0)
    assert Write().refusal({"file_path": "a.txt", "content": "x" * 100_000}, ctx) is None
