"""Writable /tmp inside the sandbox: mktemp-based tools (pre-commit hooks, backup.sh) must work."""

from __future__ import annotations

import asyncio
import shutil

import pytest
from mwm_harness.sandbox import Sandbox, bwrap_works

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not bwrap_works(), reason="bubblewrap does not run here"
)


@needs_bwrap
def test_argv_mounts_a_tmpfs_over_tmp():
    argv = Sandbox("bwrap", []).argv("true")
    pairs = list(zip(argv, argv[1:]))
    assert ("--tmpfs", "/tmp") in pairs


@needs_bwrap
def test_tmp_has_the_sticky_bit(tmp_path):
    """bwrap makes a bare --tmpfs 0755. A real /tmp is 1777, and a tool that checks for
    the sticky bit before trusting TMPDIR refuses to write into one that lacks it."""
    result = asyncio.run(Sandbox("bwrap", [tmp_path]).run("stat -c %a /tmp", tmp_path, 30))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "1777"


@needs_bwrap
def test_mktemp_writes_inside_the_sandbox(tmp_path):
    """The 2026-10-23 regression: read-only /tmp made detect-secrets' mktemp fail,
    so wmcommit could not commit from an agent session."""
    sandbox = Sandbox("bwrap", [tmp_path])
    result = asyncio.run(
        sandbox.run(
            'unset TMPDIR; f="$(mktemp --suffix=.json)" && echo ok > "$f" && cat "$f"', tmp_path, 30
        )
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "ok"


@needs_bwrap
def test_host_tmp_contents_are_hidden(tmp_path):
    marker = tmp_path / "host-tmp-marker-name"
    result = asyncio.run(Sandbox("bwrap", [tmp_path]).run("ls /tmp", tmp_path, 30))
    assert marker.name not in result.output
