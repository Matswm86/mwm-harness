"""GPU passthrough: a sandboxed command sees the same NVIDIA device nodes the host does."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from mwm_harness.sandbox import Sandbox, bwrap_works, gpu_device_binds

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not bwrap_works(), reason="bubblewrap does not run here"
)
host_gpu = sorted(Path("/dev").glob("nvidia*"))
needs_host_gpu = pytest.mark.skipif(not host_gpu, reason="this box has no NVIDIA device nodes")


def test_binds_each_node_and_skips_directories(tmp_path):
    (tmp_path / "nvidia0").touch()
    (tmp_path / "nvidiactl").touch()
    (tmp_path / "nvidia-modeset").touch()
    (tmp_path / "nvidia-ignore-me").mkdir()
    assert gpu_device_binds(tmp_path) == [
        "--dev-bind",
        str(tmp_path / "nvidia-modeset"),
        str(tmp_path / "nvidia-modeset"),
        "--dev-bind",
        str(tmp_path / "nvidia0"),
        str(tmp_path / "nvidia0"),
        "--dev-bind",
        str(tmp_path / "nvidiactl"),
        str(tmp_path / "nvidiactl"),
    ]


def test_a_box_without_nvidia_binds_nothing(tmp_path):
    (tmp_path / "null").touch()
    assert gpu_device_binds(tmp_path) == []


@needs_host_gpu
def test_argv_carries_the_binds_whatever_the_network_flag(tmp_path):
    sandbox = Sandbox("auto", [tmp_path])
    if not sandbox.enabled:
        pytest.skip("bubblewrap does not run here")
    for argv in (sandbox.argv("true"), sandbox.argv("true", network=True)):
        assert "--dev-bind" in argv
        assert str(host_gpu[0]) in argv


@needs_bwrap
@needs_host_gpu
def test_nvidia_smi_runs_inside_the_sandbox(tmp_path):
    """The regression Mats hit on 2026-10-23: this failed and read as a dead GPU."""
    sandbox = Sandbox("bwrap", [tmp_path])
    result = asyncio.run(
        sandbox.run("nvidia-smi --query-gpu=name --format=csv,noheader", tmp_path, 30)
    )
    assert result.exit_code == 0, result.output
    assert "NVIDIA" in result.output
