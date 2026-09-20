"""Run shell commands in their own process group, optionally inside bubblewrap.

A command gets a new session, so a cancel or a timeout can kill the whole tree
(``sleep 60`` started by a script started by bash) with one ``killpg``.
With bubblewrap the filesystem is read-only except for the project directory,
the scratch directory and any extra paths from settings.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShellResult:
    exit_code: int | None
    output: str
    timed_out: bool = False


def bwrap_works() -> bool:
    """True when bubblewrap is installed AND this kernel lets it make a namespace."""
    binary = shutil.which("bwrap")
    if not binary:
        return False
    try:
        probe = subprocess.run(
            [binary, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


class Sandbox:
    def __init__(self, mode: str, writable: list[Path]) -> None:
        if mode not in ("auto", "bwrap", "off"):
            raise ValueError(f"sandbox mode must be auto, bwrap or off, got {mode!r}")
        available = bwrap_works() if mode != "off" else False
        if mode == "bwrap" and not available:
            raise RuntimeError("sandbox = bwrap was requested but bubblewrap does not run here")
        self.enabled = available
        self.writable = [p.resolve() for p in writable]

    def argv(self, command: str) -> list[str]:
        if not self.enabled:
            return ["bash", "-c", command]
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        for path in self.writable:
            if path.exists():
                argv += ["--bind", str(path), str(path)]
        return [*argv, "--die-with-parent", "bash", "-c", command]

    async def run(self, command: str, cwd: Path, timeout: float) -> ShellResult:
        process = await asyncio.create_subprocess_exec(
            *self.argv(command),
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        # Output is pumped into a buffer so a timeout still returns what was printed.
        buffer = bytearray()

        async def pump() -> None:
            assert process.stdout is not None
            while chunk := await process.stdout.read(65536):
                buffer.extend(chunk)

        pump_task = asyncio.create_task(pump())
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            timed_out = True
            await _kill_group(process)
        except asyncio.CancelledError:
            await _kill_group(process)
            pump_task.cancel()
            raise
        # A background child can hold the pipe open after bash exits; do not wait on it.
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(pump_task, 1.0)
        return ShellResult(process.returncode, buffer.decode("utf-8", errors="replace"), timed_out)


async def _kill_group(process: asyncio.subprocess.Process) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
        try:
            await asyncio.wait_for(process.wait(), 2.0)
        except TimeoutError:
            continue
        return
