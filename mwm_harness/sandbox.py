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
import re
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


# A shell command is written by a model that may have just read a hostile page. It
# gets no API keys: names that look like secrets are dropped from its environment.
SECRET_ENV = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY)", re.I)
# Hidden from a sandboxed command altogether (an empty tmpfs is mounted over each).
PRIVATE_DIRS = (".ssh", ".gnupg", ".aws", ".config/mwm-harness", ".config/gh", ".qwen")


def scrubbed_env(
    secret_env: frozenset[str] = frozenset(), keep: frozenset[str] = frozenset()
) -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name in keep or not (name in secret_env or SECRET_ENV.search(name))
    }


class Sandbox:
    def __init__(
        self,
        mode: str,
        writable: list[Path],
        secret_env: frozenset[str] = frozenset(),
        env_keep: frozenset[str] = frozenset(),
    ) -> None:
        self.mode = mode
        self.secret_env = secret_env
        self.env_keep = env_keep
        if mode not in ("auto", "bwrap", "off"):
            raise ValueError(f"sandbox mode must be auto, bwrap or off, got {mode!r}")
        available = bwrap_works() if mode != "off" else False
        if mode == "bwrap" and not available:
            raise RuntimeError("sandbox = bwrap was requested but bubblewrap does not run here")
        self.enabled = available
        self.writable = [p.resolve() for p in writable]

    @property
    def warning(self) -> str:
        """Set when ``auto`` fell back to no sandbox: that must never pass in silence."""
        if self.enabled or self.mode == "off":
            return ""
        return (
            "bubblewrap does not run here, so shell commands run with NO sandbox "
            '(sandbox = auto). Install bubblewrap, or set sandbox = "off" to accept this'
        )

    def argv(self, command: str) -> list[str]:
        if not self.enabled:
            return ["bash", "-c", command]
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        for name in PRIVATE_DIRS:
            private = Path.home() / name
            if private.is_dir() and not any(
                w == private or private in w.parents for w in self.writable
            ):
                argv += ["--tmpfs", str(private)]
        for path in self.writable:
            if path.exists():
                argv += ["--bind", str(path), str(path)]
        return [*argv, "--die-with-parent", "bash", "-c", command]

    async def run(self, command: str, cwd: Path, timeout: float) -> ShellResult:
        process = await asyncio.create_subprocess_exec(
            *self.argv(command),
            cwd=str(cwd),
            env=scrubbed_env(self.secret_env, self.env_keep),
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
