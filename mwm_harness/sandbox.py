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
import resource
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


def user_scope_works() -> bool:
    """True when a user-level systemd scope can be created here.

    The process and memory caps (plan item 36) ride on a transient scope: this box's
    kernel gives a plain user no write access to cgroup control files (probed 10-24:
    mkdir under the running scope succeeds, pids.max inside it is root-owned), so
    systemd is the only cap route that needs no root. A headless caller without a bus
    (a cron-era gate) fails this probe and runs with the caps systemd carries switched
    off, which the notice says out loud.
    """
    if os.environ.get("XDG_RUNTIME_DIR") is None or shutil.which("systemd-run") is None:
        return False
    try:
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "--expand-environment=no", "true"],
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
# Settings.sandbox_private_readable exempts a named folder, which then stays visible
# and read-only; that is how a sandboxed git push reaches gh's stored credential.
PRIVATE_DIRS = (".ssh", ".gnupg", ".aws", ".config/mwm-harness", ".config/gh", ".qwen")
GPU_DEVICE_GLOB = "nvidia*"


# bwrap's --dev /dev is a fresh minimal devtmpfs holding none of the nodes the driver made
# at boot, so nvidia-smi exits non-zero and CUDA fails inside the sandbox while the same
# box is fine outside it. A machine with no NVIDIA driver matches nothing and binds nothing.
def gpu_device_binds(dev: Path = Path("/dev")) -> list[str]:
    """``--dev-bind`` pairs for every NVIDIA device node under ``dev``."""
    argv: list[str] = []
    for node in sorted(dev.glob(GPU_DEVICE_GLOB)):
        if not node.is_dir():
            argv += ["--dev-bind", str(node), str(node)]
    return argv


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
        private_readable: frozenset[str] = frozenset(),
        tasks_max: int = 512,
        memory_max_mb: int = 16384,
        file_max_mb: int = 2048,
        tmpfs_max_mb: int = 2048,
    ) -> None:
        self.mode = mode
        self.secret_env = secret_env
        self.env_keep = env_keep
        self.private_readable = private_readable
        self.tasks_max = max(0, int(tasks_max))
        self.memory_max_mb = max(0, int(memory_max_mb))
        self.file_max_mb = max(0, int(file_max_mb))
        self.tmpfs_max_mb = max(0, int(tmpfs_max_mb))
        if mode not in ("auto", "bwrap", "off"):
            raise ValueError(f"sandbox mode must be auto, bwrap or off, got {mode!r}")
        available = bwrap_works() if mode != "off" else False
        if mode == "bwrap" and not available:
            raise RuntimeError("sandbox = bwrap was requested but bubblewrap does not run here")
        self.enabled = available
        # Probed once per session, next to bwrap_works: every Bash call would otherwise
        # pay the scope round trip twice.
        self.scoped = bool(available) and user_scope_works()
        self.writable = [p.resolve() for p in writable]

    @property
    def warning(self) -> str:
        """Set when ``auto`` fell back to no sandbox, or when the process and memory caps
        could not be armed: neither may pass in silence."""
        if not self.enabled:
            if self.mode == "off":
                return ""
            return (
                "bubblewrap does not run here, so shell commands run with NO sandbox "
                "and NO resource caps (sandbox = auto). Install bubblewrap, or set "
                'sandbox = "off" to accept this'
            )
        if not self.scoped and (self.tasks_max or self.memory_max_mb):
            wanted = ", ".join(
                part
                for part, on in (
                    (f"TasksMax={self.tasks_max}", self.tasks_max),
                    (f"MemoryMax={self.memory_max_mb} MB", self.memory_max_mb),
                )
                if on
            )
            return (
                f"no user systemd scope is available, so the process and memory caps "
                f"({wanted}) are NOT in force; /tmp, per-file and session-write caps still are"
            )
        return ""

    def scope_prefix(self) -> list[str]:
        """systemd-run wrapper putting the call in a capped transient scope, or [] when
        unavailable or both caps are off. The scope costs ~13 ms per call (measured on
        this box); it holds the whole command tree, so a fork bomb gets EAGAIN and an
        unbounded allocation gets killed at the cap instead of taking the box."""
        if not self.scoped or not (self.tasks_max or self.memory_max_mb):
            return []
        argv = ["systemd-run", "--user", "--scope", "--quiet", "--expand-environment=no"]
        if self.tasks_max:
            argv += ["-p", f"TasksMax={self.tasks_max}"]
        if self.memory_max_mb:
            argv += ["-p", f"MemoryMax={self.memory_max_mb}M"]
            # A scope's default OOM policy stops the whole unit when the kernel
            # memcg-kills one process: the wrapper dies with it (rc -15, output lost)
            # and cap_notes has nothing to report. "continue" kills only the offender,
            # so the command's own exit path still runs and names the cap.
            argv += ["-p", "OOMPolicy=continue"]
        return argv

    def _apply_fsize(self) -> None:
        """preexec_fn: cap the largest single file the command may grow (RLIMIT_FSIZE).
        Inherited across exec, so it reaches bash through systemd-run and bwrap."""
        if self.enabled and self.file_max_mb:
            limit = self.file_max_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    def argv(self, command: str, network: bool = False) -> list[str]:
        """The command line for ``command``; without ``network`` it gets its own empty
        network namespace (only a loopback of its own: no internet, no host services)."""
        if not self.enabled:
            return ["bash", "-c", command]
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        # The host /tmp arrives read-only through the root bind, which breaks every tool
        # that calls mktemp (the detect-secrets pre-commit hook, backup.sh's lock file).
        # Each sandbox gets its own empty writable tmpfs instead: mktemp works, and host
        # /tmp contents stay invisible, same isolation as the PRIVATE_DIRS tmpfs mounts.
        # --perms applies to this tmpfs only. Without it bwrap makes /tmp 0755, and a
        # tool that checks for the sticky bit before trusting TMPDIR refuses to use it.
        # --size caps this tmpfs too (it applies to the next --tmpfs argument only):
        # the junk sink of a runaway command has a hard ENOSPC ceiling.
        if self.tmpfs_max_mb:
            argv += ["--size", str(self.tmpfs_max_mb * 1024 * 1024)]
        argv += ["--perms", "1777", "--tmpfs", "/tmp"]
        argv += gpu_device_binds()
        if not network:
            argv.append("--unshare-net")
        for name in PRIVATE_DIRS:
            # An exempt folder keeps the read-only root bind it already has: a command
            # can read the credential (which is what a git credential helper needs) but
            # cannot rewrite it.
            if name in self.private_readable:
                continue
            private = Path.home() / name
            if private.is_dir() and not any(
                w == private or private in w.parents for w in self.writable
            ):
                argv += ["--tmpfs", str(private)]
        for path in self.writable:
            if path.exists():
                argv += ["--bind", str(path), str(path)]
        return [*argv, "--die-with-parent", "bash", "-c", command]

    async def run(
        self, command: str, cwd: Path, timeout: float, network: bool = False
    ) -> ShellResult:
        process = await asyncio.create_subprocess_exec(
            *self.scope_prefix(),
            *self.argv(command, network),
            cwd=str(cwd),
            env=scrubbed_env(self.secret_env, self.env_keep),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=self._apply_fsize,
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
        output = buffer.decode("utf-8", errors="replace")
        notes = self.cap_notes(process.returncode, output, timed_out)
        if notes:
            output = output.rstrip("\n") + "\n" + "\n".join(notes) + "\n"
        return ShellResult(process.returncode, output, timed_out)

    def cap_notes(self, returncode: int | None, output: str, timed_out: bool) -> list[str]:
        """Name the resource cap a run hit, so an unattended run reports the cap instead
        of a bare crash (plan item 36 closing test). Signals only checked when the cap was
        actually in force; a timeout or a cancel killed the tree itself and says so
        elsewhere, so those runs get no cap note."""
        if timed_out or not self.enabled:
            return []
        notes: list[str] = []
        killed = returncode in (137, -9) or "Killed" in output
        if self.scoped and self.memory_max_mb and killed:
            notes.append(
                f"[exit was a kill: if this command allocated without bound it hit the "
                f"sandbox memory cap of {self.memory_max_mb} MB (sandbox_memory_max_mb); "
                f"if not, something else sent SIGKILL]"
            )
        if self.scoped and self.tasks_max and "Resource temporarily unavailable" in output:
            notes.append(
                f"[fork refused at the process cap of {self.tasks_max} (sandbox_tasks_max): "
                f"this command runs more processes at once than one sandboxed call may have]"
            )
        if self.file_max_mb and (
            returncode in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ)
            or "File size limit exceeded" in output
        ):
            notes.append(
                f"[a single file hit the size cap of {self.file_max_mb} MB "
                f"(sandbox_file_max_mb); split the output or raise the cap]"
            )
        if self.tmpfs_max_mb and "No space left on device" in output:
            notes.append(
                f"[disk full inside the sandbox: /tmp is capped at {self.tmpfs_max_mb} MB "
                f"(sandbox_tmpfs_max_mb); writable project paths are on the real disk]"
            )
        return notes


async def _kill_group(process: asyncio.subprocess.Process) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
        try:
            await asyncio.wait_for(process.wait(), 2.0)
        except TimeoutError:
            continue
        return
