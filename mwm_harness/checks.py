"""Check runner: compile, lint and type-check the files a turn changed, with no model call.

The chain comes from ``checks.default.toml`` (see its header for the format); a repo's
``.mwm/checks.toml`` replaces it, else the config folder's ``checks.toml``. Every check
reports one status: ``pass``, ``fail`` (with findings), ``not_run`` (with the reason, for
example the tool is missing) or ``timeout``. A missing tool is never shown as a pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mwm_harness.config import config_dir

DEFAULT_CHECKS = Path(__file__).resolve().parent / "checks.default.toml"
DEFAULT_PATTERN = r"^(?P<file>[^\s:][^:\n]*):(?P<line>\d+):(?:(?P<col>\d+):)?\s*(?P<msg>.*\S)"
OUTPUT_CAP = 200  # findings kept per check


class CheckConfigError(Exception):
    pass


@dataclass
class Check:
    name: str
    glob: list[str]
    run: list[str] | str
    pattern: str = DEFAULT_PATTERN

    def matches(self, path: Path) -> bool:
        return any(fnmatch.fnmatch(path.name, g) for g in self.glob)


@dataclass
class Finding:
    file: str  # relative to the repo root
    line: int  # 0 = the whole file
    col: int
    message: str
    check: str
    severity: str = "error"  # error | warning


@dataclass
class CheckResult:
    check: str
    status: str  # pass | fail | not_run | timeout
    files: list[str]
    findings: list[Finding] = field(default_factory=list)
    reason: str = ""
    seconds: float = 0.0

    def line(self) -> str:
        if self.status == "pass":
            return f"{self.check}: pass ({len(self.files)} file(s))"
        if self.status == "fail":
            return f"{self.check}: FAIL, {len(self.findings)} finding(s)"
        label = "NOT RUN" if self.status == "not_run" else "TIMEOUT"
        return f"{self.check}: {label}: {self.reason}"


def load_checks(root: Path) -> tuple[list[Check], float, Path]:
    """The chain for this repo, its per-command timeout, and the file it came from."""
    for path in (root / ".mwm" / "checks.toml", config_dir() / "checks.toml", DEFAULT_CHECKS):
        if path.is_file():
            break
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CheckConfigError(f"{path}: {exc}") from exc
    checks = []
    for raw in data.get("check", []):
        try:
            check = Check(
                name=str(raw["name"]),
                glob=list(raw["glob"]),
                run=raw["run"],
                pattern=str(raw.get("pattern", DEFAULT_PATTERN)),
            )
            re.compile(check.pattern, re.MULTILINE)
        except (KeyError, TypeError, re.error) as exc:
            raise CheckConfigError(f"{path}: bad check {raw!r}: {exc}") from exc
        checks.append(check)
    return checks, float(data.get("timeout", 30.0)), path


def repo_root(cwd: Path) -> Path:
    """The git top folder of ``cwd``, else ``cwd`` itself."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return cwd
    return Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else cwd


def _repo_python(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _search_path(root: Path) -> str:
    """The repo's own .venv/bin first, so its pinned ruff or mypy wins over the system one."""
    return os.pathsep.join([str(root / ".venv" / "bin"), os.environ.get("PATH", "")])


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _py_compile(files: list[Path], root: Path, name: str) -> CheckResult:
    started = time.monotonic()
    findings = []
    for path in files:
        try:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            message = f"{type(exc).__name__}: {exc.msg}"
            findings.append(
                Finding(_rel(path, root), exc.lineno or 0, exc.offset or 0, message, name)
            )
        except (OSError, ValueError) as exc:
            findings.append(Finding(_rel(path, root), 0, 0, f"cannot compile: {exc}", name))
    rels = [_rel(p, root) for p in files]
    status = "fail" if findings else "pass"
    return CheckResult(name, status, rels, findings, seconds=time.monotonic() - started)


def _parse(check: Check, output: str, files: list[Path], root: Path) -> list[Finding]:
    wanted = {p.resolve(): p for p in files}
    findings = []
    error_lines = [ln.strip() for ln in output.splitlines() if "Error" in ln]
    for match in re.finditer(check.pattern, output, re.MULTILINE):
        parts = match.groupdict()
        path = Path(parts["file"].strip())
        path = (path if path.is_absolute() else root / path).resolve()
        if path not in wanted:
            continue  # a finding in a file this turn did not change
        message = (parts.get("msg") or "").strip() or (error_lines[-1] if error_lines else "")
        if message.startswith("note:"):
            continue
        severity = "warning" if re.match(r"(warning|note)\b", message) else "error"
        findings.append(
            Finding(
                _rel(wanted[path], root),
                int(parts["line"]),
                int(parts.get("col") or 0),
                message,
                check.name,
                severity,
            )
        )
    return findings[:OUTPUT_CAP]


async def exec_capped(
    argv: list[str], root: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int | None, str]:
    """Run one command; ``None`` as the exit code means it hit the timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, 9)
        await proc.wait()
        return None, ""
    return proc.returncode, out.decode("utf-8", "replace")


async def run_check(check: Check, files: list[Path], root: Path, timeout: float) -> CheckResult:
    rels = [_rel(p, root) for p in files]
    if isinstance(check.run, str):
        if check.run == "builtin:py_compile":
            return await asyncio.to_thread(_py_compile, files, root, check.name)
        if check.run.startswith("unavailable:"):
            return CheckResult(check.name, "not_run", rels, reason=check.run.split(":", 1)[1])
        return CheckResult(check.name, "not_run", rels, reason=f"unknown run value {check.run!r}")
    program = check.run[0].replace("{python}", _repo_python(root))
    found = shutil.which(program, path=_search_path(root))
    if found is None:
        return CheckResult(check.name, "not_run", rels, reason=f"{program} missing")

    def argv(targets: list[Path]) -> list[str]:
        out = [found]
        for part in check.run[1:]:
            if part == "{files}":
                out += [str(p) for p in targets]
            else:
                out.append(part.replace("{file}", str(targets[0])).replace("{python}", found))
        return out

    per_file = "{file}" in " ".join(check.run)
    batches = [[p] for p in files] if per_file else [files]
    started = time.monotonic()
    findings: list[Finding] = []
    for batch in batches:
        code, output = await exec_capped(argv(batch), root, timeout)
        if code is None:
            return CheckResult(
                check.name,
                "timeout",
                rels,
                findings,
                f"over {timeout:g} s",
                time.monotonic() - started,
            )
        parsed = _parse(check, output, batch, root)
        if code != 0 and not parsed:
            last = next((ln for ln in reversed(output.splitlines()) if ln.strip()), "")
            note = last.strip() or f"exit code {code}, no output"
            parsed = [Finding(_rel(batch[0], root), 0, 0, note, check.name)]
        findings += parsed
    # Warnings alone leave the check passing; they still show in the gutter.
    status = "fail" if any(f.severity == "error" for f in findings) else "pass"
    return CheckResult(check.name, status, rels, findings, seconds=time.monotonic() - started)


async def run_checks(root: Path, changed: list[Path]) -> list[CheckResult]:
    """Run every check whose glob matches a changed file that still exists."""
    checks, timeout, _ = load_checks(root)
    present = [p for p in dict.fromkeys(changed) if p.is_file()]
    # Syntax checks go first: a file that does not parse is kept out of the others,
    # because one unparsable file stops mypy from checking any file in the call.
    first = [c for c in checks if c.run == "builtin:py_compile"]
    results = [
        await run_check(c, files, root, timeout)
        for c in first
        if (files := [p for p in present if c.matches(p)])
    ]
    broken = {root / f.file for r in results for f in r.findings}
    jobs, left_out = [], []
    for check in checks:
        if check in first:
            continue
        matching = [p for p in present if check.matches(p)]
        files = [p for p in matching if p not in broken]
        if files:
            jobs.append(run_check(check, files, root, timeout))
            left_out.append(sorted(_rel(p, root) for p in matching if p in broken))
    for result, skipped in zip(await asyncio.gather(*jobs), left_out, strict=True):
        if skipped:
            result.reason = f"left out (syntax error): {', '.join(skipped)}"
        results.append(result)
    return results


def as_dicts(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
