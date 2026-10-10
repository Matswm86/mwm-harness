"""Run-it view: which changed lines did a test execute, and what did one test call?

``changed_coverage`` runs the repo's tests under coverage.py with one context per test,
then reports each changed line (``git diff`` against HEAD, or the whole file when it is
new) as ``hit`` with the tests that ran it, or ``miss`` when no test ran it. Lines that
are not statements (blank, comments) are left out. ``trace_test`` runs one test with
``trace_plugin.py`` and returns the calls it made into the repo with argument values.

Both run in the repo's own Python (``.venv/bin/python``, else the harness Python); a
missing coverage or pytest there is reported as NOT RUN, never as a pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mwm_harness.checks import _repo_python, exec_capped

PLUGIN = Path(__file__).resolve().parent / "trace_plugin.py"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
TAIL_LINES = 12  # pytest output kept for the person when a run fails


@dataclass
class LineState:
    line: int
    state: str  # hit | miss
    tests: list[str] = field(default_factory=list)


@dataclass
class RunItResult:
    status: str  # ran | not_run | timeout
    python: str
    tests_passed: bool = False
    summary: str = ""  # pytest's last line, e.g. "5 passed in 0.31s"
    files: dict[str, list[LineState]] = field(default_factory=dict)  # repo-relative path
    reason: str = ""
    tail: str = ""
    seconds: float = 0.0

    def counts(self) -> tuple[int, int]:
        lines = [s for states in self.files.values() for s in states]
        return sum(s.state == "hit" for s in lines), sum(s.state == "miss" for s in lines)

    def line(self) -> str:
        if self.status != "ran":
            return f"run-it: {'NOT RUN' if self.status == 'not_run' else 'TIMEOUT'}: {self.reason}"
        hit, miss = self.counts()
        tests = "tests pass" if self.tests_passed else "TESTS FAIL"
        return f"run-it: {tests} ({self.summary}); changed lines run {hit}, never run {miss}"


@dataclass
class TraceResult:
    status: str  # ran | not_run | timeout
    test: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0  # calls over the depth or count cap
    passed: bool = False
    reason: str = ""
    tail: str = ""


def changed_lines(root: Path, files: list[Path]) -> dict[str, set[int]]:
    """Repo-relative path -> line numbers added or changed since HEAD (all lines when new)."""
    out: dict[str, set[int]] = {}
    for path in files:
        if not path.is_file() or path.suffix != ".py":
            continue
        rel = str(path.resolve().relative_to(root.resolve()))
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "-U0", "HEAD", "--", rel],
            capture_output=True,
            text=True,
            check=False,
        )
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
            capture_output=True,
            check=False,
        )
        if diff.returncode != 0 or tracked.returncode != 0:
            count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            out[rel] = set(range(1, count + 1))  # new file, or no git: every line counts
            continue
        lines: set[int] = set()
        for match in HUNK.finditer(diff.stdout):
            start, length = int(match.group(1)), int(match.group(2) or 1)
            lines.update(range(start, start + length))
        if lines:
            out[rel] = lines
    return out


def is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def git_changed(root: Path) -> list[Path]:
    """Files git reports modified against HEAD plus untracked ones; empty outside git."""
    names: list[str] = []
    for argv in (["diff", "--name-only", "HEAD"], ["ls-files", "--others", "--exclude-standard"]):
        out = subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, check=False
        )
        if out.returncode == 0:
            names += out.stdout.split("\n")
    return [root / n for n in dict.fromkeys(names) if n.strip()]


def _has_module(python: str, module: str, root: Path) -> bool:
    probe = subprocess.run(
        [python, "-c", f"import {module}"], cwd=root, capture_output=True, check=False
    )
    return probe.returncode == 0


def _tail(output: str) -> str:
    return "\n".join(output.rstrip().splitlines()[-TAIL_LINES:])


def _summary(output: str) -> str:
    rows = [r.strip("= ").strip() for r in output.strip().splitlines() if r.strip()]
    return rows[-1] if rows else ""


def node_id(root: Path, context: str) -> str:
    """coverage's ``tests.test_a.TestX.test_y`` context as a pytest node id.

    The longest dotted prefix that names an existing ``.py`` file is the module;
    the rest are class and function names. Unresolvable contexts come back as given.
    """
    parts = context.removesuffix("|run").split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module = Path(*parts[:cut]).with_suffix(".py")
        if (root / module).is_file():
            return "::".join([module.as_posix(), *parts[cut:]])
    return context


async def changed_coverage(
    root: Path, files: list[Path], pytest_args: list[str] | None = None, timeout: float = 300.0
) -> RunItResult:
    python = _repo_python(root)
    started = time.monotonic()
    for module in ("pytest", "coverage"):
        if not await _probe(python, module, root):
            return RunItResult("not_run", python, reason=f"{module} missing in {python}")
    wanted = changed_lines(root, files)
    if not wanted:
        return RunItResult("not_run", python, reason="no changed Python lines")
    with tempfile.TemporaryDirectory(prefix="mwm-runit-") as tmp:
        rc = Path(tmp) / "coveragerc"
        data = Path(tmp) / "coverage.data"
        rc.write_text(
            f"[run]\ndynamic_context = test_function\ndata_file = {data}\n"
            f"source = {root}\nomit =\n    */.venv/*\n    */site-packages/*\n",
            encoding="utf-8",
        )
        argv = [python, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "-q"]
        argv += ["-p", "no:cacheprovider", *(pytest_args or [])]
        code, output = await exec_capped(argv, root, timeout)
        if code is None:
            return RunItResult("timeout", python, reason=f"tests over {timeout:g} s")
        report = Path(tmp) / "coverage.json"
        _, json_out = await exec_capped(
            [
                python,
                "-m",
                "coverage",
                "json",
                f"--rcfile={rc}",
                "--show-contexts",
                "-o",
                str(report),
            ],
            root,
            60.0,
        )
        if not report.is_file():
            return RunItResult(
                "not_run", python, reason="coverage wrote no report", tail=_tail(output + json_out)
            )
        measured = json.loads(report.read_text(encoding="utf-8"))["files"]
    result = RunItResult(
        "ran",
        python,
        tests_passed=code == 0,
        summary=_summary(output),
        tail="" if code == 0 else _tail(output),
        seconds=time.monotonic() - started,
    )
    for rel, lines in wanted.items():
        info = measured.get(rel) or measured.get(str(root / rel))
        if info is None:
            # Not imported by any test: every statement is a miss, but we only know which
            # lines are statements from coverage, so report the changed non-blank lines.
            text = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            states = [
                LineState(n, "miss")
                for n in sorted(lines)
                if n <= len(text)
                and text[n - 1].strip()
                and not text[n - 1].lstrip().startswith("#")
            ]
        else:
            executed, missing = set(info["executed_lines"]), set(info["missing_lines"])
            contexts = info.get("contexts", {})
            states = []
            for n in sorted(lines):
                if n in executed:
                    tests = sorted({node_id(root, c) for c in contexts.get(str(n), []) if c})
                    states.append(LineState(n, "hit", tests))
                elif n in missing:
                    states.append(LineState(n, "miss"))
        if states:
            result.files[rel] = states
    return result


async def trace_test(root: Path, test: str, timeout: float = 120.0) -> TraceResult:
    """Run one pytest node id with the call tracer; calls are depth-normalised from 0."""
    python = _repo_python(root)
    if not await _probe(python, "pytest", root):
        return TraceResult("not_run", test, reason=f"pytest missing in {python}")
    with tempfile.TemporaryDirectory(prefix="mwm-trace-") as tmp:
        shutil.copy(PLUGIN, Path(tmp) / "mwm_trace_plugin.py")
        out = Path(tmp) / "trace.json"
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [tmp, os.environ.get("PYTHONPATH")])),
            "MWM_TRACE_ROOT": str(root),
            "MWM_TRACE_OUT": str(out),
        }
        argv = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "mwm_trace_plugin"]
        code, output = await exec_capped([*argv, test], root, timeout, env=env)
        if code is None:
            return TraceResult("timeout", test, reason=f"over {timeout:g} s")
        if not out.is_file():
            return TraceResult(
                "not_run", test, reason="the tracer wrote nothing", tail=_tail(output)
            )
        data = json.loads(out.read_text(encoding="utf-8"))
    calls = data["calls"]
    if calls:
        low = min(c["depth"] for c in calls)
        for call in calls:
            call["depth"] -= low
    return TraceResult(
        "ran", test, calls, data["skipped"], code == 0, tail="" if code == 0 else _tail(output)
    )


async def _probe(python: str, module: str, root: Path) -> bool:
    return await asyncio.to_thread(_has_module, python, module, root)


def as_dict(result: RunItResult | TraceResult) -> dict[str, Any]:
    return asdict(result)
