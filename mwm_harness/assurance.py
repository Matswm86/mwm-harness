"""Assurance card (plan item 22): six quality rows, filled only from evidence.

The question this card asks, written verbatim at its top:

    "Have we done what is necessary to establish that this software is safe, correct,
     reliable, performant, scalable, and fit for operation?"

Every row is PASS, FAIL, NOT RUN (with a reason) or N/A (with a reason). The card is built
from evidence the harness already has or can collect deterministically: the check chain
(item 20), run-it coverage (item 21), bug flags (item 30), a secret scan, a dependency audit
when a requirements file changed, a PyPI existence check for newly added requirement names (a
model-invented package is a FAIL on the safe row, named), and the model/MCP surface look that
puts the OWASP LLM Top 10 in scope. Building a card runs no tests and no model.

The gate: ``done_allowed`` is False while any row is FAIL or EMPTY. NOT RUN and N/A rows carry
a reason and do not hold the turn by themselves: they say "nobody checked this", which is
honest, and holding on every un-benchmarked turn would only train everyone to ignore the card.
The model may fill only three kinds of cells (the ``AssuranceReport`` tool enforces this):
N/A reasons for the performant and scalable rows, and the operation row. Those rows carry
``model_written`` and are labelled as model-written wherever they render.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mwm_harness import redact
from mwm_harness.bugflag import FlagReport
from mwm_harness.checks import CheckResult, _repo_python, repo_root

QUALITIES = ("safe", "correct", "reliable", "performant", "scalable", "fit for operation")
QUESTION = (
    "Have we done what is necessary to establish that this software is safe, correct, "
    "reliable, performant, scalable, and fit for operation?"
)

DEPS_FILES = re.compile(r"(^|/)(requirements[^/]*\.txt|pyproject\.toml|Pipfile|poetry\.lock)$")
CODE_SUFFIX = {".py", ".js", ".mjs", ".cjs", ".ts", ".gd", ".pine", ".sh", ".bash"}
REQ_LINE = re.compile(r'^\+\s*"?([A-Za-z0-9][A-Za-z0-9._-]{1,60})(?:[=<>!~\["].*)?$')
PYPI = "https://pypi.org/pypi/{name}/json"
AGE_SUSPICIOUS_DAYS = 7  # a brand-new named package is noted, not failed: age alone proves nothing
MODEL_ROWS = {"performant", "scalable", "fit for operation"}
STATUS_WORD = {"pass": "PASS", "fail": "FAIL", "not_run": "NOT RUN", "na": "N/A"}

# A changed file that talks to a model or an MCP server puts the OWASP LLM Top 10 in scope.
LLM_SURFACE = re.compile(
    r"from mwm_harness import providers|from mwm_harness\.providers|from mwm_harness import loop"
    r"|from mwm_harness\.mcp_client|import mcp_client|\bcall_model\b|mcp__[a-z0-9_]+__"
    r"|chat/completions|messages\.create\(",
    re.I,
)
FALLBACK_SECRET = re.compile(
    r"(?i)\b(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{20,}|xox[bp]-\S{10,}|AIza[0-9A-Za-z_-]{20,}"
    r"|gsk_[A-Za-z0-9]{10,})|password\s*[=:]\s*[^\s\"']{6,}|api[_-]?key\s*[=:]\s*[^\s\"']{6,}"
)


@dataclass
class Row:
    quality: str
    status: str = ""  # pass | fail | not_run | na | "" (empty: nothing filled this row)
    reason: str = ""  # required for not_run and na, shown for fail
    evidence: list[str] = field(default_factory=list)
    model_written: bool = False

    @property
    def filled(self) -> bool:
        return self.status != ""

    def label(self) -> str:
        word = STATUS_WORD.get(self.status, "EMPTY")
        extra = f" ({self.reason})" if self.reason else ""
        mark = " [model-written]" if self.model_written else ""
        return f"{self.quality}: {word}{extra}{mark}"


@dataclass
class Card:
    root: str
    files: list[str] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    built_at: float = field(default_factory=time.time)

    def row(self, quality: str) -> Row:
        for item in self.rows:
            if item.quality == quality:
                return item
        raise KeyError(quality)

    @property
    def holding(self) -> list[str]:
        """Rows that keep the turn from being presented as done: FAIL or EMPTY."""
        return [r.quality for r in self.rows if r.status == "fail" or not r.filled]

    @property
    def unanswered(self) -> list[str]:
        """NOT RUN rows: honest, but each one says "nobody checked this this turn"."""
        return [r.quality for r in self.rows if r.status == "not_run"]

    @property
    def done_allowed(self) -> bool:
        return not self.holding

    def gate_line(self) -> str:
        if self.done_allowed:
            held = f" ({', '.join(self.unanswered)} NOT RUN)" if self.unanswered else ""
            return f"assurance: done allowed{held}"
        return f"assurance: DONE HELD on {', '.join(self.holding)}"

    def render(self) -> str:
        lines = [QUESTION, ""]
        for item in self.rows:
            lines.append("  " + item.label())
            for evidence in item.evidence[:6]:
                lines.append(f"      - {evidence}")
        lines.append("")
        lines.append(self.gate_line())
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": QUESTION,
            "root": self.root,
            "files": self.files,
            "rows": [asdict(r) for r in self.rows],
            "done_allowed": self.done_allowed,
            "holding": self.holding,
            "not_run": self.unanswered,
            "built_at": self.built_at,
        }

    def fill(self, quality: str, status: str, reason: str) -> None:
        """A model-written cell: only the three declared rows, always labelled."""
        if quality not in MODEL_ROWS:
            raise ValueError(
                f"the model may not fill {quality!r}: safe, correct and reliable come from "
                "code evidence, not from a model's word"
            )
        if status not in {"na", "pass"} or not reason.strip():
            raise ValueError("a model-written row needs status na or pass and a reason")
        item = self.row(quality)
        if item.status in {"pass", "fail"} and not item.model_written:
            raise ValueError(f"{quality} already has code evidence; a model note cannot replace it")
        item.status = status
        item.reason = reason.strip()[:500]
        item.model_written = True


def _external(argv: list[str], root: Path, tool: str, timeout: float = 60.0) -> tuple[str, str]:
    """Run one scan binary. Returns (status, detail): pass, fail, error, timeout."""
    try:
        done = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", f"{tool} ran over {timeout:g} s"
    except OSError as exc:
        return "error", f"{tool} could not run: {exc}"
    if done.returncode == 0:
        return "pass", ""
    text = (done.stdout or done.stderr).strip()
    last = text.splitlines()[-1][:300] if text else ""
    return "fail", last or f"{tool} exited {done.returncode}"


def _has_module(python: str, module: str, root: Path) -> bool:
    probe = subprocess.run(
        [python, "-c", f"import {module}"], cwd=root, capture_output=True, check=False
    )
    return probe.returncode == 0


def _new_requirement_names(root: Path, files: list[Path]) -> list[str]:
    """Package names ADDED to a requirements/pyproject file this turn (git diff + lines)."""
    names: list[str] = []
    for path in files:
        if not DEPS_FILES.search(path.name):
            continue
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
            diff = subprocess.run(
                ["git", "-C", str(root), "diff", "-U0", "HEAD", "--", rel],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = (
                diff.stdout.splitlines()
                if diff.returncode == 0
                else path.read_text(encoding="utf-8", errors="replace").splitlines()
            )  # untracked: the whole file is new
        except (OSError, ValueError):
            continue
        for line in lines:
            if not line.startswith("+") or line.startswith("+++"):
                continue
            match = REQ_LINE.match(line)
            if match:
                names.append(match.group(1).lower())
    return list(dict.fromkeys(names))


def _pypi_exists(name: str, timeout: float = 5.0) -> tuple[str, str]:
    """'yes (age note)', 'no' or 'error (...)': does this name exist on PyPI right now?"""
    try:
        request = urllib.request.Request(
            PYPI.format(name=name), headers={"User-Agent": "mwm-harness"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 (fixed https host)
            data = json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return ("no", "") if exc.code == 404 else ("error", f"pypi answered {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return "error", f"pypi unreachable: {type(exc).__name__}"
    times = [
        r[0].get("upload_time")
        for r in (data.get("releases") or {}).values()
        if r and r[0].get("upload_time")
    ]
    age = ""
    if times:
        days = (time.time() - time.mktime(time.strptime(min(times)[:10], "%Y-%m-%d"))) / 86400
        age = f", first release {days:.0f} days ago"
        if days < AGE_SUSPICIOUS_DAYS:
            age += " (suspiciously new)"
    return "yes", age


async def _check_new_packages(
    root: Path, files: list[Path]
) -> tuple[list[str], list[str], list[str]]:
    """(lines, missing_names, notes) for the names this turn added to a dependency file."""
    names = await asyncio.to_thread(_new_requirement_names, root, files)
    lines, missing, notes = [], [], []
    budget = time.monotonic() + 20.0
    for name in names[:10]:
        if time.monotonic() > budget:
            notes.append(
                f"pypi check hit its 20 s budget: {len(names) - len(lines) - len(missing) - len(notes)} name(s) unchecked"
            )
            break
        state, note = await asyncio.to_thread(_pypi_exists, name)
        if state == "no":
            missing.append(name)
        elif state == "error":
            notes.append(f"{name}: {note}")
        else:
            lines.append(f"new requirement {name} exists on PyPI{note}")
    return lines, missing, notes


def _scan_secrets(root: Path, files: list[Path], python: str) -> tuple[str, str]:
    """Secret scan of the changed files: detect-secrets when installed, else the built-in look.

    The built-in look reuses redact.py's shapes plus a key=value sweep and says it is the
    fallback in its line, so its PASS never reads like a detect-secrets PASS.
    """
    targets = [f for f in files if f.is_file()]
    if not targets:
        return "pass", "secret scan: nothing to read"
    if shutil.which("detect-secrets") or _has_module(python, "detect_secrets", root):
        argv = ["detect-secrets", "scan", "--exclude-files", r"/(\.venv|node_modules|\.git)/"] + [
            str(f) for f in targets
        ]
        state, detail = _external(argv, root, "detect-secrets")
        if state == "pass":
            return "pass", "detect-secrets: no suspected secrets in the changed files"
        if state == "fail":
            return "fail", "detect-secrets reported suspected secrets on the changed files"
        return state, detail
    hits: list[str] = []
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names = {name for pattern, name in redact.SHAPES if pattern.search(text)}
        if names:
            hits.append(f"{path.name}: {'/'.join(sorted(names))}")
        elif FALLBACK_SECRET.search(text):
            hits.append(f"{path.name}: key=value secret shape")
    if hits:
        return "fail", "built-in secret look (not detect-secrets): " + "; ".join(hits[:4])
    return "pass", (
        "built-in secret look (not detect-secrets): no secret shapes in the changed files"
    )


def _pip_audit(root: Path, python: str) -> tuple[str, str]:
    """Dependency audit; only meaningful when a dependency file changed."""
    if shutil.which("pip-audit"):
        argv = ["pip-audit", "--progress-spinner", "off", "--disable-pip"]
        tool = "pip-audit"
    elif _has_module(python, "pip_audit", root):
        argv, tool = [python, "-m", "pip_audit", "--progress-spinner", "off"], "pip-audit"
    else:
        return "not_installed", "pip-audit is not installed, so the dependency audit did not run"
    return _external(argv, root, tool, timeout=120.0)


def _sast(root: Path, files: list[Path]) -> tuple[str, str]:
    """Bandit for Python, semgrep if present; neither installed is reported, never skipped."""
    py = [f for f in files if f.suffix == ".py" and f.is_file()]
    if not py:
        return "pass", "SAST: no changed Python file to scan"
    if shutil.which("semgrep"):
        argv = ["semgrep", "--quiet", "--error", "--config", "auto", *(str(f) for f in py)]
        state, detail = _external(argv, root, "semgrep", timeout=120.0)
        if state == "pass":
            return "pass", "semgrep: clean on the changed files"
        if state == "fail":
            return "fail", f"semgrep reported findings on the changed files: {detail}"
        return state, detail
    python = _repo_python(root)
    if shutil.which("bandit") or _has_module(python, "bandit", root):
        argv = (["bandit", "-q"] if shutil.which("bandit") else [python, "-m", "bandit", "-q"]) + [
            str(f) for f in py
        ]
        state, detail = _external(argv, root, "bandit", timeout=60.0)
        if state == "pass":
            return "pass", "bandit: clean on the changed files"
        if state == "fail":
            return "fail", f"bandit reported findings on the changed files: {detail}"
        return state, detail
    return "not_installed", "no SAST binary on this box (semgrep or bandit), so it did not run"


def _llm_surface(files: list[Path]) -> list[str]:
    """Changed files that call a model or an MCP server: the OWASP LLM Top 10 applies to them."""
    out: list[str] = []
    for path in files:
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            if LLM_SURFACE.search(path.read_text(encoding="utf-8", errors="replace")):
                out.append(path.name)
        except OSError:
            continue
    return out


def _row(
    label: str,
    status: str,
    reason: str = "",
    evidence: list[str] | None = None,
    model_written: bool = False,
) -> Row:
    return Row(label, status, reason, evidence or [], model_written)


async def build_card(
    cwd: Path,
    files: list[Path],
    *,
    checks: list[CheckResult],
    runit: dict[str, Any] | None,
    flags: FlagReport,
    fix: dict[str, Any] | None,
    saw_test_run: bool,
    saw_test_pass: bool,
    scans: bool = True,
) -> Card:
    """Fill the six rows from the evidence handed in. Runs no tests and no model call."""
    root = await asyncio.to_thread(repo_root, cwd)
    rel: list[str] = []
    for path in files:
        try:
            rel.append(str(path.resolve().relative_to(root.resolve())))
        except ValueError:
            rel.append(path.name)
    card = Card(str(root), files=rel)
    python = _repo_python(root)
    code_touched = any(Path(name).suffix in CODE_SUFFIX for name in rel)
    deps_touched = any(DEPS_FILES.search(name) for name in rel)

    # ---------- safe ----------
    evidence: list[str] = []
    fails: list[str] = []
    not_runs: list[str] = []
    if not scans:
        not_runs.append("assurance_scans = false: no external scan ran")
        secret_state = ("skipped", "")
    else:
        secret_state, secret_detail = await asyncio.to_thread(_scan_secrets, root, files, python)
        if secret_state == "pass":
            evidence.append(secret_detail)
        elif secret_state == "fail":
            evidence.append(secret_detail)
            fails.append(secret_detail.split(":")[0] + " found something on the changed files")
        else:
            not_runs.append(f"secret scan: {secret_detail or secret_state}")
    if deps_touched:
        audit_state, audit_detail = await asyncio.to_thread(_pip_audit, root, python)
        if audit_state == "pass":
            evidence.append("pip-audit: no known vulnerabilities in this environment")
        elif audit_state == "fail":
            evidence.append(f"pip-audit: {audit_detail}")
            fails.append("pip-audit reported vulnerable dependencies")
        else:
            not_runs.append(f"dependency audit: {audit_detail}")
    else:
        evidence.append("dependency audit: N/A, no requirements file changed this turn")
    pypi_lines, pypi_missing, pypi_notes = await _check_new_packages(root, files)
    evidence += pypi_lines + pypi_notes
    if pypi_missing:
        fails.append(
            "newly added requirement(s) do not exist on PyPI: "
            + ", ".join(f"{n} (model-invented name?)" for n in pypi_missing)
        )
    sast_state, sast_detail = await asyncio.to_thread(_sast, root, files)
    if sast_state == "pass":
        evidence.append(sast_detail)
    elif sast_state == "fail":
        evidence.append(sast_detail)
        fails.append("the SAST run reported findings on the changed files")
    else:
        not_runs.append(f"SAST: {sast_detail}")
    surface = await asyncio.to_thread(_llm_surface, files)
    if surface:
        evidence.append(
            f"model/MCP surface touched ({', '.join(surface[:3])}): the OWASP LLM Top 10 "
            "checklist applies (prompt injection, insecure output handling, excessive agency)"
        )
        not_runs.append(
            "the OWASP LLM Top 10 checklist has no runner here: review it or "
            "state why it does not apply"
        )
    if fails:
        safe = _row("safe", "fail", "; ".join(fails), evidence)
    elif not_runs:
        safe = _row("safe", "not_run", "; ".join(not_runs), evidence)
    else:
        safe = _row("safe", "pass", "", evidence)
    card.rows.append(safe)

    # ---------- correct ----------
    evidence, fails, not_runs = [], [], []
    failing_checks = [r for r in checks if r.status in ("fail", "timeout")]
    if checks:
        if failing_checks:
            fails.append(
                "the check chain reported findings: "
                + ", ".join(sorted({r.check for r in failing_checks}))
            )
        else:
            evidence.append(
                f"check chain clean on {len(files)} file(s): "
                + ", ".join(r.check for r in checks[:6])
            )
    if not code_touched:
        correct = _row(
            "correct",
            "na",
            "no code file changed this turn: docs and specs have nothing to run",
            evidence,
        )
    else:
        if not saw_test_run:
            # The plan's closing test: "a task with no tests: correct row FAIL". A code change
            # that no test executed cannot be certified, so it fails instead of merely waiting;
            # NOT RUN stays reserved for evidence a runner physically could not collect.
            fails.append("no test ran during this turn: run the repo's tests or /runit")
        elif not saw_test_pass:
            fails.append("the test run this session saw did not pass")
        else:
            line = "a test command ran this session and passed"
            if fix and fix.get("proved"):
                line += f", and the rerun of the failing tests passed ({fix['summary']})"
            evidence.append(line)
        if runit and runit.get("status") == "ran":
            hit = sum(
                1
                for states in runit.get("files", {}).values()
                for s in states
                if s["state"] == "hit"
            )
            miss = sum(
                1
                for states in runit.get("files", {}).values()
                for s in states
                if s["state"] == "miss"
            )
            if miss:
                evidence.append(f"run-it: {hit} changed line(s) executed, {miss} never executed")
                fails.append(f"run-it: {miss} changed line(s) no test executed")
            elif hit:
                evidence.append(f"run-it: all {hit} changed line(s) were executed by a test")
        else:
            not_runs.append("changed-line coverage: NOT RUN yet (press /runit)")
        evidence.append("floor guard: NOT RUN here, it runs at commit time (pre-commit)")
        evidence.append("per-criterion tests: /spec (plan item 23) is not built")
        if fails:
            correct = _row("correct", "fail", "; ".join(fails), evidence)
        elif not_runs:
            correct = _row("correct", "not_run", "; ".join(not_runs), evidence)
        else:
            correct = _row("correct", "pass", "", evidence)
    card.rows.append(correct)

    # ---------- reliable ----------
    evidence, fails, not_runs = [], [], []
    kinds = flags.counts()
    silent = kinds.get("silent_failure", 0) + kinds.get("swallowed_exception", 0)
    if not code_touched:
        reliable = _row("reliable", "na", "no code file changed this turn", [])
    elif flags.files == 0:
        reliable = _row("reliable", "not_run", "no changed Python file to look at", [])
    else:
        if silent:
            fails.append(
                f"{silent} except-block(s) swallow or fail silently, against the "
                "no-silent-failures rule"
            )
            evidence += [
                f.label()
                for f in flags.flags
                if f.kind in ("silent_failure", "swallowed_exception")
            ][:8]
        else:
            evidence.append(
                f"bug flags: no swallowed exception or silent failure in {flags.files} file(s)"
            )
        for kind, count in kinds.items():
            if kind not in {"silent_failure", "swallowed_exception"}:
                evidence.append(f"bug flag {kind}: {count} (advice, not a reliability fail)")
        if fails:
            reliable = _row("reliable", "fail", "; ".join(fails), evidence)
        else:
            reliable = _row("reliable", "pass", "", evidence)
    card.rows.append(reliable)

    # ---------- performant / scalable: no runner exists; the model may explain, not assert ----
    card.rows.append(
        _row(
            "performant",
            "not_run",
            "no benchmark ran; if no hot path changed, fill this row N/A with "
            "the reason (AssuranceReport)",
        )
    )
    card.rows.append(_row("scalable", "na", "nothing in a spec declares a scaling target"))

    # ---------- fit for operation: model-written by design (the plan says so), always labelled
    card.rows.append(
        _row(
            "fit for operation",
            "",
            evidence=[
                "needs: the real app was run, and deploy, rollback and observability paths named "
                "(AssuranceReport tool)"
            ],
        )
    )
    return card


def card_from_dict(data: dict[str, Any]) -> Card:
    card = Card(data["root"], files=data.get("files", []), built_at=data.get("built_at", 0.0))
    card.rows = [Row(**row) for row in data.get("rows", [])]
    return card
