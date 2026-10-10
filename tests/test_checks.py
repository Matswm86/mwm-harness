"""Check runner: a syntax error and a type error each give a gutter mark; a missing tool is NOT RUN."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from conftest import FixedApprover
from mwm_harness import checks
from mwm_harness import events as ev
from mwm_harness.providers import chunks_for

needs_mypy = pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy not installed")


@pytest.fixture(autouse=True)
def _no_user_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MWM_HARNESS_CONFIG", str(tmp_path / "config"))


def run(root: Path, *names: str) -> dict[str, checks.CheckResult]:
    results = asyncio.run(checks.run_checks(root, [root / n for n in names]))
    return {r.check: r for r in results}


@needs_mypy
def test_syntax_error_and_type_error_each_mark_their_line(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")
    (tmp_path / "typed.py").write_text('def f() -> int:\n    return "x"\n')
    results = run(tmp_path, "broken.py", "typed.py")

    syntax = results["py_compile"]
    assert syntax.status == "fail"
    assert [(f.file, f.line) for f in syntax.findings] == [("broken.py", 1)]

    mypy = results["mypy"]
    assert mypy.status == "fail"
    typed = [f for f in mypy.findings if f.file == "typed.py"]
    assert typed and typed[0].line == 2 and "Incompatible return value" in typed[0].message


@needs_mypy
def test_clean_edit_passes_every_check(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("def f() -> int:\n    return 1\n")
    results = run(tmp_path, "ok.py")
    assert {name: r.status for name, r in results.items()} == {
        "py_compile": "pass",
        "ruff": "pass",
        "mypy": "pass",
    }


def test_missing_tool_is_not_run_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "ok.py").write_text("x = 1\n")
    results = run(tmp_path, "ok.py")
    assert results["mypy"].status == "not_run"
    assert results["mypy"].line() == "mypy: NOT RUN: mypy missing"
    assert results["py_compile"].status == "pass"  # runs inside the harness, needs no tool


def test_bash_syntax_error_and_unwired_pine(tmp_path: Path) -> None:
    (tmp_path / "s.sh").write_text("if then\n")
    (tmp_path / "x.pine").write_text("//@version=6\n")
    results = run(tmp_path, "s.sh", "x.pine")
    assert results["bash -n"].status == "fail"
    assert results["bash -n"].findings[0].line == 1
    assert results["pine compile"].status == "not_run"
    assert "TradingView" in results["pine compile"].reason


def test_node_syntax_error_takes_message_from_error_line(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    (tmp_path / "a.js").write_text("let x = 1;\nlet = ;\n")
    finding = run(tmp_path, "a.js")["node --check"].findings[0]
    assert finding.line == 2 and "SyntaxError" in finding.message


def test_repo_file_replaces_the_chain_and_timeout_is_reported(tmp_path: Path) -> None:
    (tmp_path / ".mwm").mkdir()
    (tmp_path / ".mwm" / "checks.toml").write_text(
        'timeout = 0.5\n[[check]]\nname = "slow"\nglob = ["*.py"]\nrun = ["sleep", "5"]\n'
    )
    (tmp_path / "ok.py").write_text("x = 1\n")
    results = run(tmp_path, "ok.py")
    assert list(results) == ["slow"]
    assert results["slow"].status == "timeout"


def test_deleted_and_unmatched_files_run_nothing(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hi\n")
    assert run(tmp_path, "gone.py", "notes.md") == {}


def test_editing_turn_runs_checks_before_it_ends(make_session) -> None:
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "bad.py", "content": "def f(:\n"})]),
        chunks_for("Written."),
    ]
    session, recorder, _ = make_session(turns, approver=FixedApprover(True))
    asyncio.run(session.send("write it"))
    finished = recorder.of(ev.ChecksFinished)
    assert len(finished) == 1
    syntax = next(r for r in finished[0].results if r["check"] == "py_compile")
    assert syntax["status"] == "fail" and syntax["findings"][0]["file"] == "bad.py"
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds.index("ChecksFinished") < kinds.index("TurnEnded")


def test_turn_without_edits_runs_no_checks(make_session) -> None:
    session, recorder, _ = make_session([chunks_for("Hi.")])
    asyncio.run(session.send("hi"))
    assert recorder.of(ev.ChecksStarted) == []
