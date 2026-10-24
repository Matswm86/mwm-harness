"""Item 22: the assurance card rows, its gate, and the rows the model may and may not fill."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mwm_harness import assurance
from mwm_harness.assurance import MODEL_ROWS, build_card, card_from_dict
from mwm_harness.bugflag import Flag, FlagReport
from mwm_harness.checks import CheckResult
from mwm_harness.tools.assurance import AssuranceReport


def card_for(tmp_path: Path, files: list[Path], **over) -> assurance.Card:
    kwargs: dict = dict(
        checks=[],
        runit=None,
        flags=FlagReport(),
        fix=None,
        saw_test_run=False,
        saw_test_pass=False,
        scans=False,
    )
    kwargs.update(over)
    return asyncio.run(build_card(tmp_path, files, **kwargs))


def silent_flags() -> FlagReport:
    return FlagReport(
        flags=[Flag("loader.py", 6, "silent_failure", "except returns an empty value with no log")],
        files=1,
    )


def test_hallucinated_package_fails_safe_naming_it(tmp_path: Path, monkeypatch) -> None:
    """Plan 22 closing test: a task that adds requests-ultra -> safe row FAIL, package named."""
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (tmp_path / "requirements.txt").write_text("")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    (tmp_path / "requirements.txt").write_text("requests-ultra==1.0\n")
    monkeypatch.setattr(
        assurance,
        "_pypi_exists",
        lambda name, timeout=5.0: (
            ("no", "") if name == "requests-ultra" else ("yes", ", first release 3000 days ago")
        ),
    )
    card = card_for(tmp_path, [tmp_path / "requirements.txt"])
    row = card.row("safe")
    assert row.status == "fail"
    assert "requests-ultra" in row.reason
    assert "requests-ultra" in card.render()


def test_no_test_ran_on_code_change_fails_correct(tmp_path: Path) -> None:
    """Plan 22 closing test: a task with no tests -> correct row FAIL (not a quiet NOT RUN)."""
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    row = card.row("correct")
    assert row.status == "fail" and "no test ran" in row.reason
    assert "correct" in card.holding and not card.done_allowed


def test_docs_only_rows_na_and_done_allowed_once_operation_filled(tmp_path: Path) -> None:
    """Plan 22 closing test: a pure docs change -> N/A rows with reasons, done allowed."""
    (tmp_path / "README.md").write_text("# hello\n")
    card = card_for(tmp_path, [tmp_path / "README.md"])
    assert card.row("correct").status == "na" and card.row("correct").reason
    assert card.row("reliable").status == "na" and card.row("reliable").reason
    assert card.row("scalable").status == "na" and card.row("scalable").reason
    assert card.holding == ["fit for operation"]  # empty until named, per the gate
    card.fill("fit for operation", "na", "docs only: no app to run, nothing to deploy")
    assert card.done_allowed and "done allowed" in card.gate_line()


def test_passing_tests_and_run_it_coverage_pass_correct(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    runit_result = {
        "status": "ran",
        "files": {"app.py": [{"line": 2, "state": "hit", "tests": ["test_app.py::t"]}]},
    }
    card = card_for(
        tmp_path, [tmp_path / "app.py"], saw_test_run=True, saw_test_pass=True, runit=runit_result
    )
    assert card.row("correct").status == "pass"
    missy = {"status": "ran", "files": {"app.py": [{"line": 2, "state": "miss", "tests": []}]}}
    card = card_for(
        tmp_path, [tmp_path / "app.py"], saw_test_run=True, saw_test_pass=True, runit=missy
    )
    assert card.row("correct").status == "fail"
    assert "never executed" in " ".join(card.row("correct").evidence + [card.row("correct").reason])


def test_failing_check_chain_fails_correct(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    checks = [CheckResult(check="ruff", status="fail", files=["app.py"])]
    card = card_for(
        tmp_path, [tmp_path / "app.py"], checks=checks, saw_test_run=True, saw_test_pass=True
    )
    assert card.row("correct").status == "fail"
    assert "ruff" in card.row("correct").reason


def test_silent_failure_flag_fails_reliable(tmp_path: Path) -> None:
    (tmp_path / "loader.py").write_text("def f():\n    return None\n")
    card = card_for(
        tmp_path,
        [tmp_path / "loader.py"],
        flags=silent_flags(),
        saw_test_run=True,
        saw_test_pass=True,
    )
    assert card.row("reliable").status == "fail"
    assert (
        "silent_failure" in card.row("reliable").evidence[0]
        or "swallow" in card.row("reliable").reason
    )
    assert "loader.py:6 silent_failure" in card.row("reliable").evidence[0]


def test_clean_flags_pass_reliable_and_query_loop_is_advice(tmp_path: Path) -> None:
    (tmp_path / "dao.py").write_text("def f():\n    return 1\n")
    flags = FlagReport(
        flags=[Flag("dao.py", 3, "query_in_loop", "db.get() inside a loop")], files=1
    )
    card = card_for(tmp_path, [tmp_path / "dao.py"], flags=flags)
    assert card.row("reliable").status == "pass"  # advice kinds do not fail the row
    assert any("query_in_loop: 1" in line for line in card.row("reliable").evidence)


def test_performant_not_run_until_model_explains(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    row = card.row("performant")
    assert row.status == "not_run" and not row.model_written
    assert "performant" not in card.holding  # NOT RUN is honest, it does not hold done
    card.fill("performant", "na", "no hot path changed: a docstring edit only")
    assert card.row("performant").model_written and card.row("performant").status == "na"


def test_model_cannot_fill_safe_correct_reliable(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    for quality in ("safe", "correct", "reliable"):
        assert quality not in MODEL_ROWS
        with pytest.raises(ValueError, match="code evidence"):
            card.fill(quality, "na", "trust me")
    with pytest.raises(ValueError, match="reason"):
        card.fill("fit for operation", "pass", "")


def test_model_row_can_be_corrected_but_reason_is_never_empty(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    card.fill("performant", "na", "first reason")
    card.fill("performant", "na", "better reason after thinking")
    assert card.row("performant").reason == "better reason after thinking"
    assert card.row("performant").model_written
    with pytest.raises(ValueError, match="reason"):
        card.fill("performant", "na", "   ")


def test_gate_line_and_render_shape(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    text = card.render()
    assert text.startswith(assurance.QUESTION)
    assert "fit for operation: EMPTY" in text
    assert "DONE HELD" in card.gate_line()
    card.fill(
        "fit for operation",
        "pass",
        "ran the real app: mwm --check; deploy: wmcommit --push; rollback: git revert; "
        "logs: journalctl -u mwm",
    )
    assert "[model-written]" in card.render()


def test_card_survives_the_bus_as_a_dict(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    card = card_for(tmp_path, [tmp_path / "app.py"])
    card.fill("scalable", "na", "single-user CLI tool, no scaling target declared")
    again = card_from_dict(card.as_dict())
    assert again.done_allowed == card.done_allowed
    assert again.row("scalable").model_written
    assert again.render() == card.render()


def test_llm_surface_names_owasp_scope(tmp_path: Path) -> None:
    (tmp_path / "gw.py").write_text(
        "from mwm_harness import providers\n\n\ndef h():\n    return providers.call_model('x')\n"
    )
    card = card_for(tmp_path, [tmp_path / "gw.py"])
    row = card.row("safe")
    assert any("OWASP LLM Top 10" in line for line in row.evidence)
    assert row.status == "not_run"  # the checklist has no runner: it says so


def test_real_secret_scan_smoke_on_this_box(tmp_path: Path) -> None:
    """With scans on, the built-in or detect-secrets look must answer, never fail silently."""
    (tmp_path / "app.py").write_text("import json\nprint(json.dumps({}))\n")
    card = card_for(tmp_path, [tmp_path / "app.py"], scans=True)
    row = card.row("safe")
    assert row.status in {"pass", "not_run", "fail"}
    assert "secret" in " ".join(row.evidence + [row.reason, "scan ran"]).lower()


def test_assurance_report_tool_delegates_and_refuses() -> None:
    tool = AssuranceReport()

    async def ok_handler(rows):
        return f"filled {len(rows)} row(s)"

    async def bad_handler(rows):
        raise ValueError("the model may not fill 'safe'")

    ctx_ok = SimpleNamespace(assurance_handler=ok_handler)
    ctx_bad = SimpleNamespace(assurance_handler=bad_handler)
    result = asyncio.run(
        tool.run(
            {"rows": [{"quality": "performant", "status": "na", "reason": "no hot path"}]}, ctx_ok
        )
    )
    assert not result.is_error and "filled 1" in result.content
    refused = asyncio.run(tool.run({"rows": [{"quality": "safe"}]}, ctx_bad))
    assert refused.is_error and "may not fill" in refused.content
    empty = asyncio.run(tool.run({"rows": []}, ctx_ok))
    assert empty.is_error
    no_handler = asyncio.run(
        tool.run(
            {"rows": [{"quality": "scalable", "status": "na", "reason": "x"}]}, SimpleNamespace()
        )
    )
    assert no_handler.is_error
