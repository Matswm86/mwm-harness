"""Item 30: AST bug flags, the N+1 look over a traced test, and the prove-the-fix rerun."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mwm_harness import bugflag, runit
from mwm_harness.repl.terminal import bugflag_lines, fix_lines


def kinds(source: str) -> list[str]:
    return [flag.kind for flag in bugflag.flags_for_source(source, "x.py")]


def test_except_body_only_pass_is_swallowed() -> None:
    source = "def f():\n    try:\n        risky()\n    except ValueError:\n        pass\n"
    assert kinds(source) == ["swallowed_exception"]


def test_broad_except_without_log_or_raise_is_swallowed() -> None:
    source = "def f():\n    try:\n        risky()\n    except Exception:\n        return -1\n"
    assert kinds(source) == ["swallowed_exception"]


def test_broad_except_that_logs_and_raises_is_clean() -> None:
    source = (
        "import logging\n\n\n"
        "def f():\n    try:\n        risky()\n    except Exception:\n"
        "        logging.exception('failed')\n        raise\n"
    )
    assert kinds(source) == []


def test_except_returning_none_unlogged_is_silent_failure() -> None:
    source = "def f():\n    try:\n        risky()\n    except ValueError:\n        return None\n"
    assert kinds(source) == ["silent_failure"]


def test_except_returning_empty_list_with_log_is_clean() -> None:
    source = (
        "import logging\n\n\n"
        "def f():\n    try:\n        risky()\n    except ValueError:\n"
        "        logging.warning('bad')\n        return []\n"
    )
    assert kinds(source) == []


def test_http_call_inside_loop_is_flagged() -> None:
    source = (
        "import requests\n\n\n"
        "def f(ids):\n    out = []\n    for i in ids:\n"
        "        out.append(requests.get(f'https://x/{i}'))\n    return out\n"
    )
    assert kinds(source) == ["query_in_loop"]


def test_db_execute_in_while_loop_is_flagged() -> None:
    source = (
        "def f(db, rows):\n    while rows:\n        r = rows.pop()\n"
        "        db.execute('SELECT 1')\n"
    )
    assert kinds(source) == ["query_in_loop"]


def test_dict_get_in_loop_is_not_flagged() -> None:
    source = (
        "def f(rows, lookup):\n    out = []\n    for r in rows:\n"
        "        out.append(lookup.get(r))\n    return out\n"
    )
    assert kinds(source) == []


def test_query_outside_loop_is_not_flagged() -> None:
    source = "def f(db):\n    return db.execute('SELECT 1')\n"
    assert kinds(source) == []


def test_file_that_does_not_parse_is_flagged_not_skipped() -> None:
    flags = bugflag.flags_for_source("def broken(:\n", "b.py")
    assert [flag.kind for flag in flags] == ["parse_error"]


def test_flag_files_counts_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "def f():\n    try:\n        x()\n    except Exception:\n        pass\n"
    )
    report = bugflag.flag_files(tmp_path, [tmp_path / "mod.py", tmp_path / "notes.md"])
    assert report.files == 1 and report.unparsed == 0
    assert report.flags[0].file == "mod.py"
    assert report.counts() == {"swallowed_exception": 1}
    assert "swallowed_exception:1" in report.line()


def trace_calls(function: str, arg_name: str, arg_value: str, times: int) -> dict:
    calls = [
        {"function": function, "file": "dao.py", "line": 12, "args": {arg_name: arg_value}}
        for _ in range(times)
    ]
    return {"status": "ran", "calls": calls}


def test_same_query_three_times_is_n_plus_one() -> None:
    result = trace_calls("get_user", "sql", "SELECT * FROM users WHERE id = ?", 3)
    flags = bugflag.n_plus_one(result)
    assert len(flags) == 1
    assert flags[0].kind == "n_plus_one"
    assert "3 times" in flags[0].message and flags[0].file == "dao.py"


def test_two_repeats_and_other_arguments_are_not_flagged() -> None:
    assert bugflag.n_plus_one(trace_calls("get_user", "sql", "SELECT 1", 2)) == []
    assert bugflag.n_plus_one(trace_calls("open_file", "path", "/tmp/a", 4)) == []
    assert bugflag.n_plus_one({"status": "timeout", "calls": []}) == []


def test_n_plus_one_url_shape() -> None:
    result = trace_calls("fetch", "url", "https://api.x/items", 3)
    assert [flag.kind for flag in bugflag.n_plus_one(result)] == ["n_plus_one"]


def test_failed_node_ids_from_pytest_output() -> None:
    output = (
        "FF..\n"
        "FAILED tests/test_a.py::test_b - assert 1 == 2\n"
        "FAILED tests/test_a.py::test_b - assert 1 == 2\n"
        "ERROR tests/test_c.py::test_d - collection error\n"
        "ERROR tests/test_e.py - file level import error\n"
        "2 failed, 2 passed in 0.10s\n"
    )
    assert bugflag.failed_node_ids(output) == [
        "tests/test_a.py::test_b",
        "tests/test_c.py::test_d",
        "tests/test_e.py",
    ]


def test_bugflag_lines_summary_and_cap() -> None:
    flags = [
        {"file": "m.py", "line": n, "kind": "query_in_loop", "message": "x"} for n in range(21)
    ]
    report = {"flags": flags, "counts": {"query_in_loop": 21}, "files": 2}
    rows = bugflag_lines(report)
    assert rows[0].startswith("[flags] 21 in 2 file(s) (query_in_loop:21)")
    assert rows[-1] == "  ... 1 more" and len(rows) == 22
    assert bugflag_lines({"flags": [], "counts": {}, "files": 0}) == [
        "[flags] none in 0 changed file(s)"
    ]


def test_fix_lines_shapes() -> None:
    proved = fix_lines({"proved": True, "passed": ["t.py::a"], "seconds": 1.2})
    assert proved == ["[fix] PROVED: 1 formerly failing test(s) pass alone (1.2s)"]
    assert fix_lines({"status": "not_run", "reason": "no failing tests on record"}) == [
        "[fix] NOT_RUN: no failing tests on record"
    ]
    bad = fix_lines(
        {
            "status": "ran",
            "proved": False,
            "summary": "1 failed",
            "still_failing": ["t.py::a"],
            "missing": ["t.py::b"],
        }
    )
    assert bad[0].startswith("[fix] NOT PROVED")
    assert any("still failing" in row for row in bad)
    assert any("proves nothing" in row for row in bad)


@pytest.fixture
def fix_repo(tmp_path: Path) -> Path:
    """A throwaway repo whose .venv python is the harness's own (it has pytest)."""
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (tmp_path / "calc.py").write_text("def half(n):\n    return n * 2\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import half\n\n\ndef test_half_four():\n    assert half(4) == 2\n"
    )
    return tmp_path


def test_rerun_nodes_proves_after_fix(fix_repo: Path) -> None:
    node = "test_calc.py::test_half_four"
    result = asyncio.run(runit.rerun_nodes(fix_repo, [node]))
    assert result["status"] == "ran" and not result["proved"]
    assert result["still_failing"] == [node]
    (fix_repo / "calc.py").write_text("def half(n):\n    return n // 2\n")
    result = asyncio.run(runit.rerun_nodes(fix_repo, [node]))
    assert result["proved"] and result["passed"] == [node]


def test_rerun_deleted_test_file_is_missing_never_passed(fix_repo: Path) -> None:
    (fix_repo / "test_calc.py").unlink()
    result = asyncio.run(runit.rerun_nodes(fix_repo, ["test_calc.py::test_half_four"]))
    assert result["status"] == "ran" and not result["proved"]
    assert result["missing"] == ["test_calc.py::test_half_four"]
    assert result["passed"] == [] and result["still_failing"] == []


def test_rerun_without_nodes_is_not_run(fix_repo: Path) -> None:
    result = asyncio.run(runit.rerun_nodes(fix_repo, []))
    assert result["status"] == "not_run" and "no failing tests" in result["reason"]


def test_turn_edits_then_prove_the_fix_reruns_and_passes(make_session) -> None:
    """Plan 30 closing test (c): failing test -> edit -> the harness reruns it alone, PASS."""
    from mwm_harness import events as ev
    from mwm_harness.providers import chunks_for

    session, recorder, _ = make_session(
        [
            chunks_for(
                tool_calls=[
                    (
                        "Bash",
                        {
                            "command": "./.venv/bin/python -m pytest -q -rfEs",
                            "description": "run tests",
                        },
                    )
                ]
            ),
            chunks_for(tool_calls=[("Read", {"file_path": "calc.py"})]),
            chunks_for(
                tool_calls=[
                    (
                        "Edit",
                        {
                            "file_path": "calc.py",
                            "old_string": "    return n * 2",
                            "new_string": "    return n // 2",
                        },
                    )
                ]
            ),
            chunks_for("Fixed: integer division."),
        ],
        mode="bypassPermissions",
    )
    project = session.cwd
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (project / "calc.py").write_text("def half(n):\n    return n * 2\n")
    (project / "test_calc.py").write_text(
        "from calc import half\n\n\ndef test_half_four():\n    assert half(4) == 2\n"
    )
    ended = asyncio.run(session.send("make the tests pass"))
    assert ended.reason == "done"
    proved = recorder.of(ev.FixProved)
    assert len(proved) == 1, "the turn ended without a prove-the-fix rerun"
    assert proved[0].result["proved"], proved[0].result
    assert "test_calc.py::test_half_four" in proved[0].result["passed"]
    assert session.proof_of_fix and session.proof_of_fix["proved"]
    cards = recorder.of(ev.AssuranceFinished)
    assert len(cards) == 1
    rows = {row["quality"]: row["status"] for row in cards[0].card["rows"]}
    assert rows["correct"] == "not_run"  # tests pass, coverage was never asked for
    assert cards[0].card["holding"] == ["fit for operation"]


def test_turn_writing_swallowed_exception_flags_it(make_session) -> None:
    from mwm_harness import events as ev
    from mwm_harness.providers import chunks_for

    session, recorder, _ = make_session(
        [
            chunks_for(
                tool_calls=[
                    (
                        "Write",
                        {
                            "file_path": "loader.py",
                            "content": "import json\n\n\ndef load(text):\n    try:\n"
                            "        return json.loads(text)\n    except Exception:\n"
                            "        return None\n",
                        },
                    )
                ]
            ),
            chunks_for("Done."),
        ],
        mode="bypassPermissions",
    )
    ended = asyncio.run(session.send("add a loader"))
    assert ended.reason == "done"
    flags = recorder.of(ev.BugFlagsFinished)
    assert len(flags) == 1
    kinds = [flag["kind"] for flag in flags[0].report["flags"]]
    assert kinds == ["swallowed_exception", "silent_failure"]
    cards = recorder.of(ev.AssuranceFinished)
    rows = {row["quality"]: row["status"] for row in cards[0].card["rows"]}
    assert rows["reliable"] == "fail"  # a silent failure holds the reliable row
    assert "reliable" in cards[0].card["holding"]
