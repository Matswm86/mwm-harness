"""Run-it view: untested branch lines are misses, covered lines name their test, trace shows calls."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from mwm_harness import runit

PRICE = """\
def price(qty, vip=False, api_key=None):
    total = qty * 10
    if vip:
        total = total * 0.9
        return round(total, 2)
    return total
"""

TEST = """\
from price import price


def test_plain_price():
    assert price(3, api_key="hunter2") == 30
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The repo python must have pytest + coverage: point .venv/bin/python at the harness's own.
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (tmp_path / "price.py").write_text("def price(qty):\n    return qty * 10\n")
    (tmp_path / "test_price.py").write_text(TEST.replace(', api_key="hunter2"', ""))
    (tmp_path / ".gitignore").write_text(".venv/\n")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    (tmp_path / "price.py").write_text(PRICE)
    (tmp_path / "test_price.py").write_text(TEST)
    return tmp_path


def test_changed_lines_from_git_diff(repo: Path) -> None:
    lines = runit.changed_lines(repo, [repo / "price.py"])
    assert lines == {"price.py": {1, 2, 3, 4, 5, 6}}


def test_untested_branch_is_red_and_covered_lines_name_the_test(repo: Path) -> None:
    result = asyncio.run(runit.changed_coverage(repo, [repo / "price.py"]))
    assert result.status == "ran", result.reason
    assert result.tests_passed and "1 passed" in result.summary
    states = {s.line: s for s in result.files["price.py"]}
    assert {n for n, s in states.items() if s.state == "miss"} == {4, 5}
    assert {n for n, s in states.items() if s.state == "hit"} == {1, 2, 3, 6}
    assert states[2].tests == ["test_price.py::test_plain_price"]  # a node id /trace accepts
    assert result.counts() == (4, 2)


def test_trace_shows_test_to_function_with_masked_secret(repo: Path) -> None:
    trace = asyncio.run(runit.trace_test(repo, "test_price.py::test_plain_price"))
    assert trace.status == "ran" and trace.passed, trace.reason + trace.tail
    names = [(c["depth"], c["function"]) for c in trace.calls]
    assert names == [(0, "test_plain_price"), (1, "price")]
    args = trace.calls[1]["args"]
    assert args["qty"] == "3" and args["vip"] == "False"
    assert args["api_key"] == "*** (masked)"


def test_missing_coverage_is_not_run(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    fake = tmp_path / ".venv" / "bin" / "python"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    (tmp_path / "a.py").write_text("x = 1\n")
    result = asyncio.run(runit.changed_coverage(tmp_path, [tmp_path / "a.py"]))
    assert result.status == "not_run" and "pytest missing" in result.reason


def test_node_id_resolves_package_and_class(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    assert runit.node_id(tmp_path, "tests.test_a.TestX.test_y") == "tests/test_a.py::TestX::test_y"
    assert runit.node_id(tmp_path, "nowhere.test_z") == "nowhere.test_z"


def test_trace_records_a_generator_once_and_skips_comprehensions(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (tmp_path / "gen.py").write_text(
        "def evens(n):\n    for i in range(n):\n        yield i * 2\n\n\n"
        "def total(n):\n    return sum(x for x in evens(n))\n"
    )
    (tmp_path / "test_gen.py").write_text(
        "from gen import total\n\n\ndef test_total():\n    assert total(4) == 12\n"
    )
    trace = asyncio.run(runit.trace_test(tmp_path, "test_gen.py::test_total"))
    assert [c["function"] for c in trace.calls] == ["test_total", "total", "evens"]
