"""Failure diagnosis (plan item 31): where a failing run points, and who says so.

The deterministic half is tested with no model at all, because that half is what ships when no
critic is configured. Every critic reply is checked against the evidence: a file the model
invented must never reach the note, and the turn still ends when the model breaks.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from conftest import FixedApprover
from mwm_harness import diagnose
from mwm_harness import events as ev
from mwm_harness.config import ModelSpec
from mwm_harness.events import DenyAll
from mwm_harness.providers import chunks_for

CRITIC = ModelSpec(id="critic", base_url="http://unused.invalid/v1", key_env="UNUSED_KEY")
PLAN_CRITIC = ModelSpec(
    id="critic",
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    key_env="UNUSED_KEY",
)

BROKEN = "def first(:\n    pass\n"  # py_compile fails on this, so the check chain reports it
CLEAN = "def first() -> int:\n    return 1\n"
CRASHER = "raise RuntimeError('boom')\n"

# The plan's closing shape: a traceback whose innermost repo frame is utils.py:42. The frame
# above it is the standard library, outside this repo, so it must not appear as a suspect.
UTILS_TB = (
    "Traceback (most recent call last):\n"
    '  File "/usr/lib/python3.12/functools.py", line 975, in __get__\n'
    "    return self.fget.__get__(instance, objtype)\n"
    '  File "{ROOT}/utils.py", line 42, in first\n'
    "    return items[0]\n"
    "IndexError: list index out of range\n"
)


def utils_tb(root: Path) -> str:
    return UTILS_TB.replace("{ROOT}", str(root))


def tool(name: str, **arguments: Any) -> list[dict[str, Any]]:
    return chunks_for(tool_calls=[(name, arguments)])


def reply(data: dict[str, Any]) -> list[dict[str, Any]]:
    return chunks_for(json.dumps(data))


def critic_reply(files: list[str], steps: int = 2, cause: str = "a real reason") -> Any:
    return reply(
        {
            "root_cause": cause,
            "suspected_files": files,
            "guidance": [f"step {n}" for n in range(1, steps + 1)],
        }
    )


def broken_run(steps: int = 2, files: list[str] | None = None) -> Any:
    """A turn that writes a file the check chain cannot compile, its answer, then the critic."""
    files = files or ["broken.py"]
    return [
        tool("Write", file_path=files[0], content=BROKEN),
        chunks_for("Written."),
        critic_reply(files, steps=steps),
    ]


def diagnosed_session(make_session, turns: list[Any], critic: ModelSpec = CRITIC):
    """A session with a critic configured and no Review Desk competing for the scripted turns."""
    session, recorder, provider = make_session(turns, approver=FixedApprover(True))
    session.models[critic.id] = critic
    session.settings.review_after_turn = False
    return session, recorder, provider


def diagnose_all(root: Path, text: str, caller: Any = None) -> diagnose.Diagnosis:
    return asyncio.run(
        diagnose.run_diagnosis(root, "test run", text, "fix the crash", "", [], caller, "no critic")
    )


# ------------------------------------------------------- deterministic extraction


def test_a_failure_ending_in_utils_line_42_names_only_utils(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("def first(items):\n    return items[0]\n")
    diagnosis = diagnose_all(root, utils_tb(root))
    assert diagnosis.suspected_files == ["utils.py"]
    assert [(f.file, f.line) for f in diagnosis.frames] == [("utils.py", 42)]
    assert diagnosis.status == "diagnosed" and diagnosis.critic is False
    assert "IndexError" in diagnosis.failure


def test_a_check_finding_gives_the_file_the_check_marked(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")
    rows = [
        {
            "check": "ruff",
            "status": "fail",
            "findings": [{"file": "utils.py", "line": 2, "message": "F821 undefined name"}],
        },
        {"check": "mypy", "status": "not_run", "findings": [{"file": "x.py", "line": 1}]},
    ]
    text = diagnose.check_failure_text(rows)
    trace = diagnose.extract_failure(text, root)
    assert trace.suspected() == ["utils.py"]
    assert [(f.line, f.kind) for f in trace.frames] == [(2, "location")]
    assert "mypy" not in text  # a check that never ran is reported, not diagnosed


def test_pytest_rows_are_read_and_noise_is_not(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "utils.py").write_text("def first(items):\n    return items[0]\n")
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert 1 == 2\n")
    text = (
        "=============== FAILURES ===============\n"
        "E       IndexError: list index out of range\n"
        "\n"
        "utils.py:42: IndexError\n"
        "=========================== short test summary info ============\n"
        "FAILED tests/test_a.py::test_a - IndexError: list index out of range\n"
        "1 failed in 0.31s\n"
    )
    trace = diagnose.extract_failure(text, root)
    assert trace.suspected() == ["utils.py", "tests/test_a.py"]
    assert "IndexError" in trace.error
    assert diagnose.is_test_failure(text)
    assert not diagnose.is_test_failure("all good, nothing to see")
    assert diagnose.is_test_command("pytest -q tests") and not diagnose.is_test_command("ls -la")
    assert diagnose.extract_failure("ERROR: 404 not found\n", root).suspected() == []
    assert not diagnose.extract_failure('File "<string>", line 1, in <module>\n', root).suspected()


def test_the_frame_list_is_newest_first_and_deduped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")
    text = utils_tb(root) + utils_tb(root)
    trace = diagnose.extract_failure(text, root)
    assert [f.label() for f in trace.frames] == ["utils.py:42 in first (traceback)"]


# ------------------------------------------------------------------ the critic


def test_the_critic_fills_the_reason_and_the_order_to_check(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("def first(items):\n    return items[0]\n")
    seen: list[str] = []

    async def caller(system: str, user: str) -> str:
        seen.append(user)
        return json.dumps(
            {
                "root_cause": "first() indexes before anything checks that the list holds an item.",
                "suspected_files": ["utils.py"],
                "guidance": ["open utils.py", "guard the empty list", "add the test"],
            }
        )

    diagnosis = diagnose_all(
        root,
        utils_tb(root),
        caller,
    )
    assert diagnosis.status == "diagnosed" and diagnosis.critic
    assert diagnosis.root_cause.startswith("first() indexes")
    assert diagnosis.guidance == ["open utils.py", "guard the empty list", "add the test"]
    assert diagnosis.suspected_files == ["utils.py"]
    assert "utils.py:42" in seen[0] and "IndexError" in seen[0]
    assert "guard the empty list" in diagnosis.note()


def test_a_file_the_critic_invented_never_reaches_the_note(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")

    async def caller(system: str, user: str) -> str:
        return json.dumps(
            {
                "root_cause": "somewhere in settings.py",
                "suspected_files": ["settings.py", "utils.py"],
                "guidance": ["check settings.py"],
            }
        )

    diagnosis = diagnose_all(root, utils_tb(root), caller)
    assert diagnosis.suspected_files == ["utils.py"]
    assert diagnosis.extra_files == 1
    # The critic's own prose may still name the file it invented; the list it steers by cannot.
    listed = [row for row in diagnosis.note().splitlines() if row.startswith("Suspected files:")]
    assert listed == ["Suspected files: utils.py"]


def test_four_guidance_lines_become_three(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")

    async def caller(system: str, user: str) -> str:
        return json.dumps(
            {
                "root_cause": "c",
                "suspected_files": ["utils.py"],
                "guidance": ["one", "two", "three", "four"],
            }
        )

    assert diagnose_all(root, utils_tb(root), caller).guidance == ["one", "two", "three"]


def test_an_answer_without_json_keeps_the_location_and_says_the_why_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")

    async def caller(system: str, user: str) -> str:
        return "it looks broken, sorry"

    diagnosis = diagnose_all(root, utils_tb(root), caller)
    assert diagnosis.status == "diagnosed"  # the where survived
    assert diagnosis.suspected_files == ["utils.py"]
    assert diagnosis.critic is False
    assert "no JSON object" in diagnosis.reason
    assert "No critic model answered" in diagnosis.note()


def test_a_critic_that_raises_is_reported_and_never_raised(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")

    async def caller(system: str, user: str) -> str:
        raise RuntimeError("endpoint said no")

    diagnosis = diagnose_all(root, utils_tb(root), caller)
    assert "endpoint said no" in diagnosis.reason
    assert diagnosis.suspected_files == ["utils.py"]


def test_no_critic_configured_says_so_and_still_names_the_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")
    diagnosis = diagnose_all(root, utils_tb(root))
    assert diagnosis.critic is False
    assert "no critic model is configured" in diagnosis.reason
    note = diagnosis.note()
    assert "utils.py" in note and "No critic model answered" in note
    assert diagnosis.sent is False


def test_a_note_is_written_once_and_then_dropped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text("x = 1\n")
    trace = diagnose.extract_failure(utils_tb(root), root)
    diagnosis = diagnose.deterministic(root, "check", trace, "no critic")
    assert diagnosis.note()
    diagnosis.sent = True
    assert diagnosis.note() == ""


def test_the_budget_allows_three_and_zero_is_no_cap() -> None:
    budget = diagnose.DiagnoseBudget(limit=3)
    assert [budget.take() for _ in range(4)] == [True, True, True, False]
    open_ended = diagnose.DiagnoseBudget(limit=0)
    assert all(open_ended.take() for _ in range(9))


# --------------------------------------------------------------- the live loop


def test_a_failing_check_after_an_editing_turn_diagnoses_it(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(make_session, broken_run())
    ended = asyncio.run(session.send("write a function"))
    assert ended.reason == "done"
    finished = recorder.of(ev.DiagnoseFinished)
    assert len(finished) == 1
    payload = finished[0].diagnosis
    assert payload["source"] == "check" and payload["status"] == "diagnosed"
    assert payload["suspected_files"] == ["broken.py"]
    assert payload["root_cause"] == "a real reason"
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds.index("ChecksFinished") < kinds.index("DiagnoseFinished")
    assert kinds.index("DiagnoseFinished") < kinds.index("TurnEnded")


def test_the_note_reaches_the_next_turn_once_and_never_again(make_session, tmp_path: Path) -> None:
    session, recorder, provider = diagnosed_session(
        make_session,
        [
            *broken_run(),
            chunks_for("Read it."),
            chunks_for("Done."),
        ],
    )

    async def script() -> None:
        await session.send("write a function")
        await session.send("now read it")
        await session.send("and stop")

    asyncio.run(script())
    notes = [m for m in session.messages if m.is_meta and "FAILURE DIAGNOSIS" in m.text()]
    assert len(notes) == 1
    assert "broken.py" in notes[0].text()
    lead = [r for r in provider.requests if r["model"] != "critic"]
    with_note = [r for r in lead if any("FAILURE DIAGNOSIS" in str(m) for m in r["messages"])]
    assert len(with_note) == len(lead) - 2  # absent from the two requests before the failure
    assert all(sum("FAILURE DIAGNOSIS" in str(m) for m in r["messages"]) == 1 for r in with_note)
    assert len(recorder.of(ev.DiagnoseFinished)) == 1


def test_the_edit_that_ignores_the_diagnosis_is_noted_once(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            *broken_run(steps=2, files=["utils.py"]),
            tool("Write", file_path="other.py", content=CLEAN),
            chunks_for("Both written."),
        ],
    )

    async def script() -> None:
        await session.send("write a function")
        await session.send("now write another file")

    asyncio.run(script())
    radius = [n for n in recorder.of(ev.Notice) if "did not suspect" in n.text]
    assert len(radius) == 1
    assert "other.py" in radius[0].text and "utils.py" in radius[0].text
    assert (session.tool_ctx.resolve("other.py")).is_file()  # advice, never a block


def test_an_edit_inside_the_named_radius_is_never_noted(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            *broken_run(files=["utils.py"]),
            tool("Write", file_path="utils.py", content=CLEAN),
            chunks_for("Fixed."),
        ],
    )

    async def script() -> None:
        await session.send("write it")
        await session.send("now fix it")

    asyncio.run(script())
    assert [n for n in recorder.of(ev.Notice) if "did not suspect" in n.text] == []


def test_a_clean_turn_ends_the_failing_streak(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            *broken_run(),
            tool("Write", file_path="clean.py", content=CLEAN),
            chunks_for("Clean now."),
        ],
    )

    async def script() -> None:
        await session.send("break it")
        await session.send("fix it")

    asyncio.run(script())
    assert len(recorder.of(ev.DiagnoseFinished)) == 1
    assert session.diagnose_budget.used == 0


def test_after_three_diagnoses_it_stops_and_asks_you(make_session, tmp_path: Path) -> None:
    turns: list[Any] = []
    for _ in range(3):
        turns += broken_run()
    turns += [tool("Write", file_path="broken.py", content=BROKEN), chunks_for("Fourth try.")]
    session, recorder, provider = diagnosed_session(make_session, turns)
    session.settings.diagnose_rounds = 3

    async def script() -> None:
        for _ in range(4):
            await session.send("break it again")

    asyncio.run(script())
    assert len(recorder.of(ev.DiagnoseFinished)) == 3
    asked = [n for n in recorder.of(ev.Notice) if "asks you" in n.text]
    assert len(asked) == 1 and asked[0].level == "warn"
    assert sum(1 for r in provider.requests if r["model"] == "critic") == 3


def test_a_failing_test_run_is_diagnosed_from_its_traceback(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            tool("Write", file_path="crash.py", content=CRASHER),
            tool("Bash", command=f"{sys.executable} crash.py"),
            chunks_for("Ran it."),
        ],
    )
    asyncio.run(session.send("add a module that crashes on import"))
    finished = recorder.of(ev.DiagnoseFinished)
    assert len(finished) == 1
    assert finished[0].diagnosis["source"] == "test run"
    assert finished[0].diagnosis["suspected_files"] == ["crash.py"]


def test_a_failing_test_the_model_already_fixed_is_not_diagnosed(
    make_session, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "test_a.py").write_text("def test_a():\n    assert 1 == 1\n")
    pytest_cmd = f"{sys.executable} -m pytest -q test_a.py"
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            tool("Write", file_path="test_a.py", content="def test_a():\n    assert 1 == 2\n"),
            tool("Bash", command=pytest_cmd),
            tool("Edit", file_path="test_a.py", old_string="1 == 2", new_string="1 == 1"),
            tool("Bash", command=pytest_cmd),
            chunks_for("Fixed: the test passes now."),
        ],
    )
    asyncio.run(session.send("add a failing test, then make it pass"))
    assert recorder.of(ev.DiagnoseFinished) == []
    assert (root / "test_a.py").read_text().endswith("assert 1 == 1\n")


def test_run_it_on_a_failing_suite_diagnoses_the_run(make_session, tmp_path: Path) -> None:
    """Ctrl+S in the panel: a run that fails is diagnosed even when its turn wrote no finding."""
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "test_a.py").write_text("def test_a():\n    assert 1 == 2\n")
    session, recorder, _ = diagnosed_session(
        make_session,
        [
            tool("Write", file_path="a.py", content=CLEAN),
            chunks_for("Written."),
            critic_reply(["test_a.py"], steps=1),
        ],
    )

    async def script() -> None:
        await session.send("add a helper")
        await session.run_it()

    asyncio.run(script())
    run_it = recorder.of(ev.RunItFinished)
    assert run_it and run_it[0].result["status"] == "ran"
    assert run_it[0].result["tests_passed"] is False
    started = recorder.of(ev.DiagnoseStarted)
    assert [s.source for s in started] == ["test run"]
    finished = recorder.of(ev.DiagnoseFinished)[-1].diagnosis
    assert finished["root_cause"] == "a real reason"
    assert "test_a.py" in finished["suspected_files"]


def test_a_headless_run_on_a_subscription_endpoint_never_calls_the_critic(
    make_session, tmp_path: Path
) -> None:
    turns = [tool("Write", file_path="broken.py", content=BROKEN), chunks_for("Written.")]
    session, recorder, provider = make_session(turns, approver=DenyAll(), mode="bypassPermissions")
    session.models[PLAN_CRITIC.id] = PLAN_CRITIC
    session.settings.review_after_turn = False
    asyncio.run(session.send("write a function"))
    guard = [n for n in recorder.of(ev.Notice) if "subscription plan" in n.text]
    assert len(guard) == 1
    assert not any(r["model"] == "critic" for r in provider.requests)
    diagnosis = recorder.of(ev.DiagnoseFinished)[0].diagnosis
    assert diagnosis["critic"] is False and diagnosis["suspected_files"] == ["broken.py"]


def test_no_diagnosis_happens_when_the_setting_is_off(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(make_session, broken_run())
    session.settings.diagnose_after_fail = False
    asyncio.run(session.send("write a function"))
    assert recorder.of(ev.DiagnoseFinished) == []


def test_a_session_without_a_critic_still_reports_where_it_points(
    make_session, tmp_path: Path
) -> None:
    session, recorder, provider = make_session(
        [tool("Write", file_path="broken.py", content=BROKEN), chunks_for("Written.")],
        approver=FixedApprover(True),
    )
    session.settings.review_after_turn = False
    asyncio.run(session.send("write a function"))
    assert not any(r["model"] != "test-model" for r in provider.requests)
    diagnosis = recorder.of(ev.DiagnoseFinished)[0].diagnosis
    assert diagnosis["status"] == "diagnosed"
    assert diagnosis["critic"] is False
    assert "no critic model is configured" in diagnosis["reason"]
    assert "broken.py" in str(diagnosis["frames"])


def test_diagnosis_never_writes_a_file(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(make_session, broken_run())
    asyncio.run(session.send("write a function"))
    assert recorder.of(ev.DiagnoseFinished)
    written = sorted(p.name for p in (tmp_path / "project").iterdir() if not p.name.startswith("."))
    assert written == ["broken.py"]
    assert (tmp_path / "project" / "broken.py").read_text() == BROKEN


def test_the_diagnosis_command_reads_pasted_output_without_touching_a_file(
    make_session, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "utils.py").write_text("def first(items):\n    return items[0]\n")
    session, recorder, provider = diagnosed_session(
        make_session, [chunks_for("Looking."), critic_reply(["utils.py"])]
    )

    async def script() -> None:
        await session.send("read this failure")
        await session.diagnose_command(utils_tb(root))

    asyncio.run(script())
    finished = recorder.of(ev.DiagnoseFinished)
    assert len(finished) == 1
    assert finished[0].diagnosis["source"] == "manual"
    assert finished[0].diagnosis["root_cause"] == "a real reason"
    assert not any(m.is_meta and "FAILURE DIAGNOSIS" in m.text() for m in session.messages)


def test_the_diagnosis_command_without_a_failure_says_so(make_session, tmp_path: Path) -> None:
    session, recorder, _ = diagnosed_session(make_session, [chunks_for("Hi.")])
    asyncio.run(session.send("hi"))
    diagnosis = asyncio.run(session.diagnose_command(""))
    assert diagnosis.status == "skipped"
    assert "no failing run to read" in diagnosis.reason
    assert recorder.of(ev.DiagnoseStarted) == []


def test_the_terminal_command_prints_the_diagnosis_once(make_session, capsys) -> None:
    """/diagnose is a side command: the event paints it, the command must not paint it again."""
    from mwm_harness.repl.terminal import Printer, run_async_command

    session, recorder, _ = diagnosed_session(make_session, [chunks_for("Hi.")])
    out = Printer(color=False)
    session.bus.subscribe(out)

    async def script() -> None:
        await session.send("hi")
        assert await run_async_command(session, "/diagnose", out) is True

    asyncio.run(script())
    rows = [line for line in capsys.readouterr().out.splitlines() if "[diagnose]" in line]
    assert len(rows) == 1  # the event painted it once, the command did not paint it again
    assert "SKIPPED: no failing run to read" in rows[0]
    assert recorder.of(ev.DiagnoseStarted) == []


def test_the_terminal_prints_a_diagnosis_and_a_failure_to_read(capsys) -> None:
    from mwm_harness.repl import terminal

    frames = [
        diagnose.Frame("utils.py", 42, "first", "traceback"),
        diagnose.Frame("tests/test_a.py", 0, "", "pytest"),
    ]
    lines = terminal.diagnosis_lines(
        {
            "status": "diagnosed",
            "source": "check",
            "model": "critic",
            "root_cause": "indexes before checking",
            "suspected_files": ["utils.py"],
            "guidance": ["open utils.py"],
            "frames": [json.loads(json.dumps(f.__dict__)) for f in frames],
            "critic": True,
            "extra_files": 1,
        }
    )
    assert lines[0] == "[diagnose] check failed; read by critic"
    assert "  why: indexes before checking" in lines
    assert "  suspected: utils.py" in lines
    assert "  points at: utils.py:42 first" in lines
    assert "  points at: tests/test_a.py" in lines  # a FAILED row has no line to show
    assert "  check: open utils.py" in lines
    assert any("invented" in line for line in lines)
    assert terminal.diagnosis_lines({"status": "skipped", "reason": "nothing here"}) == [
        "[diagnose] SKIPPED: nothing here"
    ]
    no_critic = terminal.diagnosis_lines(
        {
            "status": "diagnosed",
            "source": "test run",
            "model": "no critic",
            "root_cause": "",
            "suspected_files": ["utils.py"],
            "guidance": [],
            "frames": [],
            "critic": False,
            "extra_files": 0,
        }
    )
    assert any("no critic answered" in line for line in no_critic)
