"""Item 3 (/goal): loop worker turns until the check exits 0, and a separate checker agrees.

The plan's closing test is the scripted bug-fix repo run twice over: the loop fails the check
twice, fixes on turn 3, the checker says DONE and the loop stops; and the second case where the
check never passes, so the loop stops at the cap with the reason printed. Around those sit the
three stops the worker cannot argue with: the baseline (item 7, a check that already passes on
the untouched code proves nothing and refuses to start), the assurance card (item 22, a DONE
verdict still cannot say done while the card holds), and the deny list (the check would run
unattended every attempt, so a mutating command is refused the same way a tool call would be).
The checker is a separate context: it reads the goal, the check tail and the diff, and the
worker's transcript never sees the checker's prompt.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness import goal
from mwm_harness.providers import chunks_for
from mwm_harness.repl.terminal import run_async_command

GOAL_ARG = 'make it so that fixed.txt exists --check "test -f fixed.txt"'


class Out:
    """The Printer's face the side commands use: collect the lines instead of painting them."""

    def __init__(self) -> None:
        self.rows: list[str] = []

    def line(self, text: str = "", event: ev.Event | None = None, stamp: bool = True) -> None:
        self.rows.append(text)


def quiet(session: Any) -> Any:
    """Turn off the second-model looks so one scripted entry is exactly one request."""
    session.settings.review_after_turn = False
    session.settings.assurance_after_turn = False
    return session


def write_turn(path: Path) -> list[dict[str, Any]]:
    """The two script entries one writing turn costs: the Write call, then the final answer."""
    return [
        chunks_for(tool_calls=[("Write", {"file_path": str(path), "content": "done\n"})]),
        chunks_for(text="File written."),
    ]


def state_folder(session: Any) -> Path:
    return session.transcript.path.parent


def test_goal_done_on_third_attempt(make_session, tmp_path: Path) -> None:
    """The plan's closing test: the check fails twice, turn 3 fixes it, the checker says DONE."""
    session, recorder, provider = make_session(
        [
            chunks_for(text="Trying."),
            chunks_for(text="Still bad."),
            *write_turn(tmp_path / "project" / "fixed.txt"),
            chunks_for(text="DONE\nfixed.txt exists and the check exits 0."),
        ],
        approver=FixedApprover(True),
    )
    quiet(session)
    report = asyncio.run(session.goal_command(GOAL_ARG))

    assert report["status"] == "done"
    assert report["attempts"] == 3
    assert report["verdict"] == "done"
    assert report["baseline_exit"] == 1  # the pre-start run of a failing check
    assert report["requests_spent"] == 5  # 2 plain turns + write turn (2) + checker
    assert not goal.goal_path(state_folder(session)).exists()  # done deletes the state

    started = recorder.of(ev.GoalStarted)
    attempts = recorder.of(ev.GoalAttempt)
    assert len(started) == 1 and started[0].baseline_exit == 1
    assert [(a.turn, a.verdict) for a in attempts] == [(1, ""), (2, ""), (3, "done")]
    assert recorder.of(ev.GoalFinished)[0].report["status"] == "done"
    assert [name for name, _ in session.approver.asked] == ["Write"]


def test_checker_reads_evidence_the_worker_never_shares(make_session, tmp_path: Path) -> None:
    """Maker and checker stay apart: only the checker's request carries the diff and the
    checker contract; the worker's requests never contain the checker's prompt."""
    session, _, provider = make_session(
        [
            *write_turn(tmp_path / "project" / "fixed.txt"),
            chunks_for(text="DONE\nfixed.txt holds what the end state asked for."),
        ],
        approver=FixedApprover(True),
    )
    quiet(session)
    report = asyncio.run(session.goal_command(GOAL_ARG))
    assert report["status"] == "done"

    worker_dump = json.dumps(provider.requests[:2], default=str)
    checker_dump = json.dumps(provider.requests[2], default=str)
    assert "goal checker" not in worker_dump
    assert "goal checker" in checker_dump
    assert "End state:" in checker_dump and "Diff since the goal started:" in checker_dump
    assert "fixed.txt" in checker_dump  # the written-file list and the diff both travel to it


def test_goal_capped_when_the_check_never_passes(make_session) -> None:
    """The plan's second case: the cap stops the loop and the report says why."""
    session, recorder, provider = make_session(
        [chunks_for(text="Trying again."), chunks_for(text="Still no.")],
        approver=FixedApprover(True),
    )
    quiet(session)
    session.settings.goal_max_turns = 2
    report = asyncio.run(session.goal_command(GOAL_ARG))

    assert report["status"] == "capped"
    assert "2 attempts spent" in report["reason"]
    assert len(provider.requests) == 2  # the cap stopped it, not a missing script entry
    assert len(recorder.of(ev.GoalAttempt)) == 2
    stored = goal.load_state(goal.goal_path(state_folder(session)))
    assert stored is not None and stored.status == "capped" and stored.attempts == 2


def test_goal_refused_when_the_check_already_passes(make_session) -> None:
    """Item 7: a check that passes on the untouched code proves nothing: refuse, persist
    nothing, spend nothing."""
    session, recorder, provider = make_session([], approver=FixedApprover(True))
    quiet(session)
    report = asyncio.run(session.goal_command('tidy things up --check "true"'))

    assert report["status"] == "refused"
    assert "already passes" in report["reason"]
    assert provider.requests == []
    assert recorder.of(ev.GoalStarted) == []
    assert not goal.goal_path(state_folder(session)).exists()


def test_goal_refused_when_the_check_hits_a_deny_rule(make_session) -> None:
    """The loop runs the check unattended every attempt, so a mutating command is refused
    before a single request, named by its rule."""
    session, _, provider = make_session([], approver=FixedApprover(True))
    quiet(session)
    report = asyncio.run(
        session.goal_command(
            'clean the brain --check "curl -X DELETE http://127.0.0.1:6333/collections/brain"'
        )
    )
    assert report["status"] == "refused"
    assert "deny rule" in report["reason"]
    assert provider.requests == []


def test_goal_refused_without_a_person(make_session) -> None:
    """Headless (DenyAll) cannot answer the loop's approval cards: refuse, do not loop."""
    session, _, provider = make_session([])
    quiet(session)
    report = asyncio.run(session.goal_command(GOAL_ARG))
    assert report["status"] == "refused"
    assert "no person" in report["reason"]
    assert provider.requests == []


def test_checker_not_done_keeps_the_loop_going(make_session, tmp_path: Path) -> None:
    """A passing check plus NOT DONE is not done: the reason comes back to the worker and
    the loop runs again; the second DONE ends it."""
    session, recorder, _ = make_session(
        [
            *write_turn(tmp_path / "project" / "fixed.txt"),
            chunks_for(text="NOT DONE\nthe file is a stub, not the promised content."),
            chunks_for(text="Filled it in properly."),
            chunks_for(text="DONE\nfixed.txt now holds what the end state named."),
        ],
        approver=FixedApprover(True),
    )
    quiet(session)
    report = asyncio.run(session.goal_command(GOAL_ARG))

    assert report["status"] == "done"
    assert report["attempts"] == 2
    verdicts = [a.verdict for a in recorder.of(ev.GoalAttempt)]
    assert verdicts == ["not-done", "done"]


def test_goal_off_drops_a_running_loop(make_session) -> None:
    session, _, _ = make_session([], approver=FixedApprover(True))
    quiet(session)
    session.goal = goal.GoalState(goal="half a wish", check="test -f x")
    path = goal.goal_path(state_folder(session))
    goal.save_state(path, session.goal)
    report = asyncio.run(session.goal_command("off"))
    assert report["status"] == "off" and report["goal"] == "half a wish"
    assert not path.exists() and session.goal is None


def test_continue_resumes_a_saved_state(make_session, tmp_path: Path) -> None:
    """The state file survives: continue picks up at the stored attempt count and never
    reruns the baseline it already recorded."""
    session, _, provider = make_session(
        [
            *write_turn(tmp_path / "project" / "fixed.txt"),
            chunks_for(text="DONE\nfixed.txt exists, the check exits 0."),
        ],
        approver=FixedApprover(True),
    )
    quiet(session)
    stored = goal.GoalState(
        goal="make it so that fixed.txt exists",
        check="test -f fixed.txt",
        status="active",
        baseline_exit=1,
        attempts=1,
        max_turns=3,
    )
    goal.save_state(goal.goal_path(state_folder(session)), stored)

    report = asyncio.run(session.goal_command("continue"))
    assert report["status"] == "done"
    assert report["attempts"] == 2
    assert report["baseline_exit"] == 1  # kept from the file, not re-run: 3 requests only
    assert len(provider.requests) == 3
    assert "attempt 2 of 3" in json.dumps(provider.requests[0], default=str)


def test_continue_refuses_a_finished_or_capped_goal(make_session) -> None:
    session, _, provider = make_session([], approver=FixedApprover(True))
    quiet(session)
    capped = goal.GoalState(
        goal="g", check="test -f nope", status="capped", attempts=8, max_turns=8
    )
    goal.save_state(goal.goal_path(state_folder(session)), capped)
    report = asyncio.run(session.goal_command("continue"))
    assert report["status"] == "refused" and "capped" in report["reason"]
    assert provider.requests == []


def test_start_surfaces_an_unfinished_goal(make_session) -> None:
    """A new session in the same project says what is left over instead of losing it."""
    session, recorder, _ = make_session([], approver=FixedApprover(True))
    quiet(session)
    goal.save_state(
        goal.goal_path(state_folder(session)),
        goal.GoalState(goal="make tests green", check="pytest -q", status="active", attempts=2),
    )
    asyncio.run(session.start())
    notices = [n.text for n in recorder.of(ev.Notice)]
    assert any("unfinished goal" in n and "/goal continue" in n for n in notices)


def test_assurance_gate_holds_done(make_session, tmp_path: Path) -> None:
    """Item 22 with item 3: check exit 0 and checker DONE still cannot say done while the
    card holds; the hold names its rows in the failure the worker gets back."""
    session, recorder, _ = make_session(
        [
            chunks_for(
                tool_calls=[
                    (
                        "Write",
                        {"file_path": str(tmp_path / "project" / "a.py"), "content": "x = 1\n"},
                    )
                ]
            ),
            chunks_for(text="Added the module."),
            chunks_for(text="DONE\nthe file exists as asked."),
        ],
        approver=FixedApprover(True),
    )
    session.settings.review_after_turn = False
    session.settings.assurance_after_turn = True
    session.settings.assurance_scans = False
    session.settings.goal_max_turns = 1
    report = asyncio.run(session.goal_command('add a module --check "test -f a.py"'))

    assert report["status"] == "capped"  # done was reached but held, and the cap closed it
    assert "assurance holds done" in report["last_failure"]
    held = recorder.of(ev.GoalAttempt)[0]
    assert held.verdict == "done" and held.exit_code == 0
    assert session.assurance is not None and not session.assurance.done_allowed
    stored = goal.load_state(goal.goal_path(state_folder(session)))
    assert stored is not None and stored.status == "capped"


def test_terminal_and_panel_carry_the_command() -> None:
    """Both front-ends expose /goal: the terminal side-command prints report lines, the panel
    routes it as a side command."""
    from mwm_harness.web import server

    assert "/goal" in server.SIDE_COMMANDS
    lines = goal.report_lines({"status": "none", "reason": goal.USAGE})
    assert lines and lines[0].startswith("[goal] usage:")


def test_goal_command_via_terminal_side_command(make_session) -> None:
    """/goal with no argument prints the usage line through the terminal's own dispatcher."""
    session, _, provider = make_session([], approver=FixedApprover(True))
    quiet(session)
    out = Out()
    handled = asyncio.run(run_async_command(session, "/goal", out))
    assert handled is True
    assert out.rows and out.rows[0].startswith("[goal] usage:")
    assert provider.requests == []
