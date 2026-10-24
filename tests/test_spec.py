"""Item 23 (/spec): three documents in order, each gated on a person, then every criterion linked.

Half of the point is that a shape nobody approved can never become a file, so the validators run
with no model at all: a criterion no test could quote, a task that closes nothing, or a design
with no threat model where the requirements name a password must be refused before a write is
asked for. The walk is tested against a scripted provider and a recorded approver, because "each
step waits for the person" is a real approval card per step, not a sentence saying one happened.
The last group is the link the plan closes on: delete the test that quotes AC-2 and the assurance
card's correct row FAILS and names AC-2, and it does that even when the turn wrote only markdown.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness import spec
from mwm_harness.assurance import build_card
from mwm_harness.bugflag import FlagReport
from mwm_harness.events import DenyAll
from mwm_harness.providers import chunks_for
from mwm_harness.repl.terminal import run_async_command, spec_lines
from test_panel import HOST, TOKEN, client_for, until

IDEA = "show one short line per check file in the panel"
SLUG = spec.slug_from_idea(IDEA)

REQ = """## User stories

- As a `Mats`, I want the check list to read as one line per file so that I see the verdict fast.
- As a `developer`, I want the detail to stay one click away so that I can still debug a failure.

## Acceptance criteria

- AC-1: the check list shows exactly one summary line for each file it checked.
- AC-2: a file with no findings shows a pass line and no expandable detail.
- AC-3: clicking a failing line reveals the output that produced it.
"""

# AC-1 is worded exactly as in REQ: only a criterion whose text moved may be re-flagged.
REQ_MOVED = """## User stories

- As a `Mats`, I want the check list to read as one line per file so that I see the verdict fast.

## Acceptance criteria

- AC-1: the check list shows exactly one summary line for each file it checked.
- AC-2: a failing line shows which tool reported it, beside the count.
"""

DESIGN = """## Stack

The panel's existing browser code in one file, no new dependency.

## Structure

- mwm_harness/web/static/index.html draws the list and the detail row.
- mwm_harness/web/server.py keeps sending the same ChecksFinished event.

## Data

Nothing new is stored: the summary comes from the check results already on the bus.
"""

TASKS = """## Tasks

- [ ] draw one summary line per checked file in the list (closes AC-1)
- [ ] drop the detail row when a file has no findings (closes AC-2)
- [ ] open the stored output when a failing line is clicked (closes AC-3)
"""

BAD_TASKS = """## Tasks

- [ ] draw one summary line per checked file in the list (closes AC-1)
- [ ] polish the toolbar spacing
"""

RUNIT_CLEAN = {"status": "ran", "files": {"loader.py": [{"state": "hit"}]}}
DRAFTS = [chunks_for(REQ), chunks_for(DESIGN), chunks_for(TASKS)]


class Out:
    """Stands in for the Printer: collects only what a command prints by itself."""

    def __init__(self) -> None:
        self.rows: list[str] = []

    def line(self, text: str = "", stamp: bool = True) -> None:
        self.rows.append(text)

    def __call__(self, event: ev.Event) -> None:
        pass


def quoted_test(root: Path, ids: list[str], name: str = "test_checks.py") -> Path:
    """A test file that quotes criteria inside test functions, the way a spec's test should."""
    folder = root / "tests"
    folder.mkdir(parents=True, exist_ok=True)
    body = '"""Covers the check list spec."""\n'
    for index, cid in enumerate(ids, start=1):
        body += f'\ndef test_case_{index}() -> None:\n    """{cid} holds."\n    assert True\n'
    path = folder / name
    path.write_text(body, encoding="utf-8")
    return path


def walk(make_session: Any, turns: list[Any], argument: str = IDEA, answer: bool = True):
    approver = FixedApprover(answer)
    session, recorder, provider = make_session(turns, approver=approver)
    report = asyncio.run(session.spec_command(argument))
    return session, recorder, provider, approver, report


# --------------------------------------------------------------------- shape checks, no model


def test_criteria_and_tasks_are_parsed_by_position_not_by_hope() -> None:
    assert [c.id for c in spec.parse_criteria(REQ)] == ["AC-1", "AC-2", "AC-3"]
    tasks = spec.parse_tasks(TASKS)
    assert [t.id for t in tasks] == ["T1", "T2", "T3"]
    assert tasks[0].closes == ["AC-1"]
    assert tasks[0].label().startswith("T1: draw one summary line")
    assert tasks[0].label().endswith("(closes AC-1)")
    assert spec.parse_criteria("no markers here") == []


def test_requirements_are_refused_without_both_sections_or_without_a_criterion() -> None:
    assert spec.validate_requirements(REQ) == []
    missing = spec.validate_requirements("## Acceptance criteria\n\n- AC-1: one line.\n")
    assert any("User stories" in line for line in missing)
    no_crit = spec.validate_requirements(
        "## User stories\n\n- As a user, I want X.\n\n## Acceptance criteria\n\n- it works\n"
    )
    assert any("no acceptance criterion" in line for line in no_crit)
    dup = spec.validate_requirements(REQ + "\n- AC-1: a repeat.\n")
    assert any("duplicate criterion ids" in line for line in dup)


def test_a_task_that_names_no_criterion_is_refused() -> None:
    """The plan's own wording: a task without a criterion is refused, not written."""
    errors = spec.validate_tasks(BAD_TASKS, spec.parse_criteria(REQ))
    assert any("T2 names no criterion" in line for line in errors)
    assert spec.validate_tasks(TASKS, spec.parse_criteria(REQ)) == []
    stale = spec.validate_tasks("- [ ] do a thing (closes AC-9)\n", spec.parse_criteria(REQ))
    assert any("AC-9" in line and "does not contain" in line for line in stale)
    nohead = spec.validate_tasks("- [ ] do a thing (closes AC-1)\n", spec.parse_criteria(REQ))
    assert any("matching 'task'" in line for line in nohead)


def test_design_needs_a_threat_model_only_when_the_requirements_name_a_surface() -> None:
    assert spec.threat_surface(REQ) == ""
    assert spec.validate_design(DESIGN, "") == []
    assert any("Stack" in line for line in spec.validate_design("## Data\nx\n", ""))
    auth = "- AC-1: a user signs in with a password and gets a session cookie.\n"
    surface = spec.threat_surface(auth)
    assert "password" in surface and "cookie" in surface
    assert any("threat model" in line for line in spec.validate_design(DESIGN, surface))
    empty = DESIGN + "\n## Threat model\n"
    assert any("is empty" in line for line in spec.validate_design(empty, surface))
    filled = empty + "\nAssets: the session. Way in: the sign-in form. Refused: no plain log.\n"
    assert spec.validate_design(filled, surface) == []


def test_a_folder_name_that_is_a_path_is_refused_before_anything_resolves_it() -> None:
    assert spec.is_slug("two-reading-levels") and spec.is_slug("a1")
    for bad in ("..", "../.ssh", "specs/two", "Two-Levels", "x" * 49, "-" + "x" * 49, ""):
        assert not spec.is_slug(bad), bad
    assert spec.slug_from_idea("Two reading levels!!") == "two-reading-levels"
    assert len(spec.slug_from_idea("show " + "x " * 40)) == 48, "the cap is 48 characters"
    assert spec.slug_from_idea("!!!") == "spec"


def test_coverage_joins_on_the_criterion_id_and_a_deleted_test_shows_as_a_gap(
    tmp_path: Path,
) -> None:
    quoted_test(tmp_path, ["AC-1", "AC-2"])
    hits = spec.criterion_coverage(tmp_path, ["AC-1", "AC-2", "AC-3"])
    assert hits["AC-1"] == ["tests/test_checks.py::test_case_1"]
    assert hits["AC-3"] == []
    assert spec.coverage_summary(hits)["uncovered"] == ["AC-3"]
    assert spec.coverage_summary(hits)["tested"] == 2
    (tmp_path / "tests" / "test_checks.py").unlink()
    gone = spec.criterion_coverage(tmp_path, ["AC-1", "AC-2"])
    assert gone["AC-1"] == [] and gone["AC-2"] == []


def test_only_a_criterion_whose_text_moved_is_requoted() -> None:
    old = spec.parse_criteria(REQ)
    assert spec.changed_criteria(old, spec.parse_criteria(REQ_MOVED)) == ["AC-2", "AC-3"]
    assert spec.changed_criteria(old, old) == []


def test_fences_are_unwrapped_and_a_spec_round_trips_from_disk(tmp_path: Path) -> None:
    assert spec.strip_fences("```md\n## User stories\n```") == "## User stories"
    assert spec.strip_fences("## Plain\n") == "## Plain"
    folder = tmp_path / "specs" / SLUG
    folder.mkdir(parents=True)
    (folder / "requirements.md").write_text(REQ, encoding="utf-8")
    (folder / "tasks.md").write_text(TASKS, encoding="utf-8")
    assert spec.existing_slugs(tmp_path) == [SLUG]
    loaded = spec.load_spec(tmp_path, SLUG)
    assert loaded is not None and loaded.design == ""
    assert [c.id for c in loaded.criteria] == ["AC-1", "AC-2", "AC-3"]
    assert loaded.orphan_criteria() == []
    assert loaded.tasks_without_criteria() == []
    assert loaded.files == ["requirements.md", "tasks.md"]
    assert spec.load_spec(tmp_path, "no-such-spec") is None
    assert spec.load_spec(tmp_path, "..") is None


# ---------------------------------------------------------------------------- the walk, gated


def test_three_documents_land_in_order_each_one_gated_on_an_approval(make_session) -> None:
    """Plan 23's closing shape: three files, in order, each one waiting for a yes."""
    session, recorder, provider, approver, report = walk(make_session, DRAFTS)
    assert report.status == "written" and report.slug == SLUG
    folder = session.cwd / "specs" / SLUG
    assert (folder / "requirements.md").is_file() and (folder / "design.md").is_file()
    assert (folder / "tasks.md").is_file()
    assert [s.status for s in report.steps] == ["written"] * 3
    assert [name for name, _ in approver.asked] == ["Write"] * 3
    assert [arguments["file_path"].split("/")[-1] for _, arguments in approver.asked] == [
        "requirements.md",
        "design.md",
        "tasks.md",
    ]
    assert len(provider.requests) == 3, "one draft call per step, no more"
    assert report.tasks_loaded == 3
    assert len(recorder.of(ev.SpecStarted)) == 3 and len(recorder.of(ev.SpecFinished)) == 1
    todos = recorder.of(ev.TodosUpdated)[-1].todos
    assert [t["status"] for t in todos] == ["pending"] * 3
    assert todos[0]["content"].endswith("(closes AC-1)")
    assert report.surface == "" and report.orphans == [] and report.seconds > 0
    assert [t.id for t in report.tasks] == ["T1", "T2", "T3"]


def test_answering_no_stops_the_walk_and_leaves_the_next_step_unwritten(make_session) -> None:
    session, _, provider, approver, report = walk(make_session, DRAFTS, answer=False)
    assert report.status == "refused" and len(approver.asked) == 1
    assert not (session.cwd / "specs").exists(), "a step nobody approved writes nothing"
    assert len(provider.requests) == 1, "the walk stops at the gate, it does not draft ahead"


def test_a_task_without_a_criterion_is_refused_before_it_becomes_a_file(make_session) -> None:
    """The shape check runs before the approval card: a bad draft never asks, never lands."""
    session, _, provider, approver, report = walk(
        make_session, [chunks_for(REQ), chunks_for(DESIGN), chunks_for(BAD_TASKS)]
    )
    assert report.status == "partial"
    last = report.steps[-1]
    assert last.status == "invalid" and any("no criterion" in e for e in last.errors)
    assert not (session.cwd / "specs" / SLUG / "tasks.md").exists()
    assert [s.step for s in report.steps] == ["requirements", "design", "tasks"]
    assert len(provider.requests) == 3 and len(approver.asked) == 2, "a bad draft spends no card"
    assert (session.cwd / "specs" / SLUG / "requirements.md").is_file(), "the first two stand"


def test_a_step_of_a_spec_that_is_not_on_disk_refuses_without_a_draft_call(make_session) -> None:
    """/spec <slug> tasks with no such spec: refused as a name, never started as a new idea."""
    session, _, provider, _, report = walk(make_session, [], argument=f"{SLUG} tasks")
    assert provider.requests == [] and report.status == "skipped"
    assert "folder is on disk" in report.reason
    assert not (session.cwd / "specs").exists()


def test_tasks_may_not_be_redone_before_requirements_exist(make_session) -> None:
    """The order is enforced on disk, not just asked for: a design alone is not a base for tasks."""
    session, _, _, _, first = walk(make_session, DRAFTS)
    assert first.status == "written"
    folder = session.cwd / "specs" / SLUG
    (folder / "requirements.md").unlink()
    again, recorder, provider, _, report = walk(make_session, [], argument=f"{SLUG} tasks")
    assert provider.requests == [], "a step with nothing under it spends no draft call"
    assert report.steps[0].status == "skipped"
    assert "requirements.md is not on disk" in report.steps[0].reason
    assert again.spec is not None and again.spec.requirements == ""
    assert recorder.of(ev.SpecStarted) == []


def test_plan_mode_and_headless_runs_refuse_before_a_draft_call(make_session) -> None:
    """Neither one can answer an approval card, so neither may start drafting a file."""
    plan, _, provider = make_session(DRAFTS, approver=FixedApprover(True), mode="plan")
    early = asyncio.run(plan.spec_command(IDEA))
    assert early.status == "skipped" and "plan mode" in early.reason
    assert provider.requests == [], "plan mode costs no draft call"
    assert not (plan.cwd / "specs").exists()

    headless, _, provider2 = make_session(DRAFTS, approver=DenyAll())
    gone = asyncio.run(headless.spec_command(IDEA))
    assert gone.status == "skipped" and "no person" in gone.reason
    assert provider2.requests == [], "the draft is spent after the gate, never before it"
    assert not (headless.cwd / "specs").exists()
    assert plan.spec is None and headless.spec is None, "a refused walk loads no spec"


def test_an_idea_that_looks_like_a_path_still_lands_inside_the_spec_folder(make_session) -> None:
    """``/spec <text>`` reads its argument as an idea, so its folder name is always sanitised."""
    session, _, _, _, report = walk(make_session, [chunks_for(REQ)], argument="../else/passwd")
    assert report.slug == "else-passwd"
    written = sorted(p.relative_to(session.cwd) for p in (session.cwd / "specs").rglob("*.md"))
    assert written == [Path("specs/else-passwd/requirements.md")]
    assert not (session.cwd.parent / "else").exists(), "nothing escapes the project"


def test_a_dead_draft_model_costs_the_step_not_the_session(make_session) -> None:
    """No turns scripted: the draft call fails, names the error, and nothing is written."""
    session, _, _, _, report = walk(make_session, [])
    assert report.status == "failed"
    assert report.steps[0].status == "failed" and "ProviderError" in report.steps[0].reason
    assert "ProviderError" in report.reason
    assert not (session.cwd / "specs").exists()


def test_redoing_requirements_requeues_exactly_the_tests_of_a_moved_criterion(
    make_session,
) -> None:
    """Editing a criterion re-flags its tests: the old wording's evidence is owed a rerun."""
    session, _, _, _, first = walk(make_session, DRAFTS)
    assert first.status == "written"
    quoted_test(session.cwd, ["AC-2"])
    again, _, provider, approver, report = walk(
        make_session, [chunks_for(REQ_MOVED)], argument=f"{SLUG} requirements"
    )
    assert report.status == "written" and len(provider.requests) == 1
    assert [name for name, _ in approver.asked] == ["Write"]
    assert report.reflagged == ["AC-2", "AC-3"], "AC-2 moved and AC-3 disappeared"
    assert again._pending_fix_nodes == ["tests/test_checks.py::test_case_1"]
    assert [c.id for c in report.criteria] == ["AC-1", "AC-2"]
    assert report.tasks_loaded == 0, "only requirements landed, the rail is not reloaded"
    assert report.refuted == [], "every task line still names a criterion, so nothing is refused"
    assert report.stale_tasks == ["T3"], (
        "tasks.md still closes AC-3, which requirements.md no longer has"
    )


def test_reading_a_spec_from_disk_needs_no_model_and_loads_its_tasks(make_session) -> None:
    session, _, _, _, first = walk(make_session, DRAFTS)
    assert first.status == "written"
    fresh, recorder, provider, _, report = walk(make_session, [], argument=SLUG)
    assert report.status == "loaded" and provider.requests == []
    assert [c.id for c in report.criteria] == ["AC-1", "AC-2", "AC-3"]
    assert all(s.status == "on-disk" for s in report.steps)
    assert report.tasks_loaded == 3 and recorder.of(ev.TodosUpdated)
    assert fresh.spec is not None and fresh.spec_coverage["AC-1"] == []
    assert report.uncovered() == ["AC-1", "AC-2", "AC-3"]


def test_with_no_spec_anywhere_the_status_says_so(make_session) -> None:
    _, _, provider, _, report = walk(make_session, [], argument="")
    assert provider.requests == [] and report.status == "skipped"
    assert "no spec is loaded" in report.reason and "/spec <idea>" in report.reason


def test_with_a_spec_on_disk_but_none_loaded_the_status_lists_it(make_session) -> None:
    _, _, _, _, first = walk(make_session, DRAFTS)
    assert first.status == "written"
    fresh, _, provider, _, report = walk(make_session, [], argument="")
    assert provider.requests == [] and report.status == "skipped"
    assert SLUG in report.reason


# --------------------------------------------------------------- the link to the assurance card


def card_for(root: Path, files: list[Path], ids: list[str], **over):
    """The real card, fed the payload the session builds: criteria and their measured coverage."""
    kwargs: dict[str, Any] = dict(
        checks=[],
        runit=None,
        flags=FlagReport(),
        fix=None,
        saw_test_run=True,
        saw_test_pass=True,
        scans=False,
        spec={
            "slug": SLUG,
            "dir": f"specs/{SLUG}",
            "criteria": ids,
            "coverage": spec.criterion_coverage(root, ids),
        },
    )
    kwargs.update(over)
    return asyncio.run(build_card(root, files, **kwargs))


def test_the_assurance_correct_row_names_the_criterion_whose_test_was_deleted(
    tmp_path: Path,
) -> None:
    """Plan 23's other half, through the real card: a deleted test FAILS the row and says which."""
    root = tmp_path / "repo"
    folder = root / "specs" / SLUG
    folder.mkdir(parents=True)
    (folder / "requirements.md").write_text(REQ, encoding="utf-8")
    code = root / "loader.py"
    code.write_text("def load() -> int:\n    return 1\n", encoding="utf-8")
    quoted_test(root, ["AC-1", "AC-3"])

    gap = card_for(root, [code], ["AC-1", "AC-2", "AC-3"]).row("correct")
    assert gap.status == "fail" and "AC-2" in gap.reason
    assert "have no test naming them" in gap.reason
    quoted_test(root, ["AC-2"], name="test_detail.py")
    covered = card_for(root, [code], ["AC-1", "AC-2", "AC-3"], runit=RUNIT_CLEAN).row("correct")
    assert covered.status == "pass", covered.reason
    assert "3/3 criteria quoted by a test" in " ".join(covered.evidence)
    (root / "tests" / "test_detail.py").unlink()
    assert "AC-2" in card_for(root, [code], ["AC-1", "AC-2", "AC-3"]).row("correct").reason
    assert "no AC-N criteria" in card_for(root, [code], []).row("correct").reason


def test_a_docs_only_turn_still_fails_the_correct_row_on_an_untested_criterion(
    tmp_path: Path,
) -> None:
    """A promise is a promise: writing only the spec's markdown must not certify it quietly."""
    root = tmp_path / "repo"
    (root / "notes").mkdir(parents=True)
    doc = root / "notes" / "readme.md"
    doc.write_text("# notes\n", encoding="utf-8")
    held = card_for(root, [doc], ["AC-1"], saw_test_run=False, saw_test_pass=False)
    assert held.row("correct").status == "fail" and "AC-1" in held.row("correct").reason
    quiet = card_for(root, [doc], [], saw_test_run=False, saw_test_pass=False, spec=None)
    assert quiet.row("correct").status == "na"
    assert any("no spec is loaded" in line for line in quiet.row("correct").evidence)


def test_the_session_measures_the_coverage_fresh_for_every_card(make_session) -> None:
    """The card must not report remembered coverage: deleting a test changes the next row."""
    session, _, _, _, first = walk(make_session, DRAFTS)
    assert first.status == "written"
    session.settings.assurance_scans = False
    session._saw_test_run = True
    session._saw_test_pass = True
    quoted_test(session.cwd, ["AC-1", "AC-2", "AC-3"])
    code = session.cwd / "loader.py"
    code.write_text("def load() -> int:\n    return 1\n", encoding="utf-8")
    session.written.append(code)

    covered = asyncio.run(session.assurance_card([]))
    assert covered is not None
    assert "3/3 criteria quoted by a test" in " ".join(covered.row("correct").evidence)
    assert "no test naming them" not in covered.row("correct").reason
    (session.cwd / "tests" / "test_checks.py").unlink()

    held = asyncio.run(session.assurance_card([]))
    assert held.row("correct").status == "fail", "the same card, nothing else changed but the test"
    assert "AC-1" in held.row("correct").reason and "AC-3" in held.row("correct").reason
    assert "correct" in held.holding and not held.done_allowed


# --------------------------------------------------------------------------- the two screens


def test_the_terminal_runs_the_walk_and_prints_nothing_itself(make_session) -> None:
    """A side command reports through the bus: printing it here as well would show it twice."""
    session, recorder, _ = make_session(DRAFTS, approver=FixedApprover(True))
    out = Out()
    assert asyncio.run(run_async_command(session, f"/spec {IDEA}", out)) is True
    assert out.rows == [], "the report belongs to the Printer, not to the command"
    assert len(recorder.of(ev.SpecFinished)) == 1
    report = recorder.of(ev.SpecFinished)[-1].spec
    lines = spec_lines(report)
    assert lines[0].startswith(f"[spec] {SLUG}: wrote requirements, design, tasks")
    assert any("no test names AC-1" in line for line in lines)
    assert any("3 task(s) loaded into the task list" in line for line in lines)
    assert spec_lines({"status": "skipped", "reason": "plan mode is on", "steps": []}) == [
        "[spec] SKIPPED: plan mode is on"
    ]


def test_the_terminal_shows_a_refused_task_line_as_refused() -> None:
    """The line that says what was refused is the reason the command exists."""
    refused = spec_lines(
        {
            "slug": SLUG,
            "dir": f"specs/{SLUG}",
            "status": "partial",
            "steps": [
                {
                    "step": "tasks",
                    "file": f"specs/{SLUG}/tasks.md",
                    "status": "invalid",
                    "reason": "",
                    "errors": ["T2 names no criterion"],
                }
            ],
        }
    )
    assert refused[0].startswith(f"[spec] {SLUG}: partial")
    assert any("refused: T2 names no criterion" in line for line in refused)
    stale = spec_lines(
        {
            "slug": SLUG,
            "dir": f"specs/{SLUG}",
            "status": "written",
            "steps": [
                {
                    "step": "requirements",
                    "file": f"specs/{SLUG}/requirements.md",
                    "status": "written",
                }
            ],
            "stale_tasks": ["T3"],
        }
    )
    assert any(
        "1 task(s) cannot be checked, their criterion link is gone: T3" in line for line in stale
    )


def test_the_panel_answers_one_approval_card_per_step(make_session) -> None:
    """The gate is real in the browser too: three Write cards, in order, before the walk reports."""
    session, _, _ = make_session(DRAFTS)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.send_json({"type": "prompt", "text": f"/spec {IDEA}"})
        names = []
        for _ in range(3):
            asked = until(ws, "ApprovalRequest")[-1]
            assert asked["tool"] == "Write"
            names.append(asked["input"]["file_path"].split("/")[-1])
            ws.send_json({"type": "approval", "id": asked["id"], "answer": "yes"})
        finished = until(ws, "SpecFinished")[-1]
    assert names == ["requirements.md", "design.md", "tasks.md"]
    assert finished["type"] == "SpecFinished"
    assert [s["status"] for s in finished["spec"]["steps"]] == ["written"] * 3
    assert finished["spec"]["tasks_loaded"] == 3
    folder = session.cwd / "specs" / SLUG
    assert (folder / "requirements.md").is_file() and (folder / "tasks.md").is_file()
