"""Item 25: two reading levels over one review, chosen by the reader and costing no model call.

The binding clause in the plan is that a switch is a view change, not new work: the same scripted
turn must render both views from one event log, and toggling must not re-run anything. So these
tests drive a real turn with the scripted provider, take the single ReviewFinished payload it
emitted, and show that picking the other plan moves no bytes through the provider, the checks or
the tests. The rest pins the honesty rules of the derivation: a beginner line only ever says what
a field already holds, and a file with no evidence says that instead of getting invented prose.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness import levels
from mwm_harness import review as review_mod
from mwm_harness.providers import chunks_for
from mwm_harness.repl.terminal import Printer, run_command
from test_panel import HOST, TOKEN, client_for, until

HELPER = "def helper(items):\n    if not items:\n        return None\n    return items[0]\n"
CALLER = "from a import helper\n\n\ndef run(items):\n    return helper(items)\n"

# a.py gets the reviewer's critical comment, approver.py only a suggestion, so the two beginner
# blocks differ for a reason the reader can see, and "approver" is a path that SENSITIVE matches.
ANSWER = {
    "summary": "Two new files: a helper and the one function that calls it.",
    "files": [
        {"path": "a.py", "walkthrough": "adds helper(), which returns None for an empty list"},
        {"path": "approver.py", "walkthrough": "adds decide(), which says whether a tool may run"},
    ],
    "comments": [
        {
            "file": "a.py",
            "line": 3,
            "severity": "critical",
            "text": "helper() returns None for an empty list, and run() hands that to its caller",
            "fix": "return an empty list instead of None",
        },
        {
            "file": "approver.py",
            "line": 2,
            "severity": "suggestion",
            "text": "consider logging the decision",
            "fix": "",
        },
    ],
}


def editing_session(make_session):
    """A session whose first turn writes two files, then answers the Review Desk once."""
    turns = [
        chunks_for(
            tool_calls=[
                ("Write", {"file_path": "a.py", "content": HELPER}),
                ("Write", {"file_path": "approver.py", "content": CALLER}),
            ]
        ),
        chunks_for("Written."),
        chunks_for(json.dumps(ANSWER)),
    ]
    return make_session(turns, approver=FixedApprover(True))


def one_review(make_session):
    """One real turn and the single review payload it emitted."""
    session, recorder, provider = editing_session(make_session)
    asyncio.run(session.send("add a helper and a caller"))
    finished = recorder.of(ev.ReviewFinished)
    assert len(finished) == 1, "item 25 must not add a second review to a turn"
    return session, finished[0].review, provider, recorder


def change(path: str, **fields) -> dict:
    """One file's payload row, defaults filled in, so a test names only what it varies."""
    row = asdict(review_mod.FileChange(path, "+added line\n-removed line"))
    row.update(fields)
    return row


def base_payload(rows=None, comments=(), **over) -> dict:
    """A review built from the dataclasses rather than from a model answer."""
    review = review_mod.Review(
        "/r",
        "test-model",
        over.pop("status", "reviewed"),
        over.pop("reason", ""),
        over.pop("summary", "the summary the reviewer wrote"),
        rows if rows is not None else [change("a.py", added=4, removed=1, new=True)],
        list(comments),
    )
    built = review.as_dict()
    built.update(over)
    return built


# ------------------------------------------------------------------ the setting itself


def test_the_plain_words_and_the_short_ones_both_name_a_level() -> None:
    for value in ("beginner", "Beginner", " b ", "simple", "plain"):
        assert levels.parse_level(value) == ("beginner", ""), value
    for value in ("engineer", "E", "raw", "full", "technical"):
        assert levels.parse_level(value) == ("engineer", ""), value


def test_a_value_that_names_no_level_is_said_out_loud_not_read_as_the_default() -> None:
    level, why = levels.parse_level("brginer")
    assert level == levels.DEFAULT_LEVEL == "engineer"
    assert (
        why == "brginer names no reading level: use beginner or engineer, so engineer is in force"
    )
    assert levels.parse_level("")[1].startswith("an empty value names no reading level")
    assert levels.parse_level(None)[1].startswith("an empty value names no reading level")


def test_pick_draws_the_payload_as_before_when_it_carries_no_plans() -> None:
    """A transcript written before item 25 has no views, and must still render."""
    bare = base_payload()
    assert "views" not in bare
    assert levels.pick(bare, "beginner") is None
    both = levels.views(bare)
    assert levels.pick({**bare, "views": both}, "nonsense")["level"] == "engineer"
    assert levels.pick({**bare, "views": both}, "BEGINNER")["level"] == "beginner"
    assert levels.pick({"views": "not a mapping"}, "beginner") is None


# ------------------------------------------------------------------ one pass, two plans


def test_both_views_are_plans_for_the_same_files_never_two_different_facts() -> None:
    views = levels.views(
        base_payload(
            rows=[change("a.py", added=4, removed=1), change("b/c.py", added=2, removed=9)]
        )
    )
    beginner, engineer = views["beginner"], views["engineer"]
    paths = ["a.py", "b/c.py"]
    assert [row["path"] for row in beginner["files"]] == paths
    assert [row["path"] for row in engineer["files"]] == paths
    assert beginner["fold_diff"] is True and engineer["fold_diff"] is False
    assert beginner["sections"] == list(levels.BEGINNER_SECTIONS)
    assert engineer["sections"] == list(levels.ENGINEER_SECTIONS)
    assert beginner["notes"] == engineer["notes"], "turn-wide facts may not differ between levels"
    for plan in (beginner, engineer):
        assert "+added line" not in json.dumps(plan), "a plan must not copy the diff text"


def test_the_headlines_are_the_two_numbers_and_words_a_turn_already_has() -> None:
    views = levels.views(
        base_payload(rows=[change("a.py", added=4, removed=1), change("b.py", added=2, removed=0)])
    )
    assert views["engineer"]["headline"] == "test-model read 2 file(s), +6/-1 in 0s, 0 comment(s)"
    assert views["beginner"]["headline"] == "the summary the reviewer wrote"


def test_a_skipped_review_says_so_in_both_headlines_instead_of_a_fake_verdict() -> None:
    views = levels.views(base_payload(rows=[], status="skipped", reason="no diff to read"))
    for plan in views.values():
        assert plan["headline"] == "the review desk did not read this turn: no diff to read"
        assert plan["files"] == []


# ------------------------------------------------------------------ honest derivation


def test_a_file_with_no_walkthrough_admits_it_instead_of_inventing_one() -> None:
    row = levels.views(base_payload())["beginner"]["files"][0]
    assert row["does"].startswith("no walkthrough line for this file")
    assert "a new file: 4 line(s) in, 1 out" in row["does"]


def test_the_reviewer_s_own_critical_words_carry_the_fix_text() -> None:
    comments = [
        review_mod.Comment("a.py", 3, "critical", "returns None here", "return [] instead").__dict__
    ]
    breaks = levels.views(base_payload(comments=comments))["beginner"]["files"][0]["breaks"]
    assert breaks[0] == (
        "the reviewer calls this a problem at line 3: returns None here (fix it: return [] instead)"
    )


def test_break_lines_name_the_evidence_a_reader_can_go_and_check() -> None:
    rows = [
        change(
            "approver.py",
            added=1,
            removed=8,
            new=False,
            truncated=True,
            callers=["mwm_harness.loop.Session.handle", "mwm_harness.permissions.check"],
            unresolved=["getattr(mod, name)"],
        )
    ]
    breaks = levels.views(base_payload(rows=rows))["beginner"]["files"][0]["breaks"]
    joined = " ".join(breaks)
    assert "2 function(s) reach this file (Session.handle, permissions.check)" in joined
    assert "watched code (approv)" in joined, "the path word that makes it sensitive is quoted"
    assert "it removes more than it adds (8 lines out, 1 in)" in joined
    assert "1 call(s) here are only decided while the program runs" in joined
    assert len(breaks) == levels.MAX_BREAKS + 1, (
        f"the block caps itself and names what it folded: {breaks}"
    )
    assert breaks[-1].startswith("+1 more signal(s) in the engineer view")


def test_a_brand_new_file_nobody_calls_says_that_instead_of_nothing() -> None:
    breaks = levels.views(base_payload())["beginner"]["files"][0]["breaks"]
    assert breaks == [
        "nothing in the repo calls this new file yet, so a mistake here only shows up "
        "once something starts using it"
    ]


def test_a_file_with_no_signal_at_all_says_there_is_nothing_pointing_at_a_break() -> None:
    rows = [change("plain.py", added=9, removed=0, new=False)]
    breaks = levels.views(base_payload(rows=rows))["beginner"]["files"][0]["breaks"]
    assert breaks == [
        "nothing in this file's own data points at a break: no critical comment, no caller "
        "in the map, more added than removed"
    ]


def test_a_check_that_failed_is_quoted_and_one_that_never_came_up_admits_it() -> None:
    checks = [
        {
            "check": "ruff",
            "status": "fail",
            "files": ["a.py"],
            "findings": [
                {"file": "a.py", "line": 2, "message": "F401 unused import", "check": "ruff"}
            ],
            "reason": "",
            "seconds": 0.1,
        },
        {
            "check": "mypy",
            "status": "not_run",
            "files": [],
            "findings": [],
            "reason": "no mypy config in this repo",
            "seconds": 0.0,
        },
    ]
    payload = base_payload(rows=[change("a.py", new=False), change("b.py", new=False)])
    views = levels.views(payload, checks=checks)
    assert views["beginner"]["files"][0]["check"] == [
        "ruff FAILS on it: F401 unused import (1 finding(s))"
    ]
    assert views["beginner"]["files"][1]["check"] == [
        "no check or test result names this file from this turn, so there is nothing here to "
        "re-check: /runit runs the tests and measures which of its lines they touch"
    ]
    assert "the check chain FAILED: ruff" in views["engineer"]["notes"]
    assert "the check chain did not answer: mypy" in views["engineer"]["notes"]


def test_a_passing_check_names_itself_so_the_reader_can_rerun_the_same_thing() -> None:
    checks = [
        {"check": "compileall", "status": "pass", "files": ["a.py"], "findings": [], "reason": ""}
    ]
    rows = levels.views(base_payload(), checks=checks)["beginner"]["files"][0]["check"]
    assert rows == ["compileall passes on it: run the repo's compileall to see the same result"]


def test_coverage_is_quoted_with_the_lines_a_test_never_ran() -> None:
    runit = {
        "status": "ran",
        "python": "3.12",
        "tests_passed": False,
        "summary": "1 failed, 2 passed",
        "files": {"a.py": [{"line": 3, "state": "miss"}, {"line": 4, "state": "hit"}]},
        "reason": "",
        "tail": "",
        "seconds": 0.2,
    }
    views = levels.views(base_payload(), runit=runit)
    assert views["beginner"]["files"][0]["check"] == [
        "a test run touched 1 of its changed lines and never touched 1 (never run: 3); "
        "read those lines, they are the untested part"
    ]
    assert views["engineer"]["notes"][0] == "the tests as a whole: TESTS FAIL (1 failed, 2 passed)"
    idle = {**runit, "status": "not_run", "reason": "no tests collected", "files": {}}
    views = levels.views(base_payload(), runit=idle)
    assert views["beginner"]["files"][0]["check"] == [
        "no coverage for it yet: run-it did not run (no tests collected), so /runit is the check"
    ]
    assert views["engineer"]["notes"][0] == "no test ran during this turn: no tests collected"


def test_item_23_s_criterion_map_shows_which_promises_a_test_names() -> None:
    coverage = {"AC-1": ["tests/test_a.py::test_a"], "AC-2": []}
    notes = levels.views(base_payload(), coverage=coverage)["beginner"]["notes"]
    assert notes == [
        "1/2 acceptance criteria are quoted by a test: AC-1 in tests/test_a.py::test_a",
        "1 criterion(s) have no test naming them: AC-2. "
        "Run /spec <slug> to reread what was promised.",
    ]
    assert levels.views(base_payload(), coverage=None)["beginner"]["notes"] == []


def test_dropped_comments_and_learnings_are_said_once_for_both_levels() -> None:
    views = levels.views(base_payload(dropped=2, learnings=1))
    assert views["beginner"]["notes"][:2] == [
        "2 comment(s) were dropped: they pointed at a file or line the diff does not touch",
        "1 learning(s) saved for later reviews of this repo",
    ]
    assert views["engineer"]["notes"] == views["beginner"]["notes"]


# ------------------------------------------------------------------ one event log, both views


def test_one_scripted_turn_carries_both_views_and_switching_runs_nothing(make_session) -> None:
    _session, payload, provider, recorder = one_review(make_session)
    asked = len(provider.requests)
    assert asked == 3, "two lead turns plus one reviewer pass"

    beginner = levels.pick(payload, "beginner")
    engineer = levels.pick(payload, "engineer")
    assert beginner is not None and engineer is not None
    assert len(provider.requests) == asked, "reading the other plan asked a model"
    assert len(recorder.of(ev.ReviewFinished)) == 1, "switching level reviewed the turn again"
    assert len(recorder.of(ev.ReviewStarted)) == 1

    assert payload["reading_level"] == "engineer", "the default is what every turn looked like"
    assert beginner["headline"] == ANSWER["summary"]
    paths = [row["path"] for row in payload["files"]]
    assert [row["path"] for row in beginner["files"]] == paths
    assert [row["path"] for row in engineer["files"]] == paths
    assert beginner["files"][0]["does"] == ANSWER["files"][0]["walkthrough"]
    assert beginner["files"][0]["breaks"][0].startswith(
        "the reviewer calls this a problem at line 3"
    )
    assert any(
        line.startswith("1 suggestion(s) from the reviewer")
        for line in beginner["files"][1]["breaks"]
    )
    assert any("watched code (approv)" in line for line in beginner["files"][1]["breaks"])
    assert engineer["files"][0]["sensitive"] is False
    assert engineer["files"][1]["sensitive"] is True
    assert engineer["files"][0]["added"] == payload["files"][0]["added"]


def test_the_default_level_in_the_setting_names_which_plan_a_turn_is_drawn_in(make_session) -> None:
    session, recorder, provider = editing_session(make_session)
    session.settings.reading_level = "beginner"
    asyncio.run(session.send("add a helper and a caller"))
    payload = recorder.of(ev.ReviewFinished)[0].review
    assert payload["reading_level"] == "beginner"
    assert levels.pick(payload, "beginner")["files"][0]["does"].startswith("adds helper()")
    assert len(provider.requests) == 3, "a level is not a reason to call the reviewer twice"


def test_an_unknown_level_in_the_settings_warns_once_per_session(make_session) -> None:
    session, recorder, provider = editing_session(make_session)
    session.settings.reading_level = "brginer"
    review = review_mod.Review(str(session.cwd), "test-model", "reviewed", "", "nothing to read")
    first = session._review_payload(review)
    second = session._review_payload(review)
    assert first["reading_level"] == second["reading_level"] == "engineer"
    warns = [note for note in recorder.of(ev.Notice) if "names no reading level" in note.text]
    assert len(warns) == 1, f"the typo should be said once, not once per review: {len(warns)}"
    assert "brginer names no reading level" in warns[0].text
    assert first["views"]["engineer"]["headline"].startswith("test-model read 0 file(s)")
    assert len(provider.requests) == 0


def test_the_review_payload_keeps_every_field_the_engineer_view_already_read(make_session) -> None:
    _session, payload, _provider, _recorder = one_review(make_session)
    plain = review_mod.Review(
        **{k: v for k, v in payload.items() if k not in ("reading_level", "views")}
    )
    assert plain.as_dict()["files"][0]["diff"]  # the raw diff stays on the payload itself
    for key in ("status", "model", "summary", "files", "comments", "dropped", "seconds"):
        assert key in payload, f"{key} is what item 25's plans are derived from"


# ------------------------------------------------------------------ the terminal front-end


def test_the_terminal_prints_three_answers_per_file_at_beginner(make_session) -> None:
    _session, payload, provider, _recorder = one_review(make_session)
    out = Printer(color=False)
    out.last_review = payload
    asked = len(provider.requests)

    engineer = out.review_rows(payload)
    assert engineer[0].startswith("[review] test-model read 2 file(s)")

    out.reading_level = "beginner"
    beginner = out.review_rows(payload)
    assert beginner[0] == f"[review] {ANSWER['summary']}"
    assert any(row == "  a.py" for row in beginner)
    assert any(row.startswith("    does: adds helper()") for row in beginner)
    assert "    could break:" in beginner and "    check it yourself:" in beginner
    assert beginner[-1].startswith("  advice only, nothing applied.")
    assert not any(row.startswith("@@") or row.startswith("+") for row in beginner)
    assert len(provider.requests) == asked, "the beginner view asked a model"


def test_level_command_redraws_the_last_review_without_running_anything(make_session) -> None:
    _session, payload, provider, _recorder = one_review(make_session)
    out = Printer(color=False)
    out.last_review = payload
    asked = len(provider.requests)
    printed: list[str] = []
    out.line = lambda text="", event=None, stamp=True: printed.append(text)

    assert run_command(None, {}, "/level", out) is True
    assert printed[0] == "reading level: engineer"

    printed.clear()
    run_command(None, {}, "/level beginner", out)
    assert out.reading_level == "beginner"
    assert printed[0] == "reading level: beginner"
    assert any(row.startswith("    does: adds helper()") for row in printed)

    printed.clear()
    run_command(None, {}, "/level brginer", out)
    assert out.reading_level == "engineer"
    assert "brginer names no reading level" in printed[0]
    assert len(provider.requests) == asked, "/level must not reach the provider"


# ------------------------------------------------------------------ the panel front-end


def _panel_script() -> str:
    """The page's own JavaScript: the panel is served from this file, so this is what a browser runs."""
    html = (
        Path(__file__).resolve().parents[1] / "mwm_harness" / "web" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1, (
        "the page grew a second script block; this test reads only the inline one"
    )
    return blocks[0]


def test_the_level_switch_reads_the_other_plan_per_tab_without_asking_the_server() -> None:
    """Run the shipped switch itself under node: one tab's choice must not move the others."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    script = _panel_script()
    start = script.index("const levelByTab")
    end = script.index("const toneOf")
    toggle = script[start:end]
    assert "send(" not in toggle, (
        "the level switch must never send a message: it is a view, not work"
    )
    program = (
        toggle
        + """
const store = () => {};
let session = "t1";
const stored = () => null;
const r = {reading_level: "engineer", views: {beginner: {level: "beginner"}, engineer: {level: "engineer"}}};
const out = [readPlan(r).level];
levelByTab.set("t1", "beginner");
out.push(readPlan(r).level);
session = "t2";
out.push(readPlan(r).level);
levelByTab.set("t3", "brginer");
session = "t3";
out.push(readPlan(r).level);
r.views = null;
out.push(readPlan(r) === null ? "no-plans-draws-payload" : "wrong");
console.log(out.join(","));
"""
    )
    path = Path(__file__).resolve().parents[1] / ".pytest-node-level"
    script_file = path.with_suffix(".js")
    script_file.write_text(program, encoding="utf-8")
    try:
        done = subprocess.run(
            ["node", str(script_file)], capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == (
            "engineer,beginner,engineer,engineer,no-plans-draws-payload"
        ), done.stdout
    finally:
        script_file.unlink(missing_ok=True)


def test_the_panel_page_the_browser_loads_carries_the_level_toggle() -> None:
    """A served-page check: the wiring must be in the file the panel actually reads, not a stale copy."""
    script = _panel_script()
    for needle in ("levelSwitch", "beginnerFile", "readPlan", "no reading levels in this review"):
        assert needle in script, f"the shipped page is missing {needle!r}"
    assert (
        "aria-pressed"
        in script[script.index("function levelSwitch") : script.index("const toneOf")]
    )


def test_the_rail_task_block_ticks_the_bar_and_the_strip_without_opening_anything() -> None:
    """Item 37: run the shipped setTodos itself under node with a stub DOM: the pinned block,
    the progress bar, the strip chip and the scene all move from one call, no details opened."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    script = _panel_script()
    fn = script[script.index("function setTodos") : script.index("function setPlan")]
    program = (
        """
const el = (tag, cls, text) => { const n = {tag, cls: cls || "", text: text || "", kids: []};
  n.append = (...x) => n.kids.push(...x); return n; };
const nodes = {};
const $ = (id) => (nodes[id] ||= {id, children: [], textContent: "", style: {}, hidden: true,
  replaceChildren() { this.children = []; }, append(n) { this.children.push(n); }});
let sceneCalls = null;
const scene = { setTodos: (t) => { sceneCalls = t; } };
"""
        + fn
        + """
setTodos([{content: "plan", status: "completed"},
          {content: "build", status: "in_progress", activeForm: "Building"},
          {content: "ship", status: "pending"}]);
const rows = nodes["tasks"].children.map((c) => c.cls);
const shown = nodes["tasks"].children.map((c) => c.kids[1].text);
const mid = [nodes["tasksum"].textContent, nodes["taskbar"].style.width,
  nodes["striptasks"].hidden, nodes["striptaskn"].textContent, sceneCalls.length];
setTodos([]);
const empty = nodes["tasks"].children.map((c) => c.cls + "|" + c.text);
console.log(JSON.stringify([rows, shown, mid[0], mid[1], mid[2], mid[3], mid[4], empty]));
"""
    )
    path = Path(__file__).resolve().parents[1] / ".pytest-node-tasks"
    script_file = path.with_suffix(".js")
    script_file.write_text(program, encoding="utf-8")
    try:
        done = subprocess.run(
            ["node", str(script_file)], capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 0, done.stderr
        rows, shown, sum_txt, bar, strip_hidden, strip_n, calls, empty = json.loads(done.stdout)
        assert rows == ["done", "now", ""], rows
        assert shown == ["plan", "Building", "ship"], shown  # in_progress shows its activeForm
        assert (sum_txt, bar, strip_hidden, strip_n) == ("1 of 3", "33.3%", False, "1/3")
        assert calls == 3, "the scene must receive the same list the rail draws"
        assert empty == ["empty|No tasks yet. The model keeps this list with TodoWrite."], empty
    finally:
        script_file.unlink(missing_ok=True)


def test_the_panel_script_still_parses() -> None:
    """A JavaScript syntax error bricks the whole desk silently, so it is checked as code."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    script_file = Path(__file__).resolve().parents[1] / ".pytest-node-parse.js"
    script_file.write_text(_panel_script(), encoding="utf-8")
    try:
        done = subprocess.run(
            ["node", "--check", str(script_file)], capture_output=True, text=True, timeout=60
        )
        assert done.returncode == 0, done.stderr[-2000:]
    finally:
        script_file.unlink(missing_ok=True)


def test_the_panel_replays_both_views_from_the_one_event_it_stored(make_session) -> None:
    """acceptEdits, as every panel test does: the browser answers approvals, not this session."""
    turns = [
        chunks_for(
            tool_calls=[
                ("Write", {"file_path": "a.py", "content": HELPER}),
                ("Write", {"file_path": "approver.py", "content": CALLER}),
            ]
        ),
        chunks_for("Written."),
        chunks_for(json.dumps(ANSWER)),
    ]
    session, _recorder, provider = make_session(turns, mode="acceptEdits")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "prompt", "text": "write it"})
        seen = until(ws, "TurnEnded")
        payload = next(m for m in seen if m["type"] == "ReviewFinished")["review"]
        assert set(payload["views"]) == {"beginner", "engineer"}
        assert payload["views"]["beginner"]["fold_diff"] is True
        assert set(payload["views"]["beginner"]["files"][0]) == {
            "path",
            "does",
            "breaks",
            "check",
        }, "the beginner plan reaches the page as answers only, never as a patch"
        assert "adds helper(), which returns None for an empty list" in json.dumps(
            payload["views"]["beginner"]
        ), "the walkthrough must reach the browser, not only the terminal"
        asked = len(provider.requests)

        ws.send_json({"type": "state"})
        replayed = until(ws, "State")[-1]["review"]
        assert replayed == payload, "a reconnect replays the same event, plans included"
        assert len(provider.requests) == asked, "the reload re-reviewed the turn"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
