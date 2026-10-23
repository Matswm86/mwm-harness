"""Review Desk: a second reader over the turn's diff. It advises, and never writes a file."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness import review
from mwm_harness.config import ModelSpec
from mwm_harness.events import DenyAll
from mwm_harness.providers import chunks_for

HELPER = "def helper(items):\n    if not items:\n        return None\n    return items[0]\n"
CALLER = "from a import helper\n\n\ndef run(items):\n    return helper(items)\n"

ANSWER = {
    "summary": "Two new files: a helper and the one function that calls it.",
    "files": [
        {"path": "a.py", "walkthrough": "adds helper(), which returns None for an empty list"},
        {"path": "b.py", "walkthrough": "adds run(), which passes its argument straight through"},
    ],
    "comments": [
        {
            "file": "a.py",
            "line": 3,
            "severity": "critical",
            "text": "helper() returns None for an empty list, and run() hands that to its caller",
            "fix": "return an empty list instead of None",
        }
    ],
}


def reviewer_answer(data: dict | None = None) -> list[dict]:
    return chunks_for(json.dumps(data if data is not None else ANSWER))


def editing_session(make_session, extra: list | None = None):
    """A session whose first turn writes two files, then a reviewer answer for the Review Desk."""
    turns = [
        chunks_for(
            tool_calls=[
                ("Write", {"file_path": "a.py", "content": HELPER}),
                ("Write", {"file_path": "b.py", "content": CALLER}),
            ]
        ),
        chunks_for("Written."),
        reviewer_answer(),
        *(extra or []),
    ]
    return make_session(turns, approver=FixedApprover(True))


def test_turn_that_wrote_two_files_lists_both_with_a_walkthrough_line(make_session) -> None:
    session, recorder, _ = editing_session(make_session)
    asyncio.run(session.send("add a helper and a caller"))

    finished = recorder.of(ev.ReviewFinished)
    assert len(finished) == 1
    payload = finished[0].review
    assert payload["status"] == "reviewed"
    assert [change["path"] for change in payload["files"]] == ["a.py", "b.py"]
    assert [change["walkthrough"] for change in payload["files"]] == [
        "adds helper(), which returns None for an empty list",
        "adds run(), which passes its argument straight through",
    ]
    assert all(change["new"] for change in payload["files"])
    kinds = [type(event).__name__ for event in recorder.events]
    assert kinds.index("ReviewFinished") < kinds.index("TurnEnded")


def test_the_reviewer_writes_nothing(make_session, tmp_path: Path) -> None:
    session, recorder, _ = editing_session(make_session)
    asyncio.run(session.send("add a helper and a caller"))
    assert recorder.of(ev.ReviewFinished)[0].review["status"] == "reviewed"

    project = tmp_path / "project"
    written = sorted(p.name for p in project.iterdir() if not p.name.startswith("."))
    assert written == ["a.py", "b.py"]  # the check runner's cache is the only other thing there
    assert (project / "a.py").read_text() == HELPER
    assert (project / "b.py").read_text() == CALLER


def test_a_comment_sent_becomes_the_next_turns_user_text(make_session) -> None:
    session, recorder, provider = editing_session(make_session, extra=[chunks_for("Fixed it.")])

    async def script() -> None:
        await session.send("add a helper and a caller")
        await session.send(session.comment_prompt(1, "make it an empty list"))

    asyncio.run(script())

    prompts = [m["content"] for m in provider.requests[-1]["messages"] if m["role"] == "user"]
    assert prompts[-1].startswith("Review comment on a.py:3 (critical,")
    assert "helper() returns None for an empty list" in prompts[-1]
    assert "My decision: make it an empty list" in prompts[-1]
    sent = recorder.of(ev.ReviewComment)
    assert [(c.file, c.line, c.remember) for c in sent] == [("a.py", 3, False)]


def test_remember_keeps_the_note_and_the_next_review_is_told_it(
    make_session, tmp_path: Path
) -> None:
    session, recorder, provider = editing_session(make_session, extra=[reviewer_answer()])

    async def script() -> None:
        await session.send("add a helper and a caller")
        session.remember(1)
        await session.run_review()

    asyncio.run(script())

    path = tmp_path / "project" / ".mwm" / "review-learnings.md"
    assert path.is_file()
    kept = review.read_learnings(path.parent.parent)
    assert len(kept) == 1 and "helper() returns None" in kept[0] and "a.py:3" in kept[0]

    prompt = provider.requests[-1]["messages"][-1]["content"]
    assert "STANDING NOTES FROM EARLIER REVIEWS" in prompt
    assert kept[0] in prompt
    comments = recorder.of(ev.ReviewComment)
    assert [(c.file, c.line, c.remember) for c in comments] == [("a.py", 3, True)]


def test_a_comment_number_that_does_not_exist_is_refused(make_session) -> None:
    session, _, _ = editing_session(make_session)
    asyncio.run(session.send("add a helper and a caller"))
    with pytest.raises(ValueError, match="no comment 2"):
        session.comment_prompt(2)
    with pytest.raises(ValueError, match="no comment"):
        session.remember(0)


def test_a_turn_that_wrote_nothing_runs_no_review(make_session) -> None:
    session, recorder, _ = make_session([chunks_for("Hi.")])
    asyncio.run(session.send("hi"))
    assert recorder.of(ev.ReviewStarted) == []
    assert recorder.of(ev.ReviewFinished) == []


def test_a_comment_on_a_line_the_diff_never_touched_is_dropped() -> None:
    changes = [
        review.FileChange(
            "a.py", "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,2 @@\n+one\n+two\n", 2, 0, True
        )
    ]
    reply = json.dumps(
        {
            "summary": "s",
            "files": [],
            "comments": [
                {"file": "a.py", "line": 2, "severity": "note", "text": "kept"},
                {"file": "a.py", "line": 99, "severity": "note", "text": "guessed"},
                {"file": "other.py", "line": 1, "severity": "note", "text": "not in this turn"},
            ],
        }
    )
    _, _, comments, dropped = review.parse_reply(reply, changes)
    assert [c.text for c in comments] == ["kept"]
    assert dropped == 2


def test_an_answer_with_no_json_is_a_failure_not_a_review() -> None:
    async def caller(system: str, user: str) -> str:
        return "I looked at it and it seems fine."

    changes = [review.FileChange("a.py", "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n", 1, 0, True)]
    result = asyncio.run(
        review.review_changes(Path("/repo"), "task", changes, caller, "critic", learnings=[])
    )
    assert result.status == "failed"
    assert "no JSON object" in result.reason


def test_a_reviewer_that_raises_is_reported_as_failed() -> None:
    async def caller(system: str, user: str) -> str:
        raise RuntimeError("endpoint said no")

    changes = [review.FileChange("a.py", "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n", 1, 0, True)]
    result = asyncio.run(
        review.review_changes(Path("/repo"), "task", changes, caller, "critic", learnings=[])
    )
    assert result.status == "failed"
    assert "endpoint said no" in result.reason


def test_a_sensitive_path_also_gets_the_security_reading() -> None:
    changes = [
        review.FileChange("mwm_harness/permissions.py", "@@ -1 +1 @@\n+x\n", 1, 0, False),
    ]
    prompt = review.build_prompt("task", changes, [], review.SECURITY_EXTRA)
    assert "as a security reviewer" in prompt
    assert "security reviewer" not in review.build_prompt("task", changes, [])


def test_files_over_the_cap_are_named_not_shown(tmp_path: Path) -> None:
    root = tmp_path
    before: dict[Path, bytes | None] = {}
    files = []
    for index in range(review.MAX_FILES + 3):
        path = root / f"f{index}.py"
        path.write_text("x = 1\n")
        files.append(path)
        before[path] = None
    changes = review.collect_changes(root, files, before)
    assert [change.path for change in changes] == [f"f{index}.py" for index in range(12)]
    left_out = review.skipped_names(root, files)
    assert left_out == ["f12.py", "f13.py", "f14.py"]
    assert "not reviewed" in review.build_prompt("t", changes, [], "", left_out)


def test_learnings_file_keeps_sixty_lines_newest_wins(tmp_path: Path) -> None:
    for index in range(65):
        review.remember(tmp_path, f"note {index}", "a.py:1")
    kept = review.read_learnings(tmp_path)
    assert len(kept) == 60
    assert "note 5" in kept[0]  # notes 0-4 fell off the top
    assert "note 64" in kept[-1]


def test_an_empty_note_is_refused(tmp_path: Path) -> None:
    with pytest.raises(review.ReviewError, match="empty"):
        review.remember(tmp_path, "   ")


def test_the_learnings_file_is_kept_out_of_the_clone(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert review.exclude_from_git(tmp_path) == ""
    exclude = tmp_path / ".git" / "info" / "exclude"
    assert exclude.read_text().strip() == ".mwm/review-learnings.md"

    review.exclude_from_git(tmp_path)  # a second note adds no second line
    assert exclude.read_text().strip() == ".mwm/review-learnings.md"


def test_a_worktree_pointer_is_followed(tmp_path: Path) -> None:
    real = tmp_path / "real-git"
    (real / "info").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {real}\n")
    assert review.exclude_from_git(work) == ""
    assert (real / "info" / "exclude").read_text().strip() == ".mwm/review-learnings.md"


def test_a_folder_with_no_clone_needs_no_exclude(tmp_path: Path) -> None:
    assert review.exclude_from_git(tmp_path) == ""
    assert not (tmp_path / ".git").exists()


def test_an_unwritable_exclude_is_reported_not_swallowed(tmp_path: Path) -> None:
    (tmp_path / ".git" / "info" / "exclude").mkdir(parents=True)  # a folder where the file goes
    reason = review.exclude_from_git(tmp_path)
    assert "untracked" in reason and "exclude" in reason


def test_a_live_lock_skips_the_review(tmp_path: Path, make_session) -> None:
    session, recorder, provider = editing_session(make_session)
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / f".active-{os.getpid()}").write_text("reviewing")
    session.settings.review_lock_dir = str(locks)
    asyncio.run(session.send("add a helper and a caller"))

    payload = recorder.of(ev.ReviewFinished)[0].review
    assert payload["status"] == "skipped"
    assert str(os.getpid()) in payload["reason"]
    assert len(provider.requests) == 2  # the turn's two calls; no reviewer call was made


def test_a_dead_lock_is_ignored(tmp_path: Path, make_session) -> None:
    session, recorder, _ = editing_session(make_session)
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / ".active-999999").write_text("a pid that is not running")
    session.settings.review_lock_dir = str(locks)
    asyncio.run(session.send("add a helper and a caller"))
    assert recorder.of(ev.ReviewFinished)[0].review["status"] == "reviewed"


def test_a_headless_run_never_calls_a_subscription_endpoint(make_session) -> None:
    session, recorder, provider = editing_session(make_session)
    session.model = ModelSpec(
        id="critic", base_url="https://token-plan.example.invalid/v1", key_env="UNUSED_KEY"
    )
    session.settings.review_model = ""
    session.approver = DenyAll()
    session.written = [session.cwd / "a.py"]
    (session.cwd / "a.py").write_text(HELPER)

    result = asyncio.run(session.run_review())
    assert result.status == "skipped"
    assert "subscription plan" in result.reason
    assert provider.requests == []
    assert recorder.of(ev.ReviewStarted) == []


def test_a_turn_with_the_review_switched_off_runs_no_review(make_session) -> None:
    session, recorder, provider = editing_session(make_session)
    session.settings.review_after_turn = False
    asyncio.run(session.send("add a helper and a caller"))
    assert recorder.of(ev.ReviewStarted) == []
    assert len(provider.requests) == 2


def test_a_broken_reviewer_leaves_the_turn_intact(make_session) -> None:
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "a.py", "content": HELPER})]),
        chunks_for("Written."),
    ]  # the reviewer call has no scripted turn left, so the provider raises
    session, recorder, _ = make_session(turns, approver=FixedApprover(True))
    ended = asyncio.run(session.send("write it"))

    assert ended.reason == "done"
    payload = recorder.of(ev.ReviewFinished)[0].review
    assert payload["status"] == "failed"
    assert "ran out of turns" in payload["reason"]
    assert (session.cwd / "a.py").read_text() == HELPER


def test_comment_text_quotes_the_line_and_keeps_the_decision_last() -> None:
    comment = review.Comment("a.py", 12, "suggestion", "this swallows the error", "log it")
    text = review.comment_text(comment, "no, leave it")
    assert text.splitlines() == [
        "Review comment on a.py:12 (suggestion, written by the reviewer model):",
        "this swallows the error",
        "Its proposed fix: log it",
        "My decision: no, leave it",
    ]
