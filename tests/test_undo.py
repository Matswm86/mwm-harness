"""Undo a turn: files written by Write/Edit come back; later edits and Bash changes are named."""

from __future__ import annotations

import asyncio

from conftest import FixedApprover
from mwm_harness.providers import chunks_for


def edit_turn():
    return [
        chunks_for(
            tool_calls=[
                ("Write", {"file_path": "new.txt", "content": "fresh"}),
                ("Bash", {"command": "echo side > side.txt"}),
            ]
        ),
        chunks_for(tool_calls=[("Read", {"file_path": "old.txt"})]),
        chunks_for(
            tool_calls=[
                ("Edit", {"file_path": "old.txt", "old_string": "one", "new_string": "two"})
            ]
        ),
        chunks_for("Done."),
    ]


def test_undo_restores_edited_file_deletes_created_one_and_names_bash(make_session):
    session, _, _ = make_session(
        edit_turn(), approver=FixedApprover(True), mode="bypassPermissions"
    )
    (session.cwd / "old.txt").write_text("one\n")
    asyncio.run(session.send("change things"))
    assert (session.cwd / "old.txt").read_text() == "two\n"
    report = session.undo()
    assert report.restored == ["old.txt"] and report.deleted == ["new.txt"]
    assert (session.cwd / "old.txt").read_text() == "one\n"
    assert not (session.cwd / "new.txt").exists()
    assert report.bash_calls == 1 and (session.cwd / "side.txt").exists()  # named, not undone
    assert any("NOT undone" in line for line in report.lines())
    assert session.undo().lines()[0].startswith("nothing to undo")


def test_undo_leaves_a_file_that_changed_after_the_turn(make_session):
    session, _, _ = make_session(
        edit_turn(), approver=FixedApprover(True), mode="bypassPermissions"
    )
    (session.cwd / "old.txt").write_text("one\n")
    asyncio.run(session.send("change things"))
    (session.cwd / "old.txt").write_text("mats edited this\n")
    report = session.undo()
    assert report.skipped == ["old.txt: changed again after the turn"]
    assert (session.cwd / "old.txt").read_text() == "mats edited this\n"
    assert report.deleted == ["new.txt"]


def test_undo_skips_turns_that_wrote_nothing(make_session):
    session, _, _ = make_session(
        [*edit_turn(), chunks_for("just talking")],
        approver=FixedApprover(True),
        mode="bypassPermissions",
    )
    (session.cwd / "old.txt").write_text("one\n")
    asyncio.run(session.send("change things"))
    asyncio.run(session.send("chat"))
    assert session.undo().restored == ["old.txt"]


def test_panel_undo_answers_with_the_report(make_session):
    from mwm_harness.web.server import create_app
    from starlette.testclient import TestClient

    session, _, _ = make_session(edit_turn(), approver=None, mode="bypassPermissions")
    (session.cwd / "old.txt").write_text("one\n")
    app = create_app(session, {session.model.id: session.model}, "tok", 8765)
    host = {"Host": "127.0.0.1:8765"}
    with (
        TestClient(app, base_url="http://127.0.0.1:8765") as client,
        client.websocket_connect("/ws?token=tok", headers=host) as ws,
    ):

        def until(kind):
            while (m := ws.receive_json())["type"] != kind:
                pass
            return m

        until("State")
        ws.send_json({"type": "prompt", "text": "change things"})
        until("TurnEnded")
        ws.send_json({"type": "prompt", "text": "/undo"})
        out = until("CommandOutput")
        assert "restored old.txt" in out["text"] and "unknown command" not in out["text"]
    assert (session.cwd / "old.txt").read_text() == "one\n"
