"""Diff preview, project-only file browsing, and the panel's diff-before-approval."""

from __future__ import annotations

import pytest
from mwm_harness import events as ev
from mwm_harness.preview import OutsideProject, inside, list_dir, preview_change, read_file
from mwm_harness.providers import chunks_for
from mwm_harness.repl.terminal import Printer, run_command
from test_panel import HOST, TOKEN, client_for, until

ORIGINAL = "first line\r\nsecond line ø\r\nlast line without newline".encode()


def test_preview_of_write_and_edit_does_not_touch_the_disk(make_session):
    session, _, _ = make_session([])
    target = session.cwd / "a.txt"
    target.write_bytes(b"one\ntwo\ntwo\n")
    ctx = session.tool_ctx
    new = preview_change("Write", {"file_path": "fresh.txt", "content": "hi\n"}, ctx)
    assert (
        new.is_new and new.before == "" and "+hi" in new.unified() and "/dev/null" in new.unified()
    )
    edit = preview_change(
        "Edit", {"file_path": "a.txt", "old_string": "one", "new_string": "1"}, ctx
    )
    assert edit.after == "1\ntwo\ntwo\n" and "-one\n+1\n" in edit.unified() and edit.path == "a.txt"
    ambiguous = {"file_path": "a.txt", "old_string": "two", "new_string": "2"}
    assert preview_change("Edit", ambiguous, ctx) is None
    assert preview_change("Edit", {**ambiguous, "replace_all": True}, ctx).after == "one\n2\n2\n"
    assert preview_change("Bash", {"command": "ls"}, ctx) is None
    assert target.read_bytes() == b"one\ntwo\ntwo\n" and not (session.cwd / "fresh.txt").exists()


def test_browsing_never_leaves_the_project(tmp_path):
    project, secret = tmp_path / "project", tmp_path / "secret.txt"
    (project / "src").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "src" / "main.py").write_text("print(1)\n")
    (project / "blob.bin").write_bytes(b"\xff\xfe\x00")
    secret.write_text("key")
    (project / "leak").symlink_to(secret)
    assert [e["name"] for e in list_dir(project)] == [
        "src",
        "blob.bin",
    ]  # .git and the symlink are hidden
    assert list_dir(project, "src") == [{"name": "main.py", "path": "src/main.py", "dir": False}]
    assert read_file(project, "src/main.py")["content"] == "print(1)\n"
    assert read_file(project, "blob.bin")["error"] == "binary file"
    for escape in ("../secret.txt", "leak", "/etc/passwd", "src/../../secret.txt"):
        with pytest.raises(OutsideProject):
            inside(project, escape)
        with pytest.raises(OutsideProject):
            read_file(project, escape)


def edit_turns() -> list:
    edit = {"file_path": "doc.txt", "old_string": "second line ø", "new_string": "SECOND"}
    return [
        chunks_for(tool_calls=[("Read", {"file_path": "doc.txt"})]),
        chunks_for(tool_calls=[("Edit", edit)]),
        chunks_for("ok"),
    ]


def test_rejecting_a_diff_leaves_the_file_byte_identical(make_session):
    session, recorder, _ = make_session(edit_turns())
    target = session.cwd / "doc.txt"
    target.write_bytes(ORIGINAL)
    stamp = target.stat().st_mtime_ns
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "edit it"})
        request = until(ws, "ApprovalRequest")[-1]
        diff = request["diff"]
        assert diff["path"] == "doc.txt" and diff["is_new"] is False
        assert "-second line ø" in diff["unified"] and "+SECOND" in diff["unified"]
        assert diff["before"].encode() != diff["after"].encode()
        assert target.read_bytes() == ORIGINAL  # showing the diff wrote nothing
        ws.send_json({"type": "approval", "id": request["id"], "answer": "no"})
        until(ws, "TurnEnded")
    assert target.read_bytes() == ORIGINAL
    assert target.stat().st_mtime_ns == stamp
    assert recorder.of(ev.FilesTouched)[-1].paths == ["doc.txt"]  # read, never written


def test_approving_the_diff_writes_exactly_the_previewed_text(make_session):
    session, _, _ = make_session(edit_turns())
    target = session.cwd / "doc.txt"
    target.write_text("first\nsecond line ø\nlast\n", encoding="utf-8")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "edit it"})
        request = until(ws, "ApprovalRequest")[-1]
        ws.send_json({"type": "approval", "id": request["id"], "answer": "yes"})
        until(ws, "TurnEnded")
    assert target.read_text(encoding="utf-8") == request["diff"]["after"] == "first\nSECOND\nlast\n"


def test_tree_and_open_over_the_socket(make_session):
    session, _, _ = make_session([])
    (session.cwd / "pkg").mkdir()
    (session.cwd / "pkg" / "mod.py").write_text("x = 1\n")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "tree", "path": ""})
        assert until(ws, "Tree")[-1]["entries"] == [{"name": "pkg", "path": "pkg", "dir": True}]
        ws.send_json({"type": "open", "path": "pkg/mod.py"})
        assert until(ws, "FileContent")[-1]["content"] == "x = 1\n"
        ws.send_json({"type": "prompt", "text": "/open ../../etc/passwd"})
        assert until(ws, "FileContent")[-1]["error"] == "outside the project"


def test_files_and_open_in_the_terminal(make_session, capsys):
    session, _, _ = make_session(
        [chunks_for(tool_calls=[("Read", {"file_path": "n.txt"})]), chunks_for("k")]
    )
    (session.cwd / "n.txt").write_text("alpha\nbeta\n")
    import asyncio

    asyncio.run(session.send("read"))
    out = Printer(color=False)
    run_command(session, {}, "/files", out)
    run_command(session, {}, "/open n.txt", out)
    run_command(session, {}, "/open ../x", out)
    printed = capsys.readouterr().out
    assert (
        "1 files touched" in printed
        and "    2  beta" in printed
        and "outside the project" in printed
    )
