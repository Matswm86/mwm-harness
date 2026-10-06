"""Panel tabs: several sessions in one panel, each with its own transcript and approvals."""

from __future__ import annotations

from mwm_harness.providers import chunks_for
from mwm_harness.web.server import create_app
from starlette.testclient import TestClient

PORT = 8765
TOKEN = "test-token"
HOST = {"Host": f"127.0.0.1:{PORT}"}


def until(ws, kind: str, tab: str | None = None) -> list[dict]:
    """Receive messages up to and including the first one of ``kind`` (for ``tab``, if given)."""
    seen = []
    while True:
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == kind and (tab is None or message.get("tab") == tab):
            return seen


def two_tab_client(make_session, first_turns, second_turns, max_tabs: int = 6):
    first, _, _ = make_session(first_turns)
    second, _, _ = make_session(second_turns)
    first.settings.max_tabs = max_tabs
    spare = [second]
    app = create_app(
        first,
        {first.model.id: first.model},
        TOKEN,
        PORT,
        factory=lambda cwd, model: spare.pop(),
    )
    return first, second, TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def test_a_second_tab_streams_while_the_first_waits_for_approval(make_session):
    first, second, client = two_tab_client(
        make_session,
        [
            chunks_for(tool_calls=[("Write", {"file_path": "one.txt", "content": "1"})]),
            chunks_for("Wrote it."),
        ],
        [chunks_for("Second tab answer.")],
    )
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        tabs = until(ws, "Tabs")[-1]
        assert [t["id"] for t in tabs["tabs"]] == ["t1"]
        ws.send_json({"type": "tab_new"})
        opened = until(ws, "Tabs")[-1]
        assert opened["active"] == "t2" and len(opened["tabs"]) == 2

        ws.send_json({"type": "prompt", "tab": "t1", "text": "write one"})
        request = until(ws, "ApprovalRequest", "t1")[-1]
        waiting = until(ws, "Tabs")[-1]
        assert [t["waiting"] for t in waiting["tabs"]] == [True, False]

        ws.send_json({"type": "prompt", "tab": "t2", "text": "hello"})
        seen = until(ws, "TurnEnded", "t2")
        text = "".join(m["text"] for m in seen if m["type"] == "TextDelta" and m["tab"] == "t2")
        assert text == "Second tab answer."
        assert not [m for m in seen if m["type"] == "TextDelta" and m["tab"] == "t1"]
        before = second.transcript.path.read_bytes()

        ws.send_json({"type": "approval", "tab": "t1", "id": request["id"], "answer": "yes"})
        until(ws, "TurnEnded", "t1")
        assert second.transcript.path.read_bytes() == before
    assert (first.cwd / "one.txt").read_text() == "1"
    assert first.transcript.path != second.transcript.path


def test_closing_a_tab_keeps_the_others_and_the_last_tab_stays(make_session):
    _, second, client = two_tab_client(make_session, [], [])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "tab_new"})
        until(ws, "Tabs")
        ws.send_json({"type": "tab_close", "tab": "t2"})
        until(ws, "TabClosed")
        assert [t["id"] for t in until(ws, "Tabs")[-1]["tabs"]] == ["t1"]
        ws.send_json({"type": "tab_close", "tab": "t1"})
        notice = until(ws, "Notice")[-1]
        assert "last tab" in notice["text"]
        ws.send_json({"type": "tabs"})
        assert [t["id"] for t in until(ws, "Tabs")[-1]["tabs"]] == ["t1"]


def test_the_tab_cap_refuses_one_more_tab(make_session):
    _, _, client = two_tab_client(make_session, [], [], max_tabs=1)
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "tab_new"})
        assert "max_tabs" in until(ws, "Notice")[-1]["text"]


def test_a_message_without_a_tab_goes_to_the_first_tab(make_session):
    first, _, client = two_tab_client(make_session, [chunks_for("First.")], [])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "text": "hi"})
        ended = until(ws, "TurnEnded")[-1]
        assert ended["tab"] == "t1" and ended["text"] == "First."


def test_the_tab_list_sent_with_turn_ended_shows_the_tab_idle(make_session):
    first, _, client = two_tab_client(make_session, [chunks_for("Done.")], [])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "text": "hi", "tab": "t1"})
        until(ws, "TurnEnded", "t1")
        tabs = until(ws, "Tabs")[-1]["tabs"]
        assert tabs[0]["busy"] is False


def test_switching_to_a_tab_mid_turn_replays_the_running_turn(make_session):
    first, _, client = two_tab_client(
        make_session,
        [
            chunks_for(
                "Writing now.", tool_calls=[("Write", {"file_path": "a.txt", "content": "1"})]
            )
        ],
        [],
    )
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "tab": "t1", "text": "write a"})
        until(ws, "ApprovalRequest", "t1")
        ws.send_json({"type": "state", "tab": "t1"})
        state = until(ws, "State", "t1")[-1]
        assert state["history"] == []  # the running turn is not finished history yet
        kinds = [e["type"] for e in state["live"]]
        assert kinds[0] == "TurnStarted" and state["live"][0]["prompt"] == "write a"
        assert {"type": "TextDelta", "text": "Writing now."} in state["live"]
        assert "ToolStarted" in kinds
        ws.send_json({"type": "cancel", "tab": "t1"})
        until(ws, "TurnEnded", "t1")
        ws.send_json({"type": "state", "tab": "t1"})
        after = until(ws, "State", "t1")[-1]
        assert after["live"] == [] and after["history"][0]["blocks"][0]["text"] == "write a"


def test_a_closed_tab_reopens_with_its_transcript(make_session):
    first, _, _ = make_session([chunks_for("Hi there.")])
    app = create_app(first, {first.model.id: first.model}, TOKEN, PORT)  # tabs use Session.sibling
    with (
        TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "Tabs")
        ws.send_json({"type": "tab_new", "label": "side"})
        assert until(ws, "Tabs")[-1]["active"] == "t2"
        ws.send_json({"type": "prompt", "tab": "t2", "text": "hello"})
        until(ws, "TurnEnded", "t2")
        ws.send_json({"type": "tab_close", "tab": "t2"})
        until(ws, "TabClosed")
        closed = until(ws, "Tabs")[-1]
        assert closed["closed"] == ["side"]
        ws.send_json({"type": "tab_reopen"})
        state = until(ws, "State")[-1]
        texts = [b["text"] for m in state["history"] for b in m["blocks"] if b["type"] == "text"]
        assert state["label"] == "side" and texts == ["hello", "Hi there."]
        assert until(ws, "Tabs")[-1]["closed"] == []


def test_two_tabs_on_one_local_model_take_turns(make_session):
    from mwm_harness.config import ModelSpec

    local = ModelSpec(id="local-4b", base_url="http://127.0.0.1:11434/v1", key_env="NONE")
    first, second, client = two_tab_client(
        make_session,
        [
            chunks_for(tool_calls=[("Write", {"file_path": "one.txt", "content": "1"})]),
            chunks_for("One."),
        ],
        [chunks_for("Two.")],
    )
    first.model = second.model = local
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "tab_new"})
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "tab": "t1", "text": "write"})
        request = until(ws, "ApprovalRequest", "t1")[-1]
        ws.send_json({"type": "prompt", "tab": "t2", "text": "hi"})
        notice = until(ws, "Notice", "t2")[-1]
        assert "waiting for the local model local-4b" in notice["text"]
        ws.send_json({"type": "approval", "tab": "t1", "id": request["id"], "answer": "yes"})
        seen = until(ws, "TurnEnded", "t2")
        ended_t1 = [m for m in seen if m["type"] == "TurnEnded" and m["tab"] == "t1"]
        assert ended_t1, "tab 2 ran only after tab 1 let go of the model"


def test_a_hosted_model_needs_no_lock(make_session):
    first, _, client = two_tab_client(make_session, [], [])
    panel = client.app.state.panel
    assert panel.model_lock(first.model) is None  # http://unused.invalid/v1 is not this machine


def test_an_isolated_tab_gets_its_own_worktree_and_a_plain_folder_is_refused(tmp_path):
    import asyncio
    import subprocess

    from mwm_harness.web.server import WorktreeError, make_worktree

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    folder, branch = asyncio.run(make_worktree(repo / "src", tmp_path / "trees", 3))
    assert folder.name == "src" and (folder / "a.py").read_text() == "x = 1\n"
    assert branch.startswith("mwm/tab3-")
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True
    ).stdout
    assert str(folder.parent) in listed
    plain = tmp_path / "plain"
    plain.mkdir()
    try:
        asyncio.run(make_worktree(plain, tmp_path / "trees", 4))
    except WorktreeError as exc:
        assert "git" in str(exc).lower() or "repository" in str(exc).lower()
    else:
        raise AssertionError("a folder outside git got a worktree")


def test_cancelling_one_tab_leaves_the_other_tabs_shell_running(make_session):
    first, second, client = two_tab_client(
        make_session,
        [chunks_for(tool_calls=[("Bash", {"command": "sleep 30"})]), chunks_for("x")],
        [
            chunks_for(tool_calls=[("Bash", {"command": "sleep 1; echo second-finished"})]),
            chunks_for("Done."),
        ],
    )
    first.permissions.mode = second.permissions.mode = "bypassPermissions"
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "tab_new"})
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "tab": "t1", "text": "long"})
        until(ws, "ToolStarted", "t1")
        ws.send_json({"type": "prompt", "tab": "t2", "text": "short"})
        until(ws, "ToolStarted", "t2")
        ws.send_json({"type": "cancel", "tab": "t1"})
        seen = until(ws, "TurnEnded", "t2")
        finished = [m for m in seen if m["type"] == "ToolFinished" and m["tab"] == "t2"]
        assert (
            finished and "second-finished" in finished[0]["content"] and not finished[0]["is_error"]
        )
        assert [m["reason"] for m in seen if m["type"] == "TurnEnded" and m["tab"] == "t1"] == [
            "cancelled"
        ]


def test_slash_tab_in_the_panel_lists_and_opens_tabs(make_session):
    _, _, client = two_tab_client(make_session, [], [])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "tab": "t1", "text": "/tab new"})
        assert until(ws, "Tabs")[-1]["active"] == "t2"
        ws.send_json({"type": "prompt", "tab": "t2", "text": "/tab"})
        rows = until(ws, "CommandOutput")[-1]["text"].splitlines()
        assert len(rows) == 2 and rows[1].startswith("* 2")
