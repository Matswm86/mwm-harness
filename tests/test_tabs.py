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
