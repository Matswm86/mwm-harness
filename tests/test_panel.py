"""The browser panel's server: who may connect, live events, approvals, slash commands."""

from __future__ import annotations

import pytest
from mwm_harness.providers import chunks_for
from mwm_harness.web.server import create_app
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

PORT = 8765
TOKEN = "test-token"
BASE = f"http://127.0.0.1:{PORT}"
HOST = {"Host": f"127.0.0.1:{PORT}"}  # the test client sends "testserver" on websockets


def client_for(session, models=None) -> TestClient:
    app = create_app(session, models or {session.model.id: session.model}, TOKEN, PORT)
    return TestClient(app, base_url=BASE)


def until(ws, kind: str) -> list[dict]:
    """Receive messages up to and including the first one of ``kind``."""
    seen = []
    while True:
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == kind:
            return seen


def test_the_page_is_served_and_foreign_hosts_are_refused(make_session):
    session, _, _ = make_session([])
    with client_for(session) as client:
        page = client.get("/")
        assert page.status_code == 200 and "MWM Harness" in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert TOKEN not in page.text
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 421
        assert client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize(
    ("query", "headers"),
    [
        ("", HOST),
        ("?token=wrong", HOST),
        (f"?token={TOKEN}", {"Origin": "http://evil.example", "Host": f"127.0.0.1:{PORT}"}),
        (f"?token={TOKEN}", {"Host": "evil.example"}),
    ],
)
def test_socket_needs_the_token_the_origin_and_the_host(make_session, query, headers):
    session, _, _ = make_session([])
    with client_for(session) as client, pytest.raises(WebSocketDisconnect):  # noqa: SIM117
        with client.websocket_connect(f"/ws{query}", headers=headers):
            pass


def test_tasks_tick_live_and_context_share_equals_the_usage_numbers(make_session):
    first = [{"content": "step", "status": "in_progress", "activeForm": "Stepping"}]
    second = [{"content": "step", "status": "completed"}]
    turns = [
        chunks_for(tool_calls=[("TodoWrite", {"todos": first})]),
        chunks_for(tool_calls=[("TodoWrite", {"todos": second})]),
        chunks_for("All done."),
    ]
    session, _, _ = make_session(turns)
    origin = {"Origin": BASE, "Host": f"127.0.0.1:{PORT}"}  # the test client drops Host otherwise
    with client_for(session) as client:  # noqa: SIM117
        with client.websocket_connect(f"/ws?token={TOKEN}", headers=origin) as ws:
            state = ws.receive_json()
            assert state["type"] == "State" and state["busy"] is False and state["history"] == []
            ws.send_json({"type": "prompt", "text": "go"})
            seen = until(ws, "TurnEnded")
    kinds = [m["type"] for m in seen]
    ticks = [m["todos"][0]["status"] for m in seen if m["type"] == "TodosUpdated"]
    assert ticks == ["in_progress", "completed"]
    assert kinds.index("TodosUpdated") < kinds.index("TextDelta") < kinds.index("TurnEnded")
    usage = [m for m in seen if m["type"] == "UsageUpdated"][-1]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (100, 10)
    assert usage["context_used"] == 110
    assert usage["context_fraction"] == pytest.approx(110 / session.model.context_window)
    assert usage["requests"] == 3
    assert "".join(m["text"] for m in seen if m["type"] == "TextDelta") == "All done."


def test_approval_dialog_round_trip_and_state_after_reconnect(make_session):
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "note.txt", "content": "abc"})]),
        chunks_for("Saved."),
    ]
    session, _, _ = make_session(turns)
    with client_for(session) as client:
        with client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
            ws.receive_json()
            ws.send_json({"type": "prompt", "text": "save it"})
            request = until(ws, "ApprovalRequest")[-1]
            assert request["tool"] == "Write" and request["input"]["file_path"] == "note.txt"
            ws.send_json({"type": "state"})
            assert until(ws, "State")[-1]["approvals"][0]["id"] == request["id"]
            ws.send_json({"type": "approval", "id": request["id"], "answer": "yes"})
            seen = until(ws, "TurnEnded")
            finished = next(m for m in seen if m["type"] == "ToolFinished")
            assert finished["is_error"] is False
        assert (session.cwd / "note.txt").read_text() == "abc"
        with client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
            state = ws.receive_json()
            roles = [m["role"] for m in state["history"]]
            assert roles == ["user", "assistant", "user", "assistant"]
            assert state["approvals"] == []


def test_a_declined_approval_leaves_no_file(make_session):
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "note.txt", "content": "abc"})]),
        chunks_for("Not saved."),
    ]
    session, _, _ = make_session(turns)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "save it"})
        request = until(ws, "ApprovalRequest")[-1]
        ws.send_json({"type": "approval", "id": request["id"], "answer": "no"})
        until(ws, "TurnEnded")
    assert not (session.cwd / "note.txt").exists()


def test_slash_commands_answer_into_the_page(make_session):
    session, _, _ = make_session([])
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "/mode plan"})
        seen = until(ws, "CommandOutput")
        assert any(m["type"] == "ModeChanged" and m["mode"] == "plan" for m in seen)
        assert "permission mode: plan" in seen[-1]["text"]
        ws.send_json({"type": "prompt", "text": "/help"})
        assert "/mcp" in until(ws, "CommandOutput")[-1]["text"]
