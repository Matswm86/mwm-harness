"""The browser panel's server: who may connect, live events, approvals, slash commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mwm_harness.providers import chunks_for
from mwm_harness.web.server import PANEL_SAVE, create_app
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


def test_state_lists_mcp_servers_for_the_rail(make_session):
    from mwm_harness.mcp_client import McpManager, ServerConfig
    from mwm_harness.web.server import mcp_servers

    manager = McpManager({"brain": ServerConfig("brain", "http", url="http://127.0.0.1:1/mcp")})
    manager.listings["brain"] = [{"name": "search"}, {"name": "ingest"}]
    assert mcp_servers(manager) == [
        {"name": "brain", "transport": "http", "tools": 2, "state": "idle", "error": ""}
    ]
    manager.servers["brain"].error = "HTTP 500"
    assert mcp_servers(manager)[0]["state"] == "failed"
    assert mcp_servers(None) == []
    session, _, _ = make_session([])
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        state = until(ws, "State")[-1]
        assert state["mcp"] == []


def test_the_page_can_ask_for_chart_bars(make_session):
    session, _, _ = make_session([])
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)

    async def fake_bars(symbol, minutes):
        return {"type": "Bars", "symbol": symbol, "minutes": minutes, "bars": [{"time": 1}]}

    app.state.panel.market.bars = fake_bars
    with (
        TestClient(app, base_url=BASE) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "Tabs")
        ws.send_json({"type": "bars", "symbol": "MNQ", "minutes": 5})
        bars = until(ws, "Bars")[-1]
        assert bars["symbol"] == "MNQ" and bars["minutes"] == 5 and "tab" not in bars


def test_the_mic_clip_is_transcribed_and_bad_clips_are_refused(make_session):
    import base64

    from mwm_harness.voice import Transcript

    session, _, _ = make_session([])
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)
    heard = []

    async def fake_transcribe(blob):
        heard.append(blob)
        return Transcript("open a new tab", "en", 0.4, "medium on cuda")

    app.state.panel.voice.transcribe = fake_transcribe
    with (
        TestClient(app, base_url=BASE) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "Tabs")
        ws.send_json({"type": "transcribe", "audio": base64.b64encode(b"webm-bytes").decode()})
        result = until(ws, "Transcript")[-1]
        assert result["text"] == "open a new tab" and result["model"] == "medium on cuda"
        assert heard == [b"webm-bytes"]
        ws.send_json({"type": "transcribe", "audio": "not base64!"})
        assert "base64" in until(ws, "Transcript")[-1]["error"]
        ws.send_json({"type": "transcribe"})
        assert "no audio" in until(ws, "Transcript")[-1]["error"]


def test_stale_sources_are_named_and_the_strip_carries_the_loaded_settings(make_session):
    session, _, _ = make_session([])
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)
    panel = app.state.panel
    assert panel.staleness() == []  # nothing changed since this panel started
    panel.started_at = 0  # pretend the process started before the package existed
    assert "mwm_harness/web/server.py" in panel.staleness()
    with (
        TestClient(app, base_url=BASE) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        state = until(ws, "State")[-1]
    assert state["commit"] == panel.commit
    assert state["sandbox_mode"] == session.settings.sandbox
    assert state["shell_network"] == session.settings.shell_network
    assert "mwm_harness/web/server.py" in state["stale"]


def test_check_results_reach_the_page_and_stay_in_the_tab_state(make_session):
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "bad.py", "content": "def f(:\n"})]),
        chunks_for("Written."),
    ]
    session, _, _ = make_session(turns, mode="acceptEdits")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "prompt", "text": "write it"})
        seen = until(ws, "TurnEnded")
        finished = next(m for m in seen if m["type"] == "ChecksFinished")
        syntax = next(r for r in finished["results"] if r["check"] == "py_compile")
        assert syntax["findings"][0]["file"] == "bad.py"
        ws.send_json({"type": "state"})
        state = until(ws, "State")[-1]
        assert state["checks"]["results"] == finished["results"]
        assert state["runit"] is None


REVIEW_ANSWER = json.dumps(
    {
        "summary": "One new file: a helper that can hand back None.",
        "files": [
            {"path": "a.py", "walkthrough": "adds helper(), which returns None for an empty list"}
        ],
        "comments": [
            {
                "file": "a.py",
                "line": 2,
                "severity": "critical",
                "text": "helper() returns None, and the caller passes that straight on",
                "fix": "return an empty list instead",
            }
        ],
    }
)


def test_the_review_desk_reaches_the_page_and_its_buttons_answer(make_session):
    source = "def helper(items):\n    return items[0]\n"
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "a.py", "content": source})]),
        chunks_for("Written."),
        chunks_for(REVIEW_ANSWER),
        chunks_for("Fixed it."),
    ]
    session, _, provider = make_session(turns, mode="acceptEdits")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "prompt", "text": "write it"})
        seen = until(ws, "TurnEnded")
        started = next(m for m in seen if m["type"] == "ReviewStarted")
        assert started["files"] == ["a.py"] and started["model"]
        payload = next(m for m in seen if m["type"] == "ReviewFinished")["review"]
        assert payload["status"] == "reviewed"
        assert payload["files"][0]["walkthrough"].startswith("adds helper()")
        assert payload["comments"][0]["severity"] == "critical"
        kinds = [m["type"] for m in seen]
        assert kinds.index("ReviewFinished") < kinds.index("TurnEnded")

        ws.send_json({"type": "state"})
        assert until(ws, "State")[-1]["review"] == payload  # a reconnect redraws the same desk

        ws.send_json({"type": "review_comment", "number": 1, "queue": True})
        assert until(ws, "ReviewQueued")[-1]["text"].startswith(
            "Review comment on a.py:2 (critical,"
        )

        ws.send_json({"type": "review_comment", "number": 1, "remember": True})
        assert until(ws, "ReviewSaved")[-1]["path"].endswith(".mwm/review-learnings.md")
        assert (session.cwd / ".mwm" / "review-learnings.md").is_file()

        ws.send_json({"type": "review_comment", "number": 4})
        assert "no comment 4" in until(ws, "ReviewError")[-1]["text"]

        ws.send_json({"type": "review_comment", "number": 1})
        until(ws, "TurnEnded")
    prompts = [m["content"] for m in provider.requests[-1]["messages"] if m["role"] == "user"]
    assert prompts[-1].startswith("Review comment on a.py:2 (critical,")
    assert (session.cwd / "a.py").read_text() == source  # advice only: the review wrote nothing


# ------------------------------------------------------------- saving from the code pane
#
# Mats fixes a line himself in the browser. A save is a file write, so it goes through
# permissions.decide and, in a mode that asks, through the same approval card an agent's Write
# gets. It does not re-stamp the session, so the agent is told the file changed on disk.

MARKER = "changed on disk since you last read them"


def saved_file(session, name: str = "a.py", text: str = "x = 1\n") -> Path:
    path = session.cwd / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_code_pane_save_asks_like_an_edit_and_tells_the_agent(make_session):
    turns = [
        chunks_for(tool_calls=[("Read", {"file_path": "a.py"})]),
        chunks_for("Read it."),
        chunks_for("Noted."),
    ]
    session, _, provider = make_session(turns)
    saved_file(session)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "prompt", "text": "read it"})
        until(ws, "TurnEnded")
        path = session.tool_ctx.resolve("a.py")
        stamp = session.tool_ctx.stamps[path]

        ws.send_json({"type": "save", "path": "a.py", "content": "x = 2\n"})
        request = until(ws, "ApprovalRequest")[-1]
        assert request["tool"] == "Write" and request["input"]["content"] == "x = 2\n"
        assert "code pane" in request["reason"]
        assert request["diff"]["before"] == "x = 1\n" and request["diff"]["after"] == "x = 2\n"
        ws.send_json({"type": "approval", "id": request["id"], "answer": "yes"})
        result = until(ws, "SaveResult")[-1]
        assert result["saved"] is True and result["content"] == "x = 2\n"
        assert path.read_text() == "x = 2\n"
        # The save left the session's stamp where the read put it, and the file really did move:
        # the reminder below fires because a save is not a read.
        assert session.tool_ctx.stamps[path] == stamp
        assert path.stat().st_mtime_ns != stamp

        calls = len(provider.requests)
        ws.send_json({"type": "prompt", "text": "and now"})
        until(ws, "TurnEnded")
    body = json.dumps(provider.requests[calls]["messages"], ensure_ascii=False)
    assert MARKER in body and "a.py" in body


def test_a_declined_code_pane_save_writes_nothing(make_session):
    session, _, _ = make_session([])
    target = saved_file(session)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": "a.py", "content": "x = 2\n"})
        request = until(ws, "ApprovalRequest")[-1]
        ws.send_json({"type": "approval", "id": request["id"], "answer": "no"})
        result = until(ws, "SaveResult")[-1]
        assert result["saved"] is False and "nothing was written" in result["error"]
    assert target.read_text() == "x = 1\n"


def test_a_save_outside_the_project_is_refused_without_asking(make_session):
    session, _, _ = make_session([])
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": "../escape.txt", "content": "x"})
        seen = until(ws, "SaveResult")
        assert seen[-1]["saved"] is False and "outside the project" in seen[-1]["error"]
        assert not any(m["type"] == "ApprovalRequest" for m in seen)
    assert not (session.cwd.parent / "escape.txt").exists()


def test_a_save_over_key_material_is_denied_by_the_hard_list(make_session):
    session, _, _ = make_session([])
    saved_file(session, ".ssh/id_rsa", "original key\n")
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": ".ssh/id_rsa", "content": "attacker key\n"})
        seen = until(ws, "SaveResult")
        assert seen[-1]["saved"] is False and "secret-files" in seen[-1]["error"]
        assert not any(m["type"] == "ApprovalRequest" for m in seen)
    assert (session.cwd / ".ssh" / "id_rsa").read_text() == "original key\n"


def test_acceptEdits_saves_without_a_prompt_and_plan_mode_refuses(make_session):
    session, _, _ = make_session([], mode="acceptEdits")
    target = saved_file(session)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": "a.py", "content": "x = 3\n"})
        seen = until(ws, "SaveResult")
        assert seen[-1]["saved"] is True
        assert not any(m["type"] == "ApprovalRequest" for m in seen)
    assert target.read_text() == "x = 3\n"

    planned, _, _ = make_session([], mode="plan")
    kept = saved_file(planned)
    with (
        client_for(planned) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": "a.py", "content": "x = 4\n"})
        result = until(ws, "SaveResult")[-1]
        assert result["saved"] is False and "Plan mode" in result["error"]
    assert kept.read_text() == "x = 1\n"


def test_always_on_a_save_covers_saves_and_not_the_agents_writes(make_session):
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "b.txt", "content": "agent"})]),
        chunks_for("Wrote it."),
    ]
    session, _, _ = make_session(turns)
    saved_file(session)
    with (
        client_for(session) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "State")
        ws.send_json({"type": "save", "path": "a.py", "content": "x = 5\n"})
        first = until(ws, "ApprovalRequest")[-1]
        ws.send_json({"type": "approval", "id": first["id"], "answer": "always"})
        assert until(ws, "SaveResult")[-1]["saved"] is True
        assert PANEL_SAVE in session.permissions.session_allow
        assert "Write" not in session.permissions.session_allow

        ws.send_json({"type": "save", "path": "a.py", "content": "x = 6\n"})
        second = until(ws, "SaveResult")
        assert second[-1]["saved"] is True
        assert not any(m["type"] == "ApprovalRequest" for m in second)

        ws.send_json({"type": "prompt", "text": "write b.txt"})
        asked = until(ws, "ApprovalRequest")[-1]  # the agent's own Write still asks
        assert asked["tool"] == "Write" and asked["input"]["file_path"] == "b.txt"
        ws.send_json({"type": "approval", "id": asked["id"], "answer": "no"})
        until(ws, "TurnEnded")
    assert (session.cwd / "a.py").read_text() == "x = 6\n"
    assert not (session.cwd / "b.txt").exists()
