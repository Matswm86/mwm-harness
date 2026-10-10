"""Handoffs rail: newest handoffs, read-only memory files, and archive search over the websocket."""

from __future__ import annotations

import os
import time

import pytest
from mwm_harness import memory_panel
from mwm_harness.memory_panel import OutsideMemory, list_handoffs, read_memory
from mwm_harness.web.server import create_app
from starlette.testclient import TestClient

PORT = 8765
TOKEN = "test-token"
HOST = {"Host": f"127.0.0.1:{PORT}"}


@pytest.fixture
def memory(tmp_path, monkeypatch):
    root = tmp_path / "MWM"
    mem = root / "memory"
    mem.mkdir(parents=True)
    old = mem / "handoff-old-stream.md"
    old.write_text(
        "---\nname: x\ndescription: Old stream\n---\n\n# Old\n\n## 2026-09-01 10:00: started\n\nbody\n"
    )
    new = mem / "handoff-new-stream.md"
    new.write_text(
        "---\ndescription: New stream\n---\n\n## 2026-10-09 12:00: START HERE. shipped\n\n- done\n"
    )
    (mem / "notes.md").write_text("not a handoff")
    past = time.time() - 86400
    os.utime(old, (past, past))
    monkeypatch.setenv("MWM_WORKSPACE", str(root))
    return mem


def test_handoffs_are_newest_first_with_their_top_section(memory):
    items = list_handoffs()
    assert [i["name"] for i in items] == ["new-stream", "old-stream"]
    assert items[0]["latest"] == "2026-10-09 12:00: START HERE. shipped"
    assert items[0]["date"] == "2026-10-09"
    assert items[0]["description"] == "New stream"


def test_memory_files_open_read_only_and_never_outside_memory(memory, tmp_path):
    (tmp_path / "MWM" / "secret.md").write_text("no")
    assert read_memory("memory/handoff-new-stream.md")["content"].startswith("---")
    assert read_memory("ghost.md")["error"] == "no such file"
    for bad in ("../secret.md", "/etc/passwd", "handoff-new-stream.txt"):
        with pytest.raises(OutsideMemory):
            read_memory(bad)


def test_panel_answers_handoffs_search_and_open(memory, make_session, monkeypatch):
    async def fake_search(self, query, k=8):
        return [
            {
                "path": "handoff-new-stream.md",
                "heading": "h",
                "date": "2026-10-09",
                "age_days": 1,
                "snippet": query,
            }
        ]

    monkeypatch.setattr(memory_panel.MemorySearch, "search", fake_search)
    session, _, _ = make_session([])
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as client:  # noqa: SIM117
        with client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:

            def reply(message, kind):
                ws.send_json(message)
                while True:
                    got = ws.receive_json()
                    if got["type"] == kind:
                        return got

            listed = reply({"type": "handoffs"}, "Handoffs")
            assert [i["name"] for i in listed["items"]] == ["new-stream", "old-stream"]
            hits = reply({"type": "memory_search", "query": "what shipped"}, "MemoryHits")
            assert hits["hits"][0]["snippet"] == "what shipped"
            opened = reply({"type": "memory_open", "path": "handoff-old-stream.md"}, "FileContent")
            assert (
                opened["path"] == "memory/handoff-old-stream.md" and "started" in opened["content"]
            )
            refused = reply({"type": "memory_open", "path": "../../etc/passwd"}, "FileContent")
            assert refused["error"] == "outside memory/"
