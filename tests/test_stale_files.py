"""The changed-on-disk reminder (plan item 17).

A session that read a file keeps the mtime it saw. Before each request the recorded files are
stat'ed, and anything that moved goes into one reminder at the tail of that request. The reminder
is not part of the history: it never reaches ``session.messages``, so ``/resume`` never replays it
and no gate reads it as something the person said. The request log (plan item 4) does keep its
text, because that request carried it and the log has to rebuild what the model received.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from conftest import FixedApprover
from mwm_harness.providers import chunks_for
from mwm_harness.sandbox import Sandbox
from mwm_harness.tools.base import ToolContext
from mwm_harness.transcript import load_messages

MARKER = "changed on disk since you last read them"


def run(coro):
    return asyncio.run(coro)


def sent(request: dict) -> str:
    return json.dumps(request["messages"], ensure_ascii=False)


def outside_edit(path: Path, text: str) -> None:
    """Rewrite a file the way another editor would, and make the mtime move for certain.

    The explicit stamp is here because a filesystem with coarse mtime granularity could otherwise
    let a rewrite inside one tick look unchanged, and the test would pass for the wrong reason.
    """
    path.write_text(text, encoding="utf-8")
    moved = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(moved, moved))


def project_file(session) -> Path:
    path = session.cwd / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    return path


def read_a_py() -> dict:
    return chunks_for(tool_calls=[("Read", {"file_path": "a.py"})])


def edit_a_py(old: str, new: str) -> dict:
    return chunks_for(
        tool_calls=[("Edit", {"file_path": "a.py", "old_string": old, "new_string": new})]
    )


def test_a_file_changed_on_disk_is_named_in_the_next_request_until_it_is_read_again(make_session):
    session, _, provider = make_session(
        [read_a_py(), chunks_for("read it"), read_a_py(), chunks_for("read it again")]
    )
    target = project_file(session)

    run(session.send("look at a.py"))
    assert all(MARKER not in sent(request) for request in provider.requests)

    outside_edit(target, "x = 2\n")
    run(session.send("now change it"))

    # The request that follows the outside edit carries the reminder, at the tail.
    carried = sent(provider.requests[2])
    assert MARKER in carried
    assert str(target) in carried
    tail = provider.requests[2]["messages"][-1]
    assert tail["role"] == "user"
    assert MARKER in json.dumps(tail["content"])
    # The re-read inside that same turn clears it, so the next request is clean again.
    assert MARKER not in sent(provider.requests[3])


def test_the_reminder_never_enters_the_history(make_session):
    session, _, provider = make_session([read_a_py(), chunks_for("read it"), chunks_for("done")])
    target = project_file(session)
    run(session.send("look at a.py"))
    outside_edit(target, "x = 2\n")
    run(session.send("thanks"))

    assert MARKER in sent(provider.requests[2])
    assert all(MARKER not in json.dumps(message.content) for message in session.messages)
    # The request log (plan item 4) does keep the text, as the blob of the one request that
    # carried it: model-visible means logged. What must not happen is the reminder becoming a
    # conversation entry, because then /resume would replay it and a Stop gate would read it
    # as something Mats said.
    path = session.transcript.path
    assert MARKER in path.read_text(encoding="utf-8")
    resumed = load_messages(path)
    assert resumed and all(MARKER not in json.dumps(m.content) for m in resumed)
    # The reminder changes nothing on disk either: it is request-only.
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_a_sessions_own_edit_does_not_remind_it(make_session):
    session, _, provider = make_session(
        [read_a_py(), edit_a_py("x = 1", "x = 9"), chunks_for("done")],
        approver=FixedApprover(True),
    )
    project_file(session)
    run(session.send("bump the value"))

    assert (session.cwd / "a.py").read_text(encoding="utf-8") == "x = 9\n"
    assert provider.requests, "the turn should have reached the model"
    assert all(MARKER not in sent(request) for request in provider.requests)


def test_a_second_tab_is_told_about_a_file_the_first_tab_edited(make_session):
    first, _, first_provider = make_session(
        [read_a_py(), chunks_for("read it"), edit_a_py("x = 1", "x = 5"), chunks_for("done")],
        approver=FixedApprover(True),
    )
    target = project_file(first)
    # Same project folder, its own session: this is what a second panel tab is.
    second, _, second_provider = make_session(
        [read_a_py(), chunks_for("read it"), chunks_for("done")]
    )
    assert second.cwd == first.cwd

    run(first.send("look at a.py"))
    run(second.send("look at a.py too"))
    run(first.send("bump it"))
    run(second.send("anything new?"))

    assert all(MARKER not in sent(request) for request in first_provider.requests)
    told = sent(second_provider.requests[2])
    assert MARKER in told
    assert str(target) in told


def test_a_long_list_of_changed_files_is_cut_with_a_count(make_session):
    session, _, provider = make_session([read_a_py(), chunks_for("read it"), chunks_for("done")])
    target = project_file(session)
    run(session.send("look at a.py"))

    extras = [session.cwd / f"extra{number}.py" for number in range(30)]
    for path in extras:
        path.write_text("z = 0\n", encoding="utf-8")
        session.tool_ctx.note_file(path)
    for path in [*extras, target]:
        outside_edit(path, path.read_text(encoding="utf-8") + "# touched\n")

    run(session.send("again"))
    carried = sent(provider.requests[2])
    assert MARKER in carried
    assert "and 11 more" in carried
    assert str(target) in carried
    assert "extra29.py" not in carried


def test_a_file_deleted_since_the_read_is_named_once_and_then_forgotten(tmp_path):
    ctx = ToolContext(cwd=tmp_path, scratch=tmp_path / "scratch", sandbox=Sandbox("off", []))
    gone = tmp_path / "b.py"
    gone.write_text("y = 1\n", encoding="utf-8")
    ctx.note_file(gone)
    assert ctx.stale_files() == []

    gone.unlink()
    assert ctx.stale_files() == [f"{gone} (deleted)"]
    assert ctx.stale_files() == []


def test_a_file_that_cannot_be_stat_ed_is_not_recorded(tmp_path):
    ctx = ToolContext(cwd=tmp_path, scratch=tmp_path / "scratch", sandbox=Sandbox("off", []))
    ctx.note_file(tmp_path / "never-existed.py")
    assert ctx.stamps == {}
    assert ctx.stale_files() == []
