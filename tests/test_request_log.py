"""Plan item 4: every model request can be rebuilt from the transcript alone.

The central claim is the plan's closing test: a session with one tool result over the
ObservationPack limit produces a transcript whose logged request, put back together from that
file and nothing else, is byte for byte the body the provider stub received. Everything else
here pins the properties that make that claim cheap and honest: big parts stored once, a
request line that names positions instead of copying them, and a rebuild that says what is
missing rather than inventing a request.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import MODEL
from mwm_harness import events as ev
from mwm_harness.config import ConfigError, Settings, load_settings
from mwm_harness.messages import Message, content_sha
from mwm_harness.providers import build_request, chunks_for
from mwm_harness.request_log import (
    BLOB_SUBTYPE,
    REQUEST_SUBTYPE,
    VERIFY_SUBTYPE,
    RequestLog,
    read_log,
    rebuild,
    request_lines,
    verify,
)
from mwm_harness.transcript import Transcript, load_messages

BIG = "".join(f"line {i:05d} " + "x" * 60 + "\n" for i in range(400))  # about 28 KB
OVER_BUDGET = {"prompt_tokens": 120_000, "completion_tokens": 500}  # over the 100k soft budget
TOOLS = [{"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}]


def run(coro):
    return asyncio.run(coro)


def raw_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rewrite(path: Path, lines: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    path.write_text(body, encoding="utf-8")


def blobs(path: Path, encoding: str = "") -> list[dict[str, Any]]:
    return [
        line
        for line in raw_lines(path)
        if line.get("subtype") == BLOB_SUBTYPE
        and (not encoding or line.get("encoding") == encoding)
    ]


def tool_text(body: dict[str, Any]) -> str:
    return next(m["content"] for m in body["messages"] if m["role"] == "tool")


def quiet(session):
    """The Review Desk and the assurance scan call the provider too; keep counts comparable."""
    session.settings.review_after_turn = False
    session.settings.assurance_after_turn = False
    return session


def packed_session(make_session, tmp_path: Path):
    """Four requests, the last of which carries a Read result the ObservationPack shrank."""
    turns = [
        chunks_for(tool_calls=[("Read", {"file_path": "big.txt"})]),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.txt"})]),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})]),
        chunks_for("Done."),
    ]
    session, _, provider = make_session(turns)
    quiet(session)
    (tmp_path / "project" / "big.txt").write_text(BIG, encoding="utf-8")
    assert run(session.send("read big.txt")).text == "Done."
    return session, provider


def test_every_request_rebuilds_to_the_body_the_provider_received(make_session, tmp_path):
    session, provider = packed_session(make_session, tmp_path)
    path = session.transcript.path
    lines = request_lines(path)
    assert len(lines) == len(provider.requests) == 4
    for index in range(len(lines)):
        rebuilt = rebuild(path, index)
        assert rebuilt.ok, rebuilt.reason
        assert rebuilt.body() == provider.requests[index]
    assert verify(path).ok


def test_the_packed_request_shows_the_excerpt_and_the_file_keeps_the_full_result(
    make_session, tmp_path
):
    session, provider = packed_session(make_session, tmp_path)
    path = session.transcript.path
    rebuilt = rebuild(path, -1)
    assert rebuilt.ok
    sent = tool_text(rebuilt.body())
    assert sent == tool_text(provider.requests[-1])
    assert len(sent) < 1700 and "observation-" in sent  # the excerpt is what the model saw
    line = request_lines(path)[-1]
    overrides = [part for part in line["parts"] if isinstance(part, dict) and "ref" in part]
    assert len(overrides) == 1  # named as a rewrite of a stored message, not as a copy
    full = next(
        block["content"]
        for entry in read_log(path).entries
        for block in (entry["content"] if isinstance(entry["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    assert len(full) > 20_000  # the stored result is still the whole thing


def test_repeated_parts_are_stored_once_and_a_request_line_stays_small(make_session):
    session, _, provider = make_session([chunks_for("One."), chunks_for("Two.")])
    quiet(session)
    run(session.send("first"))
    run(session.send("second"))
    path = session.transcript.path
    assert len(provider.requests) == 2
    lines = request_lines(path)
    assert len(lines) == 2
    assert all("window" in line for line in lines)  # positions, not copies of the history
    assert max(len(json.dumps(line)) for line in lines) < 700
    assert [line["content"] for line in blobs(path, "text")] == ["test system prompt"]
    parts = [json.loads(line["content"]) for line in blobs(path, "json")]
    assert sum(isinstance(part, list) for part in parts) == 1  # the tool list, once
    assert sum(isinstance(part, dict) for part in parts) == 1  # the model spec, once
    assert lines[1]["system_sha"] == lines[0]["system_sha"]


def test_verify_passes_and_then_fails_when_a_logged_message_is_edited(make_session):
    session, _, _ = make_session([chunks_for("One.")])
    quiet(session)
    run(session.send("first"))
    path = session.transcript.path
    assert verify(path).ok
    lines = raw_lines(path)
    for line in lines:
        if line.get("type") == "user" and line["message"]["content"] == "first":
            line["message"]["content"] = "first, changed after it was sent"
    rewrite(path, lines)
    failed = verify(path)
    assert not failed.ok and "the file rebuilds to" in failed.reason
    assert rebuild(path).ok  # it still rebuilds, it is just no longer what went out


def test_a_part_the_file_lost_is_named_instead_of_rebuilt_wrong(make_session):
    session, _, _ = make_session([chunks_for("One.")])
    quiet(session)
    run(session.send("first"))
    path = session.transcript.path
    lost = blobs(path, "text")
    rewrite(path, [line for line in raw_lines(path) if line not in lost])
    result = rebuild(path)
    assert not result.ok and "the system prompt is not in the file" in result.reason
    assert not verify(path).ok


def test_debug_mode_warns_when_the_file_stops_describing_the_request(make_session):
    session, recorder, _ = make_session([chunks_for("One."), chunks_for("Two.")])
    quiet(session)
    session.settings.verify_requests = True
    run(session.send("first"))
    assert [n for n in recorder.of(ev.Notice) if "does not rebuild" in n.text] == []
    path = session.transcript.path
    lines = raw_lines(path)
    for line in lines:
        if line.get("subtype") == BLOB_SUBTYPE and line.get("encoding") == "text":
            line["content"] = "a system prompt nobody ever sent"
    rewrite(path, lines)
    run(session.send("second"))
    warnings = [n.text for n in recorder.of(ev.Notice) if "request log does not rebuild" in n.text]
    assert len(warnings) == 1
    notes = [line for line in raw_lines(path) if line.get("subtype") == VERIFY_SUBTYPE]
    assert len(notes) == 1 and "the file rebuilds to" in notes[0]["content"]


def test_logging_can_be_switched_off(make_session):
    session, _, provider = make_session([chunks_for("One.")])
    quiet(session)
    session.settings.log_requests = False
    run(session.send("first"))
    path = session.transcript.path
    assert len(provider.requests) == 1  # the request still went out
    assert request_lines(path) == [] and blobs(path) == []
    assert raw_lines(path)  # the conversation itself is still written
    assert not rebuild(path).ok


def test_a_resumed_session_rebuilds_against_its_own_file(make_session):
    first, _, _ = make_session([chunks_for("Answer one.")])
    quiet(first)
    run(first.send("question one"))
    second, _, provider = make_session(
        [chunks_for("Answer two.")], resume_from=first.transcript.path
    )
    quiet(second)
    run(second.send("question two"))
    path = second.transcript.path
    assert len(request_lines(path)) == 1
    rebuilt = rebuild(path)
    assert rebuilt.ok, rebuilt.reason
    assert rebuilt.body() == provider.requests[0]
    texts = [m.content for m in rebuilt.messages if isinstance(m.content, str)]
    assert "question one" in texts and "question two" in texts


def test_the_compaction_request_is_logged_with_the_ask_it_never_stored(make_session):
    turns = [
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})], usage=OVER_BUDGET),
        chunks_for("SUMMARY: listing markdown files; next step: answer."),
        chunks_for("No markdown files here."),
    ]
    session, recorder, provider = make_session(turns)
    quiet(session)
    ended = run(session.send("list markdown files"))
    assert ended.text == "No markdown files here." and recorder.of(ev.Compacted)
    path = session.transcript.path
    lines = request_lines(path)
    assert [line["kind"] for line in lines] == ["turn", "compact", "turn"]
    rebuilt = rebuild(path, 1)
    assert rebuilt.ok, rebuilt.reason
    assert rebuilt.body() == provider.requests[1]
    assert "tools" not in rebuilt.body()  # the summary request offers no tools
    ask = lines[1]["parts"][-1]["new"]
    assert read_log(path).blobs[ask["sha"]] == rebuilt.messages[-1].content
    stored = "".join(str(entry["content"]) for entry in read_log(path).entries)
    assert str(rebuilt.messages[-1].content) not in stored


def test_a_message_only_one_request_carried_round_trips(tmp_path: Path):
    transcript = Transcript.create(tmp_path / "project", tmp_path / "sessions")
    stored = Message("user", "hello")
    transcript.append(stored)
    reminder = Message("user", "these files changed on disk", is_meta=True)
    RequestLog(transcript).record(MODEL, "sys", [stored, reminder], TOOLS, effort="high")
    rebuilt = rebuild(transcript.path)
    assert rebuilt.ok, rebuilt.reason
    assert [m.content for m in rebuilt.messages] == ["hello", "these files changed on disk"]
    assert rebuilt.messages[1].is_meta and rebuilt.messages[1].entry == -1
    assert rebuilt.effort == "high" and rebuilt.spec == MODEL
    assert rebuilt.body() == build_request(MODEL, "sys", [stored, reminder], TOOLS, "high")
    assert verify(transcript.path).ok


def test_a_request_pointing_outside_the_file_says_so(tmp_path: Path):
    transcript = Transcript.create(tmp_path / "project", tmp_path / "sessions")
    message = Message("user", "hello")
    transcript.append(message)
    RequestLog(transcript).record(MODEL, "sys", [message], TOOLS)
    lines = raw_lines(transcript.path)
    for line in lines:
        if line.get("subtype") == REQUEST_SUBTYPE:
            line["window"] = [0, 5]
    rewrite(transcript.path, lines)
    result = rebuild(transcript.path)
    assert not result.ok and "does not fit the 1 stored message(s)" in result.reason


def test_the_log_lines_stay_out_of_the_conversation(make_session):
    session, _, _ = make_session([chunks_for("One.")])
    quiet(session)
    run(session.send("first"))
    path = session.transcript.path
    assert request_lines(path) and blobs(path)
    messages = load_messages(path)
    assert len(messages) == len(session.messages)
    assert [m.text() for m in messages if m.role == "assistant"] == ["One."]
    assert not any("request 1" in str(m.content) for m in messages)


def test_verify_without_logging_is_rejected(tmp_path: Path):
    (tmp_path / "settings.toml").write_text(
        "log_requests = false\nverify_requests = true\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="log_requests"):
        load_settings(tmp_path / "settings.toml")
    assert Settings().log_requests is True and Settings().verify_requests is False


def test_a_digest_separates_text_from_structure():
    assert content_sha("[1,2]") != content_sha([1, 2])
    assert content_sha("x") == content_sha("x")
    assert content_sha({"a": 1, "b": 2}) == content_sha({"b": 2, "a": 1})
