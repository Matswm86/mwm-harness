"""ObservationPack: large tool results go in full twice, then as excerpt + archive path."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mwm_harness.messages import Message, text_block, tool_result_block, tool_use_block
from mwm_harness.observation_pack import pack_observations
from mwm_harness.providers import chunks_for

BIG = "".join(f"line {i:05d} " + "x" * 60 + "\n" for i in range(400))  # about 28 KB


def history(result: str, assistants_after: int) -> list[Message]:
    call = tool_use_block("Bash", {"command": "make test"}, "toolu_1")
    messages = [
        Message("user", "run the tests"),
        Message("assistant", [call]),
        Message("user", [tool_result_block("toolu_1", result)]),
    ]
    messages += [Message("assistant", [text_block(f"step {i}")]) for i in range(assistants_after)]
    return messages


def result_of(messages: list[Message]) -> str:
    return messages[2].content[0]["content"]


def test_result_is_sent_in_full_for_two_requests(tmp_path: Path):
    for after in (0, 1):
        packed, errors = pack_observations(history(BIG, after), tmp_path)
        assert result_of(packed) == BIG and errors == []
    assert list(tmp_path.iterdir()) == []


def test_old_large_result_becomes_excerpt_and_exact_archive(tmp_path: Path):
    original = history(BIG, 2)
    packed, errors = pack_observations(original, tmp_path)
    text = result_of(packed)
    assert errors == []
    assert len(text) < 1400
    assert text.count("line 00000") == 1 and "line 00399" in text
    archive = next(tmp_path.iterdir())
    assert str(archive) in text
    assert archive.read_text(encoding="utf-8") == BIG
    assert result_of(original) == BIG  # the stored history is untouched


def test_small_results_and_plain_text_are_never_packed(tmp_path: Path):
    packed, _ = pack_observations(history("ok\n" * 10, 5), tmp_path)
    assert result_of(packed) == "ok\n" * 10
    plain = [Message("user", BIG), Message("assistant", "a"), Message("assistant", "b")]
    assert pack_observations(plain, tmp_path)[0][0].content == BIG


def test_packed_text_is_stable_between_requests(tmp_path: Path):
    first, _ = pack_observations(history(BIG, 2), tmp_path)
    second, _ = pack_observations(history(BIG, 3), tmp_path)
    assert result_of(first) == result_of(second)
    assert len(list(tmp_path.iterdir())) == 1


def test_archive_failure_sends_the_result_in_full(tmp_path: Path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    packed, errors = pack_observations(history(BIG, 2), blocker / "scratch")
    assert result_of(packed) == BIG
    assert len(errors) == 1


def tool_message(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "tool")


def test_session_packs_on_the_third_request_and_archive_is_readable(make_session, tmp_path):
    turns = [
        chunks_for(tool_calls=[("Read", {"file_path": "big.txt"})]),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.txt"})]),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})]),
        chunks_for("Done."),
    ]
    session, _, provider = make_session(turns)
    (tmp_path / "project" / "big.txt").write_text(BIG, encoding="utf-8")

    ended = asyncio.run(session.send("read big.txt"))

    assert ended.text == "Done."
    sizes = [len(tool_message(r)) for r in provider.requests[1:]]
    assert sizes[0] == sizes[1] > 20_000  # requests 2 and 3 carry the full Read output
    assert sizes[2] < 1700  # request 4 carries the 1 KB excerpt, a header and the path
    stored = session.messages[2].content[0]["content"]
    assert len(stored) > 20_000  # the session keeps the full result
    archive = next(session.tool_ctx.scratch.glob("observation-*.txt"))
    assert archive.read_text(encoding="utf-8") == stored
