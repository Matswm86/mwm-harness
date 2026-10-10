"""Tool-call arguments reach the bus fragment by fragment while the model writes them."""

from __future__ import annotations

import asyncio
import json

from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness.providers import chunks_for


def fragmented_call(name: str, arguments: dict, pieces: int, call_id: str = "call_w") -> list[dict]:
    """One tool call streamed as ``pieces`` argument fragments, the way OpenAI-style endpoints send it."""
    raw = json.dumps(arguments)
    size = -(-len(raw) // pieces)
    parts = [raw[i : i + size] for i in range(0, len(raw), size)]
    chunks = []
    for n, part in enumerate(parts):
        head = (
            {"id": call_id, "function": {"name": name, "arguments": part}}
            if n == 0
            else {"function": {"arguments": part}}
        )
        chunks.append({"choices": [{"delta": {"tool_calls": [{"index": 0, **head}]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    chunks.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 5}})
    return chunks


def test_a_streamed_write_emits_one_delta_per_fragment_in_order(make_session):
    content = "\n".join(f"line {n}" for n in range(40))
    arguments = {"file_path": "big.txt", "content": content}
    turns = [fragmented_call("Write", arguments, 30), chunks_for("Written.")]
    session, recorder, _ = make_session(turns, approver=FixedApprover(True))
    ended = asyncio.run(session.send("write it"))
    assert ended.reason == "done"
    deltas = recorder.of(ev.ToolArgsDelta)
    assert len(deltas) == 30
    assert json.loads("".join(d.fragment for d in deltas)) == arguments
    # The id and name come with the first fragment, so every delta carries them.
    assert {(d.index, d.call_id, d.name) for d in deltas} == {(0, "call_w", "Write")}
    # All fragments arrive before the call starts running.
    started = recorder.events.index(recorder.of(ev.ToolStarted)[0])
    assert recorder.events.index(deltas[-1]) < started
    assert (session.cwd / "big.txt").read_text() == content


def test_a_plain_answer_emits_no_argument_deltas(make_session):
    session, recorder, _ = make_session([chunks_for("Hi.")])
    asyncio.run(session.send("hi"))
    assert recorder.of(ev.ToolArgsDelta) == []
