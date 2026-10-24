"""Item 1: two-step compaction. Step 1 prunes old tool output, step 2 summarises.

The plan's closing test for this item is: a scripted session with 12 tool calls over
the soft budget where step 1 alone gets under budget and NO summary request is sent,
plus the case where pruning is not enough and exactly one summary request follows.
Everything here is offline: ``ScriptedProvider`` plays the model, so a test that asserts
``provider.requests == []`` really proves no call was made.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import write_hook
from mwm_harness import compaction
from mwm_harness import events as ev
from mwm_harness.compaction import PruneResult, prune_tool_results
from mwm_harness.config import ConfigError, ModelSpec, Settings, is_local_endpoint, load_settings
from mwm_harness.messages import (
    Message,
    image_block,
    tool_result_block,
    tool_use_block,
)
from mwm_harness.providers import chunks_for
from mwm_harness.transcript import load_messages

BIG = {"prompt_tokens": 120_000, "completion_tokens": 500}  # over the 100k soft budget


def run(coro):
    return asyncio.run(coro)


def history(rounds: int, chars: int) -> list[Message]:
    """A hand-built transcript: one prompt, then ``rounds`` tool calls and their results."""
    messages = [Message("user", "do the thing")]
    for index in range(rounds):
        messages.append(
            Message(
                "assistant",
                [tool_use_block("Bash", {"command": f"run step {index}"}, f"call{index}")],
            )
        )
        messages.append(Message("user", [tool_result_block(f"call{index}", "x" * chars)]))
    return messages


def contents(messages: list[Message]) -> list[str]:
    """Every tool result's text, oldest first."""
    out = []
    for message in messages:
        for block in message.blocks():
            if block.get("type") == "tool_result":
                content = block.get("content")
                out.append(content if isinstance(content, str) else str(content))
    return out


# ---------------------------------------------------------------- step 1, the function


def test_a_result_older_than_the_window_becomes_one_line_naming_the_call(tmp_path):
    messages = history(4, 5_000)
    result = prune_tool_results(messages, keep_turns=2, scratch=tmp_path / "scratch")
    assert result.stubbed == 2  # the two oldest rounds only
    kept = result.messages
    first = contents(kept)[0]
    assert first.startswith("[pruned] Bash(") and '"run step 0"' in first
    assert "5,000 chars" in first and "Re-run the call to see it again." in first
    assert len(first) < 200
    assert result.chars_saved > 2 * 4_000
    # the input list and its messages are not modified
    assert len(contents(messages)[0]) == 5_000 and result.messages is not messages


def test_a_large_result_is_archived_and_the_stub_names_the_file(tmp_path):
    scratch = tmp_path / "scratch"
    result = prune_tool_results(
        history(3, 20_000), keep_turns=2, scratch=scratch, pack_limit=10 * 1024
    )
    assert result.stubbed == 1 and not result.archive_errors
    stub = contents(result.messages)[0]
    assert "Exact text: Read" in stub and "observation-" in stub
    archived = list(scratch.glob("observation-*.txt"))
    assert len(archived) == 1 and len(archived[0].read_text()) == 20_000
    assert result.tokens_saved == result.chars_saved // 4


def test_a_small_result_is_kept_because_a_stub_would_save_nothing():
    result = prune_tool_results(history(3, 300), keep_turns=2)
    assert result.stubbed == 0 and result == PruneResult(messages=result.messages)


def test_an_old_image_is_counted_as_tokens_even_though_its_text_is_short(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"not really a png")
    messages = [
        Message("user", "look"),
        Message("assistant", [tool_use_block("Read", {"file_path": str(frame)}, "call0")]),
        Message(
            "user",
            [tool_result_block("call0", "a chart", images=[image_block(frame)])],
        ),
        Message("assistant", "noted"),
        Message("assistant", "and again"),
    ]
    result = prune_tool_results(messages, keep_turns=2, min_chars=400)
    assert (result.stubbed, result.images_dropped) == (1, 1)
    assert "frame.png" in contents(result.messages)[0]
    # the flat image figure is what decides step 2, not the character count
    assert result.tokens_saved > result.chars_saved // 4


def test_pruning_is_idempotent_a_second_pass_saves_nothing():
    once = prune_tool_results(history(5, 5_000), keep_turns=2)
    twice = prune_tool_results(once.messages, keep_turns=2)
    assert once.stubbed == 3 and twice.stubbed == 0
    assert twice.chars_saved == 0


def test_keep_turns_zero_leaves_no_window_so_even_the_newest_is_stubbed(tmp_path):
    # A documented edge, not a useful setting: 0 means "no recent round is protected".
    result = prune_tool_results(history(3, 5_000), keep_turns=0, scratch=tmp_path / "s")
    assert result.stubbed == 3
    assert contents(result.messages)[-1].startswith("[pruned] Bash(")


def test_one_result_is_enough_and_a_single_line_result_says_no_lines():
    stub = compaction.stub_line("Bash", "{}", "x" * 500, [], "")
    assert stub.count("chars") == 1 and "lines" not in stub


# ------------------------------------------------------------------ endpoint defaults


def test_the_endpoint_decides_the_default_and_a_name_overrides_it(make_session):
    session, _, _ = make_session([])
    local = ModelSpec(id="ollama", base_url="http://localhost:11434/v1", key_env="UNUSED_KEY")
    assert is_local_endpoint(local.base_url) and not is_local_endpoint(session.model.base_url)
    assert Settings().prune_tool_results == ""  # "" = choose by endpoint
    assert session.prune_mode() == "over_budget"  # the scripted test model is not local
    session.model = local
    assert session.prune_mode() == "every_turn"
    for wanted, expect in (
        ("every_turn", "every_turn"),
        ("off", "off"),
        ("over_budget", "over_budget"),
    ):
        session.settings.prune_tool_results = wanted
        assert session.prune_mode() == expect  # an explicit name wins over the endpoint


def test_an_unknown_prune_mode_is_a_config_error_the_person_has_to_fix(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('prune_tool_results = "sometimes"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_settings(path)
    assert "prune_tool_results" in str(caught.value) and "every_turn" in str(caught.value)


# ------------------------------------------------------------- the two steps together


def load(make_session, rounds: int, chars: int, turns=None, hooks=None):
    """A session whose history is ``rounds`` tool calls, over the soft budget."""
    session, recorder, provider = make_session(turns or [], hooks=hooks)
    for message in history(rounds, chars):
        session._add(message)
    session.meter.record(BIG)
    return session, recorder, provider


def test_step_one_alone_fits_so_no_summary_request_is_sent(make_session, tmp_path):
    # The plan's closing test for item 1, first clause: 12 calls over the soft budget,
    # pruning gets the prompt under it, and not one request goes to the model.
    postcompact = tmp_path / "postcompact.json"
    hook = write_hook(tmp_path, "post.sh", f"cat > {postcompact}\n")
    session, recorder, provider = load(
        make_session, 12, 20_000, hooks={"PostCompact": [{"command": hook}]}
    )
    result = run(session.compact("auto"))
    assert result.summary == "" and not result.failed and result.pruned == 10
    assert result.tokens_saved > 20_000  # my estimate, and enough for the 100k budget
    assert provider.requests == []  # no model call at all
    assert recorder.of(ev.Compacted) == []  # no summary, so no Compacted and no hook
    assert not postcompact.exists()
    pruned = recorder.of(ev.ToolResultsPruned)
    assert len(pruned) == 1 and (pruned[0].trigger, pruned[0].stubbed) == ("auto", 10)
    assert len(session.messages) == 25  # nothing was dropped, only stubbed
    assert contents(session.messages)[-1] == "x" * 20_000  # the newest round is whole
    assert not any("history kept" in n.text for n in recorder.of(ev.Notice))


def test_the_transcript_keeps_every_full_result_after_a_prune(make_session):
    session, _, _ = load(make_session, 6, 20_000)
    run(session.compact("auto"))
    stored = load_messages(session.transcript.path)
    assert contents(stored).count("x" * 20_000) == 6
    notes = [
        json.loads(line)
        for line in session.transcript.path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("subtype") == "prune_boundary"
    ]
    assert len(notes) == 1
    assert notes[0]["trigger"] == "auto" and "4 old tool result(s) stubbed" in notes[0]["content"]


def test_pruning_is_not_enough_so_exactly_one_summary_request_follows(make_session):
    # Second clause of the plan's closing test.
    session, recorder, provider = load(
        make_session, 12, 600, turns=[chunks_for("SUMMARY: 12 steps run, next: report.")]
    )
    result = run(session.compact("auto"))
    assert result.summary.startswith("SUMMARY:") and result.pruned == 10
    assert len(provider.requests) == 1
    sent = json.dumps(provider.requests[0]["messages"])
    assert "[pruned]" in sent  # the summary is written over the pruned history
    assert recorder.of(ev.Compacted)[0].trigger == "auto"
    assert len(recorder.of(ev.ToolResultsPruned)) == 1
    assert session.messages[0].text().startswith("This session continues")
    # The flag lives on the transcript row, where resume reads it: the in-memory message
    # is the same object whether or not it was written as a summary.
    rows = [
        json.loads(line)
        for line in session.transcript.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r for r in rows if r.get("isCompactSummary")]
    assert len(load_messages(session.transcript.path)) == 1  # resume starts at the summary


def test_a_prune_at_a_turn_start_leaves_the_current_round_whole(make_session, tmp_path):
    # The plan's extra closing test for ``every_turn``: turn 3's request carries stubs
    # for turn 1 and the full text of turn 2. The cached-token half of that test cannot
    # be run offline: ScriptedProvider invents its usage, so it would prove nothing about
    # a real prompt cache. What is asserted here is the request shape that cache hits on.
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "big.txt").write_text("line\n" * 1_200, encoding="utf-8")
    read = chunks_for(tool_calls=[("Read", {"file_path": "big.txt"})])
    session, recorder, provider = make_session(
        [read, chunks_for("read it"), read, chunks_for("read it again"), chunks_for("done")]
    )
    session.settings.prune_tool_results = "every_turn"
    assert run(session.send("first")).text == "read it"
    assert run(session.send("second")).text == "read it again"
    assert run(session.send("third")).text == "done"
    assert len(recorder.of(ev.ToolResultsPruned)) == 1  # stubbed at turn 3 only
    tool_rows = [m for m in provider.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_rows) == 2
    assert tool_rows[0]["content"].startswith("[pruned] Read(")
    assert "line" in tool_rows[1]["content"] and "[pruned]" not in tool_rows[1]["content"]


def test_prune_off_means_no_stub_and_a_summary_anyway(make_session):
    session, recorder, provider = load(make_session, 8, 20_000, turns=[chunks_for("SUMMARY only.")])
    session.settings.prune_tool_results = "off"
    result = run(session.compact("auto"))
    assert result.summary == "SUMMARY only." and result.pruned == 0
    assert recorder.of(ev.ToolResultsPruned) == []
    assert len(provider.requests) == 1


# ---------------------------------------------------------------- the summary's model


def test_a_configured_compact_model_writes_the_summary(make_session):
    cheap = ModelSpec(id="cheap", base_url="http://unused.invalid/v1", key_env="UNUSED_KEY")
    session, recorder, _ = load(
        make_session, 12, 600, turns=[chunks_for("SUMMARY by the cheap model.")]
    )
    session.models = {**session.models, "cheap": cheap}
    session.settings.compact_model = "cheap"
    assert session.compact_spec()[0] is cheap
    result = run(session.compact("auto"))
    assert result.summary == "SUMMARY by the cheap model."
    assert recorder.of(ev.Notice) == []  # a name that resolves warns about nothing


def test_a_typo_in_compact_model_falls_back_and_says_so(make_session):
    session, recorder, _ = make_session(
        [chunks_for("answer"), chunks_for("SUMMARY by the session model.")]
    )
    session.settings.compact_model = "kritik"
    spec, warning = session.compact_spec()
    assert spec is session.model and "kritik" in warning
    run(session.send("q"))
    result = run(session.compact("manual"))
    assert result.summary == "SUMMARY by the session model."
    assert any("not configured" in n.text for n in recorder.of(ev.Notice))


def test_a_manual_compact_never_prunes_it_asks_for_the_whole_history(make_session):
    session, recorder, provider = load(
        make_session, 6, 20_000, turns=[chunks_for("SUMMARY of everything.")]
    )
    result = run(session.compact("manual", "keep the file names"))
    assert result.summary == "SUMMARY of everything." and result.pruned == 0
    assert recorder.of(ev.ToolResultsPruned) == []
    sent = json.dumps(provider.requests[0]["messages"])
    assert "x" * 20_000 in sent and "[pruned]" not in sent
