"""Compaction: manual, automatic at the soft budget, the PostCompact hook, resume."""

from __future__ import annotations

import asyncio
import json

from conftest import write_hook
from mwm_harness import events as ev
from mwm_harness.providers import chunks_for
from mwm_harness.repl.terminal import Printer, run_async_command, run_command
from mwm_harness.transcript import load_messages

BIG = {"prompt_tokens": 120_000, "completion_tokens": 500}  # over the 100k soft budget


def run(coro):
    return asyncio.run(coro)


def test_manual_compact_replaces_history_and_fires_postcompact(make_session, tmp_path):
    seen = tmp_path / "postcompact.json"
    hook = write_hook(tmp_path, "post.sh", f"cat > {seen}\n")
    turns = [
        chunks_for("First answer."),
        chunks_for("SUMMARY: asked X, did Y, next Z."),
        chunks_for("Onward."),
    ]
    session, recorder, provider = make_session(turns, hooks={"PostCompact": [{"command": hook}]})

    async def scenario():
        await session.send("do X")
        out = Printer(color=False)
        assert await run_async_command(session, "/compact keep the file names", out) is True
        return await session.send("continue")

    ended = run(scenario())
    assert ended.text == "Onward."
    compact_request = provider.requests[1]
    assert "tools" not in compact_request  # the summary request offers no tools
    assert "keep the file names" in compact_request["messages"][-1]["content"]
    payload = json.loads(seen.read_text())
    assert payload["hook_event_name"] == "PostCompact" and payload["trigger"] == "manual"
    assert payload["compact_summary"] == "SUMMARY: asked X, did Y, next Z."
    after = [m["content"] for m in provider.requests[2]["messages"] if m["role"] == "user"]
    assert len(after) == 1  # consecutive user messages travel as one
    assert "SUMMARY: asked X" in after[0] and after[0].endswith("continue")
    assert "do X" not in json.dumps(provider.requests[2]["messages"])
    event = recorder.of(ev.Compacted)[0]
    assert (event.trigger, event.messages_before) == ("manual", 2)
    assert session.meter.used == 100 + 10  # the meter restarts, then counts the next answer


def test_auto_compact_at_the_soft_budget_continues_the_turn(make_session):
    turns = [
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})], usage=BIG),
        chunks_for("SUMMARY: listing markdown files; next step: answer."),
        chunks_for("No markdown files here."),
    ]
    session, recorder, provider = make_session(turns)
    ended = run(session.send("list markdown files"))
    assert ended.reason == "done" and ended.text == "No markdown files here."
    assert recorder.of(ev.Compacted)[0].trigger == "auto"
    final = provider.requests[2]["messages"]
    assert [m["role"] for m in final] == ["system", "user"]
    assert "SUMMARY: listing" in final[1]["content"] and "exact next step" in final[1]["content"]


def test_a_prompt_that_stays_over_budget_compacts_once_not_on_every_request(make_session):
    # Seen live 09-20 on a 16k local model: system prompt + tools = 12,845 tokens against
    # a 12,000 budget, so the harness summarised after every single tool call.
    turns = [
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})], usage=BIG),
        chunks_for("SUMMARY: listing markdown files."),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.txt"})], usage=BIG),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.py"})], usage=BIG),
        chunks_for("Nothing found.", usage=BIG),
    ]
    session, recorder, provider = make_session(turns)
    ended = run(session.send("list files"))
    assert ended.reason == "done" and ended.text == "Nothing found."
    assert len(recorder.of(ev.Compacted)) == 1
    assert len(provider.requests) == 5
    warnings = [n.text for n in recorder.of(ev.Notice) if "right after compaction" in n.text]
    assert len(warnings) == 1 and "120,000" in warnings[0]


def test_compaction_runs_again_after_half_a_budget_of_growth(make_session):
    small = {"prompt_tokens": 20_000, "completion_tokens": 100}
    grown = {"prompt_tokens": 150_000, "completion_tokens": 100}
    turns = [
        chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})], usage=BIG),
        chunks_for("SUMMARY one."),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.txt"})], usage=small),  # floor = 20k
        chunks_for(tool_calls=[("Glob", {"pattern": "*.py"})], usage=grown),
        chunks_for("SUMMARY two."),
        chunks_for("Done.", usage=small),
    ]
    session, recorder, _ = make_session(turns)
    ended = run(session.send("list files"))
    assert ended.text == "Done."
    assert len(recorder.of(ev.Compacted)) == 2


def test_a_model_that_never_stops_calling_tools_hits_the_request_cap(make_session):
    turns = [chunks_for(tool_calls=[("Glob", {"pattern": "*.md"})]) for _ in range(4)]
    session, recorder, provider = make_session(turns)
    session.settings.max_requests_per_turn = 3
    ended = run(session.send("loop"))
    assert ended.reason == "error" and "3 model requests" in ended.text
    assert len(provider.requests) == 3
    assert any("max_requests_per_turn" in n.text for n in recorder.of(ev.Notice))


def test_auto_compact_can_be_switched_off(make_session):
    session, recorder, _ = make_session([chunks_for("a", usage=BIG), chunks_for("b")])
    session.settings.auto_compact = False
    run(session.send("one"))
    run(session.send("two"))
    assert recorder.of(ev.Compacted) == []
    assert any("over the soft context budget" in n.text for n in recorder.of(ev.Notice))


def test_a_failed_summary_keeps_the_history(make_session):
    session, recorder, _ = make_session([chunks_for("answer"), chunks_for("")])
    run(session.send("q"))
    assert run(session.compact()) == ""
    assert len(session.messages) == 2
    assert "history kept" in recorder.of(ev.Notice)[-1].text


def test_resume_starts_from_the_last_summary(make_session):
    turns = [chunks_for("old answer"), chunks_for("SUMMARY S"), chunks_for("new answer")]
    session, _, _ = make_session(turns)

    async def scenario():
        await session.send("old question")
        await session.compact()
        await session.send("new question")

    run(scenario())
    texts = [m.text() for m in load_messages(session.transcript.path)]
    assert (
        len(texts) == 3 and "SUMMARY S" in texts[0] and texts[1:] == ["new question", "new answer"]
    )


def test_agents_and_init_commands(make_session, capsys):
    session, _, _ = make_session([])
    out = Printer(color=False)
    assert run_command(session, {}, "/agents", out) is True
    prompt = run_command(session, {}, "/init focus on tests", out)
    assert "AGENTS.md" in prompt and prompt.endswith("focus on tests")
    assert "0 agents" in capsys.readouterr().out
