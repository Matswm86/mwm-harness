"""Subagents: loading agent files, tool filtering, model choice, the Task tool, cancel."""

from __future__ import annotations

import asyncio
import time

from conftest import MODEL, FixedApprover
from mwm_harness import events as ev
from mwm_harness.agents import Agent, TaskTool, load_agents, resolve_model
from mwm_harness.config import ModelSpec
from mwm_harness.providers import chunks_for

AGENT_FILE = """---
name: reviewer
description: Reviews code. Read-only.
tools: Read, Grep, Glob, mcp__brain__*
model: opus
---

You are a code reviewer. Report findings only.
"""


def run(coro):
    return asyncio.run(coro)


def test_agent_files_load_with_tools_and_model(tmp_path):
    first, second = tmp_path / "project", tmp_path / "user"
    first.mkdir()
    second.mkdir()
    (first / "reviewer.md").write_text(AGENT_FILE)
    (second / "reviewer.md").write_text(AGENT_FILE.replace("Reviews code", "Shadowed"))
    (second / "free.md").write_text("---\ndescription: No tool list\n---\nDo anything.")
    (second / "empty.md").write_text("---\nname: empty\n---\n")
    agents = load_agents([first, second, tmp_path / "missing"])
    assert sorted(agents) == ["free", "reviewer"]
    reviewer = agents["reviewer"]
    assert reviewer.description == "Reviews code. Read-only." and reviewer.model == "opus"
    assert reviewer.prompt.startswith("You are a code reviewer")
    assert reviewer.allows("Read") and reviewer.allows("mcp__brain__search_knowledge")
    assert not reviewer.allows("Bash") and not reviewer.allows("Task")
    assert agents["free"].allows("Bash") and not agents["free"].allows("Task")


def test_model_mapping_falls_back_to_the_parent_with_a_note():
    big = ModelSpec(id="qwen-big", base_url="http://x.invalid", key_env="K")
    models = {MODEL.id: MODEL, big.id: big}
    agent = Agent("a", "", "p", None, model="opus")  # type: ignore[arg-type]
    assert resolve_model(agent, MODEL, models, {"opus": "qwen-big"}) == (big, "")
    spec, note = resolve_model(agent, MODEL, models, {})
    assert spec is MODEL and "not configured" in note
    agent.model = "inherit"
    assert resolve_model(agent, MODEL, models, {"opus": "qwen-big"}) == (MODEL, "")
    agent.model = "qwen-big"
    assert resolve_model(agent, MODEL, models, {})[0] is big


def with_agent(session, text: str = AGENT_FILE) -> None:
    folder = session.cwd / "agents"
    folder.mkdir(exist_ok=True)
    (folder / "reviewer.md").write_text(text)
    session.agents = load_agents([folder])
    session.tools["Task"] = TaskTool(session.agents, session.run_agent)


def test_task_runs_a_child_session_and_returns_only_its_report(make_session):
    task = {"description": "review", "prompt": "Review note.txt", "subagent_type": "reviewer"}
    turns = [
        chunks_for(tool_calls=[("Task", task)]),  # parent
        chunks_for(tool_calls=[("Read", {"file_path": "note.txt"}), ("Bash", {"command": "ls"})]),
        chunks_for("Report: note.txt holds one line."),  # child's final answer
        chunks_for("The reviewer found one line."),  # parent
    ]
    session, recorder, provider = make_session(turns, approver=FixedApprover(True))
    (session.cwd / "note.txt").write_text("hello\n")
    with_agent(session)
    ended = run(session.send("check the note"))
    assert ended.text == "The reviewer found one line."
    child_request = provider.requests[1]
    assert child_request["messages"][0]["content"].startswith("You are a code reviewer")
    offered = {spec["function"]["name"] for spec in child_request["tools"]}
    assert offered == {"Read", "Grep", "Glob"}  # the agent's tool list, never Task
    assert [m["content"] for m in child_request["messages"] if m["role"] == "user"] == [
        "Review note.txt"
    ]
    result = next(
        block
        for message in session.messages
        for block in message.blocks()
        if block.get("type") == "tool_result"
    )
    assert result["content"] == "Report: note.txt holds one line."
    assert len(session.messages) == 4  # prompt, Task call, its result, answer: no child history
    started, finished = recorder.of(ev.SubagentStarted)[0], recorder.of(ev.SubagentFinished)[0]
    assert (started.agent, finished.reason, finished.tool_calls) == ("reviewer", "done", 2)
    assert "not configured" in recorder.of(ev.Notice)[0].text  # opus is not mapped in this test
    assert session.meter.requests == 4  # the child's two requests count towards the session


def test_cancelling_the_parent_kills_the_subagents_command(make_session):
    text = AGENT_FILE.replace("tools: Read, Grep, Glob, mcp__brain__*\n", "")
    task = {"prompt": "sleep", "subagent_type": "reviewer"}
    turns = [
        chunks_for(tool_calls=[("Task", task)]),
        chunks_for(tool_calls=[("Bash", {"command": "sleep 60"})]),
    ]
    session, recorder, _ = make_session(turns, mode="bypassPermissions")
    with_agent(session, text)

    async def scenario():
        turn = asyncio.create_task(session.send("go"))
        while not any(e.name == "Task" for e in recorder.of(ev.ToolStarted)):
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.4)  # the child has started its sleep by now
        started = time.monotonic()
        session.cancel()
        ended = await turn
        return ended, time.monotonic() - started

    ended, waited = run(scenario())
    assert ended.reason == "cancelled" and waited < 5
    assert recorder.of(ev.SubagentFinished)[0].reason == "cancelled"


def test_unknown_agent_is_a_tool_error(make_session):
    session, _, _ = make_session([])
    with_agent(session)
    result = run(
        session.tools["Task"].run({"prompt": "x", "subagent_type": "ghost"}, session.tool_ctx)
    )
    assert result.is_error and "reviewer" in result.content
