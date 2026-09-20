"""The turn loop, driven by a scripted provider: tools, approvals, deny list, hooks, cancel."""

from __future__ import annotations

import asyncio
import json
import time

from conftest import FixedApprover, write_hook
from mwm_harness import events as ev
from mwm_harness.providers import chunks_for
from mwm_harness.transcript import load_messages


def run(coro):
    return asyncio.run(coro)


def tool_results(session) -> list[dict]:
    return [
        block
        for message in session.messages
        for block in message.blocks()
        if block.get("type") == "tool_result"
    ]


def test_plain_answer_ends_turn_and_reports_usage(make_session):
    session, recorder, _ = make_session([chunks_for("Hello there.")])
    ended = run(session.send("hi"))
    assert ended.reason == "done"
    assert ended.text == "Hello there."
    usage = recorder.of(ev.UsageUpdated)[0]
    assert (usage.prompt_tokens, usage.completion_tokens) == (100, 10)


def test_write_needs_approval_and_runs_when_approved(make_session):
    approver = FixedApprover(True)
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "note.txt", "content": "abc"})]),
        chunks_for("Saved."),
    ]
    session, _, provider = make_session(turns, approver=approver)
    ended = run(session.send("save a note"))
    assert ended.reason == "done"
    assert (session.cwd / "note.txt").read_text() == "abc"
    assert [name for name, _ in approver.asked] == ["Write"]
    # The second request carries the tool result in OpenAI shape, right after the call.
    roles = [m["role"] for m in provider.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


def test_declined_tool_does_not_run(make_session):
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "note.txt", "content": "abc"})]),
        chunks_for("Understood."),
    ]
    session, _, _ = make_session(turns, approver=FixedApprover(False))
    run(session.send("save a note"))
    assert not (session.cwd / "note.txt").exists()
    assert tool_results(session)[0]["is_error"] is True


def test_read_only_tools_skip_approval(make_session):
    approver = FixedApprover(False)
    turns = [chunks_for(tool_calls=[("Glob", {"pattern": "*.txt"})]), chunks_for("None.")]
    session, _, _ = make_session(turns, approver=approver)
    run(session.send("list"))
    assert approver.asked == []
    assert "is_error" not in tool_results(session)[0]


def test_deny_list_holds_in_bypass_mode(make_session):
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "git push --force origin main"})]),
        chunks_for("I will not do that."),
    ]
    session, _, _ = make_session(turns, mode="bypassPermissions")
    run(session.send("force push"))
    result = tool_results(session)[0]
    assert result["is_error"] is True
    assert "git-force-push" in result["content"]


def test_two_parallel_calls_both_answered_in_order(make_session):
    turns = [
        chunks_for(tool_calls=[("Glob", {"pattern": "*.a"}), ("Glob", {"pattern": "*.b"})]),
        chunks_for("Done."),
    ]
    session, _, _ = make_session(turns)
    run(session.send("two globs"))
    calls = session.messages[1].tool_uses()
    assert [r["tool_use_id"] for r in tool_results(session)] == [c["id"] for c in calls]


def test_malformed_arguments_become_an_error_result(make_session):
    broken = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "function": {"name": "Read", "arguments": '{"file_path": '},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
    ]
    session, _, provider = make_session([broken, chunks_for("Retrying is pointless.")])
    ended = run(session.send("read"))
    assert ended.reason == "done"
    assert "not valid JSON" in tool_results(session)[0]["content"]
    # The stored call has an empty input, so the next request is still valid JSON.
    sent = provider.requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {}


def test_stop_hook_blocks_once_then_sees_stop_hook_active(make_session, tmp_path):
    hook = write_hook(
        tmp_path,
        "stop.sh",
        'payload=$(cat)\necho "$payload" >> "$(dirname "$0")/stop-payloads.jsonl"\n'
        'if echo "$payload" | grep -q \'"stop_hook_active": true\'; then exit 0; fi\n'
        'echo \'{"decision": "block", "reason": "remove the em dash"}\'\n',
    )
    turns = [chunks_for("First try."), chunks_for("Second try.")]
    session, recorder, provider = make_session(turns, hooks={"Stop": [{"command": hook}]})
    ended = run(session.send("write something"))
    assert ended.reason == "done"
    assert ended.text == "Second try."
    assert recorder.of(ev.HookBlocked)[0].reason == "remove the em dash"
    payloads = [
        json.loads(line) for line in (tmp_path / "stop-payloads.jsonl").read_text().splitlines()
    ]
    assert [p["stop_hook_active"] for p in payloads] == [False, True]
    assert payloads[0]["transcript_path"] == str(session.transcript.path)
    # The feedback reached the model as a meta user message, not as a typed prompt.
    feedback = [m for m in session.messages if m.is_meta]
    assert "remove the em dash" in feedback[0].text()
    assert "remove the em dash" in provider.requests[1]["messages"][-1]["content"]


def test_stop_hook_that_always_blocks_is_released(make_session, tmp_path):
    hook = write_hook(
        tmp_path, "stop.sh", "cat > /dev/null\necho 'never good enough' >&2\nexit 2\n"
    )
    turns = [chunks_for(f"Try {n}.") for n in range(3)]
    session, _, _ = make_session(turns, hooks={"Stop": [{"command": hook}]})
    ended = run(session.send("write"))
    assert ended.reason == "stop_blocks_exhausted"
    assert ended.text == "Try 2."


def test_pretooluse_rewrite_is_used_and_still_checked_by_deny_list(make_session, tmp_path):
    rewrite = write_hook(
        tmp_path,
        "rewrite.sh",
        "cat > /dev/null\n"
        'echo \'{"hookSpecificOutput": {"hookEventName": "PreToolUse", '
        '"permissionDecision": "allow", '
        '"updatedInput": {"command": "git commit --no-verify -m x"}}}\'\n',
    )
    turns = [chunks_for(tool_calls=[("Bash", {"command": "git commit -m x"})]), chunks_for("Ok.")]
    session, _, _ = make_session(
        turns, hooks={"PreToolUse": [{"matcher": "Bash", "command": rewrite}]}
    )
    run(session.send("commit"))
    result = tool_results(session)[0]
    assert result["is_error"] is True
    assert "no-verify" in result["content"]


def test_pretooluse_allow_only_skips_the_approver_when_enabled(make_session, tmp_path):
    allow = write_hook(
        tmp_path,
        "allow.sh",
        "cat > /dev/null\n"
        'echo \'{"hookSpecificOutput": {"permissionDecision": "allow", '
        '"updatedInput": {"command": "echo rewritten"}}}\'\n',
    )
    approver = FixedApprover(False)
    turns = [chunks_for(tool_calls=[("Bash", {"command": "echo original"})]), chunks_for("Ok.")]
    session, _, _ = make_session(
        turns, approver=approver, hooks={"PreToolUse": [{"matcher": "Bash", "command": allow}]}
    )
    run(session.send("echo"))
    # A hook's "allow" does not replace the person by default, and the person is
    # shown the rewritten command, the one that would really run.
    assert approver.asked == [("Bash", {"command": "echo rewritten"})]
    session.settings.hooks_may_approve = True
    session.provider = type(session.provider)(
        [chunks_for(tool_calls=[("Bash", {"command": "echo original"})]), chunks_for("Ok.")]
    )
    run(session.send("echo again"))
    assert len(approver.asked) == 1
    assert tool_results(session)[-1]["content"] == "rewritten"


def test_permission_request_hook_can_approve(make_session, tmp_path):
    hook = write_hook(
        tmp_path,
        "approve.sh",
        "cat > /dev/null\n"
        'echo \'{"hookSpecificOutput": {"hookEventName": "PermissionRequest", '
        '"decision": {"behavior": "allow"}}}\'\n',
    )
    approver = FixedApprover(False)
    turns = [chunks_for(tool_calls=[("Bash", {"command": "echo hi"})]), chunks_for("Ok.")]
    session, _, _ = make_session(
        turns,
        approver=approver,
        hooks={"PermissionRequest": [{"matcher": "Bash|Write", "command": hook}]},
    )
    run(session.send("echo"))
    assert approver.asked == []
    assert tool_results(session)[0]["content"] == "hi"


def test_prompt_hook_context_reaches_the_model_as_meta(make_session, tmp_path):
    hook = write_hook(tmp_path, "ctx.sh", "cat > /dev/null\necho 'TODAY IS SATURDAY'\n")
    session, _, provider = make_session(
        [chunks_for("Noted.")], hooks={"UserPromptSubmit": [{"command": hook}]}
    )
    run(session.send("what day is it"))
    sent = provider.requests[0]["messages"][1]["content"]
    assert "TODAY IS SATURDAY" in sent and sent.endswith("what day is it")
    assert [m.is_meta for m in session.messages[:2]] == [True, False]


def test_posttooluse_exit_2_feedback_is_appended_to_the_result(make_session, tmp_path):
    hook = write_hook(
        tmp_path, "lint.sh", "cat > /dev/null\necho 'F841 unused variable' >&2\nexit 2\n"
    )
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "a.py", "content": "x = 1\n"})]),
        chunks_for("Fixed."),
    ]
    session, _, _ = make_session(
        turns,
        mode="acceptEdits",
        hooks={"PostToolUse": [{"matcher": "Write|Edit", "command": hook}]},
    )
    run(session.send("write"))
    assert "F841 unused variable" in tool_results(session)[0]["content"]


def test_cancel_during_bash_kills_the_process_tree(make_session):
    turns = [chunks_for(tool_calls=[("Bash", {"command": "sleep 60 & wait"})])]
    session, _, _ = make_session(turns, mode="bypassPermissions")

    async def scenario():
        task = asyncio.create_task(session.send("sleep"))
        await asyncio.sleep(0.5)
        assert session.cancel() is True
        return await task

    started = time.monotonic()
    ended = run(scenario())
    assert ended.reason == "cancelled"
    assert time.monotonic() - started < 10
    assert "Interrupted" in tool_results(session)[0]["content"]


def test_cancel_mid_stream_keeps_partial_text(make_session):
    class SlowProvider:
        async def stream(self, spec, system, messages, tools):
            yield {"choices": [{"delta": {"content": "Partial ans"}}]}
            await asyncio.sleep(30)
            yield {"choices": [{"delta": {"content": "never arrives"}}]}

    session, _, _ = make_session([])
    session.provider = SlowProvider()

    async def scenario():
        task = asyncio.create_task(session.send("talk"))
        await asyncio.sleep(0.3)
        session.cancel()
        return await task

    ended = run(scenario())
    assert (ended.reason, ended.text) == ("cancelled", "Partial ans")
    assert session.messages[-1].text() == "Partial ans"


def test_resume_restores_history_and_repairs_unanswered_calls(make_session):
    turns = [chunks_for(tool_calls=[("Bash", {"command": "sleep 60"})])]
    first, _, _ = make_session(turns, mode="bypassPermissions")

    async def scenario():
        task = asyncio.create_task(first.send("sleep"))
        await asyncio.sleep(0.5)
        first.cancel()
        await task

    run(scenario())
    restored = load_messages(first.transcript.path)
    assert [m.role for m in restored] == ["user", "assistant", "user"]

    second, _, provider = make_session(
        [chunks_for("Welcome back.")], resume_from=first.transcript.path
    )
    ended = run(second.send("continue"))
    assert ended.reason == "done"
    roles = [m["role"] for m in provider.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]


def test_plan_mode_blocks_edits_until_the_plan_is_approved(make_session):
    approver = FixedApprover(True)
    turns = [
        chunks_for(tool_calls=[("Write", {"file_path": "a.txt", "content": "x"})]),
        chunks_for(tool_calls=[("ExitPlanMode", {"plan": "1. write a.txt"})]),
        chunks_for(tool_calls=[("Write", {"file_path": "a.txt", "content": "x"})]),
        chunks_for("written"),
    ]
    session, recorder, _ = make_session(turns, approver=approver, mode="acceptEdits")
    session.set_mode("plan")
    ended = run(session.send("make a.txt"))
    assert ended.reason == "done"
    results = tool_results(session)
    assert results[0]["is_error"] and "Plan mode is on" in results[0]["content"]
    assert "approved the plan" in results[1]["content"]
    assert not results[2].get("is_error")
    assert (session.cwd / "a.txt").read_text() == "x"
    assert session.permissions.mode == "acceptEdits"  # back to the mode before /plan
    assert recorder.of(ev.PlanProposed)[0].plan == "1. write a.txt"
    assert [e.mode for e in recorder.of(ev.ModeChanged)] == ["plan", "acceptEdits"]
    assert approver.asked[0][0] == "ExitPlanMode"


def test_a_rejected_plan_keeps_plan_mode_on(make_session):
    turns = [chunks_for(tool_calls=[("ExitPlanMode", {"plan": "p"})]), chunks_for("ok")]
    session, recorder, _ = make_session(turns, approver=FixedApprover(False))
    session.set_mode("plan")
    run(session.send("plan it"))
    assert tool_results(session)[0]["is_error"]
    assert session.permissions.mode == "plan"
    assert recorder.of(ev.PlanResolved)[0].approved is False
