"""Item 5: a session-end pass writes candidate lessons for a person to review.

The plan's closing test is
test_one_repeated_failed_command_writes_one_candidate_and_a_second_run_writes_nothing: a
scripted transcript with the same failing command twice produces exactly one candidate that
names it, and rerunning on the same session id adds no line. Everything else here guards the
two promises the item makes: candidates only, never a rule, and a pass that cannot run says
why, because a silent skip looks exactly like a session with nothing to learn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness import lessons
from mwm_harness.config import ModelSpec, Settings
from mwm_harness.messages import Message, tool_result_block
from mwm_harness.providers import chunks_for

GROQ = ModelSpec(
    id="qwen/qwen3.8-27b",
    base_url="https://api.groq.com/openai/v1",
    key_env="UNUSED_KEY",
    context_window=131_072,
    soft_budget=100_000,
)

BAD_CMD = "cd $HOME/MWM/projects/mwm-harness && .venv/bin/python -m pytest tests/test_x.py"
MAIL = "someone@example.org"


def run(coro):
    return asyncio.run(coro)


def answer(entries: list[dict[str, str]]):
    return chunks_for(
        json.dumps({"lessons": entries}),
        usage={"prompt_tokens": 4213, "completion_tokens": 190},
    )


def one_entry() -> list[dict[str, str]]:
    return [
        {
            "lesson": f"the same command failed twice and was retried unchanged: {BAD_CMD}",
            "evidence": "exit code 1, AssertionError: 2 failed",
        }
    ]


def bash_fail_turn(index: int = 0) -> list[dict[str, Any]]:
    """A failing Bash turn.

    The command differs every time on purpose: consecutive identical calls trip the loop's
    repeat brake and end the turn early, which is a different behavior than this test needs.
    """
    return chunks_for(tool_calls=[("Bash", {"command": f"exit 3  # step {index}"})])


def session_messages(repeats: int = 2, with_identity: bool = False) -> list[Message]:
    """The shape the closing test names: one ask, the same failing command N times."""
    ask = "fix the failing test" + (f" and mail {MAIL}" if with_identity else "")
    out = [Message("user", ask)]
    for i in range(repeats):
        call_id = f"call_{i}"
        out.append(
            Message(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "Bash",
                        "input": {"command": BAD_CMD},
                    }
                ],
            )
        )
        out.append(
            Message(
                "user",
                [
                    tool_result_block(
                        call_id,
                        "exit code 1\nAssertionError: 2 failed"
                        + (f"\ncontact {MAIL}\n" if with_identity else ""),
                        is_error=True,
                    )
                ],
            )
        )
    out.append(Message("assistant", [{"type": "text", "text": "Giving up for now."}]))
    return out


LONG_CALLS = 600
LONG_FAILS = sum(1 for index in range(LONG_CALLS) if index % 4 == 0)


def long_messages(calls: int = LONG_CALLS) -> list[Message]:
    """Long enough to force a cut, so the priority rules can be checked under pressure."""
    out = [Message("user", "do the work")]
    for i in range(calls):
        call_id = f"c{i}"
        out.append(
            Message(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "Read",
                        "input": {"file_path": f"/repo/file_{i}.py"},
                    }
                ],
            )
        )
        fail = i % 4 == 0
        out.append(
            Message("user", [tool_result_block(call_id, f"boom number {i}" if fail else "ok")])
            if not fail
            else Message("user", [tool_result_block(call_id, f"boom number {i}", is_error=True)])
        )
    return out


def make_caller(entries: list[dict[str, str]]):
    """A caller that records what it was asked, so the prompt itself can be inspected."""

    async def caller(system: str, user: str) -> tuple[str, dict[str, Any]]:
        caller.prompts.append((system, user))
        return json.dumps({"lessons": entries}), {
            "prompt_tokens": 4213,
            "completion_tokens": 190,
        }

    caller.prompts = []
    return caller


# ---------------------------------------------------------------- closing test


def test_one_repeated_failed_command_writes_one_candidate_and_a_second_run_writes_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data/lessons/LESSONS.md"
    messages = session_messages(repeats=2)
    caller = make_caller(one_entry())
    result = run(lessons.run_lessons(messages, "sess-1", caller, GROQ.id, path, min_calls=2))
    assert result.status == "recorded", result.reason
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(lines) == 1, f"expected one candidate line, got {lines}"
    assert BAD_CMD in lines[0], "the candidate has to name the command that failed"
    assert "session=sess-1" in lines[0]

    second_caller = make_caller(one_entry())
    again = run(lessons.run_lessons(messages, "sess-1", second_caller, GROQ.id, path, min_calls=2))
    assert again.status == "skipped"
    assert "already recorded" in again.reason
    # Proven against the file and the request count, not only the status string.
    after = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")
    ]
    assert len(after) == 1
    assert second_caller.prompts == [], "a deduped pass must not spend a model request"


def test_dedupe_is_per_session_not_per_file(tmp_path: Path) -> None:
    path = tmp_path / "LESSONS.md"
    messages = session_messages(repeats=2)
    run(lessons.run_lessons(messages, "sess-1", make_caller(one_entry()), "m", path, min_calls=2))
    second = run(
        lessons.run_lessons(messages, "sess-2", make_caller(one_entry()), "m", path, min_calls=2)
    )
    assert second.status == "recorded"
    assert len([line for line in path.read_text().splitlines() if line.startswith("- ")]) == 2


# ---------------------------------------------------------------- size and priority


def test_a_cut_digest_keeps_every_ask_and_every_failure() -> None:
    """The cap may drop the model's prose and its routine calls, never a failure or an ask."""
    text, dropped = lessons.digest(long_messages(600))
    assert dropped > 0, "this transcript must be big enough to force a cut"
    assert len(text) <= lessons.MAX_DIGEST_CHARS + 400
    assert "[person] do the work" in text
    failures = [line for line in text.splitlines() if line.startswith("[FAILED]")]
    assert len(failures) == LONG_FAILS, "every failure has to survive the trim"
    assert "left out" in text, "a cut says so inside the digest rather than passing as whole"


def test_the_digest_cap_sits_under_the_measured_input_ceiling() -> None:
    """Groq refused 27b at 7,000 input tokens per minute, and 12,000 chars measured 4,487.

    That is 2.65 characters per token, not the usual four: chars/4 would have promised 3,000
    and hidden how close the real number sits, and the 24,000-char digest that drew HTTP 413
    (8,403 tokens) proves the ceiling is per request minute rather than per window.
    """
    assert lessons.MAX_DIGEST_CHARS / 2.65 < 7_000
    text, _dropped = lessons.digest(long_messages(600))
    assert len(text) <= lessons.MAX_DIGEST_CHARS + 400


def test_an_email_and_a_key_never_reach_the_lessons_prompt(tmp_path: Path) -> None:
    """This digest leaves the machine for a hosted endpoint, and identity is not evidence."""
    caller = make_caller(one_entry())
    run(
        lessons.run_lessons(
            session_messages(repeats=3, with_identity=True),
            "sess-mail",
            caller,
            "m",
            tmp_path / "L.md",
            min_calls=2,
        )
    )
    sent = caller.prompts[0][1]
    assert MAIL not in sent
    assert "[redacted:email]" in sent


# ---------------------------------------------------------------- skip and failure


def test_a_short_session_is_skipped_before_any_call(tmp_path: Path) -> None:
    caller = make_caller(one_entry())
    result = run(
        lessons.run_lessons(
            session_messages(repeats=1), "sess-short", caller, "m", tmp_path / "L.md", min_calls=8
        )
    )
    assert result.status == "skipped"
    assert "under the 8 floor" in result.reason
    assert caller.prompts == []
    assert not (tmp_path / "L.md").exists()


def test_an_unparseable_answer_is_a_failure_that_names_itself(tmp_path: Path) -> None:
    async def garbage(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return "I learned a lot today", {}

    result = run(
        lessons.run_lessons(
            session_messages(repeats=3), "sess-garbage", garbage, "m", tmp_path / "L.md", 2
        )
    )
    assert result.status == "failed"
    assert "no JSON object" in result.reason
    assert not (tmp_path / "L.md").exists()


def test_an_empty_list_is_quiet_and_writes_no_line(tmp_path: Path) -> None:
    path = tmp_path / "L.md"
    result = run(
        lessons.run_lessons(
            session_messages(repeats=3), "sess-quiet", make_caller([]), "m", path, min_calls=2
        )
    )
    assert result.status == "quiet"
    assert "kept nothing" in result.reason
    assert not path.exists(), "a quiet pass leaves no empty entry behind"


def test_a_provider_error_becomes_a_status_not_a_crash(tmp_path: Path) -> None:
    async def boom(system: str, user: str) -> tuple[str, dict[str, Any]]:
        raise RuntimeError("HTTP 413: input tokens per minute")

    result = run(
        lessons.run_lessons(
            session_messages(repeats=3), "sess-413", boom, "m", tmp_path / "L.md", 2
        )
    )
    assert result.status == "failed"
    assert "413" in result.reason


def test_more_than_three_lessons_are_cut_to_three() -> None:
    entries = [{"lesson": f"lesson {i}", "evidence": "x"} for i in range(6)]
    parsed, error = lessons.parse_lessons(json.dumps({"lessons": entries}))
    assert error == ""
    assert len(parsed) == lessons.MAX_LESSONS


def test_a_string_lesson_is_accepted_too() -> None:
    parsed, error = lessons.parse_lessons(json.dumps({"lessons": ["just one line"]}))
    assert error == "" and parsed == [{"lesson": "just one line", "evidence": ""}]


# ---------------------------------------------------------------- session wiring


def arm_lessons(session, ledger: Path) -> None:
    session.models[GROQ.id] = GROQ
    session.settings.review_after_turn = False
    session.settings.assurance_after_turn = False
    session.settings.lessons_at_session_end = True
    session.settings.lessons_model = GROQ.id
    session.settings.lessons_min_calls = 2
    session.settings.lessons_file = str(ledger)


def test_session_end_runs_the_pass_once_and_off_runs_nothing(make_session, tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.md"
    turns = [bash_fail_turn(i) for i in range(9)] + [chunks_for("done"), answer(one_entry())]
    session, recorder, provider = make_session(turns, approver=FixedApprover(True))
    arm_lessons(session, ledger)
    run(session.send("work"))
    run(session.close())
    finished = recorder.of(ev.LessonsFinished)
    assert len(finished) == 1
    assert finished[0].result["status"] == "recorded", finished[0].result
    assert finished[0].result["prompt_tokens"] == 4213
    text = ledger.read_text(encoding="utf-8")
    assert BAD_CMD in text, "the candidate line is the model's text, written through"
    assert f"session={session.transcript.session_id}" in text, "the line names its session"
    assert text.startswith("# Candidate lessons"), "the queue explains that it is not a rule"

    # Off in the default settings: an ordinary session end makes no extra request at all.
    assert Settings().lessons_at_session_end is False
    plain, plain_recorder, plain_provider = make_session(
        [chunks_for("hi")], approver=FixedApprover(True)
    )
    run(plain.send("hi"))
    run(plain.close())
    assert plain_recorder.of(ev.LessonsFinished) == []
    assert len(plain_provider.requests) == 1


def test_a_headless_session_writes_no_candidate(make_session, tmp_path: Path) -> None:
    """A headless turn is a script call, and the gate scripts would bury the queue."""
    ledger = tmp_path / "lessons.md"
    turns = [bash_fail_turn(i) for i in range(9)] + [chunks_for("done")]
    # approver=None is the headless wiring: DenyAll refuses every tool, and no person is there.
    session, recorder, provider = make_session(turns)
    arm_lessons(session, ledger)
    run(session.send("work"))
    during = len(provider.requests)
    run(session.close())
    finished = recorder.of(ev.LessonsFinished)
    assert len(finished) == 1
    assert finished[0].result["status"] == "skipped"
    assert "headless" in finished[0].result["reason"]
    assert not ledger.exists()
    # The claim measured, not assumed: session end sent no request at all. The turn list has
    # no lessons answer in it, so a pass that ran would have failed on an exhausted provider.
    assert len(provider.requests) == during


def test_the_queue_lands_in_data_never_in_memory(tmp_path: Path, monkeypatch) -> None:
    """The default is a review queue under data/, and a pass touches nothing else.

    Item 5 promises the harness never edits QWEN.md, memory or skills. A grep of the source
    would only prove the words are absent, so this builds the whole forbidden set, runs a pass
    that writes, and compares bytes: the queue is the only file that changes.
    """
    rules = tmp_path / "QWEN.md"
    rules.write_text("# rules\nnever weaken a test\n", encoding="utf-8")
    memory = tmp_path / "memory" / "MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# index\n- one line\n", encoding="utf-8")
    skill = tmp_path / "core" / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# skill\n", encoding="utf-8")
    before = {p: p.read_bytes() for p in (rules, memory, skill)}

    monkeypatch.setenv("MWM_WORKSPACE", str(tmp_path))
    path = lessons.lessons_path()
    assert path == tmp_path / "data" / "lessons" / "LESSONS.md"
    assert "memory" not in path.parts

    result = run(
        lessons.run_lessons(
            session_messages(repeats=3),
            "sess-place",
            make_caller(one_entry()),
            "m",
            path,
            min_calls=2,
        )
    )
    assert result.status == "recorded"
    for file, data in before.items():
        assert file.read_bytes() == data, f"{file.name} changed, and item 5 must not touch it"
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert written == [
        "QWEN.md",
        "core/skills/review/SKILL.md",
        "data/lessons/LESSONS.md",
        "memory/MEMORY.md",
    ], "the pass may create the queue and nothing else"


def test_a_bad_model_name_falls_back_with_a_warning(make_session, tmp_path: Path) -> None:
    session, _recorder, _provider = make_session([chunks_for("hi")], approver=FixedApprover(True))
    session.settings.lessons_model = "glm-9.9-nope"
    spec, warn = session.lessons_spec()
    assert spec.id == session.model.id
    assert "not configured" in warn


def test_lessons_path_honours_an_absolute_setting(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MWM_WORKSPACE", str(tmp_path))
    absolute = tmp_path.parent / "elsewhere.md"
    assert lessons.lessons_path(str(absolute)) == absolute
    assert lessons.lessons_path("notes/custom.md") == tmp_path / "notes/custom.md"
