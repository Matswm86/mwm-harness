"""The reasoning eval: case file, graders, the three arms, and the shipped playbooks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mwm_harness.agents import agent_roots, load_agents
from mwm_harness.reason_eval import (
    AREAS,
    REPO_CASES,
    Case,
    EvalError,
    Result,
    answer_case,
    command_caller,
    judge_grade,
    keyword_grade,
    load_cases,
    main,
    parse_judge,
    playbook_text,
    run,
    summarise,
    wilson,
)
from mwm_harness.skills import load_skills, skill_roots

PLAYBOOK_NAMES = {"reason-audit", "reason-challenge", "reason-debug", "reason-predelivery"}


def case(**overrides) -> Case:
    fields = {
        "id": "c1",
        "area": "verify",
        "prompt": "Is it down?",
        "trap": "grep output is unreliable",
        "signals": [["wrapper", "unreliable"], ["pgrep"]],
        "fail_if": [r"^\s*yes"],
    }
    return Case(**{**fields, **overrides})


def scripted(*replies: str):
    """A caller that returns the replies in order and records what it was sent."""
    queue = list(replies)
    seen: list[tuple[str, str]] = []

    async def call(system: str, user: str) -> str:
        seen.append((system, user))
        return queue.pop(0)

    return call, seen


def test_the_shipped_case_file_has_52_valid_cases_in_every_area():
    cases = load_cases(REPO_CASES)
    assert len(cases) == 52
    assert {c.area for c in cases} == set(AREAS)
    assert all(len(c.prompt) > 80 and len(c.trap) > 40 for c in cases)


def test_the_shipped_case_file_names_no_private_place():
    text = REPO_CASES.read_text(encoding="utf-8")
    for needle in ("/home/", "~/", "192.168.", "10.0."):
        assert needle.lower() not in text.lower(), needle


def test_a_trap_sentence_passes_its_own_keyword_grader_mostly():
    """The grader must be reachable: a reply that states the trap should usually pass."""
    cases = load_cases(REPO_CASES)
    reachable = sum(keyword_grade(c, c.trap)[0] for c in cases)
    assert reachable >= 40, reachable


def test_bad_case_files_are_refused(tmp_path: Path):
    good = 'id="a"\narea="verify"\nprompt="p"\ntrap="t"\nsignals=[["x"]]\n'
    for body, word in [
        (f"[[case]]\n{good}[[case]]\n{good}", "duplicate"),
        ("[[case]]\n" + good.replace("verify", "vibes"), "area"),
        ("[[case]]\n" + good.replace('[["x"]]', "[[]]"), "signals"),
        ("[[case]]\n" + good.replace('[["x"]]', '[["("]]'), "regex"),
        ('[[case]]\nid="a"\n', "missing"),
        ("", "no cases"),
    ]:
        path = tmp_path / "cases.toml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(EvalError, match=word):
            load_cases(path)


def test_keyword_grader_needs_every_group_and_no_fail_pattern():
    assert keyword_grade(case(), "The wrapper eats grep, so I ran pgrep.") == (True, "")
    assert keyword_grade(case(), "The wrapper eats grep.")[1] == "signal group 2 missing"
    passed, note = keyword_grade(case(), "Yes. The wrapper is odd but pgrep agrees.")
    assert not passed and note.startswith("fail_if")


def test_judge_verdict_parsing_and_unreadable_verdict_is_a_miss():
    assert parse_judge('noise {"caught": true, "why": "names it"} tail') == (True, "names it")
    assert parse_judge('{"caught": "yes"}') is None
    assert parse_judge("caught!") is None
    judge, seen = scripted("I think it is fine")
    assert asyncio.run(judge_grade(case(), "reply", judge)) == (False, "judge reply unreadable")
    assert "TRAP:\ngrep output is unreliable" in seen[0][1]


def test_playbooks_ship_with_the_harness_and_a_workspace_skill_shadows_them(tmp_path: Path):
    skills = load_skills(skill_roots(tmp_path, []))
    assert set(skills) >= PLAYBOOK_NAMES
    assert all(skills[name].description for name in PLAYBOOK_NAMES)
    own = tmp_path / ".claude" / "skills" / "reason-debug"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("---\nname: reason-debug\ndescription: mine\n---\nbody")
    assert load_skills(skill_roots(tmp_path, []))["reason-debug"].description == "mine"


def test_the_critic_agent_ships_read_only_and_asks_for_its_own_model(tmp_path: Path):
    critic = load_agents(agent_roots(tmp_path))["critic"]
    assert critic.model == "critic"
    assert critic.allows("Read") and not critic.allows("Bash") and not critic.allows("Edit")
    assert "VERDICT: PASS | REVISE | BLOCK" in critic.prompt


def test_the_three_arms_send_what_they_promise():
    writer, sent = scripted("bare reply")
    assert asyncio.run(answer_case(case(), "bare", writer, None)) == "bare reply"
    assert "Pre-delivery check" not in sent[0][0]

    writer, sent = scripted("with playbooks")
    asyncio.run(answer_case(case(), "playbooks", writer, None))
    assert "Pre-delivery check" in sent[0][0] and "Debugging procedure" in sent[0][0]
    assert playbook_text().count("\n---\n") == 3

    writer, sent = scripted("draft", "final")
    critic, critic_sent = scripted("VERDICT: REVISE")
    assert asyncio.run(answer_case(case(), "critic", writer, critic)) == "final"
    assert "You are the critic" in critic_sent[0][0] and "DRAFT:\ndraft" in critic_sent[0][1]
    assert "VERDICT: REVISE" in sent[1][1] and "Your first draft:\ndraft" in sent[1][1]

    with pytest.raises(EvalError, match="--critic"):
        asyncio.run(answer_case(case(), "critic", writer, None))


def test_run_writes_one_line_per_answer_and_a_failed_request_is_a_recorded_miss(tmp_path: Path):
    async def broken(system: str, user: str) -> str:
        raise EvalError("endpoint down")

    out = tmp_path / "r.jsonl"
    good, _ = scripted("the wrapper is unreliable, use pgrep")
    results = asyncio.run(run([case()], ["bare"], "m", good, None, None, out))
    results += asyncio.run(run([case(id="c2")], ["bare"], "m", broken, None, None, out))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
    assert rows[1]["note"] == "ERROR: endpoint down" and rows[1]["grader"] == "none"
    assert "1/2" in summarise(results)


def test_wilson_interval_is_wide_at_50_cases():
    low, high = wilson(25, 50)
    assert 0.36 < low < 0.37 and 0.63 < high < 0.64
    assert wilson(0, 0) == (0.0, 0.0)


def test_command_caller_pipes_the_prompt_and_reports_a_failing_command():
    assert asyncio.run(command_caller("cat")("SYS", "USER")) == "SYS\n\nUSER"
    with pytest.raises(EvalError, match="exit 3"):
        asyncio.run(command_caller("sh -c 'exit 3'")("s", "u"))


def test_cli_compare_and_bad_input(tmp_path: Path, capsys):
    path = tmp_path / "a.jsonl"
    row = Result("c1", "verify", "bare", "model-a", True, "keyword", "x")
    path.write_text(json.dumps(row.__dict__) + "\n")
    assert main(["--compare", str(path)]) == 0
    assert "model-a" in capsys.readouterr().out
    assert main(["--model", "cmd:cat", "--ids", "nope"]) == 2
    assert main(["--model", "cmd:cat", "--arms", "critic", "--limit", "1"]) == 2
