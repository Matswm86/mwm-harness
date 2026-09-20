"""Jev: request shape, the decision log, outcomes and the hit-rate report, the Judge tool."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mwm_harness import jev
from mwm_harness.jev import DecisionLog, JevClient, JevError, check_questions, report
from mwm_harness.providers import chunks_for
from mwm_harness.tools.judge import Judge

QUESTIONS = {
    "is_setup": {"type": "noul", "instructions": "Is a liquidity sweep described?"},
    "kind": {
        "type": "choice",
        "instructions": "What is the note about?",
        "criteria": {"trading": "markets", "infra": "servers"},
    },
    "grade": {"type": "score", "instructions": "Quality", "criteria": ["poor", "fair", "good"]},
}
ANSWERS = {
    "is_setup": {"type": "noul", "noul": 0.9},
    "kind": {
        "type": "choice",
        "choice": "trading",
        "probabilities": {"trading": 0.8, "infra": 0.2},
        "confidence": 0.7,
    },
    "grade": {"type": "score", "score": 1.6, "probabilities": {}, "confidence": 0.5},
}


def make_client(tmp_path, responses, seen=None):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    log = DecisionLog(tmp_path / "jev" / "decisions.jsonl")
    client = JevClient(api_key="k-test", log=log, transport=httpx.MockTransport(handler))
    return client, log


def ok(answers=ANSWERS):
    usage = {"input_tokens": 300, "output_tokens": 40}
    return (200, {"model": "jev-1.13.0", "answers": answers, "usage": usage})


def test_a_call_sends_the_documented_body_and_logs_the_decision(tmp_path):
    seen = []
    client, log = make_client(tmp_path, [ok()], seen)
    verdict = asyncio.run(client.ask("swept the Asia low", QUESTIONS, domain="trading", label="x"))
    body = json.loads(seen[0].content)
    assert seen[0].url == jev.API_URL and seen[0].headers["authorization"] == "Bearer k-test"
    assert body == {"state": "swept the Asia low", "model": "jev-latest", "questions": QUESTIONS}
    assert verdict.decisions() == {"is_setup": True, "kind": "trading", "grade": 2}
    (row,) = log.records()
    assert (row["kind"], row["id"], row["domain"], row["label"]) == (
        "decision",
        verdict.id,
        "trading",
        "x",
    )
    assert row["state_preview"] == "swept the Asia low" and len(row["state_sha256"]) == 64
    assert "k-test" not in log.path.read_text()


def test_outcomes_give_a_hit_rate_and_a_brier_score(tmp_path):
    miss = {"is_setup": {"type": "noul", "noul": 0.8}}
    client, log = make_client(tmp_path, [ok(), ok(miss)])
    first = asyncio.run(client.ask("a", QUESTIONS, domain="trading", label="sweep"))
    only_noul = {"is_setup": QUESTIONS["is_setup"]}
    second = asyncio.run(client.ask("b", only_noul, domain="trading", label="sweep"))
    log.outcome(first.id, "is_setup", True)
    log.outcome(first.id, "kind", "infra")
    log.outcome(first.id, "grade", 2)
    log.outcome(second.id, "is_setup", "false", note="no sweep on the chart")
    data = report(log)
    assert (data["calls"], data["questions_asked"], data["questions_with_outcome"]) == (2, 4, 4)
    group = data["groups"]["trading/sweep"]
    assert (group["judged"], group["hits"], group["hit_rate"]) == (4, 2, 0.5)
    assert group["brier"] == pytest.approx((0.1**2 + 0.8**2) / 2, abs=1e-4)
    assert "50.0%" in jev.format_report(data)


def test_a_later_outcome_for_the_same_question_replaces_the_earlier_one(tmp_path):
    client, log = make_client(tmp_path, [ok()])
    verdict = asyncio.run(client.ask("a", QUESTIONS))
    log.outcome(verdict.id, "is_setup", False)
    log.outcome(verdict.id, "is_setup", True)
    assert report(log)["groups"]["all"] == {"judged": 1, "hits": 1, "hit_rate": 1.0, "brier": 0.01}


def test_an_outcome_for_an_unknown_decision_or_question_is_refused(tmp_path):
    client, log = make_client(tmp_path, [ok()])
    verdict = asyncio.run(client.ask("a", QUESTIONS))
    with pytest.raises(JevError, match="no decision"):
        log.outcome("nope", "is_setup", True)
    with pytest.raises(JevError, match="has no question"):
        log.outcome(verdict.id, "typo", True)


def test_overload_is_retried_and_a_hard_failure_is_logged_not_hidden(tmp_path, monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(jev.asyncio, "sleep", no_sleep)
    client, log = make_client(tmp_path, [(529, {}), ok(), (401, {"detail": "bad key"})])
    assert asyncio.run(client.ask("a", QUESTIONS)).model == "jev-1.13.0"
    with pytest.raises(JevError, match="HTTP 401"):
        asyncio.run(client.ask("a", QUESTIONS, domain="brain"))
    kinds = [r["kind"] for r in log.records()]
    assert kinds == ["decision", "failure"]
    assert report(log)["failures"] == 1


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": {"type": "essay", "instructions": "x"}},
        {"q": {"type": "noul"}},
        {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}},
        {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}},
    ],
)
def test_malformed_questions_never_reach_the_api(questions):
    assert check_questions(questions)


def test_the_judge_tool_asks_for_approval_and_returns_the_decision_id(make_session, tmp_path):
    from conftest import FixedApprover

    client, log = make_client(tmp_path, [ok()])
    approver = FixedApprover(True)
    call = ("Judge", {"state": "note text", "questions": QUESTIONS, "domain": "research"})
    turns = [chunks_for(tool_calls=[call]), chunks_for("Jev says trading.")]
    session, _, _ = make_session(turns, approver=approver)
    session.tools["Judge"] = Judge(client)
    asyncio.run(session.send("classify this"))
    assert [name for name, _ in approver.asked] == ["Judge"]  # state leaves the machine
    (row,) = log.records()
    results = [b for m in session.messages for b in m.blocks() if b["type"] == "tool_result"]
    assert row["id"] in str(results[0]["content"]) and row["caller"] == "harness:Judge"
