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


def make_client(tmp_path, responses, seen=None, cloudflare=()):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    log = DecisionLog(tmp_path / "jev" / "decisions.jsonl")
    client = JevClient(
        api_key="k-test", log=log, transport=httpx.MockTransport(handler), cloudflare=cloudflare
    )
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


CHOICE_ONLY = {"kind": QUESTIONS["kind"]}
CF = ("acct-1", "cf-test")


def clef_ok():
    result = {"model": "clef", "answers": {"kind": ANSWERS["kind"]}, "usage": {"input_tokens": 200}}
    return (200, {"success": True, "result": result, "errors": []})


def test_choice_only_calls_go_to_jev_even_with_cloudflare(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, [ok({"kind": ANSWERS["kind"]})], seen, cloudflare=CF)
    asyncio.run(client.ask("a", CHOICE_ONLY, domain="brain"))
    assert seen[0].url == jev.API_URL


def test_mixed_questions_stay_on_jev_even_with_cloudflare(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, [ok()], seen, cloudflare=CF)
    asyncio.run(client.ask("a", QUESTIONS))
    assert seen[0].url == jev.API_URL


def test_an_explicit_clef_model_goes_to_cloudflare(tmp_path):
    seen = []
    client, log = make_client(tmp_path, [clef_ok()], seen, cloudflare=CF)
    verdict = asyncio.run(client.ask("a", CHOICE_ONLY, domain="brain", model="clef"))
    assert str(seen[0].url) == jev.CLEF_URL.format(account="acct-1", model="clef")
    assert seen[0].headers["authorization"] == "Bearer cf-test"
    assert verdict.model == "clef" and verdict.decisions() == {"kind": "trading"}
    assert [r["kind"] for r in log.records()] == ["decision"]


ROUTE = {
    "route": {
        "type": "choice",
        "instructions": "Which knowledge-base collection does this text chunk belong in?",
        "criteria": {k: f"{k} text" for k in jev.ROUTE_KEYS},
    }
}


def ollama_ok(letter="D"):
    tops = [{"token": letter, "logprob": -0.1}, {"token": "B", "logprob": -2.5}]
    return (
        200,
        {"message": {"content": letter}, "logprobs": [{"token": letter, "top_logprobs": tops}]},
    )


def test_the_routing_question_goes_to_the_local_model(tmp_path, monkeypatch):
    monkeypatch.delenv(jev.ROUTE_MODEL_ENV, raising=False)
    seen = []
    client, log = make_client(tmp_path, [ollama_ok("D")], seen, cloudflare=CF)
    verdict = asyncio.run(client.ask("a chunk about physics", ROUTE, domain="brain"))
    assert str(seen[0].url) == jev.OLLAMA_URL
    body = json.loads(seen[0].content)
    assert (
        body["model"] == jev.ROUTE_MODEL_DEFAULT
        and "D. science_writing text" in body["messages"][0]["content"]
    )
    assert verdict.model == jev.ROUTE_MODEL_DEFAULT
    assert verdict.decisions() == {"route": "science_writing"}
    probs = verdict.answers["route"]["probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-9 and probs["science_writing"] > 0.9
    assert [r["kind"] for r in log.records()] == ["decision"]


def test_a_failed_local_route_is_logged_and_retried_on_clef(tmp_path, monkeypatch):
    monkeypatch.delenv(jev.ROUTE_MODEL_ENV, raising=False)
    answer = {"route": {"type": "choice", "choice": "research", "probabilities": {}}}
    result = {"model": "clef", "answers": answer, "usage": {}}
    seen = []
    client, log = make_client(
        tmp_path, [(500, {"error": "busy"}), (200, {"success": True, "result": result})], seen, CF
    )
    verdict = asyncio.run(client.ask("a", ROUTE))
    assert "cloudflare" in str(seen[1].url) and verdict.decisions() == {"route": "research"}
    kinds = [r["kind"] for r in log.records()]
    assert kinds == ["failure", "decision"] and "retried on clef" in log.records()[0]["error"]


def test_a_failed_local_route_without_cloudflare_falls_back_to_jev(tmp_path, monkeypatch):
    monkeypatch.delenv(jev.ROUTE_MODEL_ENV, raising=False)
    answer = {"route": {"type": "choice", "choice": "codebase", "probabilities": {}}}
    seen = []
    client, log = make_client(tmp_path, [(200, {"logprobs": []}), ok(answer)], seen)
    verdict = asyncio.run(client.ask("a", ROUTE))
    assert seen[1].url == jev.API_URL and verdict.decisions() == {"route": "codebase"}
    assert [r["kind"] for r in log.records()] == ["failure", "decision"]


def test_an_empty_route_model_setting_sends_routing_to_jev(tmp_path, monkeypatch):
    monkeypatch.setenv(jev.ROUTE_MODEL_ENV, "")
    answer = {"route": {"type": "choice", "choice": "codebase", "probabilities": {}}}
    seen = []
    client, _ = make_client(tmp_path, [ok(answer)], seen)
    asyncio.run(client.ask("a", ROUTE))
    assert seen[0].url == jev.API_URL


def test_a_choice_over_other_options_is_not_the_routing_question():
    assert not jev.is_route_question(CHOICE_ONLY)
    assert not jev.is_route_question({**ROUTE, "x": QUESTIONS["is_setup"]})
    assert jev.is_route_question(ROUTE)


def test_an_explicit_clef_model_does_not_fall_back(tmp_path):
    client, log = make_client(tmp_path, [(401, {"success": False})], cloudflare=CF)
    with pytest.raises(JevError, match="Clef request failed"):
        asyncio.run(client.ask("a", CHOICE_ONLY, model="clef"))
    assert [r["kind"] for r in log.records()] == ["failure"]


REL_Q = {
    "rel": {
        "type": "choice",
        "instructions": "How well does the PASSAGE answer the QUERY?",
        "criteria": {"2": "answers it", "1": "partly", "0": "not relevant"},
    }
}


def rel_answer(choice, p):
    rest = {k: (1 - p) / 2 for k in ("2", "1", "0") if k != choice}
    return {"rel": {"type": "choice", "choice": choice, "probabilities": {choice: p, **rest}}}


def test_unsure_jev_is_escalated_to_sonnet_and_both_answers_are_logged(tmp_path):
    asked = []
    client, log = make_client(tmp_path, [ok(rel_answer("1", 0.60))])
    client._sonnet = lambda system, prompt: asked.append(prompt) or "2"
    verdict = asyncio.run(client.ask("QUERY: q\n\nPASSAGE: p", REL_Q, domain="brain"))
    assert verdict.decisions() == {"rel": "2"}
    assert "PASSAGE: p" in asked[0] and "2: answers it" in asked[0]
    (row,) = log.records()
    assert row["model"].startswith(jev.SONNET_MODEL)
    assert row["answers"]["rel"]["jev"]["choice"] == "1"
    assert row["answers"]["rel"]["escalated"]["cutoff"] == 0.78


def test_sure_jev_is_kept_and_sonnet_is_not_called(tmp_path):
    client, log = make_client(tmp_path, [ok(rel_answer("1", 0.91))])
    client._sonnet = lambda system, prompt: pytest.fail("Sonnet must not be asked")
    assert asyncio.run(client.ask("x", REL_Q)).decisions() == {"rel": "1"}


def test_failed_sonnet_keeps_jevs_answer_and_logs_the_failure(tmp_path):
    client, log = make_client(tmp_path, [ok(rel_answer("0", 0.50))])
    client._sonnet = lambda system, prompt: "I think it is somewhat relevant"
    assert asyncio.run(client.ask("x", REL_Q)).decisions() == {"rel": "0"}
    kinds = [r["kind"] for r in log.records()]
    assert kinds == ["failure", "decision"]
    assert "kept Jev" in log.records()[0]["error"]


def test_escalation_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv(jev.ESCALATE_ENV, "0")
    client, _ = make_client(tmp_path, [ok(rel_answer("1", 0.40))])
    client._sonnet = lambda system, prompt: pytest.fail("Sonnet must not be asked")
    assert asyncio.run(client.ask("x", REL_Q)).decisions() == {"rel": "1"}


def test_unmeasured_question_shapes_are_never_escalated():
    assert jev.escalation_cutoff(QUESTIONS) is None
    assert jev.escalation_cutoff({"kind": QUESTIONS["kind"]}) is None
    answers_q = {
        "a": {
            "type": "choice",
            "instructions": "i",
            "criteria": {"full": "", "partial": "", "no": ""},
        }
    }
    assert jev.escalation_cutoff(answers_q) == 0.69


def test_jev_down_retries_on_clef_and_does_not_escalate(tmp_path):
    seen = []
    clef = (200, {"success": True, "result": {"model": "clef", "answers": rel_answer("2", 0.4)}})
    client, log = make_client(
        tmp_path, [(401, {"error": "bad key"}), clef], seen, cloudflare=("acct", "tok")
    )
    client._sonnet = lambda system, prompt: pytest.fail("Clef answers are not escalated")
    verdict = asyncio.run(client.ask("x", REL_Q))
    assert verdict.decisions() == {"rel": "2"} and "cloudflare" in str(seen[1].url)
    assert [r["kind"] for r in log.records()] == ["failure", "decision"]
