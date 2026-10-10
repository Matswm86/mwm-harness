"""Jev (TypeSafe AI, a "System One" model): a second opinion, never the decider.

Jev answers typed questions about a piece of state: a yes/no probability
(noul), one option out of a closed set (choice), or a level on an ordered scale
(score). It writes no text and calls no tools, so it cannot be a chat model
here. It is a judge: a classifier for research and knowledge-base items, an
extra vote beside other evidence.

Every call is written to a decision log. The real result is added later with
``record_outcome``; ``report`` joins the two and gives the hit rate, so the
weight Jev gets is earned by its record and not by its vendor's claims.

API reference read 2026-09-20: POST https://api.typesafe.ai/v1/systemone,
``Authorization: Bearer``, body ``{state, model, questions}``; 401 bad key,
422 bad request, 429 rate limit, 529 overloaded.

Which model answers (``model="auto"``), per the 2026-10-10 bake-off scored
against Sonnet 5.5 labels:

- The brain-routing question (one choice over the five collections in
  ``ROUTE_KEYS``) goes to the local fine-tuned 4B in Ollama: 0.920 on n=200
  against Clef 0.885 and Jev 0.850. If Ollama fails, the call is logged and
  retried on Clef, else Jev.
- Every other question goes to Jev: it beat Clef on passage relevance (0.688
  vs 0.647, n=400) and on answer checks (0.733 vs 0.633, n=150). Yes/no and
  score questions were not re-tested and stay on Jev.
- When Jev is less sure than the cut-off in ``ESCALATION_CUTOFFS`` on a
  question shape listed there, Sonnet answers instead. On a held-out half of
  the same items: relevance 0.921 with 56% sent to Sonnet (cut-off 0.78),
  answer checks 0.889 with 29% sent (cut-off 0.69). Every blend with Clef
  needed as many Sonnet calls or more. If Sonnet fails, Jev's answer stands
  and the failure is logged. ``MWM_JEV_ESCALATE=0`` turns escalation off.
- If Jev itself fails, the call is logged and retried on Clef when Clef
  credentials exist. A Clef answer is never escalated (no measured cut-off).

Clef is otherwise used only when a caller asks for it by name.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mwm_harness.config import ConfigError, load_secrets, workspace_root

API_URL = "https://api.typesafe.ai/v1/systemone"
KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-latest"
AUTO_MODEL = "auto"
CLEF_MODELS = ("clef", "clef-flash")
CLEF_DEFAULT = "clef"
CLEF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
CF_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
CF_ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"
ROUTE_KEYS = ("codebase", "creative_writing", "research", "science_writing", "textbooks")
ROUTE_MODEL_ENV = "MWM_JEV_ROUTE_MODEL"
ROUTE_MODEL_DEFAULT = "routing-4b-ft-q4km"
OLLAMA_URL = "http://localhost:11434/api/chat"
LETTERS = "ABCDE"
RETRY_STATUSES = (429, 529)
QUESTION_TYPES = ("noul", "choice", "score")
STATE_PREVIEW_CHARS = 400
SONNET_MODEL = "claude-sonnet-5-5"
ESCALATE_ENV = "MWM_JEV_ESCALATE"
# Option set of a one-question choice -> lowest Jev confidence that is kept.
ESCALATION_CUTOFFS = {
    frozenset({"2", "1", "0"}): 0.78,  # passage relevance
    frozenset({"full", "partial", "no"}): 0.69,  # does the top passage answer the question
}
SONNET_SYSTEM = (
    "You are a judge. You get a STATE and one QUESTION with lettered options. "
    "Reply with only the option key that answers the question, nothing else."
)


class JevError(Exception):
    """The request failed or was malformed. The caller carries on without Jev's vote."""


def log_path() -> Path:
    """One log for every caller, so the hit rate covers all of them."""
    override = os.environ.get("MWM_JEV_LOG")
    if override:
        return Path(override).expanduser()
    workspace = workspace_root()
    if workspace.is_dir():
        return workspace / "data" / "jev" / "decisions.jsonl"
    return Path.home() / ".local" / "share" / "mwm-harness" / "jev" / "decisions.jsonl"


def check_questions(questions: Any) -> str | None:
    """Error text for a question map the API would reject, else None."""
    if not isinstance(questions, dict) or not questions:
        return "questions must be a non-empty object: {name: {type, instructions, criteria}}"
    for name, question in questions.items():
        if not isinstance(question, dict):
            return f"question {name!r} must be an object"
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            return f"question {name!r}: type must be one of {', '.join(QUESTION_TYPES)}"
        if not question.get("instructions"):
            return f"question {name!r}: instructions are required"
        criteria = question.get("criteria")
        if kind == "choice" and not (isinstance(criteria, dict) and 2 <= len(criteria) <= 255):
            return f"question {name!r}: a choice needs criteria = {{option: description}}, 2-255"
        if kind == "score" and not (isinstance(criteria, list) and 2 <= len(criteria) <= 10):
            return f"question {name!r}: a score needs criteria = [level descriptions], 2-10"
    return None


def is_route_question(questions: dict[str, Any]) -> bool:
    """True for one choice question over exactly the five brain collections."""
    if len(questions) != 1:
        return False
    (question,) = questions.values()
    criteria = question.get("criteria")
    return (
        question.get("type") == "choice"
        and isinstance(criteria, dict)
        and set(criteria) == set(ROUTE_KEYS)
    )


def escalation_cutoff(questions: dict[str, Any]) -> float | None:
    """The Jev confidence below which Sonnet answers, for a measured question shape; else None."""
    if os.environ.get(ESCALATE_ENV, "1").strip() == "0" or len(questions) != 1:
        return None
    (question,) = questions.values()
    criteria = question.get("criteria")
    if question.get("type") != "choice" or not isinstance(criteria, dict):
        return None
    return ESCALATION_CUTOFFS.get(frozenset(str(k) for k in criteria))


def choice_confidence(answer: dict[str, Any]) -> float:
    """Jev's probability for the option it chose (0.0 when it gave none)."""
    probs = answer.get("probabilities") or {}
    try:
        return float(probs.get(answer.get("choice"), 0.0))
    except (TypeError, ValueError):
        return 0.0


def route_model() -> str:
    """The Ollama model for routing; MWM_JEV_ROUTE_MODEL overrides it, an empty value sends routing to Jev."""
    return os.environ.get(ROUTE_MODEL_ENV, ROUTE_MODEL_DEFAULT)


def decision_of(answer: dict[str, Any]) -> Any:
    """The one value a caller acts on: a bool, an option name, or a level number."""
    kind = answer.get("type")
    if kind == "noul":
        return float(answer.get("noul", 0.0)) >= 0.5
    if kind == "choice":
        return answer.get("choice")
    if kind == "score":
        return round(float(answer.get("score", 0.0)))
    return None


@dataclass
class Verdict:
    """One logged call. ``id`` is what ``record_outcome`` needs later."""

    id: str
    answers: dict[str, dict[str, Any]]
    model: str
    usage: dict[str, int]
    latency_ms: int

    def decisions(self) -> dict[str, Any]:
        return {name: decision_of(answer) for name, answer in self.answers.items()}


@dataclass
class DecisionLog:
    path: Path = field(default_factory=log_path)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def decision(
        self,
        verdict: Verdict,
        state: Any,
        questions: dict[str, Any],
        domain: str,
        label: str,
        caller: str,
    ) -> None:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        self._append(
            {
                "kind": "decision",
                "id": verdict.id,
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "domain": domain,
                "label": label,
                "caller": caller,
                "model": verdict.model,
                "latency_ms": verdict.latency_ms,
                "usage": verdict.usage,
                "state_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "state_preview": text[:STATE_PREVIEW_CHARS],
                "questions": questions,
                "answers": verdict.answers,
                "decisions": verdict.decisions(),
            }
        )

    def failure(self, error: str, domain: str, label: str, caller: str) -> None:
        self._append(
            {
                "kind": "failure",
                "id": uuid.uuid4().hex[:12],
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "domain": domain,
                "label": label,
                "caller": caller,
                "error": error[:500],
            }
        )

    def outcome(self, decision_id: str, question: str, truth: Any, note: str = "") -> None:
        """What really happened. Appended, never edited in: the log stays append-only."""
        known = {r["id"]: r for r in self.records() if r.get("kind") == "decision" and "id" in r}
        if decision_id not in known:
            raise JevError(f"no decision with id {decision_id} in {self.path}")
        if question not in known[decision_id]["answers"]:
            names = ", ".join(known[decision_id]["answers"])
            raise JevError(f"decision {decision_id} has no question {question!r}; it has: {names}")
        self._append(
            {
                "kind": "outcome",
                "id": decision_id,
                "question": question,
                "truth": truth,
                "note": note,
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a killed writer; the rest still counts
        return rows


def _same(decision: Any, truth: Any) -> bool:
    if isinstance(decision, bool) or isinstance(truth, bool):
        return bool(decision) == _as_bool(truth)
    if isinstance(decision, int):
        try:
            return decision == round(float(truth))
        except (TypeError, ValueError):
            return False
    return str(decision) == str(truth)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def report(log: DecisionLog | None = None) -> dict[str, Any]:
    """Hit rate per domain and label, over decisions whose outcome is known.

    For yes/no questions the Brier score is given too (mean squared gap between
    the stated probability and what happened; 0.25 = a coin that always says
    50%, lower is better). It tests the vendor's "calibrated probabilities".
    """
    log = log or DecisionLog()
    rows = log.records()
    decisions = {r["id"]: r for r in rows if r.get("kind") == "decision"}
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == "outcome":
            outcomes[(row["id"], row["question"])] = row  # the last word wins
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"judged": 0, "hits": 0, "brier_sum": 0.0, "brier_n": 0}
    )
    for (decision_id, question), outcome in outcomes.items():
        record = decisions.get(decision_id)
        if record is None or question not in record["answers"]:
            continue
        answer = record["answers"][question]
        hit = _same(decision_of(answer), outcome["truth"])
        for key in ("all", f"{record['domain']}", f"{record['domain']}/{record['label']}"):
            group = groups[key]
            group["judged"] += 1
            group["hits"] += int(hit)
            if answer.get("type") == "noul":
                gap = float(answer.get("noul", 0.0)) - float(_as_bool(outcome["truth"]))
                group["brier_sum"] += gap * gap
                group["brier_n"] += 1
    asked = sum(len(r["answers"]) for r in decisions.values())
    table = {}
    for key, group in sorted(groups.items()):
        table[key] = {
            "judged": group["judged"],
            "hits": group["hits"],
            "hit_rate": round(group["hits"] / group["judged"], 4),
            "brier": round(group["brier_sum"] / group["brier_n"], 4) if group["brier_n"] else None,
        }
    return {
        "log": str(log.path),
        "calls": len(decisions),
        "failures": sum(1 for r in rows if r.get("kind") == "failure"),
        "questions_asked": asked,
        "questions_with_outcome": len(outcomes),
        "groups": table,
    }


def format_report(data: dict[str, Any]) -> str:
    lines = [
        f"Jev decision log: {data['log']}",
        f"{data['calls']} calls, {data['questions_asked']} questions, "
        f"{data['questions_with_outcome']} with a known outcome, {data['failures']} failed calls",
    ]
    if not data["groups"]:
        lines.append("No outcomes recorded yet, so there is no hit rate. Record one with:")
        lines.append("  mwm-jev outcome DECISION_ID QUESTION TRUTH")
        return "\n".join(lines)
    lines.append(f"{'group':40} {'judged':>6} {'hits':>5} {'hit rate':>9} {'Brier':>7}")
    for key, row in data["groups"].items():
        brier = "" if row["brier"] is None else f"{row['brier']:.3f}"
        lines.append(
            f"{key[:40]:40} {row['judged']:>6} {row['hits']:>5} {row['hit_rate']:>8.1%} {brier:>7}"
        )
    lines.append(
        "At 30 judged the 95% interval is still about +/-18pp: read a rate below that as noise."
    )
    return "\n".join(lines)


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        log: DecisionLog | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retries: int = 3,
        timeout: float = 30.0,
        cloudflare: tuple[str, str] | tuple[()] | None = None,
        sonnet: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self._key = api_key
        self.log = log or DecisionLog()
        self._transport = transport
        self._retries = retries
        self._timeout = timeout
        self._cf = cloudflare
        self._sonnet = sonnet

    def _api_key(self) -> str:
        key = self._key or os.environ.get(KEY_ENV) or load_secrets().get(KEY_ENV, "")
        if not key:
            raise ConfigError(f"no Jev key: set {KEY_ENV} in the environment or in secrets.env")
        return key

    async def ask(
        self,
        state: Any,
        questions: dict[str, Any],
        domain: str = "other",
        label: str = "",
        caller: str = "",
        model: str = AUTO_MODEL,
    ) -> Verdict:
        """Ask, log, return. A failed call is logged too and raises JevError.

        ``model="auto"`` sends the brain-routing question to the local
        routing model and everything else to Jev (see the module docstring).
        """
        problem = check_questions(questions)
        if problem:
            raise JevError(problem)
        if state in ("", None, [], {}):
            raise JevError("state is empty: Jev needs the thing to judge")
        auto = model == AUTO_MODEL
        if auto:
            model = route_model() if is_route_question(questions) else DEFAULT_MODEL
            model = model or DEFAULT_MODEL
        body = {"state": state, "model": model, "questions": questions}
        try:
            if auto and model != DEFAULT_MODEL:
                try:
                    verdict = await self._post_ollama_route(body)
                except JevError as exc:
                    fallback = CLEF_DEFAULT if self._cloudflare() else DEFAULT_MODEL
                    self.log.failure(f"{exc}; retried on {fallback}", domain, label, caller)
                    body["model"] = fallback
                    if fallback == CLEF_DEFAULT:
                        verdict = await self._post_clef(body)
                    else:
                        verdict = await self._post(body)
            elif model in CLEF_MODELS:
                verdict = await self._post_clef(body)
            elif auto and self._cloudflare():
                try:
                    verdict = await self._post(body)
                except JevError as exc:
                    self.log.failure(f"{exc}; retried on {CLEF_DEFAULT}", domain, label, caller)
                    body["model"] = CLEF_DEFAULT
                    verdict = await self._post_clef(body)
            else:
                verdict = await self._post(body)
        except JevError as exc:
            self.log.failure(str(exc), domain, label, caller)
            raise
        if auto and body["model"] == DEFAULT_MODEL:
            verdict = await self._maybe_escalate(verdict, state, questions, domain, label, caller)
        self.log.decision(verdict, state, questions, domain, label, caller)
        return verdict

    async def _maybe_escalate(
        self,
        verdict: Verdict,
        state: Any,
        questions: dict[str, Any],
        domain: str,
        label: str,
        caller: str,
    ) -> Verdict:
        """Swap in Sonnet's answer when Jev is below the measured cut-off; keep Jev's if Sonnet fails."""
        cutoff = escalation_cutoff(questions)
        if cutoff is None:
            return verdict
        ((name, question),) = questions.items()
        if name not in verdict.answers:
            return verdict
        jev_answer = verdict.answers[name]
        confidence = choice_confidence(jev_answer)
        if confidence >= cutoff:
            return verdict
        started = time.monotonic()
        try:
            choice = await asyncio.to_thread(self._ask_sonnet, state, question)
        except JevError as exc:
            self.log.failure(
                f"Sonnet escalation failed, kept Jev's answer: {exc}", domain, label, caller
            )
            return verdict
        answer = {
            "type": "choice",
            "choice": choice,
            "escalated": {"from": verdict.model, "confidence": confidence, "cutoff": cutoff},
            "jev": jev_answer,
        }
        return Verdict(
            id=verdict.id,
            answers={name: answer},
            model=f"{SONNET_MODEL} (escalated from {verdict.model})",
            usage=verdict.usage,
            latency_ms=verdict.latency_ms + int((time.monotonic() - started) * 1000),
        )

    def _ask_sonnet(self, state: Any, question: dict[str, Any]) -> str:
        """One option key from Sonnet through the workspace's `claude -p` wrapper."""
        criteria = {str(k): v for k, v in question["criteria"].items()}
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        options = "\n".join(f"{key}: {desc}" for key, desc in criteria.items())
        prompt = (
            f"STATE:\n{text}\n\nQUESTION: {question['instructions']}\n\n"
            f"OPTIONS:\n{options}\n\nReply with one key from: {', '.join(criteria)}"
        )
        call = self._sonnet or _workspace_sonnet()
        reply = (call(SONNET_SYSTEM, prompt) or "").strip().strip("`'\".").strip()
        if reply in criteria:
            return reply
        hits = [k for k in criteria if reply.lower().split()[:1] == [k.lower()]]
        if len(hits) == 1:
            return hits[0]
        raise JevError(f"Sonnet gave no option key: {reply[:80]!r}")

    async def _post_ollama_route(self, body: dict[str, Any]) -> Verdict:
        """Ask the local routing model for the option letter; probabilities come from its first-token logprobs."""
        ((name, question),) = body["questions"].items()
        criteria = question["criteria"]
        opts = "\n".join(f"{LETTERS[i]}. {criteria[k]}" for i, k in enumerate(ROUTE_KEYS))
        prompt = (
            f"Text chunk:\n<<<\n{str(body['state'])[:1500]}\n>>>\n\n"
            f"Which knowledge-base collection does this text chunk belong in?\n{opts}\n\n"
            "Reply with exactly one letter."
        )
        request = {
            "model": body["model"],
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {"temperature": 0, "num_predict": 1},
            "messages": [{"role": "user", "content": prompt}],
        }
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(OLLAMA_URL, json=request)
            response.raise_for_status()
            tops = response.json()["logprobs"][0]["top_logprobs"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise JevError(f"local routing model failed: {type(exc).__name__}: {exc}") from exc
        raw = {k: 0.0 for k in ROUTE_KEYS}
        for top in tops:
            letter = str(top.get("token", "")).strip().upper()
            if letter and letter in LETTERS:
                raw[ROUTE_KEYS[LETTERS.index(letter)]] += math.exp(float(top["logprob"]))
        total = sum(raw.values())
        if not total:
            raise JevError("local routing model answered without an option letter")
        probs = {k: v / total for k, v in raw.items()}
        choice = max(probs, key=probs.get)
        return Verdict(
            id=uuid.uuid4().hex[:12],
            answers={name: {"type": "choice", "choice": choice, "probabilities": probs}},
            model=body["model"],
            usage={},
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _cloudflare(self) -> tuple[str, str] | None:
        """(account id, token) for Workers AI, or None when either is missing."""
        if self._cf is None:
            secrets = load_secrets()
            token = os.environ.get(CF_TOKEN_ENV) or secrets.get(CF_TOKEN_ENV, "")
            account = os.environ.get(CF_ACCOUNT_ENV) or secrets.get(CF_ACCOUNT_ENV, "")
            self._cf = (account, token) if token and account else ()
        return self._cf or None

    async def _post_clef(self, body: dict[str, Any]) -> Verdict:
        creds = self._cloudflare()
        if creds is None:
            raise JevError(f"no Clef credentials: set {CF_TOKEN_ENV} and {CF_ACCOUNT_ENV}")
        account, token = creds
        url = CLEF_URL.format(account=account, model=body["model"])
        return await self._post(body, url=url, token=token, wrapped=True, vendor="Clef")

    async def _post(
        self,
        body: dict[str, Any],
        url: str = API_URL,
        token: str | None = None,
        wrapped: bool = False,
        vendor: str = "Jev",
    ) -> Verdict:
        """POST with retries. ``wrapped`` reads Cloudflare's {success, result: {...}} envelope."""
        headers = {"Authorization": f"Bearer {token or self._api_key()}"}
        started = time.monotonic()
        last = ""
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            for attempt in range(self._retries):
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    last = f"{type(exc).__name__}: {exc}"
                else:
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if wrapped:
                                data = data["result"]
                            answers = data["answers"]
                        except (ValueError, KeyError, TypeError) as exc:
                            raise JevError(f"unreadable answer from {vendor}: {exc}") from exc
                        return Verdict(
                            id=uuid.uuid4().hex[:12],
                            answers=answers,
                            model=str(data.get("model", "")) or body["model"],
                            usage=data.get("usage") or {},
                            latency_ms=int((time.monotonic() - started) * 1000),
                        )
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code not in RETRY_STATUSES:
                        break
                if attempt + 1 < self._retries:
                    await asyncio.sleep(0.5 * 2**attempt)
        raise JevError(f"{vendor} request failed: {last}")


def _workspace_sonnet() -> Callable[[str, str], str | None]:
    """The workspace's `claude -p` wrapper (core/anthropic_via_claude_cli.py), bound to Sonnet."""
    root = str(workspace_root())
    if root not in sys.path:
        sys.path.append(root)
    try:
        from core.anthropic_via_claude_cli import call_claude_cli  # noqa: PLC0415
    except ImportError as exc:
        raise JevError(f"no Sonnet path: {exc}") from exc

    def call(system: str, prompt: str) -> str | None:
        return call_claude_cli(model=SONNET_MODEL, system_prompt=system, user_prompt=prompt)

    return call


def judge(state: Any, questions: dict[str, Any], **fields: Any) -> Verdict:
    """Blocking form for scripts: ``judge(text, {"q": {...}}, domain="trading", label="x")``."""
    return asyncio.run(JevClient().ask(state, questions, **fields))


# ------------------------------------------------------------------ command


def _parse_truth(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mwm-jev", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask Jev; the state comes from --state or stdin")
    ask.add_argument("--questions", required=True, help="JSON object, or @file.json")
    ask.add_argument("--state", help="text to judge (default: read stdin)")
    ask.add_argument("--domain", default="other", help="research, brain, trading, ...")
    ask.add_argument("--label", default="", help="what is being judged, e.g. a strategy name")
    ask.add_argument("--caller", default="mwm-jev")
    outcome = sub.add_parser("outcome", help="record what really happened for one question")
    outcome.add_argument("decision_id")
    outcome.add_argument("question")
    outcome.add_argument("truth", help="true/false, an option name, or a level number")
    outcome.add_argument("--note", default="")
    rep = sub.add_parser("report", help="hit rate per domain and label")
    rep.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.command == "ask":
            raw = args.questions
            questions = json.loads(Path(raw[1:]).read_text() if raw.startswith("@") else raw)
            state = args.state if args.state is not None else sys.stdin.read()
            verdict = judge(
                state, questions, domain=args.domain, label=args.label, caller=args.caller
            )
            print(
                json.dumps(
                    {
                        "id": verdict.id,
                        "decisions": verdict.decisions(),
                        "answers": verdict.answers,
                        "latency_ms": verdict.latency_ms,
                    },
                    ensure_ascii=False,
                )
            )
        elif args.command == "outcome":
            DecisionLog().outcome(
                args.decision_id, args.question, _parse_truth(args.truth), args.note
            )
            print(f"recorded: {args.decision_id} {args.question} = {args.truth}")
        else:
            data = report()
            print(json.dumps(data, indent=2) if args.json else format_report(data))
    except (JevError, ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"mwm-jev: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
