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
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mwm_harness.config import ConfigError, load_secrets, workspace_root

API_URL = "https://api.typesafe.ai/v1/systemone"
KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-latest"
RETRY_STATUSES = (429, 529)
QUESTION_TYPES = ("noul", "choice", "score")
STATE_PREVIEW_CHARS = 400


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
    ) -> None:
        self._key = api_key
        self.log = log or DecisionLog()
        self._transport = transport
        self._retries = retries
        self._timeout = timeout

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
        model: str = DEFAULT_MODEL,
    ) -> Verdict:
        """Ask, log, return. A failed call is logged too and raises JevError."""
        problem = check_questions(questions)
        if problem:
            raise JevError(problem)
        if state in ("", None, [], {}):
            raise JevError("state is empty: Jev needs the thing to judge")
        try:
            verdict = await self._post({"state": state, "model": model, "questions": questions})
        except JevError as exc:
            self.log.failure(str(exc), domain, label, caller)
            raise
        self.log.decision(verdict, state, questions, domain, label, caller)
        return verdict

    async def _post(self, body: dict[str, Any]) -> Verdict:
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        started = time.monotonic()
        last = ""
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            for attempt in range(self._retries):
                try:
                    response = await client.post(API_URL, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    last = f"{type(exc).__name__}: {exc}"
                else:
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            answers = data["answers"]
                        except (ValueError, KeyError) as exc:
                            raise JevError(f"unreadable answer from Jev: {exc}") from exc
                        return Verdict(
                            id=uuid.uuid4().hex[:12],
                            answers=answers,
                            model=str(data.get("model", "")),
                            usage=data.get("usage") or {},
                            latency_ms=int((time.monotonic() - started) * 1000),
                        )
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code not in RETRY_STATUSES:
                        break
                if attempt + 1 < self._retries:
                    await asyncio.sleep(0.5 * 2**attempt)
        raise JevError(f"Jev request failed: {last}")


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
