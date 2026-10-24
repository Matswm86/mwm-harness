"""Reasoning eval: does a model catch the trap in a situation, and does scaffolding help?

Each case in ``evals/cases.toml`` is an ordinary request with one trap in its
facts. Four arms answer it:

- ``bare``       the model with a one-line system prompt
- ``playbooks``  the same model with the four reasoning playbooks in its system prompt
- ``critic``     the model drafts, a second model criticises with the critic prompt,
                 the first model writes the final reply
- ``diagnose``   the item 31 diagnosis runs on the case text before the model answers:
                 a second model reads the failure and its note lands in the request as
                 the ``<system-reminder>`` meta message the session injects

A reply passes when it names the trap. Two graders: ``keyword`` (regex groups from
the case file, offline and repeatable, crude) and ``judge`` (a model reads the
trap sentence and the reply and answers caught or not).

A model is a ``models.toml`` id, or ``cmd:<shell command>`` for anything that
reads a prompt on stdin and prints a reply, which is how a reference model
behind another CLI takes the same test.

Single turn, no tools: this measures noticing, not agent behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import shlex
import sys
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from mwm_harness.agents import BUILTIN_AGENTS
from mwm_harness.config import ConfigError, ModelSpec, load_models
from mwm_harness.diagnose import run_diagnosis
from mwm_harness.loop import reminder
from mwm_harness.messages import Message
from mwm_harness.providers import OpenAICompatProvider, Provider, ProviderError
from mwm_harness.skills import PLAYBOOKS, split_frontmatter
from mwm_harness.streaming import StreamAssembler

REPO_CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.toml"
AREAS = ("verify", "debug", "audit", "challenge")
ARMS = ("bare", "playbooks", "critic", "diagnose")
BASE_SYSTEM = (
    "You are an engineering assistant working with one person on their own systems. "
    "Reply to the request below as you would in the session."
)
COMMAND_TIMEOUT = 600.0
# A small model can fall into a loop and write until its context is full (seen live:
# 6,900 tokens and 20 minutes on one case). A reply to these cases needs a few hundred.
MAX_TOKENS = 2000
# The diagnoser is a reasoning model and its thinking is billed against the same budget as its
# answer: at 2,000 it stopped mid-object on 4 of 9 live diagnoses. The JSON shape needs hundreds
# of tokens at most, so the room goes on the thinking.
CRITIC_MAX_TOKENS = 6000

# (system, user) -> reply
Caller = Callable[[str, str], Awaitable[str]]


class EvalError(Exception):
    """The case file is malformed or a model could not be reached."""


@dataclass
class Case:
    id: str
    area: str
    prompt: str
    trap: str
    signals: list[list[str]]
    fail_if: list[str] = field(default_factory=list)


@dataclass
class Result:
    case: str
    area: str
    arm: str
    model: str
    passed: bool
    grader: str
    answer: str
    note: str = ""


# ------------------------------------------------------------------ cases


def load_cases(path: Path) -> list[Case]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EvalError(f"{path}: {exc}") from exc
    cases: list[Case] = []
    seen: set[str] = set()
    for index, raw in enumerate(data.get("case") or []):
        try:
            case = Case(
                id=raw["id"],
                area=raw["area"],
                prompt=raw["prompt"].strip(),
                trap=raw["trap"].strip(),
                signals=[list(group) for group in raw["signals"]],
                fail_if=list(raw.get("fail_if") or []),
            )
        except (KeyError, TypeError) as exc:
            raise EvalError(f"{path}: case {index}: missing or bad field {exc}") from exc
        if case.id in seen:
            raise EvalError(f"{path}: duplicate case id {case.id}")
        if case.area not in AREAS:
            raise EvalError(f"{path}: {case.id}: area {case.area!r} is not one of {AREAS}")
        if not case.signals or any(not group for group in case.signals):
            raise EvalError(f"{path}: {case.id}: signals needs at least one non-empty group")
        for pattern in [p for group in case.signals for p in group] + case.fail_if:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise EvalError(f"{path}: {case.id}: bad regex {pattern!r}: {exc}") from exc
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvalError(f"{path}: no cases")
    return cases


# ---------------------------------------------------------------- graders


def keyword_grade(case: Case, answer: str) -> tuple[bool, str]:
    """Every signal group needs one hit, and no ``fail_if`` pattern may hit."""
    flags = re.IGNORECASE | re.MULTILINE
    for pattern in case.fail_if:
        if re.search(pattern, answer, flags):
            return False, f"fail_if hit: {pattern}"
    for number, group in enumerate(case.signals, start=1):
        if not any(re.search(pattern, answer, flags) for pattern in group):
            return False, f"signal group {number} missing"
    return True, ""


JUDGE_SYSTEM = (
    "You grade one reply. You are given a situation, the trap hidden in it, and a reply. "
    "Decide whether the reply NAMES the trap and lets it change what the reply says or does. "
    "A reply that walks into the trap fails even if it is well written. A reply that hedges in "
    "general terms without naming this trap fails. "
    'Answer with one JSON object and nothing else: {"caught": true or false, "why": "<one sentence>"}'
)


def parse_judge(text: str) -> tuple[bool, str] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("caught"), bool):
        return None
    return data["caught"], str(data.get("why", ""))


async def judge_grade(case: Case, answer: str, judge: Caller) -> tuple[bool, str]:
    user = f"SITUATION:\n{case.prompt}\n\nTRAP:\n{case.trap}\n\nREPLY:\n{answer}"
    verdict = parse_judge(await judge(JUDGE_SYSTEM, user))
    if verdict is None:
        # An unreadable verdict is recorded as a miss and says so; it never counts as a pass.
        return False, "judge reply unreadable"
    return verdict


# ----------------------------------------------------------------- callers


def model_caller(spec: ModelSpec, provider: Provider) -> Caller:
    async def call(system: str, user: str) -> str:
        assembler = StreamAssembler()
        async for chunk in provider.stream(spec, system, [Message("user", user)], []):
            assembler.feed(chunk)
        return assembler.finish().text.strip()

    return call


def command_caller(command: str) -> Caller:
    argv = shlex.split(command)
    if not argv:
        raise EvalError("cmd: needs a command")

    async def call(system: str, user: str) -> str:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(f"{system}\n\n{user}".encode()), COMMAND_TIMEOUT
            )
        except TimeoutError as exc:
            process.kill()
            raise EvalError(f"{argv[0]}: no reply in {COMMAND_TIMEOUT:.0f} s") from exc
        if process.returncode != 0:
            detail = err.decode("utf-8", errors="replace")[:300]
            raise EvalError(f"{argv[0]}: exit {process.returncode}: {detail}")
        return out.decode("utf-8", errors="replace").strip()

    return call


def make_caller(
    name: str, models: dict[str, ModelSpec], provider: Provider, max_tokens: int = MAX_TOKENS
) -> Caller:
    if name.startswith("cmd:"):
        return command_caller(name[4:])
    if name not in models:
        raise EvalError(f"unknown model {name}; known: {', '.join(models)}")
    spec = models[name]
    if max_tokens > 0:
        spec = replace(spec, extra_body={**spec.extra_body, "max_tokens": max_tokens})
    return model_caller(spec, provider)


# -------------------------------------------------------------------- arms


def playbook_text(root: Path = PLAYBOOKS) -> str:
    bodies = [
        split_frontmatter(path.read_text(encoding="utf-8"))[1].strip()
        for path in sorted(root.glob("*/SKILL.md"))
    ]
    if not bodies:
        raise EvalError(f"no playbooks under {root}")
    return "\n\n---\n\n".join(bodies)


def critic_text(path: Path = BUILTIN_AGENTS / "critic.md") -> str:
    return split_frontmatter(path.read_text(encoding="utf-8"))[1].strip()


# What the session hands `run_diagnosis` as `known_files`: the repo's own file list. A single-turn
# eval has no repo, so the file names the case text mentions stand in for it. A diagnosis naming
# anything else is dropped by `parse_reply`, exactly as it is in a live turn.
FILE_MENTION = re.compile(
    r"[\w][\w./-]*\.(?:py|js|ts|tsx|go|rs|sql|html|css|toml|ya?ml|json|md|sh)\b"
)


def case_files(prompt: str, limit: int = 24) -> list[str]:
    """File names mentioned in the case text, deduped and capped."""
    return sorted(set(FILE_MENTION.findall(prompt)))[:limit]


async def answer_case(
    case: Case, arm: str, writer: Caller, critic: Caller | None, diag_model: str = "critic"
) -> str:
    if arm == "bare":
        return await writer(BASE_SYSTEM, case.prompt)
    if arm == "playbooks":
        system = (
            f"{BASE_SYSTEM}\n\nBefore you reply, apply whichever of these procedures fits. "
            f"Do the steps in your head; the reply itself stays short.\n\n{playbook_text()}"
        )
        return await writer(system, case.prompt)
    if arm == "critic":
        if critic is None:
            raise EvalError("the critic arm needs --critic")
        draft = await writer(BASE_SYSTEM, case.prompt)
        review = await critic(critic_text(), f"REQUEST:\n{case.prompt}\n\nDRAFT:\n{draft}")
        final_prompt = (
            f"{case.prompt}\n\n---\nYour first draft:\n{draft}\n\n"
            f"A second reader's review of that draft:\n{review}\n\n"
            "Write your final reply to the person. Fix what the review got right, "
            "ignore what it got wrong."
        )
        return await writer(BASE_SYSTEM, final_prompt)
    if arm == "diagnose":
        if critic is None:
            raise EvalError("the diagnose arm needs --critic: it runs the item 31 diagnosis")
        with tempfile.TemporaryDirectory(prefix="mwm-eval-") as folder:
            diagnosis = await run_diagnosis(
                root=Path(folder),
                source="eval",
                failure_text=case.prompt,
                task=case.prompt,
                changes_text="",
                known_files=case_files(case.prompt),
                caller=critic,
                model_id=diag_model,
            )
        note = diagnosis.note()
        # The note is the treatment: keep it in the run log so a later read of the result file
        # can see what the writer was actually handed, not only how it scored.
        print(
            f"  [{case.id}] diagnosis note:\n{note or '(empty: ' + diagnosis.reason + ')'}\n",
            file=sys.stderr,
            flush=True,
        )
        if not note:
            # Nothing to say is an outcome, not a silent fallback: say which one it degraded to.
            return await writer(BASE_SYSTEM, f"{case.prompt}\n\n(diagnosis: {diagnosis.reason})")
        return await writer(BASE_SYSTEM, f"{reminder(note)}\n\n{case.prompt}")
    raise EvalError(f"unknown arm {arm}")


# ------------------------------------------------------------------ report


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval for a pass rate; with 50 cases it is about +/- 13 points wide."""
    if total == 0:
        return 0.0, 0.0
    rate = passed / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(results: list[Result]) -> str:
    groups: dict[tuple[str, str], list[Result]] = defaultdict(list)
    for result in results:
        groups[(result.model, result.arm)].append(result)
    lines = [
        f"{'model':<28} {'arm':<10} {'pass':>7} {'rate':>6} {'95% interval':>14}  "
        + "  ".join(f"{a:>9}" for a in AREAS)
    ]
    for (model, arm), rows in sorted(groups.items()):
        passed = sum(r.passed for r in rows)
        low, high = wilson(passed, len(rows))
        per_area = []
        for area in AREAS:
            in_area = [r for r in rows if r.area == area]
            per_area.append(f"{sum(r.passed for r in in_area):>4}/{len(in_area):<4}")
        lines.append(
            f"{model[:28]:<28} {arm:<10} {passed:>3}/{len(rows):<3} {passed / len(rows):>6.0%} "
            f"{low:>6.0%} - {high:<5.0%}  " + "  ".join(per_area)
        )
    return "\n".join(lines)


def read_results(paths: list[Path]) -> list[Result]:
    results: list[Result] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(Result(**json.loads(line)))
    return results


# --------------------------------------------------------------------- run


async def run(
    cases: list[Case],
    arms: list[str],
    model_name: str,
    writer: Caller,
    critic: Caller | None,
    judge: Caller | None,
    out: Path | None,
    concurrency: int = 1,
    done: list[Result] | None = None,
    diag_model: str = "critic",
) -> list[Result]:
    """``done`` holds answers from an interrupted run; those (arm, case) pairs are not asked again."""
    gate = asyncio.Semaphore(max(1, concurrency))
    results: list[Result] = list(done or [])
    have = {(r.arm, r.case) for r in results}

    async def one(case: Case, arm: str) -> None:
        async with gate:
            try:
                answer = await answer_case(case, arm, writer, critic, diag_model)
                if judge is not None:
                    passed, note = await judge_grade(case, answer, judge)
                    grader = "judge"
                else:
                    passed, note = keyword_grade(case, answer)
                    grader = "keyword"
            except (ProviderError, EvalError, ConfigError) as exc:
                # A failed request is a recorded miss, never a silent skip.
                answer, passed, note, grader = "", False, f"ERROR: {exc}", "none"
            result = Result(case.id, case.area, arm, model_name, passed, grader, answer, note)
            results.append(result)
            if out is not None:
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            mark = "PASS" if passed else "miss"
            print(f"  {mark}  {arm:<10} {case.id}  {note}", file=sys.stderr)

    todo = [(case, arm) for arm in arms for case in cases if (arm, case.id) not in have]
    await asyncio.gather(*(one(case, arm) for case, arm in todo))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mwm-eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="models.toml id, or cmd:<command reading stdin>")
    parser.add_argument("--arms", default="bare,playbooks", help=f"comma list of {ARMS}")
    parser.add_argument(
        "--critic",
        help="second model for the critic and diagnose arms (another family than --model)",
    )
    parser.add_argument("--judge", help="model that grades; without it the keyword grader runs")
    parser.add_argument("--cases", type=Path, default=REPO_CASES)
    parser.add_argument("--ids", help="comma list of case ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("evals/out"))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="0 = no cap")
    parser.add_argument(
        "--critic-max-tokens",
        type=int,
        default=CRITIC_MAX_TOKENS,
        help="room for the second model, whose thinking is billed against its own answer",
    )
    parser.add_argument("--resume", type=Path, help="result file of an interrupted run")
    parser.add_argument("--compare", nargs="+", type=Path, help="print a table from result files")
    args = parser.parse_args(argv)

    try:
        if args.compare:
            print(summarise(read_results(args.compare)))
            return 0
        if not args.model:
            parser.error("--model is required")
        cases = load_cases(args.cases)
        if args.ids:
            wanted = set(args.ids.split(","))
            unknown = wanted - {c.id for c in cases}
            if unknown:
                raise EvalError(f"unknown case ids: {sorted(unknown)}")
            cases = [c for c in cases if c.id in wanted]
        if args.limit:
            cases = cases[: args.limit]
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        for arm in arms:
            if arm not in ARMS:
                raise EvalError(f"unknown arm {arm}; known: {ARMS}")
        models = load_models()
        provider = OpenAICompatProvider()
        cap = args.max_tokens
        writer = make_caller(args.model, models, provider, cap)
        critic = (
            make_caller(args.critic, models, provider, args.critic_max_tokens)
            if args.critic
            else None
        )
        judge = make_caller(args.judge, models, provider, cap) if args.judge else None
        if "critic" in arms and critic is None:
            raise EvalError("the critic arm needs --critic")
        if "diagnose" in arms and critic is None:
            raise EvalError("the diagnose arm needs --critic (the model that runs the diagnosis)")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        label = re.sub(r"[^\w.-]+", "_", args.model)[:40]
        out = args.resume or args.out_dir / f"{stamp}-{label}.jsonl"
        done = [r for r in read_results([out]) if r.model == args.model] if args.resume else []
        results = asyncio.run(
            run(
                cases,
                arms,
                args.model,
                writer,
                critic,
                judge,
                out,
                args.concurrency,
                done,
                args.critic,
            )
        )
    except (EvalError, ConfigError) as exc:
        print(f"mwm-eval: {exc}", file=sys.stderr)
        return 2
    print(summarise(results))
    print(f"\nanswers: {out}")
    errors = sum(r.note.startswith("ERROR") for r in results)
    if errors:
        print(f"{errors} of {len(results)} requests failed and count as misses", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
