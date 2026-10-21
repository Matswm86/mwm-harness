"""Spike: does the endpoint take an image, and does the model read the chart?

Sends gold-set frames (``data/videos/gold/graded.jsonl``, graded by a human or by
an Opus round) to each named model as one ``image_url`` data URL in a user
message, asks for the numbers written on the chart, and scores the answer
against the graded values. A value counts as found when the answer holds a
number within 0.05% of it.

This is a script call, so it refuses any model on a subscription plan endpoint:
the Token Plan terms allow interactive use inside a tool only. Use the ``payg-``
registry entries. Spend is recorded into the same ledger the providers write.

Usage:
    .venv/bin/python spike/spike_vision.py
    .venv/bin/python spike/spike_vision.py --models payg-qwen3.8-max --frames 5
    .venv/bin/python spike/spike_vision.py --image /path/to/frame.jpg

Raw chunks go to ``spike/out/vision-<model>.<n>.jsonl``.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from mwm_harness import spend
from mwm_harness.config import (
    ConfigError,
    ModelSpec,
    api_key_for,
    is_subscription_endpoint,
    load_models,
)
from mwm_harness.streaming import StreamAssembler, parse_sse_line

REPO = Path(__file__).resolve().parent.parent
GOLD = Path("/home/mm/MWM/data/videos/gold/graded.jsonl")
FRAMES = Path("/home/mm/MWM/data/videos")
OUT = REPO / "spike" / "out"

DEFAULT_MODELS = ["payg-qwen3.8-flash", "payg-qwen3.8-max"]
PROMPT = (
    "This is one frame from a trading video. Read the chart and list every price level "
    "you can see: the value of each horizontal line, each price label on the right axis "
    "and each number drawn on the chart. Answer with the numbers only, one per line, "
    "exactly as written."
)
RELATIVE_TOLERANCE = 0.0005
NUMBER = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:\.\d+)?|\d+")


@dataclass
class Case:
    gold_id: str
    path: Path
    truth: list[float]


@dataclass
class Result:
    model: str
    case: str
    ok: bool
    note: str = ""
    found: int = 0
    total: int = 0
    answer: str = ""
    usage: dict | None = None
    seconds: float = 0.0


def gold_cases(limit: int) -> list[Case]:
    """Distinct frames that exist on disk and carry at least two graded numbers."""
    cases: list[Case] = []
    seen: set[tuple[str, str]] = set()
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video, frame = row.get("video") or "", row.get("frame") or ""
        if not video or not frame or (video, frame) in seen:
            continue
        path = FRAMES / video / "frames" / frame
        truth = [
            float(n["value"])
            for n in row.get("numbers") or []
            if isinstance(n.get("value"), (int, float))
        ]
        if not path.is_file() or len(truth) < 2:
            continue
        seen.add((video, frame))
        cases.append(Case(row.get("id") or f"{video}/{frame}", path, truth))
        if len(cases) >= limit:
            break
    return cases


def data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else f"image/{path.suffix[1:]}"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for raw in NUMBER.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def score(answer: str, truth: list[float]) -> tuple[int, int]:
    """How many graded values the answer holds, within the tolerance."""
    seen = numbers_in(answer)
    found = sum(
        1
        for value in truth
        if any(abs(candidate - value) <= abs(value) * RELATIVE_TOLERANCE for candidate in seen)
    )
    return found, len(truth)


def ask(client: httpx.Client, spec: ModelSpec, key: str, case: Case, raw: Path) -> Result:
    body = {
        "model": spec.api_model or spec.id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(case.path)}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    assembler = StreamAssembler()
    try:
        with (
            client.stream(
                "POST",
                f"{spec.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            ) as response,
            raw.open("w", encoding="utf-8") as handle,
        ):
            if response.status_code != 200:
                detail = response.read().decode(errors="replace")[:400]
                return Result(
                    spec.id, case.gold_id, False, f"HTTP {response.status_code}: {detail}"
                )
            for line in response.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                handle.write(json.dumps(chunk) + "\n")
                assembler.feed(chunk)
    except httpx.HTTPError as exc:
        return Result(spec.id, case.gold_id, False, f"{type(exc).__name__}: {exc}")
    turn = assembler.finish()
    answer = turn.text or ""
    found, total = score(answer, case.truth)
    usage = turn.usage or {}
    if spend.is_metered(spec.base_url) and usage:
        spend.record(spec.api_model or spec.id, usage)
    return Result(
        spec.id,
        case.gold_id,
        True,
        "",
        found,
        total,
        answer,
        usage,
        time.monotonic() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--frames", type=int, default=3, help="how many gold frames per model")
    parser.add_argument("--image", type=Path, help="one frame instead of the gold set")
    parser.add_argument("--show", action="store_true", help="print each answer")
    args = parser.parse_args()

    if args.image:
        cases = [Case(args.image.name, args.image, [])]
    else:
        if not GOLD.is_file():
            print(f"no gold set at {GOLD}", file=sys.stderr)
            return 2
        cases = gold_cases(args.frames)
    if not cases:
        print("no usable gold frames", file=sys.stderr)
        return 2

    try:
        models = load_models()
    except ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=20.0)) as client:
        for model_id in args.models:
            spec = models.get(model_id)
            if spec is None:
                print(f"{model_id}: not in the registry", file=sys.stderr)
                return 2
            if is_subscription_endpoint(spec.base_url):
                print(
                    f"{model_id}: on a subscription plan endpoint. A spike is a script call, "
                    "which the plan terms forbid; use a payg- model.",
                    file=sys.stderr,
                )
                return 3
            try:
                key = api_key_for(spec)
            except ConfigError as exc:
                print(f"{model_id}: {exc}", file=sys.stderr)
                return 2
            for index, case in enumerate(cases, 1):
                raw = OUT / f"vision-{model_id.replace('/', '_')}.{index}.jsonl"
                result = ask(client, spec, key, case, raw)
                results.append(result)
                print(
                    f"{model_id} {case.gold_id}: "
                    + (
                        f"{result.found}/{result.total} graded values, "
                        f"{(result.usage or {}).get('prompt_tokens', '?')} in / "
                        f"{(result.usage or {}).get('completion_tokens', '?')} out, "
                        f"{result.seconds:.1f}s"
                        if result.ok
                        else f"FAILED {result.note}"
                    )
                )
                if args.show and result.ok:
                    print("  answer: " + result.answer.replace("\n", " | ")[:600])

    scored = [r for r in results if r.ok and r.total]
    if scored:
        found = sum(r.found for r in scored)
        total = sum(r.total for r in scored)
        print(f"\ntotal: {found}/{total} graded values found across {len(scored)} frame runs")
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"{len(failed)} of {len(results)} runs failed")
    return 1 if failed and not scored else 0


if __name__ == "__main__":
    raise SystemExit(main())
