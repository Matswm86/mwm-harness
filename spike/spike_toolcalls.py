"""Phase 0 spike: measure how each model streams tool calls on the shared endpoint.

For every model it sends one prompt that needs TWO tools at once, streams the
answer, and records: did the model call both tools in one turn, were the
arguments valid JSON, did it send reasoning text, did the endpoint report token
usage and cached tokens. A second identical request follows so a prompt cache,
if the endpoint has one, shows up as cached_tokens > 0.

Results go to COMPAT.md (the table the provider adapter is written against) and
the raw chunks of every request go to spike/out/ for inspection.

Usage:
    .venv/bin/python spike/spike_toolcalls.py
    .venv/bin/python spike/spike_toolcalls.py --models qwen3.8-max glm-5.2

The API key is read from the MWM_HARNESS_API_KEY environment variable, or from
~/.config/mwm-harness/secrets.env (a line MWM_HARNESS_API_KEY=...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from mwm_harness.streaming import AssembledTurn, StreamAssembler, parse_sse_line

REPO = Path(__file__).resolve().parent.parent
SECRETS = Path.home() / ".config" / "mwm-harness" / "secrets.env"
KEY_ENV = "MWM_HARNESS_API_KEY"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current local time in an IANA time zone.",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]
PROMPT = (
    "What is the weather in Oslo and what is the local time in Europe/Oslo? "
    "Call both tools in this one turn, then wait for their results."
)


class SpikeError(Exception):
    """A problem the person running the spike has to fix (key, config)."""


@dataclass
class Result:
    model: str
    ok: bool
    note: str = ""
    turn: AssembledTurn | None = None
    second_cached: int | None = None
    seconds: float = 0.0


def load_key() -> str:
    key = os.environ.get(KEY_ENV, "")
    if not key and SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_ENV:
                key = value.strip().strip("\"'")
    if not key:
        raise SpikeError(f"No API key. Set {KEY_ENV} or add a line {KEY_ENV}=... to {SECRETS}")
    if key.endswith((".com", ".cn")) or "://" in key:
        raise SpikeError(
            "The configured key looks like a hostname or URL, not an API key. "
            "Copy the key itself from the Model Studio console."
        )
    return key


def stream_once(client: httpx.Client, base_url: str, model: str, raw_path: Path) -> AssembledTurn:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assembler = StreamAssembler()
    with (
        client.stream("POST", f"{base_url}/chat/completions", json=body) as response,
        raw_path.open("w") as raw,
    ):
        if response.status_code != 200:
            detail = response.read().decode(errors="replace")[:300]
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {detail}",
                request=response.request,
                response=response,
            )
        for line in response.iter_lines():
            chunk = parse_sse_line(line)
            if chunk is None:
                continue
            raw.write(json.dumps(chunk) + "\n")
            assembler.feed(chunk)
    return assembler.finish()


def run_model(client: httpx.Client, base_url: str, model: str, out_dir: Path) -> Result:
    started = time.monotonic()
    safe = model.replace("/", "_")
    try:
        first = stream_once(client, base_url, model, out_dir / f"{safe}.1.jsonl")
        second = stream_once(client, base_url, model, out_dir / f"{safe}.2.jsonl")
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return Result(model, ok=False, note=str(exc)[:200], seconds=time.monotonic() - started)
    return Result(
        model,
        ok=True,
        turn=first,
        second_cached=second.cached_tokens,
        seconds=time.monotonic() - started,
    )


def compat_row(result: Result) -> str:
    if not result.ok or result.turn is None:
        return f"| {result.model} | FAILED | | | | | | {result.note} |"
    turn = result.turn
    names = sorted(call.name for call in turn.tool_calls)
    parallel = "yes" if names == ["get_time", "get_weather"] else f"no ({len(names)} call)"
    valid = "yes" if turn.tool_calls and all(c.error is None for c in turn.tool_calls) else "NO"
    usage = turn.usage or {}
    tokens = f"{usage.get('prompt_tokens', '?')} / {usage.get('completion_tokens', '?')}"
    cached = "not reported" if result.second_cached is None else str(result.second_cached)
    return (
        f"| {result.model} | {parallel} | {valid} | {'yes' if turn.reasoning else 'no'} "
        f"| {turn.finish_reason} | {tokens} | {cached} | {result.seconds:.1f}s for 2 requests |"
    )


def write_compat(results: list[Result], base_url: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# COMPAT: measured model behaviour on the shared endpoint",
        "",
        f"Written by `spike/spike_toolcalls.py` on {stamp}. Endpoint: `{base_url}`.",
        "Every value below is measured, none is assumed. Raw chunks: `spike/out/`.",
        "",
        "| model | 2 parallel tool calls | arguments valid JSON | sent reasoning text "
        "| finish_reason | prompt / completion tokens | cached tokens on repeat | note |",
        "|---|---|---|---|---|---|---|---|",
        *[compat_row(result) for result in results],
        "",
    ]
    path = REPO / "COMPAT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    registry = tomllib.loads((REPO / "models.toml").read_text())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(registry["models"]))
    parser.add_argument("--base-url", default=registry["defaults"]["base_url"])
    args = parser.parse_args()

    try:
        key = load_key()
    except SpikeError as exc:
        print(f"[spike] {exc}", file=sys.stderr)
        return 2

    out_dir = REPO / "spike" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        results = []
        for model in args.models:
            print(f"[spike] {model} ...", flush=True)
            results.append(run_model(client, args.base_url, model, out_dir))

    path = write_compat(results, args.base_url)
    print(path.read_text())
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
