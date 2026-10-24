"""Ask the real Groq 27b endpoint for session-end lessons on a real transcript.

One call. It measures the thing the cap in lessons.py is based on: how many INPUT tokens the
provider counts a capped digest as, against the 7,000 tokens-per-minute ceiling that 413'd a
24,000-character digest on 2026-10-24. Nothing here writes to the live queue unless --write
is passed, and a skip or a failure prints its reason rather than an empty line.

    cd ~/MWM/projects/mwm-harness && .venv/bin/python spike/real_lessons_check.py
    .venv/bin/python spike/real_lessons_check.py <session-id-prefix> [--write]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mwm_harness.config import load_models, load_secrets, load_settings, workspace_root
from mwm_harness.lessons import (
    MAX_DIGEST_CHARS,
    call_model,
    digest,
    lessons_path,
    run_lessons,
    tool_call_count,
)
from mwm_harness.providers import OpenAICompatProvider
from mwm_harness.transcript import load_messages, project_slug, sessions_root

ITPM_LIMIT = 7_000


def find(project: str, prefix: str) -> Path:
    folder = sessions_root() / project
    files = sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    if prefix:
        hits = [p for p in files if p.stem.startswith(prefix)]
        if not hits:
            # A session id is enough on its own: search every project folder rather than
            # making the caller work out which cwd wrote the transcript.
            everywhere = sorted(
                sessions_root().glob("*/*.jsonl"), key=lambda p: p.stat().st_size, reverse=True
            )
            hits = [p for p in everywhere if p.stem.startswith(prefix)]
        if not hits:
            raise SystemExit(f"no session starts with {prefix!r} under {sessions_root()}")
        return hits[0]
    if not files:
        # Whatever project this box has transcripts for: the check wants the largest real
        # session, because that is the one that stresses the digest cap.
        every = sorted(
            sessions_root().glob("*/*.jsonl"), key=lambda p: p.stat().st_size, reverse=True
        )
        if not every:
            raise SystemExit(f"no transcripts under {sessions_root()}")
        return every[0]
    return files[0]


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    path = find(project_slug(Path.cwd()), args[0] if args else "")
    settings = load_settings()
    models, secrets = load_models(), load_secrets()
    model_id = settings.lessons_model
    if model_id not in models:
        raise SystemExit(f"{model_id!r} is not in the model registry, so this check cannot run")
    if not secrets.get(models[model_id].key_env):
        raise SystemExit(f"no {models[model_id].key_env} in secrets.env, so this check cannot run")

    messages = load_messages(path)
    text, dropped = digest(messages)
    tokens = len(text) // 4
    print(f"session   : {path.stem[:8]}  ({path.stat().st_size:,} bytes on disk)")
    print(f"transcript: {len(messages)} messages, {tool_call_count(messages)} tool calls")
    print(
        f"digest    : {len(text):,} chars (cap {MAX_DIGEST_CHARS:,}), ~{tokens:,} by chars/4, "
        f"{dropped} lines cut"
    )
    print(f"model     : {model_id}, ITPM ceiling {ITPM_LIMIT:,} input tokens per minute")

    out = (
        lessons_path(settings.lessons_file)
        if write
        else workspace_root() / "scratch" / f"lessons-check-{path.stem[:8]}.md"
    )
    provider = OpenAICompatProvider(secrets=secrets)
    result = await run_lessons(
        messages,
        path.stem,
        lambda system, user: call_model(provider, models[model_id], system, user),
        model_id,
        out,
        min_calls=settings.lessons_min_calls,
        timeout=settings.lessons_timeout,
    )
    print(f"\nstatus    : {result.status}  ({result.reason})")
    print(
        f"counted   : {result.prompt_tokens:,} input / {result.completion_tokens:,} output "
        f"tokens, {result.seconds:.1f} s"
    )
    if result.status == "recorded":
        print(f"written to: {out}")
        for entry in result.entries:
            print(f"  - {entry['lesson']}")
            print(f"    evidence: {entry['evidence']}")
    if result.prompt_tokens:
        margin = ITPM_LIMIT - result.prompt_tokens
        print(
            f"headroom  : {margin:,} tokens under the cap "
            f"({100 * result.prompt_tokens / ITPM_LIMIT:.0f}% used)"
        )
    return 0 if result.status in {"recorded", "quiet"} else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
