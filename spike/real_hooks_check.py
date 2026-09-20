"""Phase 3 closing check: do the workspace's REAL hooks work under the harness?

No model and no API key are involved. A scripted provider plays the model:
its first answer contains an em dash (a rule the workspace's Stop gate
enforces), its second answer is clean. With the real hook settings loaded the
expected result is: the first answer is blocked by a Stop hook, the reason is
fed back, the second answer passes.

It also runs one Bash tool call, so the PreToolUse shell-rewrite hook and the
PostToolUse hooks fire against a real payload.

Usage:
    .venv/bin/python spike/real_hooks_check.py [--cwd DIR]

Exit 0 = the gate blocked the em dash and released the clean answer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

from mwm_harness import events as ev
from mwm_harness.config import ModelSpec, Settings
from mwm_harness.loop import Session, default_hook_settings
from mwm_harness.providers import ScriptedProvider, chunks_for

EM_DASH = "—"
FIRST = f"The file holds three lines {EM_DASH} nothing else is in it."
SECOND = "The file holds three lines, and nothing else is in it."


class ApproveAll:
    async def ask(self, tool_name: str, tool_input: dict, reason: str) -> bool:
        return True


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path)
    args = parser.parse_args()
    cwd = args.cwd or Path(tempfile.mkdtemp(prefix="mwm-harness-hooks-"))
    (cwd / "three.txt").write_text("a\nb\nc\n")

    provider = ScriptedProvider(
        [
            chunks_for(tool_calls=[("Bash", {"command": "wc -l three.txt"})]),
            chunks_for(FIRST),
            chunks_for(SECOND),
            chunks_for(SECOND),
        ]
    )
    bus = ev.EventBus()
    seen: list[ev.Event] = []
    bus.subscribe(seen.append)
    session = Session(
        cwd=cwd,
        model=ModelSpec(id="scripted", base_url="http://unused.invalid", key_env="UNUSED"),
        provider=provider,
        settings=Settings(),
        approver=ApproveAll(),
        bus=bus,
        system_prompt="scripted run",
    )
    per_event: dict[str, int] = {}
    for registration in session.hooks.registrations:
        per_event[registration.event] = per_event.get(registration.event, 0) + 1
    print(f"hook settings files: {[str(p) for p in default_hook_settings(cwd) if p.is_file()]}")
    print(f"registrations: {len(session.hooks.registrations)} {per_event}")
    print(f"shell sandbox: {'bubblewrap' if session.tool_ctx.sandbox.enabled else 'off'}")

    started = time.monotonic()
    await session.start()
    ended = await session.send("How many lines does three.txt hold?")
    await session.close()
    print(f"turn took {time.monotonic() - started:.1f} s, ended: {ended.reason}")

    for event in seen:
        if isinstance(event, ev.HookContext):
            print(f"  context from {event.event}: {len(event.text)} chars")
        elif isinstance(event, ev.ToolFinished):
            print(f"  tool {event.name}: error={event.is_error} -> {event.content[:120]!r}")
        elif isinstance(event, ev.HookBlocked):
            print(f"  BLOCKED by {event.event}: {event.reason[:300]!r}")
        elif isinstance(event, ev.Notice):
            print(f"  notice [{event.level}]: {event.text[:300]}")
    print(f"transcript: {session.transcript.path}")

    blocked = [e for e in seen if isinstance(e, ev.HookBlocked) and e.event == "Stop"]
    ok = bool(blocked) and ended.reason == "done" and EM_DASH not in ended.text
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
