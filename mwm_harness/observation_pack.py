"""ObservationPack: stop re-sending large tool results on every request.

A tool result stays in the history for the rest of the session, so a 40 KB
log is paid for again on every later request. Here a result over the size
limit is sent in full for its first ``full_requests`` requests; after that the
request carries an excerpt and the path of a file holding the exact text. The
model reads that file with Read (offset and limit) when it needs a part back.

Only the request is changed. ``Session.messages`` and the transcript keep the
full text, so hooks, compaction and resume see what the tool returned.

Mechanism from SoL-Pi (Liu et al. 2026, arXiv 2609.20519, section on
ObservationPack): 10 KiB limit, full for two requests, 1 KB excerpt.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from mwm_harness.messages import Block, Message, _result_text

PACK_LIMIT = 10 * 1024  # characters
FULL_REQUESTS = 2
EXCERPT = 1024  # characters: the first half and the last half of the result


def archive_path(scratch: Path, text: str) -> Path:
    """Same text, same file: the packed request text stays byte-stable across
    requests, which keeps a provider's prompt cache valid."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return scratch / f"observation-{digest}.txt"


def packed_text(text: str, path: Path, excerpt: int = EXCERPT) -> str:
    half = excerpt // 2
    lines = text.count("\n") + 1
    return (
        f"[Large tool result packed: {len(text):,} characters, {lines:,} lines. "
        f"The exact text is in {path}; Read it with offset and limit for any part. "
        f"First and last {half} characters follow.]\n"
        f"{text[:half]}\n[...]\n{text[-half:]}"
    )


def pack_observations(
    messages: list[Message],
    scratch: Path,
    limit: int = PACK_LIMIT,
    full_requests: int = FULL_REQUESTS,
    excerpt: int = EXCERPT,
) -> tuple[list[Message], list[str]]:
    """Return the history to send, with old large tool results packed, and the
    errors of archive files that could not be written. A result whose archive
    fails is sent in full: the model must never get a path to a missing file.

    A result's age is the number of assistant messages after it: 0 on the
    request right after the tool ran, 1 on the next one. At ``full_requests``
    and older it is packed. Messages that change are copied; the input list
    and its messages are not modified.
    """
    assistants_after = 0
    out: list[Message] = []
    errors: list[str] = []
    for message in reversed(messages):
        if message.role == "assistant":
            assistants_after += 1
            out.append(message)
            continue
        if assistants_after < full_requests or isinstance(message.content, str):
            out.append(message)
            continue
        blocks: list[Block] = []
        changed = False
        for block in message.content:
            text = _result_text(block.get("content")) if block.get("type") == "tool_result" else ""
            if len(text) > limit:
                path = archive_path(scratch, text)
                try:
                    if not path.exists():
                        scratch.mkdir(parents=True, exist_ok=True)
                        path.write_text(text, encoding="utf-8")
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
                else:
                    block = {**block, "content": packed_text(text, path, excerpt)}
                    changed = True
            blocks.append(block)
        out.append(replace(message, content=blocks) if changed else message)
    out.reverse()
    return out, errors
