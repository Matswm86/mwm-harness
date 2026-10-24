"""Two-step compaction, step 1: replace old tool output with a one-line stub.

The summary step costs a model call and throws away detail the model can get back
by re-running a tool. Both teardowns cited in the plan (Claude Code, Pi) clear old
tool output first for that reason: nothing is lost that the model cannot fetch
again, and it costs no request.

This module is pure and deterministic: it never calls a provider and never touches
disk except to write an ObservationPack archive. ``Session`` decides when to run it
(``prune_tool_results`` in ``config.py``) and summarises only when the estimate here
says the prompt is still over budget.

The estimate is ``chars // 4``, a rough token count used ONLY for that go/no-go
choice. The context meter keeps reporting the endpoint's own numbers and is never
written with a figure from here (see ``context.py``). An image gets a flat
``IMAGE_TOKENS`` because its bytes are not in the block's text.

Only the live history changes. The transcript keeps every full tool result, so
``--resume``, the Stop gates and an audit still see exactly what the tool returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from mwm_harness.messages import Block, Message, _result_text, images_in
from mwm_harness.observation_pack import archive_path

CHARS_PER_TOKEN = 4  # rough, for the go/no-go decision only (see the module docstring)
IMAGE_TOKENS = 1_000  # what observation_pack measured per frame held in a request
ARGS_HEAD = 160  # characters of the call's JSON arguments kept in the stub


@dataclass
class PruneResult:
    """What step 1 did. ``messages`` is the history to keep; the input is unchanged."""

    messages: list[Message]
    stubbed: int = 0
    chars_saved: int = 0
    images_dropped: int = 0
    archive_errors: tuple[str, ...] = ()

    @property
    def tokens_saved(self) -> int:
        return self.chars_saved // CHARS_PER_TOKEN + self.images_dropped * IMAGE_TOKENS

    @property
    def changed(self) -> bool:
        return self.stubbed > 0

    def line(self) -> str:
        if not self.stubbed:
            return "nothing to prune"
        return (
            f"{self.stubbed} old tool result(s) stubbed, ~{self.tokens_saved:,} tokens "
            f"saved ({self.chars_saved:,} characters"
            + (f", {self.images_dropped} image(s)" if self.images_dropped else "")
            + " dropped)"
        )


@dataclass
class CompactResult:
    """What one compaction did: a summary, a prune, both, or neither.

    ``tokens_saved`` is step 1's estimate, ``summary`` the length of what step 2 wrote.
    ``failed`` names a summary request that errored or came back empty, so a caller can
    tell "step 1 was enough" apart from "nothing worked" instead of reading an empty
    string as both.
    """

    summary: str = ""
    pruned: int = 0
    tokens_saved: int = 0
    failed: bool = False


def _call_names(messages: list[Message]) -> dict[str, tuple[str, str]]:
    """``tool_use_id -> (tool name, arguments head)``, read off the assistant calls."""
    out: dict[str, tuple[str, str]] = {}
    for message in messages:
        for call in message.tool_uses():
            arguments = json.dumps(call.get("input") or {}, default=str, sort_keys=True)
            key = str(call.get("id") or "")
            out[key] = (str(call.get("name") or "unknown_tool"), arguments[:ARGS_HEAD])
    return out


def _image_paths(block: Block) -> list[str]:
    return [str((image.get("source") or {}).get("path", "")) for image in images_in(block)]


def _archive(text: str, scratch: Path | None, pack_limit: int) -> tuple[str, str]:
    """The path to name in the stub, and the error when archiving failed.

    Small results are not archived: the stub alone is cheaper than a file. Same text,
    same file, so the stubbed request text stays byte-stable across requests and a
    provider's prompt cache survives (the rule ``observation_pack.archive_path`` keeps).
    """
    if scratch is None or pack_limit <= 0 or len(text) <= pack_limit:
        return "", ""
    path = archive_path(scratch, text)
    if path.exists():
        return str(path), ""
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return "", f"{path}: {exc}"
    return str(path), ""


def stub_line(name: str, arguments: str, text: str, paths: list[str], pointer: str) -> str:
    """The one line that replaces a result. Names the call, its size and how to get it back."""
    head = f"[pruned] {name}({arguments}) -> {len(text):,} chars"
    lines = text.count("\n") + 1 if text else 0
    if lines > 1:
        head += f", {lines:,} lines"
    if paths:
        head += f", {len(paths)} image(s): {', '.join(paths)}"
    if pointer:
        return f"{head}. Exact text: Read {pointer} with offset and limit."
    return f"{head}. Re-run the call to see it again."


def prune_tool_results(
    messages: list[Message],
    keep_turns: int = 2,
    min_chars: int = 400,
    scratch: Path | None = None,
    pack_limit: int = 0,
) -> PruneResult:
    """Stub every tool result older than ``keep_turns`` assistant rounds.

    A result's age is the number of assistant messages after it, the same count
    ``observation_pack.pack_observations`` uses. ``keep_turns`` of 0 or less stubs
    nothing: the recent window is the whole history.

    A result shorter than ``min_chars`` is kept verbatim, because stubbing it saves
    almost nothing and a short error message is the one thing a model cannot re-fetch
    cheaply. An image-only result has no text to measure, so it is pruned on age alone:
    a frame costs about ``IMAGE_TOKENS`` on every request it stays in.
    """
    names = _call_names(messages)
    out: list[Message] = []
    errors: list[str] = []
    stubbed = chars_saved = images_dropped = 0
    assistants_after = 0
    for message in reversed(messages):
        if message.role == "assistant":
            assistants_after += 1
            out.append(message)
            continue
        if assistants_after < keep_turns or isinstance(message.content, str):
            out.append(message)
            continue
        blocks: list[Block] = []
        changed = False
        for block in message.content:
            if block.get("type") != "tool_result":
                blocks.append(block)
                continue
            text = _result_text(block.get("content"))
            paths = _image_paths(block)
            if len(text) < min_chars and not paths:
                blocks.append(block)
                continue
            pointer, error = _archive(text, scratch, pack_limit)
            if error:
                errors.append(error)
                pointer = ""
            name, arguments = names.get(str(block.get("tool_use_id") or ""), ("unknown_tool", ""))
            stub = stub_line(name, arguments, text, paths, pointer)
            if len(stub) < len(text):
                chars_saved += len(text) - len(stub)
            images_dropped += len(paths)
            stubbed += 1
            blocks.append({**block, "content": stub})
            changed = True
        out.append(replace(message, content=blocks) if changed else message)
    out.reverse()
    return PruneResult(
        messages=out,
        stubbed=stubbed,
        chars_saved=chars_saved,
        images_dropped=images_dropped,
        archive_errors=tuple(errors),
    )
