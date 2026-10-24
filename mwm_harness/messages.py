"""The conversation model: Anthropic-style content blocks.

Everything inside the harness (history, transcript, hooks) speaks this shape:

    {"role": "user" | "assistant", "content": str | [block, ...]}

with blocks ``text``, ``thinking``, ``tool_use``, ``tool_result`` and ``image``.
The existing Stop gates parse exactly this shape, so it is the storage format.
Conversion to the OpenAI wire format happens here and only for the request
that leaves the process.

An image block names a FILE, it never holds the bytes:

    {"type": "image", "source": {"type": "file", "path": ..., "media_type": ...}}

The bytes are read and base64-encoded when a request is built, so the transcript
stays small enough for the gates and ``/resume`` to parse, and an image costs
tokens only while it is recent enough to be sent (see observation_pack).
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mwm_harness.events import now_stamp
from mwm_harness.streaming import AssembledTurn

Block = dict[str, Any]


def canonical_json(payload: Any) -> str:
    """One JSON spelling for everything the harness hashes or compares byte for byte.

    Key order and whitespace would otherwise make the same payload hash differently in
    two places, so the request log (plan item 4), the transcript stamp and the tests all
    go through this function.
    """
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha(payload: Any) -> str:
    """A stable digest of a message content, a tool list, a model spec or a whole request.

    Text is tagged apart from structure so the string ``"[1,2]"`` and the list ``[1, 2]``
    cannot share a digest: a shared digest would hand one blob's payload to the other.
    """
    text = f"s:{payload}" if isinstance(payload, str) else f"j:{canonical_json(payload)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass
class Message:
    role: str
    content: str | list[Block]
    # Meta messages reach the model but are not something the person typed
    # (hook feedback, injected context). Gates skip them when looking for the prompt.
    is_meta: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=now_stamp, compare=False)
    # Where this message lives in the transcript, and the digest of the content it was
    # written with (plan item 4: the request log points at stored messages by position
    # instead of copying them into every request line). Both are stamped by
    # ``Transcript.append`` and carried along by ``dataclasses.replace``, so a copy whose
    # content was rewritten for one request (ObservationPack, a prune stub) still knows
    # which stored message it came from and that its text is no longer the stored text.
    # -1 means "not in the transcript": a message that only exists inside one request.
    entry: int = field(default=-1, compare=False, repr=False)
    stored_sha: str = field(default="", compare=False, repr=False)

    def blocks(self) -> list[Block]:
        if isinstance(self.content, str):
            return [text_block(self.content)] if self.content else []
        return self.content

    def text(self) -> str:
        return "\n".join(
            b.get("text") or "" for b in self.blocks() if b.get("type") == "text"
        ).strip()

    def tool_uses(self) -> list[Block]:
        return [b for b in self.blocks() if b.get("type") == "tool_use"]


def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def tool_use_block(name: str, tool_input: dict[str, Any], call_id: str = "") -> Block:
    return {
        "type": "tool_use",
        "id": call_id or f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": tool_input,
    }


def tool_result_block(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
    images: Sequence[Block] = (),
) -> Block:
    """A result the model can see. With images the content becomes a block list,
    which is the shape Anthropic uses and the shape ``_result_text`` already reads."""
    block: Block = {"type": "tool_result", "tool_use_id": tool_use_id}
    if images:
        parts: list[Block] = [text_block(content)] if content else []
        parts.extend(images)
        block["content"] = parts
    else:
        block["content"] = content
    if is_error:
        block["is_error"] = True
    return block


MAX_IMAGE_BYTES = 8 * 1024 * 1024  # the DashScope per-image limit is 10 MB
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def media_type_for(path: Path | str) -> str:
    guessed = mimetypes.guess_type(str(path))[0] or ""
    return guessed if guessed.startswith("image/") else "image/jpeg"


def image_problem(path: Path) -> str | None:
    """Why this file cannot be sent as an image, or None when it can."""
    if not path.is_file():
        return f"file not found: {path}"
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return f"not an image ({path.suffix or 'no extension'}); supported: png jpg gif webp bmp"
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        return (
            f"the image is {size:,} bytes, over the {MAX_IMAGE_BYTES:,} byte limit; shrink it first"
        )
    return None


def image_block(path: Path | str, media_type: str = "") -> Block:
    return {
        "type": "image",
        "source": {
            "type": "file",
            "path": str(path),
            "media_type": media_type or media_type_for(path),
        },
    }


def images_in(block: Block) -> list[Block]:
    """The images a tool_result block carries."""
    content = block.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "image"]


def encode_image(block: Block) -> tuple[str | None, str]:
    """The ``data:`` URL for an image block, and the note to send instead when it fails."""
    source = block.get("source") or {}
    path = Path(str(source.get("path") or ""))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"[image {path} could not be read: {exc}]"
    if len(raw) > MAX_IMAGE_BYTES:
        return None, f"[image {path} is {len(raw):,} bytes, over the send limit]"
    media = str(source.get("media_type") or media_type_for(path))
    return f"data:{media};base64,{base64.b64encode(raw).decode()}", ""


def assistant_from_turn(turn: AssembledTurn) -> tuple[Message, dict[str, str]]:
    """Build the assistant message for a streamed turn.

    Returns the message and a map ``tool_use_id -> error`` for calls whose
    arguments did not parse. Those calls are kept with an empty input so the
    history stays valid; the loop answers them with an error result instead of
    running anything.
    """
    blocks: list[Block] = []
    errors: dict[str, str] = {}
    if turn.reasoning:
        blocks.append({"type": "thinking", "thinking": turn.reasoning})
    if turn.text:
        blocks.append(text_block(turn.text))
    for call in turn.tool_calls:
        block = tool_use_block(call.name or "unknown_tool", call.arguments or {}, call.id)
        blocks.append(block)
        if call.error:
            errors[block["id"]] = f"{call.error}. Raw arguments: {call.raw_arguments[:500]}"
    return Message("assistant", blocks), errors


def to_openai(
    system: str, messages: list[Message], echo_reasoning: bool = False, vision: bool = False
) -> list[dict[str, Any]]:
    """Convert block history into OpenAI chat messages.

    Tool results become ``role: tool`` messages and must directly follow the
    assistant message that made the calls, so they are emitted before any text
    that shares their user message. Adjacent plain user messages are merged
    because some endpoints reject two user turns in a row.

    Images cannot ride in a ``role: tool`` message: the endpoints take image
    parts in a user turn only. So every tool result of one message goes out
    first and its images follow in one multipart user message. A model without
    ``vision`` gets a text note naming the file instead of the bytes, because an
    image sent to a text-only model fails the whole request.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "assistant":
            out.append(_assistant_to_openai(message, echo_reasoning))
            continue
        texts: list[str] = []
        pending: list[Block] = []
        from_tools: list[str] = []
        for block in message.blocks():
            kind = block.get("type")
            if kind == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _result_text(block.get("content")),
                    }
                )
                carried = images_in(block)
                pending += carried
                from_tools += [str((i.get("source") or {}).get("path", "")) for i in carried]
            elif kind == "image":
                pending.append(block)
            elif kind == "text" and block.get("text"):
                texts.append(block["text"])
        if pending and not vision:
            texts += [_no_vision_note(block) for block in pending]
            pending = []
        parts: list[dict[str, Any]] = []
        for block in pending:
            url, note = encode_image(block)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif note:
                texts.append(note)
        if parts and from_tools:
            # A tool message cannot carry an image, so the picture lands in a turn
            # of its own: say which result it came from, or several calls in one
            # turn leave the model guessing which image answered which.
            texts.insert(0, f"[image(s) above from the tool result(s): {', '.join(from_tools)}]")
        _append_user(out, texts, parts)
    return out


def _no_vision_note(block: Block) -> str:
    path = (block.get("source") or {}).get("path", "")
    return f"[image not sent, this model takes text only: {path}. Use a vision model to read it.]"


def _append_user(out: list[dict[str, Any]], texts: list[str], parts: list[dict[str, Any]]) -> None:
    """Add one user turn, merging into the previous one: two user turns in a row
    are rejected by some endpoints, and a multipart turn has to stay multipart."""
    joined = "\n\n".join(texts)
    if not parts and not joined:
        return
    if out and out[-1]["role"] == "user":
        entry = out[-1]
        content = entry["content"]
        if isinstance(content, str):
            if not parts:
                entry["content"] = f"{content}\n\n{joined}" if content else joined
                return
            content = [{"type": "text", "text": content}] if content else []
            entry["content"] = content
        content.extend(parts)
        if joined:
            content.append({"type": "text", "text": joined})
        return
    if parts:
        parts = parts + ([{"type": "text", "text": joined}] if joined else [])
    out.append({"role": "user", "content": parts or joined})


def _assistant_to_openai(message: Message, echo_reasoning: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": message.text() or None}
    calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
            },
        }
        for block in message.tool_uses()
    ]
    if calls:
        entry["tool_calls"] = calls
    if echo_reasoning:
        reasoning = "".join(
            b.get("thinking") or "" for b in message.blocks() if b.get("type") == "thinking"
        )
        if reasoning:
            entry["reasoning_content"] = reasoning
    return entry


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Blocks without text (an image) are skipped, not joined as empty lines.
        return "\n".join(
            block["text"] for block in content if isinstance(block, dict) and block.get("text")
        )
    return "" if content is None else str(content)


def tools_to_openai(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``specs`` are ``{"name", "description", "input_schema"}`` dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in specs
    ]
