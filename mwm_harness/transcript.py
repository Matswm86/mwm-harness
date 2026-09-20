"""Session transcript: one JSON object per line, in the shape Claude Code writes.

The workspace's Stop gates read ``transcript_path`` and look for entries with
``type`` "user"/"assistant" and ``message.content`` blocks. They exit 0 when a
line does not parse, so a wrong shape would switch every gate off without a
sound. This module is the single writer of that file, and the reader used by
``/resume``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mwm_harness import __version__
from mwm_harness.messages import Message


def sessions_root() -> Path:
    return Path.home() / ".local" / "share" / "mwm-harness" / "sessions"


def project_slug(cwd: Path) -> str:
    return str(cwd.resolve()).replace("/", "-")


class Transcript:
    def __init__(self, path: Path, session_id: str, cwd: Path) -> None:
        self.path = path
        self.session_id = session_id
        self.cwd = cwd
        self._last_uuid: str | None = None
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, cwd: Path, root: Path | None = None) -> Transcript:
        session_id = str(uuid.uuid4())
        folder = (root or sessions_root()) / project_slug(cwd)
        return cls(folder / f"{session_id}.jsonl", session_id, cwd)

    def append(self, message: Message, **fields: Any) -> str:
        """Write one user or assistant message. Returns the entry uuid."""
        entry_uuid = str(uuid.uuid4())
        body: dict[str, Any] = {"role": message.role, "content": message.content}
        body.update(message.extra)
        entry: dict[str, Any] = {
            "parentUuid": self._last_uuid,
            "isSidechain": False,
            "type": message.role,
            "message": body,
            "uuid": entry_uuid,
            "timestamp": _now(),
            "cwd": str(self.cwd),
            "sessionId": self.session_id,
            "version": f"mwm-harness/{__version__}",
        }
        if message.is_meta:
            entry["isMeta"] = True
        entry.update(fields)
        self._write(entry)
        self._last_uuid = entry_uuid
        return entry_uuid

    def note(self, subtype: str, content: str, **fields: Any) -> None:
        """A ``system`` line: visible to a person reading the file, ignored by the gates."""
        entry = {
            "type": "system",
            "subtype": subtype,
            "content": content,
            "timestamp": _now(),
            "sessionId": self.session_id,
        }
        entry.update(fields)
        self._write(entry)

    def _write(self, entry: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def usage_to_anthropic(usage: dict[str, Any] | None) -> dict[str, int]:
    """Map OpenAI usage numbers onto the field names the transcript shape uses."""
    usage = usage or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": usage.get("completion_tokens") or 0,
    }


def load_messages(path: Path) -> list[Message]:
    """Rebuild the model-visible history from a transcript file.

    Lines that do not parse are skipped: a session killed mid-write leaves a
    torn last line, and that must not make the session unresumable.
    """
    messages: list[Message] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("type") not in ("user", "assistant"):
            continue
        body = entry.get("message") or {}
        content = body.get("content")
        if not isinstance(content, (str, list)):
            continue
        if entry.get("isCompactSummary"):
            messages = []  # everything before a compaction was replaced by its summary
        extra = {k: v for k, v in body.items() if k not in ("role", "content")}
        messages.append(Message(entry["type"], content, bool(entry.get("isMeta")), extra))
    return repair_history(messages)


def repair_history(messages: list[Message]) -> list[Message]:
    """Give every unanswered tool call an error result.

    A cancel or a crash can leave an assistant message whose tool calls never
    got results; endpoints reject such a history.
    """
    repaired: list[Message] = []
    for position, message in enumerate(messages):
        repaired.append(message)
        if message.role != "assistant" or not message.tool_uses():
            continue
        following = messages[position + 1] if position + 1 < len(messages) else None
        answered = {
            b.get("tool_use_id")
            for b in (following.blocks() if following and following.role == "user" else [])
            if b.get("type") == "tool_result"
        }
        missing = [b for b in message.tool_uses() if b["id"] not in answered]
        if not missing:
            continue
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": b["id"],
                "content": "Interrupted before this tool ran.",
                "is_error": True,
            }
            for b in missing
        ]
        if following and following.role == "user" and answered:
            following.content = blocks + following.blocks()
        else:
            repaired.append(Message("user", blocks, is_meta=True))
    return repaired


def list_sessions(cwd: Path, root: Path | None = None) -> list[Path]:
    folder = (root or sessions_root()) / project_slug(cwd)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
