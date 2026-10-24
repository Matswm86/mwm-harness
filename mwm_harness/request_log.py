"""Every model request, rebuildable from the transcript (plan item 4).

The rule this follows is "model-visible means logged". Before this module the
transcript stored each message once, as the tool returned it, while the request
that left the process carried something else: ObservationPack replaces an old
large result with an excerpt and a scratch path, a prune replaces it with a
one-line stub, and the reminder that files changed on disk is appended to one
request and to nothing else. Reading the file therefore did not tell you what the
model saw, and an audit of a wrong answer had to guess.

The log lives inside the session transcript:

* one ``system`` line with subtype ``request`` per model request: the model, the
  reasoning effort, the message count, a digest of the system prompt, of the tool
  list and of the model spec, the messages by position, and ``sent_sha``, the
  fingerprint of the request as it went out;
* one ``system`` line with subtype ``request-blob`` per unique part: the system
  prompt, the tool list, the spec, and the content of a message the request
  rewrote or carried without storing. A blob is written the first time its digest
  appears, so a part that never changes costs one line per session, not one per
  request.

Messages are named by their position in this file's own user/assistant sequence
(``Message.entry``, stamped by ``Transcript.append``), so a request line costs a
few hundred bytes instead of a copy of the history. A bare position means "sent
exactly as stored"; ``{"ref": n, "sha": ...}`` means "stored message ``n``, with
this content instead"; ``{"new": {...}}`` is a message that exists only inside
that one request. A contiguous run of unchanged positions collapses to
``{"window": [first, count]}``.

Covered: the requests a session's own conversation makes, which is every turn and
the compaction summary. NOT covered: the single-shot critic calls (Review Desk,
advisor, failure diagnosis, ``/spec`` drafts, reading levels), which build a
one-message request of their own outside the conversation.

Known limit: an image part is read from its file and base64-encoded while the
request is built, so a request that carried image bytes rebuilds byte-identically
only while those files are unchanged. Text is not affected, because a packed
result names a scratch archive that stays on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mwm_harness.config import ModelSpec
from mwm_harness.messages import Message, canonical_json, content_sha
from mwm_harness.providers import build_request
from mwm_harness.transcript import Transcript

REQUEST_SUBTYPE = "request"
BLOB_SUBTYPE = "request-blob"
VERIFY_SUBTYPE = "request-verify"


class _Missing(Exception):
    """A part the file cannot supply. Its message becomes the rebuild's reason."""


def message_payload(message: Message) -> dict[str, Any]:
    """The four fields a request depends on, in the spelling both sides hash."""
    return {
        "role": message.role,
        "content": message.content,
        "is_meta": message.is_meta,
        "extra": message.extra,
    }


def request_fingerprint(
    spec: ModelSpec,
    system: str,
    messages: list[Message],
    tools: list[dict[str, Any]],
    effort: str = "",
) -> str:
    """One digest for the whole request. ``build_request`` is pure, so this pins the
    body the provider builds from these five inputs without building it twice."""
    return content_sha(
        {
            "spec": asdict(spec),
            "system": system,
            "messages": [message_payload(message) for message in messages],
            "tools": tools,
            "effort": effort,
        }
    )


class RequestLog:
    """Writes the request and blob lines of one session into its transcript."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.count = 0
        self._blobs: set[str] = set()

    def record(
        self,
        spec: ModelSpec,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        effort: str = "",
        kind: str = "turn",
    ) -> str:
        """Log one request as it goes out. Returns the fingerprint of what was sent."""
        sent = request_fingerprint(spec, system, messages, tools, effort)
        parts = [self._part(message) for message in messages]
        self.count += 1
        line: dict[str, Any] = {
            "request": self.count,
            "kind": kind,
            "model": spec.id,
            "api_model": spec.api_model or spec.id,
            "effort": effort,
            "messages": len(messages),
            "system_sha": self._blob(system),
            "tools_sha": self._blob(tools),
            "spec_sha": self._blob(asdict(spec)),
            "sent_sha": sent,
        }
        window = _window(parts)
        if window:
            line["window"] = window
        else:
            line["parts"] = parts
        self.transcript.note(REQUEST_SUBTYPE, summary_of(line), **line)
        return sent

    def _part(self, message: Message) -> Any:
        """How one sent message is named in the request line."""
        if message.entry < 0:
            # Not in the transcript at all: a reminder appended to this request only.
            return {
                "new": {
                    "role": message.role,
                    "is_meta": message.is_meta,
                    "extra": message.extra,
                    "sha": self._blob(message.content),
                }
            }
        if content_sha(message.content) == message.stored_sha:
            return message.entry
        # ObservationPack or a prune stub rewrote it for the request: log the sent text,
        # keep the stored text where it is.
        return {"ref": message.entry, "sha": self._blob(message.content)}

    def _blob(self, payload: Any) -> str:
        digest = content_sha(payload)
        if digest in self._blobs:
            return digest
        self._blobs.add(digest)
        as_text = isinstance(payload, str)
        self.transcript.note(
            BLOB_SUBTYPE,
            payload if as_text else canonical_json(payload),
            sha256=digest,
            encoding="text" if as_text else "json",
        )
        return digest


def summary_of(line: dict[str, Any]) -> str:
    """The human-readable text of a request line, for somebody reading the file."""
    window = line.get("window")
    where = (
        f"stored messages {window[0]}..{window[0] + window[1] - 1}"
        if window
        else f"{line.get('messages', 0)} message part(s)"
    )
    return (
        f"request {line.get('request')} ({line.get('kind')}): {line.get('model')}, {where}, "
        f"system {str(line.get('system_sha'))[:8]}, tools {str(line.get('tools_sha'))[:8]}, "
        f"sent {str(line.get('sent_sha'))[:8]}"
    )


def _window(parts: list[Any]) -> list[int]:
    """``[first, count]`` when the request sent a contiguous run of messages as stored."""
    if not parts or not all(isinstance(part, int) for part in parts):
        return []
    first = parts[0]
    return [first, len(parts)] if parts == list(range(first, first + len(parts))) else []


@dataclass
class Stored:
    """What one transcript file holds for the request log."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    blobs: dict[str, Any] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)


def read_log(path: Path) -> Stored:
    """The stored messages, the blobs and the request lines of one transcript.

    Lines that do not parse are skipped, like ``load_messages`` does: a session
    killed mid-write leaves a torn last line and that must not make the file
    unreadable.
    """
    stored = Stored()
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stored
    for text in raw.splitlines():
        if not text.strip():
            continue
        try:
            line = json.loads(text)
        except json.JSONDecodeError:
            continue
        kind = line.get("type")
        if kind in ("user", "assistant"):
            body = line.get("message") or {}
            stored.entries.append(
                {
                    "role": kind,
                    "content": body.get("content"),
                    "is_meta": bool(line.get("isMeta")),
                    "extra": {k: v for k, v in body.items() if k not in ("role", "content")},
                }
            )
            continue
        if kind != "system":
            continue
        subtype = line.get("subtype")
        if subtype == BLOB_SUBTYPE:
            payload = line.get("content")
            stored.blobs[str(line.get("sha256") or "")] = (
                payload if line.get("encoding") == "text" else _json(payload)
            )
        elif subtype == REQUEST_SUBTYPE:
            stored.requests.append(line)
    return stored


def request_lines(path: Path) -> list[dict[str, Any]]:
    """Every logged request of one transcript, oldest first."""
    return read_log(path).requests


def _json(payload: Any) -> Any:
    if not isinstance(payload, str):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


@dataclass
class Rebuilt:
    """One logged request put back together, or the reason it could not be."""

    ok: bool
    reason: str = ""
    line: dict[str, Any] | None = None
    spec: ModelSpec | None = None
    system: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    effort: str = ""

    def fingerprint(self) -> str:
        return request_fingerprint(self.spec, self.system, self.messages, self.tools, self.effort)

    def body(self) -> dict[str, Any]:
        """The wire body this request went out as."""
        if not self.ok or self.spec is None:
            raise ValueError(f"this request did not rebuild: {self.reason}")
        return build_request(self.spec, self.system, self.messages, self.tools, self.effort)


def rebuild(path: Path, index: int = -1) -> Rebuilt:
    """Put one logged request back together from the transcript alone."""
    stored = read_log(path)
    if not stored.requests:
        return Rebuilt(False, "this transcript holds no request line")
    try:
        line = stored.requests[index]
    except IndexError:
        return Rebuilt(
            False,
            f"this transcript holds {len(stored.requests)} request line(s) and none at {index}",
        )
    number = line.get("request")
    for name, label in (("spec_sha", "model spec"), ("system_sha", "system prompt")):
        if line.get(name) not in stored.blobs:
            return Rebuilt(False, f"request {number}: the {label} is not in the file")
    if line.get("tools_sha") not in stored.blobs:
        return Rebuilt(False, f"request {number}: the tool list is not in the file")
    try:
        spec = ModelSpec(**stored.blobs[line["spec_sha"]])
    except TypeError as exc:
        return Rebuilt(False, f"request {number}: the stored model spec does not build: {exc}")
    system = stored.blobs[line["system_sha"]]
    tools = stored.blobs[line["tools_sha"]]
    if not isinstance(system, str):
        return Rebuilt(False, f"request {number}: the stored system prompt is not text")
    if not isinstance(tools, list):
        return Rebuilt(False, f"request {number}: the stored tool list is not a list")
    try:
        messages = [_message(part, stored) for part in _parts(line, len(stored.entries))]
    except _Missing as exc:
        return Rebuilt(False, f"request {number}: {exc}")
    return Rebuilt(
        True,
        "",
        line,
        spec,
        system,
        messages,
        tools,
        str(line.get("effort") or ""),
    )


def verify(path: Path, index: int = -1) -> Rebuilt:
    """Rebuild one logged request and compare it with what was sent.

    The comparison is against ``sent_sha``, the fingerprint taken from the live
    request at send time, so a file that no longer describes what the model
    received fails here instead of looking fine.
    """
    result = rebuild(path, index)
    if not result.ok:
        return result
    sent = str((result.line or {}).get("sent_sha") or "")
    built = result.fingerprint()
    if built != sent:
        number = (result.line or {}).get("request")
        return Rebuilt(
            False,
            f"request {number}: the file rebuilds to {built[:8]}, "
            f"the request that went out was {sent[:8]}",
            result.line,
        )
    return result


def _parts(line: dict[str, Any], count: int) -> list[Any]:
    window = line.get("window")
    if window:
        first, size = window
        if first < 0 or size < 0 or first + size > count:
            raise _Missing(
                f"its window [{first}, {size}] does not fit the {count} stored message(s)"
            )
        return list(range(first, first + size))
    parts = line.get("parts")
    return parts if isinstance(parts, list) else []


def _message(part: Any, stored: Stored) -> Message:
    if isinstance(part, int):
        return _stored(part, stored, None)
    if not isinstance(part, dict):
        raise _Missing(f"message part {part!r} is neither a position nor an object")
    if "ref" in part:
        return _stored(part["ref"], stored, part.get("sha"))
    new = part.get("new")
    if not isinstance(new, dict):
        raise _Missing(f"message part {part!r} names no stored message and holds no content")
    return Message(
        str(new.get("role") or "user"),
        _payload(new.get("sha"), stored),
        bool(new.get("is_meta")),
        dict(new.get("extra") or {}),
    )


def _stored(position: Any, stored: Stored, sha: Any) -> Message:
    if not isinstance(position, int) or not 0 <= position < len(stored.entries):
        raise _Missing(
            f"message position {position!r} is outside the {len(stored.entries)} stored message(s)"
        )
    entry = stored.entries[position]
    content = entry["content"] if sha is None else _payload(sha, stored)
    message = Message(entry["role"], content, entry["is_meta"], dict(entry["extra"]))
    # Stamp it the way the transcript would, so a rebuilt history can be logged again.
    message.entry = position
    message.stored_sha = content_sha(entry["content"])
    return message


def _payload(sha: Any, stored: Stored) -> Any:
    if not isinstance(sha, str) or sha not in stored.blobs:
        raise _Missing(f"the content blob {sha!r} is not in the file")
    return stored.blobs[sha]
