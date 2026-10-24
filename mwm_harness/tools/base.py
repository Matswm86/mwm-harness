"""Tool contract shared by every built-in tool."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mwm_harness.sandbox import Sandbox


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # Image blocks (see messages.image_block) to attach to the tool result. The
    # loop puts them in the result's content list, so a vision model sees the
    # picture and a text model gets a note naming the file instead.
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolContext:
    cwd: Path
    scratch: Path
    sandbox: Sandbox
    output_cap: int = 30_000
    # Total bytes this session's Write and Edit calls may put on disk (plan item 36). A
    # runaway unattended --yolo run has no human to notice the disk filling; wall clock,
    # /tmp size and RLIMIT_FSIZE cover one command, this covers the session's own tools.
    # 0 switches it off. Bash-written bytes are not counted here: no cheap honest counter
    # exists for many small files from a shell loop, and saying so beats a df guess that
    # would blame the session for whatever the services on this box write next to it.
    session_write_cap: int = 0
    session_written: int = 0
    read_files: set[Path] = field(default_factory=set)
    # The mtime each read or written file had when this session last saw it, so a file somebody
    # else changed can be named before the model plans an edit on text that is no longer there.
    stamps: dict[Path, int] = field(default_factory=dict)
    todos: list[dict[str, Any]] = field(default_factory=list)
    plan_handler: Callable[[str], Awaitable[bool]] | None = None  # set by the session
    # The AssuranceReport tool's sink: fills this session's assurance card (plan item 22).
    assurance_handler: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None

    def resolve(self, file_path: str) -> Path:
        path = Path(file_path).expanduser()
        return (path if path.is_absolute() else self.cwd / path).resolve()

    def note_file(self, path: Path) -> None:
        """Remember the state of a file this session just read or wrote."""
        try:
            self.stamps[path] = path.stat().st_mtime_ns
        except OSError:
            self.stamps.pop(path, None)

    def write_over_cap(self, payload_bytes: int) -> str | None:
        """Refusal text when writing payload_bytes would take this session past its disk
        cap, or None. Asked before the write and before the approval card, so a refused
        write never touches disk and never shows a dialog the human would answer in vain."""
        if (
            not self.session_write_cap
            or self.session_written + payload_bytes <= self.session_write_cap
        ):
            return None
        return (
            f"refused: this write of {payload_bytes:,} bytes would take the session past its "
            f"disk cap of {self.session_write_cap:,} bytes "
            f"(session_write_cap_mb = {self.session_write_cap // (1024 * 1024)}, "
            f"{self.session_written:,} written so far). This is the runaway-write guard, "
            "not a permission problem"
        )

    def stale_files(self) -> list[str]:
        """Recorded files whose mtime moved, so the model is told before it edits stale text.

        Only the recorded files are stat'ed, never a folder. A file that is gone is named as
        deleted and then forgotten: there is nothing left to read again, and an edit of it fails
        on its own.
        """
        changed: list[str] = []
        gone: list[Path] = []
        for path, stamp in self.stamps.items():
            try:
                now = path.stat().st_mtime_ns
            except OSError:
                changed.append(f"{path} (deleted)")
                gone.append(path)
                continue
            if now != stamp:
                changed.append(str(path))
        for path in gone:
            del self.stamps[path]
        return changed

    def cap(self, text: str) -> str:
        """Trim long output for the model; the full text is kept in a scratch file."""
        if len(text) <= self.output_cap:
            return text
        self.scratch.mkdir(parents=True, exist_ok=True)
        spill = self.scratch / f"output-{uuid.uuid4().hex[:8]}.txt"
        spill.write_text(text, encoding="utf-8")
        head = text[: self.output_cap]
        return (
            f"{head}\n\n[output cut at {self.output_cap} of {len(text)} characters; "
            f"full text saved to {spill}]"
        )


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]
    read_only: ClassVar[bool] = False

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def refusal(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        """Why this call would fail without changing anything, or None.

        The loop asks before it asks the person: nobody should have to approve
        a call the tool is going to refuse anyway.
        """
        return None

    def check(self, tool_input: dict[str, Any]) -> str | None:
        """Return an error text when a required argument is missing or mistyped."""
        properties = self.input_schema.get("properties", {})
        for key in self.input_schema.get("required", []):
            if key not in tool_input:
                return f"missing required argument: {key}"
        kinds = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "array": list,
            "number": (int, float),
        }
        for key, value in tool_input.items():
            expected = kinds.get(properties.get(key, {}).get("type", ""))
            if expected and not isinstance(value, expected):
                return f"argument {key} must be of type {properties[key]['type']}"
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError
