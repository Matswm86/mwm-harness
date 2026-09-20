"""Tool contract shared by every built-in tool."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mwm_harness.sandbox import Sandbox


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolContext:
    cwd: Path
    scratch: Path
    sandbox: Sandbox
    output_cap: int = 30_000
    read_files: set[Path] = field(default_factory=set)
    todos: list[dict[str, Any]] = field(default_factory=list)

    def resolve(self, file_path: str) -> Path:
        path = Path(file_path).expanduser()
        return (path if path.is_absolute() else self.cwd / path).resolve()

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
