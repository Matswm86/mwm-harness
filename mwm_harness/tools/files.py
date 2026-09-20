"""File tools: Read, Write, Edit, Glob, Grep.

Names and argument names match Claude Code's, because hook matchers such as
``Write|Edit`` and hook scripts that read ``tool_input.file_path`` key on them.
Grep is native Python: a workspace hook rewrites shell ``grep``, and its output
under that rewrite undercounts.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult

SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000


class Read(Tool):
    name = "Read"
    description = (
        "Read a text file. Returns numbered lines. Use offset and limit for a part of a large file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path of the file to read"},
            "offset": {"type": "integer", "description": "First line to return, 1-based"},
            "limit": {"type": "integer", "description": "Number of lines to return"},
        },
        "required": ["file_path"],
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(tool_input["file_path"])
        if path.is_dir():
            return ToolResult(f"{path} is a directory; use Glob or Bash ls", True)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ToolResult(f"file not found: {path}", True)
        except PermissionError:
            return ToolResult(f"permission denied: {path}", True)
        ctx.read_files.add(path)
        lines = text.splitlines()
        start = max(int(tool_input.get("offset") or 1), 1)
        count = int(tool_input.get("limit") or MAX_READ_LINES)
        chosen = lines[start - 1 : start - 1 + count]
        if not chosen:
            return ToolResult(f"(no lines: the file has {len(lines)} lines)")
        body = "\n".join(
            f"{number:>6}\t{line[:MAX_LINE_CHARS]}" for number, line in enumerate(chosen, start)
        )
        left = len(lines) - (start - 1 + len(chosen))
        if left > 0:
            body += f"\n[{left} more lines; continue with offset={start + len(chosen)}]"
        return ToolResult(ctx.cap(body))


class Write(Tool):
    name = "Write"
    description = "Create a file or replace its whole content. An existing file must be Read first."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(tool_input["file_path"])
        if path.exists() and path not in ctx.read_files:
            return ToolResult(f"{path} exists; Read it before overwriting it", True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tool_input["content"], encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"could not write {path}: {exc}", True)
        ctx.read_files.add(path)
        return ToolResult(f"wrote {len(tool_input['content'])} characters to {path}")


class Edit(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in a file. old_string must occur once, "
        "unless replace_all is true. The file must be Read first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def refusal(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        path = ctx.resolve(tool_input["file_path"])
        old, new = tool_input["old_string"], tool_input["new_string"]
        if not path.is_file():
            return f"file not found: {path}"
        if path not in ctx.read_files:
            return f"Read {path} before editing it"
        if old == new:
            return "old_string and new_string are identical"
        try:
            found = path.read_text(encoding="utf-8").count(old)
        except (OSError, UnicodeDecodeError) as exc:
            return f"could not read {path}: {exc}"
        if found == 0:
            return (
                "old_string was not found in the file. The file may have changed since "
                "you read it: Read it again and copy the exact current text"
            )
        if found > 1 and not tool_input.get("replace_all"):
            return f"old_string occurs {found} times; add context or set replace_all"
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        problem = self.refusal(tool_input, ctx)
        if problem:
            return ToolResult(problem, True)
        path = ctx.resolve(tool_input["file_path"])
        old, new = tool_input["old_string"], tool_input["new_string"]
        text = path.read_text(encoding="utf-8")
        try:
            path.write_text(text.replace(old, new), encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"could not write {path}: {exc}", True)
        return ToolResult(f"replaced {text.count(old)} occurrence(s) in {path}")


def _walk(root: Path):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield Path(folder) / name


class Glob(Tool):
    name = "Glob"
    description = "List files matching a glob pattern such as **/*.py, newest first."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search; default is cwd"},
        },
        "required": ["pattern"],
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.resolve(tool_input.get("path") or ".")
        if not root.is_dir():
            return ToolResult(f"not a directory: {root}", True)
        pattern = tool_input["pattern"]
        matches = [
            p
            for p in root.glob(pattern)
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts)
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return ToolResult("no files matched")
        return ToolResult(ctx.cap("\n".join(str(p) for p in matches[:1000])))


class Grep(Tool):
    name = "Grep"
    description = (
        "Search file contents with a Python regular expression. output_mode: "
        "files_with_matches (default), content (path:line:text) or count."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "File or directory; default is cwd"},
            "glob": {"type": "string", "description": "Only search files whose name matches"},
            "output_mode": {"type": "string"},
            "case_insensitive": {"type": "boolean"},
            "head_limit": {"type": "integer"},
        },
        "required": ["pattern"],
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            flags = re.IGNORECASE if tool_input.get("case_insensitive") else 0
            regex = re.compile(tool_input["pattern"], flags)
        except re.error as exc:
            return ToolResult(f"bad regular expression: {exc}", True)
        root = ctx.resolve(tool_input.get("path") or ".")
        if not root.exists():
            return ToolResult(f"path not found: {root}", True)
        mode = tool_input.get("output_mode") or "files_with_matches"
        if mode not in ("files_with_matches", "content", "count"):
            return ToolResult(f"unknown output_mode: {mode}", True)
        name_glob = tool_input.get("glob")
        limit = int(tool_input.get("head_limit") or 500)
        out: list[str] = []
        for path in [root] if root.is_file() else _walk(root):
            if name_glob and not fnmatch.fnmatch(path.name, name_glob):
                continue
            hits = _search_file(path, regex)
            if not hits:
                continue
            if mode == "files_with_matches":
                out.append(str(path))
            elif mode == "count":
                out.append(f"{path}:{len(hits)}")
            else:
                out.extend(f"{path}:{number}:{line}" for number, line in hits)
            if len(out) >= limit:
                out = out[:limit] + [f"[stopped at head_limit={limit}]"]
                break
        return ToolResult(ctx.cap("\n".join(out)) if out else "no matches")


def _search_file(path: Path, regex: re.Pattern[str]) -> list[tuple[int, str]]:
    try:
        with path.open("rb") as handle:
            if b"\0" in handle.read(2048):
                return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [
        (number, line[:MAX_LINE_CHARS])
        for number, line in enumerate(text.splitlines(), 1)
        if regex.search(line)
    ]
