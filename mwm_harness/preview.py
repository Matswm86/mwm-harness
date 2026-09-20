"""What a front-end shows about files: the change an edit would make, a folder, a file.

``preview_change`` computes the result of a Write or Edit without touching the
disk, so the person approves a diff, not a JSON blob. The browsing helpers only
ever answer for paths inside the project directory.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mwm_harness.tools.base import ToolContext
from mwm_harness.tools.files import SKIP_DIRS

MAX_VIEW_BYTES = 1_000_000
MAX_DIFF_LINES = 2_000


@dataclass
class Change:
    path: str
    before: str
    after: str
    is_new: bool

    def unified(self) -> str:
        rows = list(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile="/dev/null" if self.is_new else f"a/{self.path}",
                tofile=f"b/{self.path}",
            )
        )
        text = "".join(row if row.endswith("\n") else row + "\n" for row in rows[:MAX_DIFF_LINES])
        if len(rows) > MAX_DIFF_LINES:
            text += f"[diff cut at {MAX_DIFF_LINES} of {len(rows)} lines]\n"
        return text


def preview_change(tool_name: str, tool_input: dict[str, Any], ctx: ToolContext) -> Change | None:
    """The change a Write or Edit call would make, or None when it cannot be worked out."""
    if tool_name not in ("Write", "Edit") or not tool_input.get("file_path"):
        return None
    path = ctx.resolve(str(tool_input["file_path"]))
    try:
        shown = str(path.relative_to(ctx.cwd))
    except ValueError:
        shown = str(path)
    before = ""
    if path.is_file():
        try:
            before = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    if tool_name == "Write":
        return Change(shown, before, str(tool_input.get("content") or ""), not path.exists())
    old, new = str(tool_input.get("old_string") or ""), str(tool_input.get("new_string") or "")
    found = before.count(old) if old else 0
    if found == 0 or (found > 1 and not tool_input.get("replace_all")):
        return None  # the tool itself will refuse this call; there is nothing to show
    return Change(shown, before, before.replace(old, new), False)


class OutsideProject(Exception):
    pass


def inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; symlinks and ``..`` may not lead outside."""
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise OutsideProject(relative)
    return path


def list_dir(root: Path, relative: str = "") -> list[dict[str, Any]]:
    folder = inside(root, relative)
    if not folder.is_dir():
        return []
    entries = []
    for child in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name in SKIP_DIRS:
            continue
        try:
            inside(root, str(child.relative_to(root)))
        except (OutsideProject, ValueError):
            continue  # a symlink that points out of the project
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(root)),
                "dir": child.is_dir(),
            }
        )
    return entries


def read_file(root: Path, relative: str) -> dict[str, Any]:
    path = inside(root, relative)
    if not path.is_file():
        return {"path": relative, "error": "not a file"}
    size = path.stat().st_size
    if size > MAX_VIEW_BYTES:
        return {"path": relative, "error": f"{size:,} bytes; the viewer stops at 1 MB"}
    try:
        return {"path": relative, "content": path.read_text(encoding="utf-8")}
    except UnicodeDecodeError:
        return {"path": relative, "error": "binary file"}
    except OSError as exc:
        return {"path": relative, "error": str(exc)}
