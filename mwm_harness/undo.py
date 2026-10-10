"""Undo a turn: put back every file the turn's Write and Edit calls changed.

Before a turn's first write to a file, its bytes (or "did not exist") are saved.
When the turn ends, the hash of what it left is saved too. ``undo`` restores the
newest turn that changed files, but leaves alone any file that changed again after
the turn (someone else edited it) and says so. Changes made by Bash commands are
not recorded; the report names the turn's Bash calls so nothing is assumed undone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

KEEP_TURNS = 10
MAX_FILE_BYTES = 5 * 1024 * 1024  # bigger files are not snapshotted; undo says so


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


@dataclass
class TurnSnapshot:
    prompt: str
    before: dict[Path, bytes | None] = field(default_factory=dict)  # None = did not exist
    after: dict[Path, str | None] = field(default_factory=dict)  # hash when the turn ended
    too_big: list[Path] = field(default_factory=list)
    bash_calls: int = 0


@dataclass
class UndoReport:
    prompt: str = ""
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # "path: why"
    bash_calls: int = 0

    def lines(self) -> list[str]:
        if not (self.restored or self.deleted or self.skipped):
            return ["nothing to undo: no recent turn changed a file through Write or Edit"]
        out = [f"undid the turn for: {self.prompt[:80]}"]
        out += [f"  restored {p}" for p in self.restored]
        out += [f"  deleted {p} (the turn created it)" for p in self.deleted]
        out += [f"  left alone {s}" for s in self.skipped]
        if self.bash_calls:
            out.append(
                f"  {self.bash_calls} Bash call(s) in that turn: their changes are NOT undone"
            )
        return out


class UndoLog:
    def __init__(self) -> None:
        self.turns: list[TurnSnapshot] = []

    def start(self, prompt: str) -> None:
        self.turns.append(TurnSnapshot(prompt))
        del self.turns[:-KEEP_TURNS]

    def before_write(self, path: Path) -> None:
        if not self.turns:
            self.start("")
        turn = self.turns[-1]
        if path in turn.before or path in turn.too_big:
            return
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            turn.too_big.append(path)
            return
        turn.before[path] = path.read_bytes() if path.is_file() else None

    def note_bash(self) -> None:
        if self.turns:
            self.turns[-1].bash_calls += 1

    def end(self) -> None:
        if self.turns:
            turn = self.turns[-1]
            turn.after = {p: _digest(p) for p in turn.before}

    def undo(self, root: Path) -> UndoReport:
        while self.turns and not (self.turns[-1].before or self.turns[-1].too_big):
            self.turns.pop()  # turns that wrote nothing
        if not self.turns:
            return UndoReport()
        turn = self.turns.pop()
        report = UndoReport(turn.prompt, bash_calls=turn.bash_calls)

        def shown(path: Path) -> str:
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        for path, old in turn.before.items():
            if turn.after and _digest(path) != turn.after.get(path):
                report.skipped.append(f"{shown(path)}: changed again after the turn")
                continue
            if old is None:
                path.unlink(missing_ok=True)
                report.deleted.append(shown(path))
            else:
                path.write_bytes(old)
                report.restored.append(shown(path))
        report.skipped += [f"{shown(p)}: over 5 MB, not snapshotted" for p in turn.too_big]
        return report
