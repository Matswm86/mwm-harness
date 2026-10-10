"""Mods: one key (F1-F9) or button, one action, through the same approvals as the model.

``mods.toml`` in the config folder overrides the copy that ships with the
harness. Four kinds:

- ``shell``: ``run`` is a shell command, run by the Bash tool (deny list, hooks,
  approval prompt). ``confirm = true`` asks even when the mode would not.
- ``agent``: ``agent`` runs on ``on`` = ``last_answer`` or ``uncommitted_diff``;
  its answer becomes a result card and is not added to the conversation.
- ``command``: ``builtin`` (a slash command such as ``/compact``) or ``file``
  (a command file name, run as ``/name args``).
- ``prompt``: ``text`` is sent as a message.

Templates: ``{tab.cwd}``, ``{tab.label}``, ``{tab.repo}`` and ``{tab.repo_dir}``
(from the ``[repos]`` table: folder = name, longest match wins),
``{tab.changed}`` (files this tab wrote that git reports changed, relative to the
repo folder) and ``{prompt}`` (the text in the message box). Shell values are quoted.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mwm_harness.config import config_dir

DEFAULT_MODS = Path(__file__).resolve().parent / "mods.default.toml"
KINDS = ("shell", "agent", "command", "prompt")
TEMPLATE = re.compile(r"\{(tab\.[a-z_]+|prompt)\}")


class ModError(Exception):
    pass


@dataclass
class Mod:
    key: str
    label: str
    kind: str
    note: str = ""
    confirm: bool = False
    network: bool = False  # shell mods: the command may reach the network (asks first)
    ask: str = ""  # what {prompt} means, shown when the message box is empty
    lock: str = ""  # mods with the same lock name never run in two tabs at once
    run: str = ""
    agent: str = ""
    on: str = "last_answer"
    instruction: str = ""
    builtin: str = ""
    file: str = ""
    args: str = ""
    text: str = ""

    def templates(self) -> set[str]:
        body = " ".join((self.run, self.args, self.text, self.instruction))
        return set(TEMPLATE.findall(body))


@dataclass
class ModSet:
    mods: list[Mod] = field(default_factory=list)
    repos: dict[Path, str] = field(default_factory=dict)
    path: Path | None = None

    def get(self, key: str) -> Mod | None:
        return next((m for m in self.mods if m.key.upper() == key.upper()), None)

    def repo_for(self, cwd: Path) -> tuple[str, Path] | None:
        matches = [
            (folder, name)
            for folder, name in self.repos.items()
            if cwd == folder or folder in cwd.parents
        ]
        if not matches:
            return None
        folder, name = max(matches, key=lambda item: len(item[0].parts))
        return name, folder


def load_mods(path: Path | None = None) -> ModSet:
    path = path or config_dir() / "mods.toml"
    if not path.is_file():
        path = DEFAULT_MODS
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ModError(f"{path}: {exc}") from exc
    known = set(Mod.__dataclass_fields__)
    mods = []
    for entry in data.get("mod") or []:
        unknown = set(entry) - known
        if unknown:
            raise ModError(f"{path}: mod {entry.get('key')}: unknown keys {sorted(unknown)}")
        mod = Mod(**entry)
        if mod.kind not in KINDS:
            raise ModError(f"{path}: mod {mod.key}: kind must be one of {', '.join(KINDS)}")
        mods.append(mod)
    repos = {
        Path(folder).expanduser().resolve(): str(name)
        for folder, name in (data.get("repos") or {}).items()
    }
    return ModSet(mods, repos, path)


def git_changed(repo_dir: Path) -> set[Path]:
    """Absolute paths git reports as modified, added or untracked under ``repo_dir``."""
    try:
        top = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", top, "status", "--porcelain", "-z", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    changed = set()
    for entry in status.split("\0"):
        if len(entry) > 3:
            changed.add((Path(top) / entry[3:]).resolve())
    return changed


def tab_values(
    mods: ModSet, cwd: Path, label: str, written: set[Path], prompt: str
) -> dict[str, str]:
    """Template values for one tab; a value that cannot be known is left out."""
    values = {"tab.cwd": str(cwd), "tab.label": label}
    if prompt.strip():
        values["prompt"] = prompt.strip()
    repo = mods.repo_for(cwd)
    base = repo[1] if repo else cwd
    if repo:
        values["tab.repo"], values["tab.repo_dir"] = repo[0], str(repo[1])
    mine = sorted(p for p in written & git_changed(base) if p == base or base in p.parents)
    if mine:
        values["tab.changed"] = " ".join(shlex.quote(str(p.relative_to(base))) for p in mine)
    return values


def why_not(mod: Mod, values: dict[str, str]) -> str:
    """Why ``mod`` cannot run in this tab now, or an empty string."""
    missing = mod.templates() - set(values)
    if "tab.repo" in missing or "tab.repo_dir" in missing:
        return "this folder is not in any repo of the [repos] table"
    if "tab.changed" in missing:
        return "this tab has written no file that git reports as changed"
    if "prompt" in missing:
        return f"type {mod.ask or 'the text'} in the message box first"
    return ""


def fill(template: str, values: dict[str, str], shell: bool) -> str:
    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        text = values[name]
        return text if (name == "tab.changed" or not shell) else shlex.quote(text)

    return TEMPLATE.sub(value, template)
