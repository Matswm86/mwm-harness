"""Shared test helpers: sessions wired to a scripted provider and throwaway hook files."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
from mwm_harness import events as ev
from mwm_harness.config import ModelSpec, Settings
from mwm_harness.loop import Session
from mwm_harness.providers import ScriptedProvider

MODEL = ModelSpec(id="test-model", base_url="http://unused.invalid/v1", key_env="UNUSED_KEY")


class Recorder:
    """Collects every bus event so a test can assert on what a front-end would see."""

    def __init__(self) -> None:
        self.events: list[ev.Event] = []

    def __call__(self, event: ev.Event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


class FixedApprover:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool:
        self.asked.append((tool_name, tool_input))
        return self.answer


def write_hook(folder: Path, name: str, body: str) -> Path:
    """Write an executable bash hook script and return its path."""
    path = folder / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def write_settings(folder: Path, hooks: dict[str, list[dict[str, Any]]]) -> Path:
    """``hooks`` maps event -> [{"matcher": ..., "command": ..., "async": ...}]."""
    data: dict[str, Any] = {"hooks": {}}
    for event, entries in hooks.items():
        data["hooks"][event] = [
            {
                "matcher": entry.get("matcher", ""),
                "hooks": [
                    {
                        "type": "command",
                        "command": str(entry["command"]),
                        **{k: entry[k] for k in ("async", "timeout") if k in entry},
                    }
                ],
            }
            for entry in entries
        ]
    path = folder / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def make_session(tmp_path: Path):
    def build(
        turns: list[Any],
        hooks: dict[str, list[dict[str, Any]]] | None = None,
        approver: Any = None,
        mode: str = "default",
        sandbox: str = "off",
        resume_from: Path | None = None,
    ) -> tuple[Session, Recorder, ScriptedProvider]:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        hook_files = [write_settings(tmp_path, hooks)] if hooks else []
        provider = ScriptedProvider(turns)
        recorder = Recorder()
        bus = ev.EventBus()
        bus.subscribe(recorder)
        session = Session(
            cwd=project,
            model=MODEL,
            provider=provider,
            settings=Settings(permission_mode=mode, sandbox=sandbox, hook_timeout=10.0),
            approver=approver,
            bus=bus,
            hook_settings=hook_files,
            resume_from=resume_from,
            sessions_dir=tmp_path / "sessions",
            system_prompt="test system prompt",
        )
        return session, recorder, provider

    return build


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the real config directory or the real workspace."""
    monkeypatch.setenv("MWM_HARNESS_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("MWM_WORKSPACE", str(tmp_path / "workspace"))
