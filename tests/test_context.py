"""The fixed prompt: what stays resident, what is read on demand, what travels as a delta.

Plan item 2. The two constraints that pull against each other are both asserted here: the
hard rules stay resident, and the prompt stays byte-stable across turns so a hosted prompt
cache holds.
"""

from __future__ import annotations

import asyncio
from datetime import date as real_date
from pathlib import Path

import pytest
from conftest import FixedApprover  # noqa: F401  (kept for parity with the other loop tests)
from mwm_harness import context
from mwm_harness.config import ModelSpec
from mwm_harness.context import (
    EnvironmentTracker,
    build_system_prompt,
    digest_sources,
    digest_text,
)
from mwm_harness.providers import chunks_for

MODEL = ModelSpec(id="some-model", base_url="http://x/v1", key_env="K")

RULES = """\
# Project rules

## CRITICAL RULES
1. **Never force-push**: the deny list blocks it in every mode.
2. **Pine Script**: always V6, never V5.

## Style
Write f-strings, not percent formatting. This paragraph is prose, and prose is what the
digest leaves behind: it is long, it says nothing a hard rule says, and the model can Read
this file when it needs it.
"""

MEMORY = """\
# Memory index

## Hard Rules
- ⛔ [plain_rule](feedback/plain.md) — no bare except
- 🆕⛔ [emoji_rule](feedback/emoji.md) — a mark behind another emoji still counts
- 📋 a pointer line with no mark at all

## Active handoffs
- a handoff line, prose, not resident
"""


def workspace_with(tmp_path: Path, imports: bool = False) -> Path:
    """A workspace shaped like the real one: a rules file, a memory index, a user rules file."""
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "MEMORY.md").write_text(MEMORY, encoding="utf-8")
    body = RULES + ("\n@memory/MEMORY.md\n" if imports else "")
    (ws / "QWEN.md").write_text(body, encoding="utf-8")
    (ws / "harness").mkdir()
    (ws / "harness" / "user-rules.md").write_text(
        "# RTK\nPrefix noisy commands.\n", encoding="utf-8"
    )
    return ws


# --------------------------------------------------------------------- digest


def test_digest_keeps_a_critical_section_whole_and_drops_prose(tmp_path):
    ws = workspace_with(tmp_path)
    prompt = build_system_prompt(ws, ws, MODEL, "digest")
    assert "1. **Never force-push**" in prompt  # numbered prose inside CRITICAL RULES
    assert "2. **Pine Script**: always V6" in prompt
    assert "## Style" in prompt  # the heading is the map of what the file holds
    assert "Write f-strings" not in prompt  # the prose under it is read on demand
    assert str(ws / "QWEN.md") in prompt  # and the prompt names the file to Read


def test_digest_keeps_a_hard_rule_marked_behind_another_emoji(tmp_path):
    ws = workspace_with(tmp_path)
    prompt = build_system_prompt(ws, ws, MODEL, "digest")
    assert "plain_rule" in prompt
    assert "emoji_rule" in prompt  # the index writes 🆕⛔ and 🗣⛔, not only ⛔
    assert "a pointer line with no mark at all" not in prompt
    assert "a handoff line" not in prompt


def test_a_cut_that_drops_hard_rules_says_how_many():
    text = "# Rules\n\n" + "".join(f"- ⛔ rule {n} " + "x" * 60 + "\n" for n in range(20))
    digest = digest_text(text, cap=400)
    assert "[ digest cut: Read the file for the rest" in digest
    assert "hard-rule lines are NOT resident" in digest


def test_a_cut_of_plain_prose_does_not_claim_lost_rules():
    # Headings are always resident, so a file that is all headings still overflows a small cap.
    digest = digest_text("# Rules\n" + "".join(f"## section {n}\n" for n in range(200)), cap=400)
    assert "[ digest cut: Read the file for the rest ]" in digest
    assert "NOT resident" not in digest


def test_a_rules_file_with_no_markers_keeps_its_head(tmp_path):
    (tmp_path / "RULES.md").write_text("Just prose.\n" * 400, encoding="utf-8")
    prompt = build_system_prompt(tmp_path, tmp_path, MODEL, "digest")
    assert "Just prose." in prompt  # an unmarked file still gets a resident opening


def test_full_mode_pastes_everything_and_is_the_escape_hatch(tmp_path):
    ws = workspace_with(tmp_path)
    prompt = build_system_prompt(ws, ws, MODEL, "full")
    assert "Write f-strings" in prompt
    assert "a handoff line" in prompt
    assert "resident digest" not in prompt


def test_an_imported_file_is_digested_once_not_twice(tmp_path):
    ws = workspace_with(tmp_path, imports=True)
    sources = [path for path, _ in digest_sources(ws, ws)]
    assert sources.count((ws / "memory" / "MEMORY.md").resolve()) == 1
    prompt = build_system_prompt(ws, ws, MODEL, "digest")
    assert prompt.count("# Memory index from") == 1
    assert "emoji_rule" in prompt  # it still arrived, through the import


def test_an_imported_file_gets_its_own_section_and_cap(tmp_path):
    ws = workspace_with(tmp_path, imports=True)
    own = dict(digest_sources(ws, ws))[(ws / "QWEN.md").resolve()]
    assert "@memory/MEMORY.md" not in own  # the import line is not resident text
    assert "emoji_rule" not in own  # the index is its own source, so its own cap applies


# ------------------------------------------------------------- byte stability


class _FakeDate:
    """Stands in for ``datetime.date`` so a test can move the calendar."""

    def __init__(self, today: real_date) -> None:
        self.today_value = today

    def today(self) -> real_date:
        return self.today_value


def test_the_digest_prompt_is_byte_stable_across_a_day_rolling_over(tmp_path, monkeypatch):
    ws = workspace_with(tmp_path)
    monkeypatch.setattr(context, "date", _FakeDate(real_date(2026, 10, 24)))
    before = build_system_prompt(ws, ws, MODEL, "digest")
    monkeypatch.setattr(context, "date", _FakeDate(real_date(2026, 10, 25)))
    after = build_system_prompt(ws, ws, MODEL, "digest")
    assert before == after  # the hosted prompt cache only holds on an identical prefix
    assert build_system_prompt(ws, ws, MODEL, "full") != after  # full mode carries the date


def test_the_digest_prompt_leaves_the_volatile_fields_out(tmp_path):
    ws = workspace_with(tmp_path)
    prompt = build_system_prompt(ws, ws, MODEL, "digest")
    assert "Today's date" not in prompt
    assert f"Working directory: {ws}" not in prompt
    assert "- Model: some-model" in prompt  # the model is not volatile inside one session
    assert "- Platform:" in prompt


def test_the_digest_is_much_smaller_than_the_paste(tmp_path):
    ws = workspace_with(tmp_path)
    (ws / "QWEN.md").write_text(RULES + "\n" + ("prose line " * 40 + "\n") * 200, encoding="utf-8")
    full = build_system_prompt(ws, ws, MODEL, "full")
    digest = build_system_prompt(ws, ws, MODEL, "digest")
    assert len(digest) * 3 < len(full)


# ------------------------------------------------------------- environment


def test_the_first_note_names_every_field(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "git_branch", lambda cwd: "main")
    note = EnvironmentTracker(tmp_path).note("default")
    assert note.startswith("# Environment\n")
    assert f"- Working directory: {tmp_path}" in note
    assert f"- Today's date: {real_date.today().isoformat()}" in note
    assert "- Git branch: main" in note
    assert "- Permission mode: default" in note


def test_nothing_changed_means_no_message_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "git_branch", lambda cwd: "main")
    tracker = EnvironmentTracker(tmp_path)
    assert tracker.note("default") != ""
    assert tracker.note("default") == ""
    assert tracker.note("default") == ""


def test_only_the_field_that_moved_is_resent(tmp_path, monkeypatch):
    branch = ["main"]
    monkeypatch.setattr(context, "git_branch", lambda cwd: branch[0])
    tracker = EnvironmentTracker(tmp_path)
    tracker.note("default")

    branch[0] = "worktree-item2"
    note = tracker.note("default")
    assert note.startswith("# Environment changed\n")
    assert note == "# Environment changed\n- Git branch: worktree-item2"

    note = tracker.note("plan")
    assert note == "# Environment changed\n- Permission mode: plan"

    monkeypatch.setattr(context, "date", _FakeDate(real_date(2026, 12, 25)))
    assert tracker.note("plan") == "# Environment changed\n- Today's date: 2026-12-25"


def test_a_branchless_folder_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "git_branch", lambda cwd: "")
    assert "- Git branch: (none)" in EnvironmentTracker(tmp_path).note("default")


# ------------------------------------------------------------------- in a turn


def run(coro):
    return asyncio.run(coro)


def test_the_reminder_lands_in_the_transcript_before_the_prompt(make_session):
    session, _, provider = make_session([chunks_for("Hello.")])
    run(session.send("hi"))
    assert session.messages[0].is_meta
    assert "# Environment" in session.messages[0].text()
    prompt = next(m for m in session.messages if m.text() == "hi")
    assert not prompt.is_meta
    assert session.messages.index(prompt) == 1  # reminder first, then the person's words
    # It is inside the request too, ahead of the prompt.
    sent = [m["content"] for m in provider.requests[0]["messages"] if m["role"] == "user"]
    assert "# Environment" in sent[0] and sent[0].endswith("hi")


def test_a_second_turn_with_no_change_sends_no_second_reminder(make_session):
    session, _, provider = make_session([chunks_for("One."), chunks_for("Two.")])
    run(session.send("first"))
    run(session.send("second"))
    assert len(provider.requests) == 2
    assert "# Environment changed" not in str(provider.requests[1]["messages"])


def test_full_mode_sends_no_reminder_because_the_prompt_carries_the_fields(make_session):
    session, _, _ = make_session([chunks_for("Hello.")])
    session.settings.rules_prompt = "full"
    run(session.send("hi"))
    assert "# Environment" not in session.messages[0].text()
    assert session.messages[0].text() == "hi"


def test_a_mode_change_reaches_the_model_on_the_next_turn(make_session):
    session, _, provider = make_session(
        [chunks_for("One."), chunks_for("Two.")], approver=FixedApprover(True)
    )
    run(session.send("first"))
    session.set_mode("plan")
    run(session.send("second"))
    last = [m["content"] for m in provider.requests[1]["messages"] if m["role"] == "user"]
    assert any("- Permission mode: plan" in text for text in last)


def test_rules_prompt_rejects_an_unknown_value(tmp_path):
    from mwm_harness.config import ConfigError, load_settings

    (tmp_path / "settings.toml").write_text('rules_prompt = "summary"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="rules_prompt"):
        load_settings(tmp_path / "settings.toml")
