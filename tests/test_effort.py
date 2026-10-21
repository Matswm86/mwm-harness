"""Reasoning effort: the wire field, the family clamp, and the per-session dial."""

import pytest
from mwm_harness import events as ev
from mwm_harness.config import ConfigError, ModelSpec, Settings, load_settings
from mwm_harness.providers import build_request

MAX = ModelSpec(
    id="qwen3.8-max",
    base_url="https://example.invalid/v1",
    key_env="K",
    thinking_field="enable_thinking",
)
FLASH = ModelSpec(
    id="payg-qwen3.8-flash",
    api_model="qwen3.8-flash",
    base_url="https://example.invalid/v1",
    key_env="K",
    thinking_field="enable_thinking",
)
GLM = ModelSpec(id="glm-5.3", base_url="https://example.invalid/v1", key_env="K")


def body(spec: ModelSpec, effort: str) -> dict:
    return build_request(spec, "system", [], [], effort)


def test_effort_becomes_reasoning_effort_on_the_qwen38_family() -> None:
    assert body(MAX, "medium")["reasoning_effort"] == "medium"
    assert body(FLASH, "low")["reasoning_effort"] == "low"


def test_the_thinking_toggle_is_not_sent_next_to_effort() -> None:
    assert "enable_thinking" not in body(MAX, "medium")


def test_the_family_ladder_stops_at_xhigh() -> None:
    assert body(MAX, "max")["reasoning_effort"] == "xhigh"


def test_no_effort_keeps_the_thinking_toggle() -> None:
    sent = body(MAX, "")
    assert sent.get("enable_thinking") is True
    assert "reasoning_effort" not in sent


def test_other_families_ignore_effort() -> None:
    assert "reasoning_effort" not in body(GLM, "medium")


def test_session_starts_at_the_settings_default(make_session) -> None:
    session, _, _ = make_session([])
    assert session.effort == Settings().reasoning_effort == "medium"


def test_set_effort_moves_the_dial_and_says_so(make_session) -> None:
    session, recorder, _ = make_session([])
    session.set_effort("low")
    assert session.effort == "low"
    assert [e.effort for e in recorder.of(ev.EffortChanged)] == ["low"]


def test_set_effort_refuses_an_unknown_tier(make_session) -> None:
    session, _, _ = make_session([])
    with pytest.raises(ValueError):
        session.set_effort("turbo")
    assert session.effort == "medium"


def test_a_sibling_session_inherits_the_dial(make_session) -> None:
    session, _, _ = make_session([])
    session.set_effort("xhigh")
    assert session.sibling(session.cwd, session.model).effort == "xhigh"


def test_settings_reject_an_effort_outside_the_ladder(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('reasoning_effort = "turbo"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)
