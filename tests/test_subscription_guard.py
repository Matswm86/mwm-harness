"""A subscription plan key must never serve a headless, script-initiated turn.

Alibaba's Token Plan terms allow interactive use inside a coding or agent tool
and ban automation scripts, custom application backends and non-interactive
batch calls, with subscription suspension or key banning as the penalty. The
interactive REPL and the agents it spawns stay on the plan, because the terms
treat calls initiated interactively within a tool as normal use. ``mwm -p`` is
a script call and does not.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mwm_harness import cli
from mwm_harness.config import ConfigError, ModelSpec, is_subscription_endpoint

PLAN_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
LOCAL_URL = "http://localhost:11434/v1"


class _Stop(Exception):
    """Raised by a stub to end ``run`` once the guard has been passed."""


def _spec(model_id: str, base_url: str, key_env: str = "MWM_HARNESS_API_KEY") -> ModelSpec:
    return ModelSpec(id=model_id, base_url=base_url, key_env=key_env)


def _settings(default_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        permission_mode="default",
        mcp_allow=[],
        default_model=default_model,
        mcp_enabled=False,
        mcp_files=[],
    )


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (PLAN_URL, True),
        ("https://TOKEN-PLAN.example/v1", True),
        ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", False),
        (LOCAL_URL, False),
        ("", False),
    ],
)
def test_recognises_the_plan_endpoint(base_url: str, expected: bool) -> None:
    assert is_subscription_endpoint(base_url) is expected


def test_headless_turn_refuses_the_plan_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings("qwen3.8-max"))
    monkeypatch.setattr(cli, "load_models", lambda: {"qwen3.8-max": _spec("qwen3.8-max", PLAN_URL)})
    monkeypatch.setattr(
        cli,
        "load_secrets",
        lambda: pytest.fail("the guard must fire before any key is read"),
    )
    args = cli.parse_args(["-p", "hi", "--model", "qwen3.8-max"])
    with pytest.raises(ConfigError, match="subscription plan endpoint"):
        asyncio.run(cli.run(args))


def test_headless_agent_run_refuses_the_plan_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Stop gate calls ``mwm -p --agent``; that is a script call too."""
    monkeypatch.setattr(cli, "load_settings", lambda: _settings("qwen3.8-max"))
    monkeypatch.setattr(cli, "load_models", lambda: {"qwen3.8-max": _spec("qwen3.8-max", PLAN_URL)})
    monkeypatch.setattr(
        cli,
        "load_secrets",
        lambda: pytest.fail("the guard must fire before any key is read"),
    )
    args = cli.parse_args(["-p", "review this", "--agent", "critic", "--model", "qwen3.8-max"])
    with pytest.raises(ConfigError, match="subscription plan endpoint"):
        asyncio.run(cli.run(args))


def test_interactive_session_is_allowed_on_the_plan_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings("qwen3.8-max"))
    monkeypatch.setattr(cli, "load_models", lambda: {"qwen3.8-max": _spec("qwen3.8-max", PLAN_URL)})
    reached: list[bool] = []

    def stop(*_a: object, **_k: object) -> str:
        reached.append(True)
        raise _Stop

    monkeypatch.setattr(cli, "api_key_for", stop)
    args = cli.parse_args(["--model", "qwen3.8-max"])
    with pytest.raises(_Stop):
        asyncio.run(cli.run(args))
    assert reached == [True]


def test_headless_turn_allows_a_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    local = "ornith-1.5-35b-16k"
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(local))
    monkeypatch.setattr(
        cli,
        "load_models",
        lambda: {local: _spec(local, LOCAL_URL, key_env="OLLAMA_API_KEY")},
    )
    reached: list[bool] = []

    def stop(*_a: object, **_k: object) -> str:
        reached.append(True)
        raise _Stop

    monkeypatch.setattr(cli, "api_key_for", stop)
    args = cli.parse_args(["-p", "hi", "--model", local])
    with pytest.raises(_Stop):
        asyncio.run(cli.run(args))
    assert reached == [True]
