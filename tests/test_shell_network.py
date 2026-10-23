"""Shell network boundary: no network unless the call asks; who decides about asking."""

from __future__ import annotations

import asyncio
import shutil
import socket
import threading
from types import SimpleNamespace

import pytest
from conftest import FixedApprover
from mwm_harness import cli
from mwm_harness.config import ConfigError, ModelSpec, Settings, load_settings
from mwm_harness.providers import chunks_for
from mwm_harness.sandbox import Sandbox, bwrap_works

LOCAL_URL = "http://localhost:11434/v1"


def _spec(model_id: str, base_url: str = LOCAL_URL) -> ModelSpec:
    return ModelSpec(id=model_id, base_url=base_url, key_env="OLLAMA_API_KEY")


needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not bwrap_works(), reason="bubblewrap does not run here"
)


def tool_result(session) -> dict:
    return next(b for m in session.messages for b in m.blocks() if b.get("type") == "tool_result")


@pytest.fixture
def local_server():
    """A TCP server on the host's loopback that answers one line."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.sendall(b"hello-from-host\n")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1]
    server.close()


def test_argv_cuts_the_network_unless_asked(tmp_path):
    sandbox = Sandbox("auto", [tmp_path])
    if not sandbox.enabled:
        pytest.skip("bubblewrap does not run here")
    assert "--unshare-net" in sandbox.argv("true")
    assert "--unshare-net" not in sandbox.argv("true", network=True)


@needs_bwrap
def test_a_command_without_network_cannot_reach_a_host_service(tmp_path, local_server):
    sandbox = Sandbox("bwrap", [tmp_path])
    probe = f"exec 3<>/dev/tcp/127.0.0.1/{local_server} && head -1 <&3"
    cut = asyncio.run(sandbox.run(probe, tmp_path, 10))
    assert cut.exit_code != 0 and "hello-from-host" not in cut.output
    open_ = asyncio.run(sandbox.run(probe, tmp_path, 10, network=True))
    assert open_.exit_code == 0 and "hello-from-host" in open_.output


def test_network_call_runs_unasked_in_bypass_mode(make_session):
    """The 2026-10-23 change: bypass is the "I am not watching" mode, so it must not
    stop at a network Bash. Before this, shell_network = "ask" forced a prompt in every
    mode, and a headless run's DenyAll approver turned that prompt into a refusal that
    killed git push."""
    approver = FixedApprover(False)
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "echo hi", "network": True})]),
        chunks_for("ok"),
    ]
    session, _, _ = make_session(turns, approver=approver, mode="bypassPermissions")
    asyncio.run(session.send("go"))
    assert approver.asked == []
    assert tool_result(session)["content"] == "hi"


def test_network_call_still_asks_in_default_mode(make_session):
    """What the bypass change must not touch: a session with a person at it still asks."""
    approver = FixedApprover(True)
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "echo hi", "network": True})]),
        chunks_for("ok"),
    ]
    session, _, _ = make_session(turns, approver=approver, mode="default")
    asyncio.run(session.send("go"))
    assert [name for name, _ in approver.asked] == ["Bash"]
    assert tool_result(session)["content"] == "hi"


def test_network_call_declined_in_default_mode(make_session):
    approver = FixedApprover(False)
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "echo hi", "network": True})]),
        chunks_for("ok"),
    ]
    session, _, _ = make_session(turns, approver=approver, mode="default")
    asyncio.run(session.send("go"))
    assert [name for name, _ in approver.asked] == ["Bash"]
    assert "declined" in tool_result(session)["content"]


def test_plain_call_in_bypass_mode_does_not_ask(make_session):
    approver = FixedApprover(False)
    turns = [chunks_for(tool_calls=[("Bash", {"command": "echo hi"})]), chunks_for("ok")]
    session, _, _ = make_session(turns, approver=approver, mode="bypassPermissions")
    asyncio.run(session.send("go"))
    assert approver.asked == []
    assert tool_result(session)["content"] == "hi"


def test_shell_network_deny_refuses_and_bad_value_is_an_error(make_session, tmp_path):
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "echo hi", "network": True})]),
        chunks_for("ok"),
    ]
    session, _, _ = make_session(turns, approver=FixedApprover(True), mode="bypassPermissions")
    session.settings.shell_network = "deny"
    asyncio.run(session.send("go"))
    assert "switched off" in tool_result(session)["content"]
    bad = tmp_path / "settings.toml"
    bad.write_text('shell_network = "sometimes"\n')
    with pytest.raises(ConfigError):
        load_settings(bad)
    assert Settings().shell_network == "ask"


class _Stop(Exception):
    """Ends ``cli.run`` once the flags have been applied to the settings."""


def _cli_settings(shell_network: str) -> SimpleNamespace:
    return SimpleNamespace(
        permission_mode="default",
        shell_network=shell_network,
        mcp_allow=[],
        default_model="m",
        mcp_enabled=False,
        mcp_files=[],
    )


def _run_cli(monkeypatch, argv: list[str], shell_network: str) -> SimpleNamespace:
    """Run ``cli.run`` up to the key check and hand back the settings it built."""
    settings = _cli_settings(shell_network)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_models", lambda: {"m": _spec("m", LOCAL_URL)})

    def stop(*_a: object, **_k: object) -> str:
        raise _Stop

    monkeypatch.setattr(cli, "api_key_for", stop)
    with pytest.raises(_Stop):
        asyncio.run(cli.run(cli.parse_args(argv)))
    return settings


def test_yolo_switches_off_both_prompts(monkeypatch):
    """One flag for an unattended run: bypass mode AND no network prompt. Either half
    alone still stops the job, because a headless approver is DenyAll."""
    settings = _run_cli(monkeypatch, ["-p", "task", "--yolo"], "ask")
    assert settings.permission_mode == "bypassPermissions"
    assert settings.shell_network == "allow"


def test_yolo_leaves_an_explicit_network_deny_alone(monkeypatch):
    settings = _run_cli(monkeypatch, ["-p", "task", "--yolo"], "deny")
    assert settings.permission_mode == "bypassPermissions"
    assert settings.shell_network == "deny"


def test_yolo_overrides_mode(monkeypatch):
    settings = _run_cli(monkeypatch, ["-p", "task", "--mode", "plan", "--yolo"], "ask")
    assert settings.permission_mode == "bypassPermissions"


def test_no_yolo_leaves_the_settings_as_loaded(monkeypatch):
    settings = _run_cli(monkeypatch, ["-p", "task"], "ask")
    assert settings.permission_mode == "default"
    assert settings.shell_network == "ask"
