"""Shell network boundary: no network unless the call asks, and asking always reaches the person."""

from __future__ import annotations

import asyncio
import shutil
import socket
import threading

import pytest
from conftest import FixedApprover
from mwm_harness.config import ConfigError, Settings, load_settings
from mwm_harness.providers import chunks_for
from mwm_harness.sandbox import Sandbox, bwrap_works

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


def test_network_call_asks_even_in_bypass_mode(make_session):
    approver = FixedApprover(False)
    turns = [
        chunks_for(tool_calls=[("Bash", {"command": "echo hi", "network": True})]),
        chunks_for("ok"),
    ]
    session, _, _ = make_session(turns, approver=approver, mode="bypassPermissions")
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
