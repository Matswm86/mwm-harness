"""Mods (F1-F9): loading mods.toml, template values, and running each kind from the panel."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mwm_harness.mods import DEFAULT_MODS, Mod, fill, load_mods, tab_values, why_not
from mwm_harness.providers import chunks_for
from mwm_harness.web.server import create_app
from starlette.testclient import TestClient

PORT = 8765
TOKEN = "test-token"
HOST = {"Host": f"127.0.0.1:{PORT}"}


def until(ws, kind: str) -> list[dict]:
    seen = []
    while True:
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == kind:
            return seen


def git_repo(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a.txt").write_text("a\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(["git", "-C", str(folder), *args], check=True)
    return folder


def write_mods(body: str) -> Path:
    path = Path(os.environ["MWM_HARNESS_CONFIG"]) / "mods.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_the_default_set_has_f1_to_f9():
    mods = load_mods(DEFAULT_MODS)
    assert [m.key for m in mods.mods] == [f"F{n}" for n in range(1, 10)]
    assert mods.get("f2").confirm and mods.get("F6").lock == "review"


def test_the_longest_repo_folder_wins_and_values_are_quoted(tmp_path):
    path = tmp_path / "mods.toml"
    path.write_text(f'[repos]\n"{tmp_path}" = "infra"\n"{tmp_path / "projects/x"}" = "x"\n')
    mods = load_mods(path)
    assert mods.repo_for(tmp_path / "projects/x/src") == ("x", tmp_path / "projects/x")
    assert mods.repo_for(tmp_path / "other") == ("infra", tmp_path)
    assert (
        fill("commit -m {prompt}", {"prompt": "it's done; rm -rf /"}, shell=True)
        == "commit -m 'it'\"'\"'s done; rm -rf /'"
    )


def test_changed_means_written_by_this_tab_and_reported_by_git(tmp_path):
    repo = git_repo(tmp_path / "repo")
    (repo / "a.txt").write_text("changed\n")
    (repo / "b.txt").write_text("new\n")
    (repo / "c.txt").write_text("someone else\n")
    mods = load_mods(DEFAULT_MODS)
    values = tab_values(mods, repo, "t", {repo / "a.txt", repo / "b.txt", repo / "gone.txt"}, "")
    assert values["tab.changed"] == "a.txt b.txt"
    assert "tab.changed" not in tab_values(mods, repo, "t", set(), "")


def test_why_not_names_the_missing_piece():
    shell = Mod(
        key="F1", label="Save", kind="shell", run="cd {tab.repo_dir} && commit {tab.changed}"
    )
    assert "repo" in why_not(shell, {"tab.changed": "a"})
    assert "no file" in why_not(Mod(key="F1", label="S", kind="shell", run="x {tab.changed}"), {})
    assert why_not(Mod(key="F9", label="B", kind="prompt", text="hi"), {}) == ""


def panel(make_session, turns, mode="default"):
    session, _, _ = make_session(turns, mode=mode)
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)
    return session, TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def test_state_lists_mods_and_greys_a_save_with_nothing_written(make_session):
    write_mods(
        '[[mod]]\nkey = "F1"\nlabel = "Save"\nkind = "shell"\nrun = "git commit {tab.changed}"\n'
        '[[mod]]\nkey = "F2"\nlabel = "Push"\nkind = "shell"\nask = "a message"\nrun = "echo {prompt}"\n'
    )
    _, client = panel(make_session, [])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        mods = until(ws, "State")[-1]["mods"]
        assert [m["key"] for m in mods] == ["F1", "F2"]
        assert (
            "no file" in mods[0]["why"] and mods[1]["why"] == ""
        )  # the prompt comes with the press
        ws.send_json({"type": "mod", "key": "F2"})
        assert "type a message" in until(ws, "Notice")[-1]["text"]


def test_a_prompt_mod_sends_its_text_and_a_shell_mod_runs_through_bash(make_session):
    write_mods(
        '[[mod]]\nkey = "F9"\nlabel = "Brief"\nkind = "prompt"\ntext = "brief me"\n'
        '[[mod]]\nkey = "F1"\nlabel = "Say"\nkind = "shell"\nrun = "echo said {prompt}"\n'
    )
    _, client = panel(make_session, [chunks_for("Brief.")], mode="bypassPermissions")
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "mod", "key": "F9"})
        assert until(ws, "TurnStarted")[-1]["prompt"] == "brief me"
        until(ws, "TurnEnded")
        ws.send_json({"type": "mod", "key": "F1", "prompt": "two words"})
        seen = until(ws, "TurnEnded")
        finished = [m for m in seen if m["type"] == "ToolFinished"]
        assert finished[0]["name"] == "Bash" and "said two words" in finished[0]["content"]
        started = [m for m in seen if m["type"] == "TurnStarted"]
        assert seen[-1]["reason"] == "done" and started[0]["prompt"] == "[F1 Say]"


def test_a_confirm_mod_asks_even_when_the_mode_would_not(make_session):
    write_mods(
        '[[mod]]\nkey = "F2"\nlabel = "Push"\nkind = "shell"\nconfirm = true\nrun = "echo pushed"\n'
    )
    _, client = panel(make_session, [], mode="bypassPermissions")
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "mod", "key": "F2"})
        request = until(ws, "ApprovalRequest")[-1]
        assert request["tool"] == "Bash" and "always asks" in request["reason"]
        ws.send_json({"type": "approval", "id": request["id"], "answer": "no"})
        assert until(ws, "TurnEnded")[-1]["reason"] == "error"


def test_an_agent_mod_without_the_agent_says_so_and_a_held_lock_skips(make_session):
    write_mods(
        '[[mod]]\nkey = "F4"\nlabel = "Critic"\nkind = "agent"\nagent = "nobody"\n'
        '[[mod]]\nkey = "F6"\nlabel = "Review"\nkind = "agent"\nagent = "nobody"\nlock = "review"\n'
    )
    session, client = panel(make_session, [chunks_for("An answer.")])
    with client, client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws:
        until(ws, "Tabs")
        ws.send_json({"type": "prompt", "text": "hi"})
        until(ws, "TurnEnded")
        ws.send_json({"type": "mod", "key": "F4"})
        assert "no agent named nobody" in until(ws, "TurnEnded")[-1]["text"]
        lock = client.app.state.panel.mod_locks.setdefault("review", __import__("asyncio").Lock())
        lock._locked = True  # another tab is reviewing
        ws.send_json({"type": "mod", "key": "F6"})
        assert until(ws, "TurnEnded")[-1]["text"].startswith("SKIPPED")
