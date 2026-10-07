import json
from types import SimpleNamespace

from mwm_harness import events as ev
from mwm_harness import postmortem
from mwm_harness.repl.terminal import Printer, run_postmortem


def _line(kind, content, ts, **extra):
    entry = {"type": kind, "message": {"role": kind, "content": content}, "timestamp": ts}
    entry.update(extra)
    return json.dumps(entry)


def _use(call_id, name, tool_input):
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def _result(call_id, text, is_error=True):
    return {"type": "tool_result", "tool_use_id": call_id, "content": text, "is_error": is_error}


def transcript(tmp_path):
    lines = [
        _line("user", "fix the build", "2026-10-07T09:00:00.000Z"),
        _line(
            "assistant",
            [_use("a", "Bash", {"command": "make"}), _use("b", "Edit", {"file_path": "x"})],
            "2026-10-07T09:00:01.000Z",
        ),
        _line(
            "user",
            [
                _result("a", "make: *** No rule to make target"),
                _result("b", "Blocked by a PreToolUse hook: edits need a read first"),
            ],
            "2026-10-07T09:00:45.000Z",
        ),
        _line("assistant", [_use("c", "Write", {"file_path": "y"})], "2026-10-07T09:00:46.000Z"),
        _line(
            "user",
            [_result("c", "The user declined this action. Ask before trying another way.")],
            "2026-10-07T09:00:47.000Z",
        ),
        _line("assistant", [_use("d", "Read", {"file_path": "z"})], "2026-10-07T09:00:48.000Z"),
        _line("user", [_result("d", "ok", is_error=False)], "2026-10-07T09:00:49.000Z"),
        _line(
            "user",
            "Stop hook feedback:\nno test was run",
            "2026-10-07T09:00:50.000Z",
            isMeta=True,
        ),
        json.dumps({"type": "system", "subtype": "compact_boundary", "content": "40 messages"}),
        "{torn line",
    ]
    path = tmp_path / "abcdef12-0000.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_postmortem_sorts_every_failure_into_its_layer(tmp_path):
    report = postmortem.analyze(transcript(tmp_path))
    found = {(f.layer, f.kind): f.count for f in report.ranked()}
    assert found[("environment", "shell command failed")] == 1
    assert found[("orchestration", "PreToolUse hook blocked a call")] == 1
    assert found[("permissions", "person declined a call")] == 1
    assert found[("verification", "Stop gate sent the answer back")] == 1
    assert found[("context", "history compacted")] == 1
    assert found[("observability", "tool call over 30s")] == 2  # Bash and Edit both took 44 s
    assert report.turns == 1 and report.tool_calls == 4
    assert [round(s.seconds) for s in report.slow] == [44, 44]


def test_postmortem_report_proposes_fixes_and_changes_nothing(tmp_path):
    path = transcript(tmp_path)
    before = path.read_text(encoding="utf-8")
    target = postmortem.write(postmortem.analyze(path), tmp_path / "out")
    text = target.read_text(encoding="utf-8")
    assert "## What failed, by harness layer" in text
    assert "## Proposed fixes (drafts; nothing was changed)" in text
    assert "**verification**" in text and "no test was run" in text
    assert "(1 min)" in text
    assert path.read_text(encoding="utf-8") == before


def test_postmortem_on_a_clean_session_says_so(tmp_path):
    path = tmp_path / "clean.jsonl"
    path.write_text(_line("user", "hi", "2026-10-07T09:00:00Z") + "\n", encoding="utf-8")
    assert "No failures recorded" in postmortem.render(postmortem.analyze(path))


def test_classify_error_reads_the_loop_texts():
    assert postmortem.classify_error("Read", "The call was not run: bad path")[0] == "tools"
    assert (
        postmortem.classify_error("Bash", "Not run: this exact call was made 3 times")[0] == "loop"
    )
    assert postmortem.classify_error("Grep", "Grep crashed: ValueError: x")[1] == "Grep crashed"


def test_postmortem_command_writes_report_and_draft_returns_a_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(postmortem, "postmortems_root", lambda: tmp_path / "pm")
    session = SimpleNamespace(transcript=SimpleNamespace(path=transcript(tmp_path)), cwd=tmp_path)
    out = Printer(color=False)
    assert run_postmortem(session, "", out) is True
    printed = capsys.readouterr().out
    assert "4 tool calls" in printed and "observability" in printed
    assert (tmp_path / "pm" / "abcdef12-0000.md").is_file()
    prompt = run_postmortem(session, "draft", out)
    assert isinstance(prompt, str) and "abcdef12-0000.draft.md" in prompt
    assert "Do not edit any hook" in prompt


def test_terminal_shows_how_long_a_tool_ran(capsys):
    out = Printer(color=False)
    out(ev.ToolStarted("t1", "Bash", {"command": "ls"}, ts="2026-10-07T09:00:00.000Z"))
    out(ev.ToolFinished("t1", "Bash", "a.txt", False, ts="2026-10-07T09:00:02.500Z"))
    assert "a.txt  (2.5s)" in capsys.readouterr().out
