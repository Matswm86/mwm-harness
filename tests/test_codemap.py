"""Item 32: who calls this. The static map, and the two places it is shown."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mwm_harness import codemap, review
from mwm_harness.config import ModelSpec
from mwm_harness.providers import chunks_for

# The plan's fixture: b.helper has two callers, and both are found through an import.
A = "from b import helper\n\n\ndef run():\n    return helper()\n"
B = "def helper():\n    return 1\n"
C = "from b import helper\n\n\ndef main():\n    return helper()\n"


def tree(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def test_changing_a_function_names_the_functions_that_call_it(tmp_path: Path) -> None:
    code = codemap.index(tree(tmp_path, {"a.py": A, "b.py": B, "c.py": C}))
    callers, unresolved = code.radius("b.py", [1, 2])
    assert callers == ["a.run", "c.main"]
    assert unresolved == []
    assert code.block("b.helper").calls == []


def test_a_call_decided_while_the_program_runs_is_listed_not_guessed(tmp_path: Path) -> None:
    body = (
        "handlers = {}\n"
        "\n"
        "\n"
        "def fire(spec, kind):\n"
        "    getattr(spec, kind)()\n"
        "    handlers[kind]()\n"
        "    return kind\n"
    )
    code = codemap.index(tree(tmp_path, {"d.py": body}))
    block = code.block("d.fire")
    assert block.calls == []
    assert len(block.unresolved) == 2
    assert any("getattr" in item for item in block.unresolved)
    assert any("handlers[kind]" in item for item in block.unresolved)
    # every unresolved call carries the line it happened on, so a reader can go and look
    assert all(item.rsplit(":", 1)[-1].isdigit() for item in block.unresolved)


def test_a_method_on_self_is_found_and_an_attribute_this_class_lacks_is_not(tmp_path: Path) -> None:
    body = (
        "class Box:\n"
        "    def run(self):\n"
        "        self.helper()\n"
        "        self.nope()\n"
        "\n"
        "    def helper(self):\n"
        "        return 1\n"
    )
    code = codemap.index(tree(tmp_path, {"box.py": body}))
    run = code.block("box.Box.run")
    assert run.calls == ["box.Box.helper"]
    assert run.unresolved == ["self.nope:4"]
    assert code.block("box.Box.helper").callers == ["box.Box.run"]


def test_a_name_handed_to_a_worker_counts_as_reaching_it(tmp_path: Path) -> None:
    body = "import asyncio\n\nfrom b import helper\n\n\nasync def go():\n    await asyncio.to_thread(helper)\n"
    code = codemap.index(tree(tmp_path, {"a.py": body, "b.py": B}))
    assert code.block("b.helper").callers == ["a.go"]


def test_a_file_that_will_not_parse_is_reported_and_the_rest_of_the_map_still_builds(
    tmp_path: Path,
) -> None:
    files = {"a.py": A, "b.py": B, "broken.py": "def oops(:\n"}
    code = codemap.index(tree(tmp_path, files))
    assert "broken.py" in code.errors
    assert "SyntaxError" in code.errors["broken.py"]
    assert code.radius("b.py", [1])[0] == ["a.run"]
    assert "not parsed: broken.py" in code.text()


def test_the_map_text_is_one_line_per_function_and_fits_a_small_window() -> None:
    package = Path(__file__).resolve().parents[1] / "mwm_harness"
    text = codemap.text(package)
    assert text.startswith("# code map of mwm_harness:")
    assert "\n  collect_changes " in text
    assert len(text) <= codemap.MAX_TEXT_CHARS
    tokens = len(text) // 4  # my calc: the four-characters-to-a-token estimate the plan uses
    assert tokens < 8_000, f"the map of mwm_harness/ is {tokens} tokens"


def test_a_review_carries_the_caller_line_for_the_functions_its_diff_touched(
    tmp_path: Path,
) -> None:
    root = tree(tmp_path, {"a.py": A, "b.py": B, "c.py": C, "notes.md": "hi\n"})
    changed = review.FileChange(
        "b.py", "@@ -1,2 +1,2 @@\n def helper():\n-    return 1\n+    return 2\n", 1, 1
    )
    prose = review.FileChange("notes.md", "@@ -1 +1 @@\n-hi\n+hey\n", 1, 1)
    broken = review.FileChange("broken.py", "@@ -0,0 +1 @@\n+def oops(:\n", 1, 0)
    root.joinpath("broken.py").write_text("def oops(:\n", encoding="utf-8")

    warnings: list[str] = []
    review.attach_radius(root, [changed, prose, broken], warnings.append)

    assert changed.callers == ["a.run", "c.main"]
    assert prose.callers == []  # a markdown file has no callers to find
    assert (
        broken.callers == []
    )  # it will not parse, and that is the indexer's business, not a crash
    assert warnings == []
    prompt = review.build_prompt("change the answer", [changed], [])
    assert "called by 2: a.run, c.main" in prompt


def test_a_model_with_a_small_window_is_offered_the_map(make_session, tmp_path: Path) -> None:
    session, _, _ = make_session([chunks_for("hi")])
    (tmp_path / "project" / "thing.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    tiny = ModelSpec(
        id="tiny", base_url="http://unused.invalid/v1", key_env="UNUSED_KEY", context_window=8_000
    )

    session.set_model(tiny)
    asyncio.run(session._load_code_map())

    assert "# Code map" in session.system_prompt
    assert "thing.py" in session.system_prompt
    assert "go 1-2" in session.system_prompt


def test_a_model_with_a_roomy_window_gets_no_map(make_session) -> None:
    session, _, _ = make_session([chunks_for("hi")])
    asyncio.run(session._load_code_map())
    assert session._code_map is None
    assert session._code_map_prompt() == ""
    assert "# Code map" not in session.system_prompt
