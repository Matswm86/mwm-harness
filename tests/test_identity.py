"""The agent must not introduce itself as Claude's helper: the rule files it inherits say Claude."""

from mwm_harness.config import ModelSpec
from mwm_harness.context import build_system_prompt


def test_the_prompt_names_the_agent_and_reframes_inherited_claude_wording(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Claude Code hooks rewrite every shell command.\n")
    model = ModelSpec(id="some-model", base_url="http://x/v1", key_env="K")
    prompt = build_system_prompt(tmp_path, tmp_path, model)
    identity = prompt.index("You are not Claude")
    assert identity < prompt.index("Claude Code hooks rewrite")  # said before the files are read
    assert "read those words as meaning you and this harness" in prompt
    assert "- Model: some-model" in prompt
