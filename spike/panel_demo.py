"""Serve the browser panel with a scripted model, so the page can be looked at without an API key.

    .venv/bin/python spike/panel_demo.py [port]

Type anything in the page: the scripted model ticks a task list, asks to write a
file (approval dialog), then answers. With PANEL_DEMO_AUTORUN=1 the turn starts
by itself at startup, so a page that opens later shows the pending approval.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import uvicorn
from mwm_harness.config import ModelSpec, Settings
from mwm_harness.loop import Session
from mwm_harness.providers import ScriptedProvider, chunks_for
from mwm_harness.web.server import create_app

MODELS = {
    name: ModelSpec(id=name, base_url="http://unused.invalid/v1", key_env="UNUSED")
    for name in ("scripted-qwen", "scripted-kimi")
}


def todos(*states: str) -> dict:
    names = ["Read the failing test", "Fix the off-by-one in parse()", "Run the test suite"]
    forms = ["Reading the failing test", "Fixing the off-by-one", "Running the test suite"]
    return {
        "todos": [
            {"content": n, "status": s, "activeForm": f}
            for n, s, f in zip(names, states, forms, strict=True)
        ]
    }


def script() -> list:
    one_round = [
        chunks_for(tool_calls=[("TodoWrite", todos("in_progress", "pending", "pending"))]),
        chunks_for(tool_calls=[("Glob", {"pattern": "*.py"})]),
        chunks_for(tool_calls=[("TodoWrite", todos("completed", "in_progress", "pending"))]),
        chunks_for(
            "The loop stops one item early. I will write the fix:\n\n```python\n"
            "for index in range(len(items)):\n    handle(items[index])\n```\n",
            tool_calls=[
                (
                    "Write",
                    {
                        "file_path": "parse.py",
                        "content": "def parse(items):\n    return list(items)\n",
                    },
                )
            ],
        ),
        chunks_for(tool_calls=[("TodoWrite", todos("completed", "completed", "in_progress"))]),
        chunks_for(tool_calls=[("TodoWrite", todos("completed", "completed", "completed"))]),
        chunks_for("Fixed `parse()` and the suite is green: 3 of 3 tasks done."),
    ]
    return one_round * 20


async def main(port: int) -> None:
    with tempfile.TemporaryDirectory() as folder:
        project = Path(folder) / "scratch-repo"
        project.mkdir()
        (project / "parse.py").write_text("def parse(items):\n    return list(items)[:-1]\n")
        session = Session(
            cwd=project,
            model=MODELS["scripted-qwen"],
            provider=ScriptedProvider(script()),
            settings=Settings(sandbox="off"),
            hook_settings=[],
            sessions_dir=Path(folder) / "sessions",
            system_prompt="demo",
            skills={},
            commands={},
            agents={},
        )
        token = os.environ.get("PANEL_DEMO_TOKEN", "demo")
        app = create_app(session, MODELS, token, port)
        panel = app.state.panel
        if os.environ.get("PANEL_DEMO_AUTORUN"):

            async def autorun() -> None:
                await asyncio.sleep(0.3)  # let the server's startup run session.start() first
                panel.start_turn("The parse test fails, please fix it.", asyncio.Queue())

            asyncio.get_running_loop().create_task(autorun())
        print(f"http://127.0.0.1:{port}/#token={token}")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765))
