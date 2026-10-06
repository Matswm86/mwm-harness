"""The terminal's /tab command: several sessions in one terminal, one on screen."""

from __future__ import annotations

import asyncio

from mwm_harness.repl.terminal import Tabs


class Out:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def line(self, text: str = "") -> None:
        self.rows.append(text)

    def __call__(self, event) -> None:  # subscribed to each new session's bus
        pass


def test_tab_new_switch_list_and_close(make_session, tmp_path):
    first, _, _ = make_session([])
    other = tmp_path / "other"
    other.mkdir()
    out = Out()
    tabs = Tabs(first, out)

    async def run() -> None:
        assert await tabs.command(f"/tab new {other}", out)
        assert tabs.current.cwd == other.resolve() and len(tabs.sessions) == 2
        await tabs.command("/tab 1", out)
        assert tabs.current is first
        out.rows.clear()
        await tabs.command("/tab", out)
        assert out.rows[0].startswith("* 1") and out.rows[1].startswith("  2")
        await tabs.command("/tab close 2", out)
        assert tabs.sessions == [first]
        await tabs.command("/tab close", out)
        assert "last tab stays open" in out.rows[-1]
        assert not await tabs.command("/tasks", out)
        await tabs.close()

    asyncio.run(run())
