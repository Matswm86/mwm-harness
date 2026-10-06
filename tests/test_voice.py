"""Local Whisper for the mic button: errors come back as text, never as a crash."""

from __future__ import annotations

import asyncio

from mwm_harness.voice import Transcriber


def test_a_load_failure_is_reported_as_an_error(monkeypatch):
    transcriber = Transcriber("medium", "cpu")

    def broken():
        raise RuntimeError("no Whisper model could load: medium/cpu: disk full")

    monkeypatch.setattr(transcriber, "_load", broken)
    result = asyncio.run(transcriber.transcribe(b"x"))
    assert result.text == "" and "disk full" in result.error
