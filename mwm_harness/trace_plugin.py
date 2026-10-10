"""pytest plugin that traces one test for the run-it view.

The harness copies this file into a temp folder and loads it as ``-p mwm_trace_plugin``.

While the test body runs, every Python call into a file under ``MWM_TRACE_ROOT`` (minus
virtualenvs and site-packages) is recorded with its depth and argument values, and the
list is written as JSON to ``MWM_TRACE_OUT``. Caps keep the output small; values whose
argument name looks like a secret are masked. Standard library only: it runs in the
project's own Python, which may not have the harness installed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

SECRET = re.compile(r"(pass(word)?|secret|token|api_?key|auth|cookie|credential)", re.I)
MAX_CALLS = int(os.environ.get("MWM_TRACE_MAX_CALLS", "400"))
MAX_DEPTH = int(os.environ.get("MWM_TRACE_MAX_DEPTH", "12"))
REPR_CAP = 80

_root = Path(os.environ.get("MWM_TRACE_ROOT", ".")).resolve()
_calls: list[dict[str, Any]] = []
_skipped = 0
# Generator frames already recorded once, kept alive so a freed frame's id is never reused.
_resumed: dict[int, Any] = {}
GENERATOR_FLAGS = 0x20 | 0x80 | 0x200  # CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR


def _ours(filename: str) -> bool:
    if not filename.startswith(str(_root)):
        return False
    return not any(part in filename for part in ("/.venv/", "/site-packages/", "/node_modules/"))


def _value(name: str, value: Any) -> str:
    if SECRET.search(name):
        return "*** (masked)"
    try:
        text = repr(value)
    except Exception as exc:  # a broken __repr__ in the code under test
        text = f"<repr failed: {type(exc).__name__}>"
    return text if len(text) <= REPR_CAP else text[: REPR_CAP - 1] + "…"


def _tracer(base: int):
    def trace(frame, event, arg):
        global _skipped
        if event != "call" or not _ours(frame.f_code.co_filename):
            return None
        if frame.f_code.co_name.startswith("<"):
            return None  # comprehensions and generator expressions: noise in a call list
        if frame.f_code.co_flags & GENERATOR_FLAGS:
            if id(frame) in _resumed:
                return None  # a generator resuming is not a new call
            _resumed[id(frame)] = frame
        depth, walk = 0, frame.f_back
        while walk is not None:
            depth += 1
            walk = walk.f_back
        depth -= base
        if depth > MAX_DEPTH or len(_calls) >= MAX_CALLS:
            _skipped += 1
            return None
        code = frame.f_code
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        _calls.append(
            {
                "depth": depth,
                "function": getattr(code, "co_qualname", code.co_name),
                "file": str(Path(code.co_filename).resolve().relative_to(_root)),
                "line": frame.f_lineno,
                "args": {n: _value(n, frame.f_locals.get(n)) for n in names},
            }
        )
        return None  # call events only; no per-line tracing

    return trace


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    walk, base = sys._getframe(), 0
    while walk is not None:
        base += 1
        walk = walk.f_back
    sys.settrace(_tracer(base))
    try:
        yield
    finally:
        sys.settrace(None)


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("MWM_TRACE_OUT")
    if out:
        data = {"calls": _calls, "skipped": _skipped, "exit": int(exitstatus)}
        Path(out).write_text(json.dumps(data), encoding="utf-8")
