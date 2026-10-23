"""Cost-spike watch: unit tests for TurnCostWatch plus one Session-level test."""

from __future__ import annotations

import asyncio

from conftest import FixedApprover
from mwm_harness import events as ev
from mwm_harness.costwatch import MIN_TURNS, SPIKE_RATIO, TurnCostWatch
from mwm_harness.providers import chunks_for


def test_no_spike_until_min_turns():
    watch = TurnCostWatch()
    for _ in range(MIN_TURNS - 1):
        assert watch.record(100) is None
    # The fifth turn is the first with enough history; 100 vs median 100 is normal.
    assert watch.record(100) is None


def test_spike_over_ratio_is_flagged():
    watch = TurnCostWatch()
    for _ in range(MIN_TURNS):
        watch.record(100)
    spike = watch.record(1000)
    assert spike is not None
    assert spike["ratio"] == 10.0
    assert spike["median_tokens"] == 100.0


def test_just_under_ratio_is_clean():
    watch = TurnCostWatch()
    for _ in range(MIN_TURNS):
        watch.record(100)
    assert watch.record(int(SPIKE_RATIO * 100)) is None  # >3x required, not >=


def test_sustained_expensive_stretch_rebaselines_eventually():
    watch = TurnCostWatch()
    for _ in range(MIN_TURNS):
        watch.record(100)
    # The first expensive turns still spike against the cheap median.
    assert watch.record(10_000) is not None
    for _ in range(5):
        watch.record(10_000)
    # Once expensive turns are the majority the median has caught up and the
    # same cost is normal again: the watch alarms on a CHANGE, not on a level.
    assert watch.record(10_000) is None


def test_zero_median_never_divides():
    watch = TurnCostWatch()
    for _ in range(MIN_TURNS):
        watch.record(0)
    assert watch.record(500) is None


def test_session_emits_notice_and_logs_on_spike(make_session, monkeypatch) -> None:
    logged: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "mwm_harness.loop.log_spike", lambda sid, model, spike: logged.append((sid, spike)) or True
    )
    turns = []
    # Six cheap turns then one expensive turn; each turn is one scripted answer.
    for completion in [10, 12, 11, 10, 12, 11, 5000]:
        turns.append(
            chunks_for(text="ok", usage={"prompt_tokens": 100, "completion_tokens": completion})
        )
    session, recorder, _ = make_session(turns, approver=FixedApprover(True))
    for i in range(7):
        asyncio.run(session.send(f"turn {i}"))
    notices = [n for n in recorder.of(ev.Notice) if "cost spike" in n.text]
    assert len(notices) == 1
    assert "5,000" in notices[0].text
    assert len(logged) == 1
    assert logged[0][1]["ratio"] > 3
