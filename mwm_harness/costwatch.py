"""Turn cost-spike watch: anomaly when one turn bills >3x the session median.

From the r/AI_Agents "60 engineers" post (2026-10-23 audit, DO item 1b): the
runaway-loop failure shows up in the bill before it shows up in the output.
A stuck agent re-reading files or looping tool calls turns cheap into
expensive one request at a time; the session median stays normal while the
current turn explodes, so the ratio is the signal.

Token counts are the cost measure here, not dollars: prompt tokens are priced
per call but cached input dominates, and completion tokens track the real
spend of a looping turn. The check fires only after enough turns exist for a
median to mean something (MIN_TURNS), so a two-turn session never alarms.

Writes one line to the shared anomaly log (core/observability/log_event.py ->
data/runners/anomalies.jsonl) so it lands next to the hook-side anomalies, and
emits a Notice the panel shows. Best effort: a missing core module or a full
disk must never break a turn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("mwm.costwatch")

SPIKE_RATIO = 3.0
MIN_TURNS = 5  # a median over fewer turns is noise


class TurnCostWatch:
    """Remembers per-turn completion tokens; flags a turn >3x the median."""

    def __init__(self, ratio: float = SPIKE_RATIO, min_turns: int = MIN_TURNS) -> None:
        self.ratio = ratio
        self.min_turns = min_turns
        self.turn_tokens: list[int] = []

    def record(self, tokens: int) -> dict[str, float] | None:
        """Store one finished turn. Returns spike details, or None when normal.

        The comparison median covers the turns BEFORE this one, so a sustained
        expensive stretch re-baselines instead of alarming forever.
        """
        spike = None
        history = self.turn_tokens
        if len(history) >= self.min_turns:
            ordered = sorted(history)
            mid = len(ordered) // 2
            median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            if median > 0 and tokens > self.ratio * median:
                spike = {
                    "turn_tokens": float(tokens),
                    "median_tokens": float(median),
                    "ratio": round(tokens / median, 2),
                }
        self.turn_tokens.append(tokens)
        return spike


def log_spike(session_id: str, model: str, spike: dict[str, float]) -> bool:
    """One anomaly-log line via core.observability.log_event. Never raises."""
    root = str(Path(os.environ.get("MWM_AI_ROOT", str(Path.home() / "MWM"))))
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from core.observability.log_event import SEV_WARN, bump, log_event

        written = log_event(
            source="harness_cost_watch",
            kind="turn_cost_spike",
            severity=SEV_WARN,
            session=session_id,
            model=model,
            **spike,
        )
        bump("harness_cost_watch", "turn_cost_spike")
        return bool(written)
    except Exception:  # noqa: BLE001 - observability must never break a turn
        logger.warning("cost-spike anomaly not logged", exc_info=True)
        return False


__all__ = ["MIN_TURNS", "SPIKE_RATIO", "TurnCostWatch", "log_spike"]
