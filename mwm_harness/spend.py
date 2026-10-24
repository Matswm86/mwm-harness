"""Spend metering for harness calls to the pay-as-you-go Alibaba endpoint.

The harness talks to models through ``providers.py``, not through
``core/llm_backends.py``, so the ledger and monthly cap that guard the text
lanes did not cover it. That left every headless ``mwm -p`` run on a paid model
unmetered, which is the exact failure the cap exists to prevent: a runaway agent
loop billing freely with nothing watching.

This module writes the SAME ledger file in the SAME schema as
``core/llm_backends.py``, so one budget covers both paths and one command shows
the month's total. The price table is imported from core when it is importable
and falls back to a copy; ``tests/test_spend.py`` asserts the copy matches core
so the two cannot drift apart silently.

Only the Alibaba pay-as-you-go hosts are metered. Local Ollama costs nothing,
the Token Plan is a flat subscription, and Groq has its own account and its own
limits, so pricing those with Alibaba's sheet would be wrong.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("mwm.spend")

METERED_HOSTS = frozenset({"dashscope-intl.aliyuncs.com", "dashscope.aliyuncs.com"})
LEDGER_SUBDIR = Path("data") / "bailian-spend"
DEFAULT_BUDGET_USD = 12.00

# Fallback copy of core.llm_backends.BAILIAN_PRICES_USD_PER_M, USD per 1M tokens,
# Singapore sheet read 2026-10-20. tests/test_spend.py fails if core is
# importable and disagrees with this.
FALLBACK_PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "qwen3.8-max": (2.00, 6.00),
    "qwen3.8-max-0902": (2.00, 6.00),
    "qwen3.7-max": (2.50, 7.50),
    "qwen-max": (1.60, 6.40),
    "qwen3.8-flash": (0.15, 0.47),
    "qwen3.5-flash": (0.10, 0.40),
    "qwen-flash": (0.05, 0.40),
    "qwen3-vl-plus": (0.20, 1.60),
}
FALLBACK_UNPRICED_USD_PER_M = (3.301, 9.902)
# A cache-hit input token bills at this fraction of the model's standard input
# price (Alibaba sheet, read 2026-10-24). Copy of core.llm_backends' constant of
# the same name; core/tests/test_llm_backends_cutover.py::
# test_harness_spend_table_matches_core fails if the two drift apart.
FALLBACK_CACHE_HIT_INPUT_FACTOR = 0.10


def workspace_root() -> Path:
    return Path(os.environ.get("MWM_AI_ROOT", str(Path.home() / "MWM")))


def _core_tables() -> tuple[dict[str, tuple[float, float]], tuple[float, float], float]:
    """core.llm_backends' price tables when reachable, else the local copies."""
    root = str(workspace_root())
    if root not in os.sys.path:
        os.sys.path.insert(0, root)
    try:
        from core.llm_backends import (  # noqa: PLC0415
            BAILIAN_CACHE_HIT_INPUT_FACTOR,
            BAILIAN_PRICES_USD_PER_M,
            BAILIAN_UNPRICED_USD_PER_M,
        )

        return (
            BAILIAN_PRICES_USD_PER_M,
            BAILIAN_UNPRICED_USD_PER_M,
            BAILIAN_CACHE_HIT_INPUT_FACTOR,
        )
    except Exception:  # noqa: BLE001 - metering must never break a harness turn
        return (
            FALLBACK_PRICES_USD_PER_M,
            FALLBACK_UNPRICED_USD_PER_M,
            FALLBACK_CACHE_HIT_INPUT_FACTOR,
        )


def is_metered(base_url: str) -> bool:
    return urlparse(base_url).netloc.lower() in METERED_HOSTS


def ledger_path() -> Path:
    return workspace_root() / LEDGER_SUBDIR / f"{datetime.now(UTC):%Y-%m}.jsonl"


def budget_usd() -> float:
    raw = os.environ.get("BAILIAN_MONTHLY_BUDGET_USD", "").strip()
    if not raw:
        return DEFAULT_BUDGET_USD
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.error("BAILIAN_MONTHLY_BUDGET_USD=%r is not a number; failing closed at 0", raw)
        return 0.0


def month_spend_usd() -> tuple[float, int]:
    """(USD spent this calendar month, calls recorded). Unreadable ledger = inf."""
    path = ledger_path()
    if not path.is_file():
        return 0.0, 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.error("cannot read spend ledger %s: %s", path, exc)
        return float("inf"), 0
    total, calls = 0.0, 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            total += float(json.loads(line).get("cost_usd") or 0.0)
            calls += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("skipping corrupt line in spend ledger %s", path)
    return total, calls


def budget_ok(model: str) -> bool:
    """False when the month's spend has reached the cap. Never raises."""
    try:
        budget = budget_usd()
        spent, calls = month_spend_usd()
    except Exception as exc:  # noqa: BLE001
        logger.error("spend check failed for %s: %s; refusing the call", model, exc)
        return False
    if spent < budget:
        return True
    logger.error(
        "bailian monthly budget exhausted (spent=%s budget=%s calls=%d model=%s); refusing",
        spent,
        budget,
        calls,
        model,
    )
    return False


def record(model: str, usage: dict | None) -> None:
    """Append one priced call. A write failure is logged, never raised.

    `input_tokens` stays the API's own `prompt_tokens`, cache-hit part included.
    That part is reported beside it in `cached_input_tokens` and billed at
    FALLBACK_CACHE_HIT_INPUT_FACTOR (core's BAILIAN_CACHE_HIT_INPUT_FACTOR) of the
    input price, because a cache-hit input token costs a tenth of a new one on the
    Alibaba sheet. Measured on this box 2026-10-24 from the harness' own session
    transcripts: 151 pay-as-you-go flash calls sent 4,604,942 prompt tokens of
    which 3,399,424 (73.8%) came back as cached_tokens, so billing every input
    token at full rate over-counted the harness' biggest ledger share by 2.2x,
    which fires the monthly cap early and hides a real overspend in a phantom one.
    """
    if not usage:
        return
    prices, unpriced, cache_factor = _core_tables()
    price_in, price_out = prices.get(model, unpriced)
    try:
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
    except (ValueError, TypeError):
        in_tok = out_tok = 0
    # Parsed apart from the token counts on purpose: a malformed details field must
    # cost the call cached=0, not zero the counts too, which would bill a paid call
    # at nothing and drain the budget unwatched.
    cached_in = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        try:
            cached_in = int(details.get("cached_tokens") or 0)
        except (ValueError, TypeError):
            cached_in = 0
    # A provider that reports more cached than prompted tokens must not be able to
    # turn a paid call free, so the cached share is clamped inside the input count.
    cached_in = max(0, min(cached_in, in_tok))
    cost = (
        (in_tok - cached_in) / 1_000_000 * price_in
        + cached_in / 1_000_000 * price_in * cache_factor
        + out_tok / 1_000_000 * price_out
    )
    path = ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                        "model": model,
                        "input_tokens": in_tok,
                        "cached_input_tokens": cached_in,
                        "output_tokens": out_tok,
                        "price_in_per_m": price_in,
                        "price_out_per_m": price_out,
                        "cost_usd": round(cost, 8),
                        "source": "harness",
                    }
                )
                + "\n"
            )
    except OSError as exc:
        logger.error("cannot write spend ledger %s: %s", path, exc)


__all__ = [
    "FALLBACK_CACHE_HIT_INPUT_FACTOR",
    "FALLBACK_PRICES_USD_PER_M",
    "FALLBACK_UNPRICED_USD_PER_M",
    "METERED_HOSTS",
    "budget_ok",
    "budget_usd",
    "is_metered",
    "ledger_path",
    "month_spend_usd",
    "record",
]
