"""Spend metering for harness calls to the pay-as-you-go endpoint.

The harness reaches models through providers.py rather than core/llm_backends.py,
so before this module every headless `mwm -p` run on a paid model was unmetered
and outside BAILIAN_MONTHLY_BUDGET_USD. Both paths now write the same ledger.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mwm_harness import spend
from mwm_harness.config import ModelSpec
from mwm_harness.providers import OpenAICompatProvider, ProviderError

PAYG = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@pytest.fixture(autouse=True)
def _isolate_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Never write to the real ~/MWM/data/bailian-spend ledger from a test."""
    monkeypatch.setenv("MWM_AI_ROOT", str(tmp_path))
    monkeypatch.delenv("BAILIAN_MONTHLY_BUDGET_USD", raising=False)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (PAYG, True),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", True),
        ("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", False),
        ("http://localhost:11434/v1", False),
        ("https://api.groq.com/openai/v1", False),
    ],
)
def test_only_the_payg_alibaba_hosts_are_metered(base_url: str, expected: bool) -> None:
    """Local is free, the plan is a flat subscription, Groq is another account."""
    assert spend.is_metered(base_url) is expected


def test_record_prices_a_call_into_the_shared_ledger() -> None:
    spend.record("qwen3.8-flash", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    rows = [
        json.loads(x)
        for f in spend.ledger_path().parent.glob("*.jsonl")
        for x in f.read_text().splitlines()
        if x.strip()
    ]
    assert len(rows) == 1
    # 1M in at $0.15 + 1M out at $0.47
    assert rows[0]["cost_usd"] == pytest.approx(0.62)
    assert rows[0]["source"] == "harness"
    assert rows[0]["model"] == "qwen3.8-flash"


def test_record_ignores_an_empty_usage_block() -> None:
    spend.record("qwen3.8-flash", None)
    spend.record("qwen3.8-flash", {})
    assert not spend.ledger_path().exists()


def test_an_unpriced_model_is_costed_at_the_most_expensive_pair() -> None:
    spend.record("qwen9-brand-new", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    total, calls = spend.month_spend_usd()
    assert calls == 1
    assert total == pytest.approx(3.301 + 9.902)


def _rows() -> list[dict]:
    paths = sorted(spend.ledger_path().parent.glob("*.jsonl"))
    assert len(paths) == 1, f"expected one ledger, got {paths}"
    return [json.loads(x) for x in paths[0].read_text().splitlines() if x.strip()]


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        # The harness' own steady state: every turn resends the system prompt and
        # the tool list, so almost all of the input is a cache hit. Numbers from
        # this box's session transcripts, 151 pay-as-you-go flash calls.
        (
            "qwen3.8-flash",
            {
                "prompt_tokens": 30_456,
                "completion_tokens": 225,
                "prompt_tokens_details": {"cached_tokens": 22_464, "text_tokens": 30_456},
            },
            7_992 * 0.15 / 1e6 + 22_464 * 0.15 * 0.10 / 1e6 + 225 * 0.47 / 1e6,
        ),
        (
            "qwen3.8-max",
            {
                "prompt_tokens": 2011,
                "completion_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 1792, "text_tokens": 2011},
            },
            219 * 2.00 / 1e6 + 1792 * 2.00 * 0.10 / 1e6 + 300 * 6.00 / 1e6,
        ),
    ],
)
def test_cached_input_is_billed_at_a_tenth_of_the_input_price(
    model: str, usage: dict, expected: float
) -> None:
    spend.record(model, usage)
    row = _rows()[0]
    assert row["cost_usd"] == pytest.approx(round(expected, 8))
    assert row["cached_input_tokens"] == usage["prompt_tokens_details"]["cached_tokens"]
    # input_tokens stays the provider's own prompt_tokens, cached part included.
    assert row["input_tokens"] == usage["prompt_tokens"]


def test_a_cache_hit_is_cheaper_than_the_same_call_without_one() -> None:
    """The direction of the fix, as a positive control.

    Both calls report the same prompt_tokens, so a record() that ignored the cache
    details would price them identically and this test would fail rather than pass
    blindly.
    """
    base = {"prompt_tokens": 30_456, "completion_tokens": 225}
    spend.record("qwen3.8-flash", dict(base))
    spend.record("qwen3.8-flash", dict(base, prompt_tokens_details={"cached_tokens": 22_464}))
    cold, warm = (r["cost_usd"] for r in _rows())
    assert warm < cold
    # 22,464 tokens at 10% of $0.15 instead of $0.15 is $0.304 off a $0.0068 call.
    assert cold - warm == pytest.approx(22_464 * 0.15 * 0.90 / 1e6)


def test_no_cache_hit_costs_what_it_cost_before() -> None:
    """A zero hit must not move the price, so the old rows stay comparable."""
    spend.record(
        "qwen3.8-flash",
        {
            "prompt_tokens": 1715,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 1715},
        },
    )
    row = _rows()[0]
    assert row["cost_usd"] == pytest.approx(1715 * 0.15 / 1e6 + 1 * 0.47 / 1e6)
    assert row["cached_input_tokens"] == 0


def test_a_usage_block_without_the_details_field_is_not_a_crash() -> None:
    """An endpoint that omits the details block still gets priced, at full rate."""
    spend.record("qwen3.8-flash", {"prompt_tokens": 100, "completion_tokens": 5})
    row = _rows()[0]
    assert row["cached_input_tokens"] == 0
    assert row["cost_usd"] == pytest.approx(100 * 0.15 / 1e6 + 5 * 0.47 / 1e6)


def test_more_cached_tokens_than_prompt_tokens_cannot_make_a_call_free() -> None:
    spend.record(
        "qwen3.8-flash",
        {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 999_999},
        },
    )
    row = _rows()[0]
    assert row["input_tokens"] == 100
    assert row["cached_input_tokens"] == 100
    assert row["cost_usd"] == pytest.approx(100 * 0.15 * 0.10 / 1e6)


def test_a_details_field_that_is_not_a_mapping_is_not_a_crash() -> None:
    """A malformed usage block prices as zero, it never raises into the turn."""
    spend.record(
        "qwen3.8-flash",
        {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": "none"},
    )
    row = _rows()[0]
    assert row["cached_input_tokens"] == 0
    assert row["cost_usd"] == pytest.approx(100 * 0.15 / 1e6 + 5 * 0.47 / 1e6)


def test_budget_ok_refuses_once_the_month_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    spend.record("qwen3.8-flash", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    assert spend.month_spend_usd()[0] == pytest.approx(0.62)
    monkeypatch.setenv("BAILIAN_MONTHLY_BUDGET_USD", "1.00")
    assert spend.budget_ok("qwen3.8-flash") is True
    monkeypatch.setenv("BAILIAN_MONTHLY_BUDGET_USD", "0.50")
    assert spend.budget_ok("qwen3.8-flash") is False


def test_an_unreadable_budget_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAILIAN_MONTHLY_BUDGET_USD", "twelve")
    assert spend.budget_usd() == 0.0
    assert spend.budget_ok("qwen3.8-flash") is False


def test_a_corrupt_ledger_line_is_skipped_not_fatal() -> None:
    path = spend.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model": "qwen3.8-flash", "cost_usd": 0.10}) + "\n" + "not json at all\n"
    )
    total, calls = spend.month_spend_usd()
    assert total == pytest.approx(0.10)
    assert calls == 1


def test_the_fallback_table_is_what_runs_here() -> None:
    """This venv has no `requests`, so core.llm_backends cannot be imported and
    the fallback table is the operative one at runtime, not a rare path.

    The cross-check that the two tables agree lives in
    core/tests/test_llm_backends_cutover.py, where both modules are importable.
    """
    prices, unpriced, cache_factor = spend._core_tables()
    assert prices is spend.FALLBACK_PRICES_USD_PER_M
    assert unpriced == spend.FALLBACK_UNPRICED_USD_PER_M
    assert cache_factor == spend.FALLBACK_CACHE_HIT_INPUT_FACTOR
    assert prices["qwen3.8-flash"] == (0.15, 0.47)
    assert prices["qwen3.8-max"] == (2.00, 6.00)
    # Every model the harness registers on a metered endpoint must be priced, or
    # it silently bills at the most expensive fallback rate.
    from mwm_harness.config import load_models

    metered = [
        (mid, s.api_model or s.id)
        for mid, s in load_models().items()
        if spend.is_metered(s.base_url)
    ]
    assert metered, "no metered model is registered, so this test proves nothing"
    for model_id, wire in metered:
        assert wire in prices, f"{model_id} sends {wire}, which has no price"


def test_the_provider_refuses_a_call_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap has to stop the request, not merely note it afterwards."""
    spec = ModelSpec(id="payg-qwen3.8-flash", base_url=PAYG, key_env="X", api_model="qwen3.8-flash")
    monkeypatch.setenv("X", "sk-test")
    monkeypatch.setattr(spend, "budget_ok", lambda model: False)
    provider = OpenAICompatProvider(secrets={})

    async def drain() -> None:
        async for _ in provider.stream(spec, "sys", [], []):
            pass

    with pytest.raises(ProviderError) as exc:
        asyncio.run(drain())
    assert exc.value.status == 402
    assert "budget exhausted" in str(exc.value)


def test_an_unmetered_model_is_never_budget_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local model must keep working when the paid budget is exhausted.

    Port 9 is discard and nothing listens on it, so the call fails on the
    connection. The point is that it fails THERE, having got past the budget
    gate, rather than being refused with 402. A real Ollama on 11434 would
    answer the request, which is why the spec does not point at it.
    """
    spec = ModelSpec(id="spark-x2.5-4b-32k", base_url="http://127.0.0.1:9/v1", key_env="X")
    monkeypatch.setenv("X", "ollama")
    monkeypatch.setattr(
        spend, "budget_ok", lambda model: pytest.fail("a local model must not be budget-checked")
    )
    provider = OpenAICompatProvider(secrets={}, timeout=5.0)

    async def drain() -> None:
        async for _ in provider.stream(spec, "sys", [], []):
            pass

    with pytest.raises(ProviderError) as exc:
        asyncio.run(drain())
    assert exc.value.status != 402
