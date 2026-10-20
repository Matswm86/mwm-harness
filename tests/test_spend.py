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
    prices, unpriced = spend._core_tables()
    assert prices is spend.FALLBACK_PRICES_USD_PER_M
    assert unpriced == spend.FALLBACK_UNPRICED_USD_PER_M
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
