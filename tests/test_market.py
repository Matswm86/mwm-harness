"""The panel chart feed: TopstepX bars first, Yahoo when the gateway fails."""

from __future__ import annotations

import asyncio
import json

import httpx
from mwm_harness.web.market import MarketFeed

GATEWAY_BARS = [  # the gateway answers newest first
    {"t": "2026-10-06T08:20:00+00:00", "o": 2.0, "h": 3.0, "l": 1.5, "c": 2.5, "v": 7},
    {"t": "2026-10-06T08:15:00+00:00", "o": 1.0, "h": 2.0, "l": 0.5, "c": 2.0, "v": 9},
]


def gateway(calls: list[str], fail: bool = False):
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        body = json.loads(request.content or b"{}")
        if request.url.host == "query1.finance.yahoo.com":
            return httpx.Response(
                200,
                json={
                    "chart": {
                        "result": [
                            {
                                "timestamp": [100, 400],
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": [1, 2],
                                            "high": [2, 3],
                                            "low": [0, 1],
                                            "close": [None, 2.5],
                                            "volume": [5, 6],
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                },
            )
        if fail:
            return httpx.Response(401, json={})
        if request.url.path == "/api/Auth/loginKey":
            assert body == {"userName": "me", "apiKey": "k"}
            return httpx.Response(200, json={"success": True, "token": "tok"})
        assert request.headers["authorization"] == "Bearer tok"
        if request.url.path == "/api/Contract/search":
            return httpx.Response(
                200,
                json={
                    "contracts": [
                        {"id": "CON.F.US.MNQ.U26", "activeContract": False},
                        {"id": "CON.F.US.MNQ.Z26", "activeContract": True},
                    ]
                },
            )
        assert (
            body["unit"] == 2
            and body["unitNumber"] == 5
            and body["contractId"] == "CON.F.US.MNQ.Z26"
        )
        return httpx.Response(200, json={"success": True, "bars": GATEWAY_BARS})

    return httpx.MockTransport(handle)


def test_topstepx_bars_come_oldest_first_and_are_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("PROJECT_X_USERNAME", raising=False)
    monkeypatch.delenv("PROJECT_X_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# creds\nPROJECT_X_USERNAME=me\nPROJECT_X_API_KEY='k'\n")
    calls: list[str] = []
    feed = MarketFeed(str(env), transport=gateway(calls))
    first = asyncio.run(feed.bars("mnq", 5))
    assert (
        first["source"] == "TopstepX"
        and first["contract"] == "CON.F.US.MNQ.Z26"
        and not first["delayed"]
    )
    assert [b["close"] for b in first["bars"]] == [2.0, 2.5]
    assert first["errors"] == []
    asyncio.run(feed.bars("MNQ", 5))
    assert calls.count("/api/History/retrieveBars") == 1  # the second call came from the cache


def test_a_failing_gateway_falls_back_to_delayed_yahoo(monkeypatch):
    monkeypatch.setenv("PROJECT_X_USERNAME", "me")
    monkeypatch.setenv("PROJECT_X_API_KEY", "k")
    feed = MarketFeed("", transport=gateway([], fail=True))
    result = asyncio.run(feed.bars("MNQ", 5))
    assert result["source"] == "Yahoo NQ=F" and result["delayed"] is True
    assert [b["close"] for b in result["bars"]] == [2.5]  # the bar with no close is dropped
    assert result["errors"] and result["errors"][0].startswith("gateway:")


def test_no_credentials_says_so_and_still_draws_yahoo(monkeypatch):
    monkeypatch.delenv("PROJECT_X_USERNAME", raising=False)
    monkeypatch.delenv("PROJECT_X_API_KEY", raising=False)
    result = asyncio.run(MarketFeed("", transport=gateway([])).bars("MNQ", 1))
    assert "no TopstepX credentials" in result["errors"][0] and result["delayed"]
