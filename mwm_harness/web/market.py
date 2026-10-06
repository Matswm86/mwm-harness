"""Candles for the panel's chart: TopstepX (ProjectX gateway) first, Yahoo as a delayed fallback.

Read-only: the only gateway calls are ``Auth/loginKey``, ``Contract/search`` and
``History/retrieveBars``. Credentials come from ``PROJECT_X_USERNAME`` and
``PROJECT_X_API_KEY``, in the environment or in the file named by the
``chart_env_file`` setting; they never reach the page.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

GATEWAY = "https://api.topstepx.com"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_SYMBOLS = {"MNQ": "NQ=F", "MES": "ES=F", "MGC": "GC=F", "MCL": "CL=F"}
TIMEFRAMES = (1, 5)  # minutes
LOOKBACK = timedelta(hours=20)  # covers Asia, London and New York of the current day
CACHE_SECONDS = 15.0
TOKEN_SECONDS = 20 * 3600.0


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


class MarketFeed:
    def __init__(self, env_file: str = "", transport: httpx.AsyncBaseTransport | None = None):
        self.env_file = Path(env_file).expanduser() if env_file else None
        self._client = httpx.AsyncClient(transport=transport, timeout=20.0)
        self._token = ""
        self._token_at = 0.0
        self._contracts: dict[str, tuple[str, float]] = {}
        self._cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}

    def credentials(self) -> tuple[str, str]:
        values = dict(os.environ)
        if self.env_file and self.env_file.is_file():
            values = {**read_env_file(self.env_file), **{k: v for k, v in values.items() if v}}
        return values.get("PROJECT_X_USERNAME", ""), values.get("PROJECT_X_API_KEY", "")

    async def _post(self, path: str, body: dict[str, Any], auth: bool = True) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._token}"} if auth else {}
        response = await self._client.post(GATEWAY + path, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("success") is False or data.get("errorCode"):
            raise RuntimeError(
                f"{path}: errorCode {data.get('errorCode')} {data.get('errorMessage') or ''}"
            )
        return data

    async def _login(self) -> None:
        if self._token and time.monotonic() - self._token_at < TOKEN_SECONDS:
            return
        user, key = self.credentials()
        if not user or not key:
            raise RuntimeError("no TopstepX credentials (PROJECT_X_USERNAME, PROJECT_X_API_KEY)")
        data = await self._post("/api/Auth/loginKey", {"userName": user, "apiKey": key}, auth=False)
        self._token, self._token_at = data["token"], time.monotonic()

    async def _contract(self, symbol: str) -> str:
        cached = self._contracts.get(symbol)
        if cached and time.monotonic() - cached[1] < 3600:
            return cached[0]
        data = await self._post("/api/Contract/search", {"searchText": symbol, "live": False})
        prefix = f"CON.F.US.{symbol}."
        found = [c for c in data.get("contracts") or [] if str(c.get("id", "")).startswith(prefix)]
        active = [c for c in found if c.get("activeContract")] or found
        if not active:
            raise RuntimeError(f"no {symbol} contract on TopstepX")
        self._contracts[symbol] = (active[0]["id"], time.monotonic())
        return active[0]["id"]

    async def _gateway(self, symbol: str, minutes: int) -> dict[str, Any]:
        await self._login()
        contract = await self._contract(symbol)
        end = datetime.now(UTC)
        data = await self._post(
            "/api/History/retrieveBars",
            {
                "contractId": contract,
                "live": False,
                "startTime": (end - LOOKBACK).isoformat(),
                "endTime": end.isoformat(),
                "unit": 2,  # minutes
                "unitNumber": minutes,
                "limit": 2000,
                "includePartialBar": True,
            },
        )
        bars = [
            {
                "time": int(datetime.fromisoformat(b["t"]).timestamp()),
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b.get("v", 0),
            }
            for b in data.get("bars") or []
        ]
        bars.sort(key=lambda b: b["time"])
        return {"source": "TopstepX", "contract": contract, "delayed": False, "bars": bars}

    async def _yahoo(self, symbol: str, minutes: int) -> dict[str, Any]:
        ticker = YAHOO_SYMBOLS.get(symbol, symbol)
        response = await self._client.get(
            YAHOO.format(symbol=ticker),
            params={"interval": f"{minutes}m", "range": "1d", "includePrePost": "true"},
            headers={"User-Agent": "Mozilla/5.0 mwm-harness"},
        )
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        bars = []
        for i, stamp in enumerate(result.get("timestamp") or []):
            row = [
                quote[k][i] if i < len(quote.get(k) or []) else None
                for k in ("open", "high", "low", "close")
            ]
            if None in row:
                continue
            volume = (quote.get("volume") or [0] * (i + 1))[i] or 0
            bars.append(
                {
                    "time": int(stamp),
                    "open": row[0],
                    "high": row[1],
                    "low": row[2],
                    "close": row[3],
                    "volume": volume,
                }
            )
        return {"source": f"Yahoo {ticker}", "contract": ticker, "delayed": True, "bars": bars}

    async def bars(self, symbol: str = "MNQ", minutes: int = 5) -> dict[str, Any]:
        symbol = symbol.upper()
        minutes = minutes if minutes in TIMEFRAMES else 5
        key = (symbol, minutes)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            return cached[1]
        errors = []
        result: dict[str, Any] | None = None
        for fetch in (self._gateway, self._yahoo):
            try:
                result = await fetch(symbol, minutes)
                if result["bars"]:
                    break
                errors.append(f"{result['source']}: no bars")
            except (
                httpx.HTTPError,
                RuntimeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                if fetch is self._gateway:
                    self._token = ""  # a rejected token is fetched again next time
                errors.append(f"{fetch.__name__.strip('_')}: {exc}")
        payload = {
            "type": "Bars",
            "symbol": symbol,
            "minutes": minutes,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "errors": errors,
            **(result or {"source": "", "contract": "", "delayed": False, "bars": []}),
        }
        if payload["bars"]:
            self._cache[key] = (time.monotonic(), payload)
        return payload

    async def close(self) -> None:
        await self._client.aclose()
