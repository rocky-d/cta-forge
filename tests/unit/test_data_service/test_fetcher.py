"""Tests for Binance kline fetching helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from data_service.fetcher import _drop_unclosed_bars, _parse_klines, fetch_klines


def _raw_kline(open_ms: int, close_ms: int) -> list:
    return [
        open_ms,
        "100.0",
        "105.0",
        "95.0",
        "102.0",
        "1000.0",
        close_ms,
        "100000.0",
        500,
        "500.0",
        "50000.0",
        "0",
    ]


def test_drop_unclosed_bars_keeps_only_closed_bars():
    closed = _raw_kline(0, 1_000)
    unclosed = _raw_kline(2_000, 10_000)
    df = _parse_klines([closed, unclosed])

    result = _drop_unclosed_bars(
        df,
        now=datetime.fromtimestamp(5, tz=UTC),
    )

    assert len(result) == 1
    assert result["open_time"].to_list() == [datetime.fromtimestamp(0, tz=UTC)]


class _FakeClient:
    def __init__(self):
        self.calls = 0

    async def get(self, url, params):
        self.calls += 1
        request = httpx.Request("GET", url, params=params)
        if self.calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )
        return httpx.Response(
            200,
            json=[_raw_kline(0, 1_000)],
            request=request,
        )


@pytest.mark.asyncio
async def test_fetch_klines_retries_binance_rate_limit():
    client = _FakeClient()

    result = await fetch_klines(client, "BTCUSDT")

    assert client.calls == 2
    assert len(result) == 1
