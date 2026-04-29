"""Binance USDS-M Futures kline fetcher."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from core.constants import (
    BINANCE_EXCHANGE_INFO_ENDPOINT,
    BINANCE_FUTURES_BASE,
    BINANCE_KLINE_LIMIT,
    BINANCE_KLINES_ENDPOINT,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# Binance kline columns in order
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


async def fetch_symbols(client: httpx.AsyncClient) -> list[str]:
    """Fetch all active USDT-margined perpetual symbols from Binance."""
    resp = await client.get(f"{BINANCE_FUTURES_BASE}{BINANCE_EXCHANGE_INFO_ENDPOINT}")
    resp.raise_for_status()
    data = resp.json()
    symbols = [
        s["symbol"]
        for s in data["symbols"]
        if s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
    ]
    logger.info("Fetched %d active USDT perpetual symbols", len(symbols))
    return sorted(symbols)


async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str = "6h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = BINANCE_KLINE_LIMIT,
    *,
    include_incomplete: bool = False,
) -> pl.DataFrame:
    """Fetch klines for a single symbol. Returns closed bars by default."""
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    resp = await client.get(
        f"{BINANCE_FUTURES_BASE}{BINANCE_KLINES_ENDPOINT}",
        params=params,
    )
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        return _empty_bars_df()

    df = _parse_klines(raw)
    if not include_incomplete:
        df = _drop_unclosed_bars(df)
    return df


async def fetch_all_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str = "6h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    *,
    include_incomplete: bool = False,
) -> pl.DataFrame:
    """Fetch all closed klines for a symbol by paginating through Binance API."""
    frames: list[pl.DataFrame] = []
    cursor = start_ms

    while True:
        df = await fetch_klines(
            client,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=end_ms,
            limit=BINANCE_KLINE_LIMIT,
            include_incomplete=include_incomplete,
        )
        if df.is_empty():
            break

        frames.append(df)
        last_open_time = df["open_time"].max()

        # Move cursor past the last bar
        if not isinstance(last_open_time, datetime):
            break
        cursor = int(last_open_time.timestamp() * 1000) + 1

        if len(df) < BINANCE_KLINE_LIMIT:
            break

        # Rate limit courtesy
        await asyncio.sleep(0.1)

    if not frames:
        return _empty_bars_df()

    result = pl.concat(frames).unique(subset=["open_time"]).sort("open_time")
    if not include_incomplete:
        result = _drop_unclosed_bars(result)
    logger.info("Fetched %d bars for %s", len(result), symbol)
    return result


def _drop_unclosed_bars(
    df: pl.DataFrame, *, now: datetime | None = None
) -> pl.DataFrame:
    """Drop bars whose scheduled close time has not passed yet."""
    if df.is_empty():
        return df
    now = now or datetime.now(tz=UTC)
    return df.filter(pl.col("close_time") < now)


def _parse_klines(raw: list[list]) -> pl.DataFrame:
    """Parse raw Binance kline response into a typed polars DataFrame."""
    rows = []
    for k in raw:
        rows.append(
            {
                "open_time": datetime.fromtimestamp(k[0] / 1000, tz=UTC),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": datetime.fromtimestamp(k[6] / 1000, tz=UTC),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
            }
        )
    return pl.DataFrame(rows)


def _empty_bars_df() -> pl.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    return pl.DataFrame(
        schema={
            "open_time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Datetime("us", "UTC"),
            "quote_volume": pl.Float64,
            "trades": pl.Int64,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
        }
    )
