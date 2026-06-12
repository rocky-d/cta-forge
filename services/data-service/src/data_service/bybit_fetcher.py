"""Bybit USDT perpetual futures kline fetcher.

Stores data in data-bybit/ (separate from Binance's data/) using the same
ParquetStore layout.  Normalises Bybit's response to match the schema
expected by the signal pipeline (open_time, open, high, low, close, volume,
quote_volume).  Missing fields (close_time, trades) are populated as null/zero.

Usage:
    uv run python -m data_service.bybit_fetcher   # download all V10G_SYMBOLS
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from core.constants import V10G_SYMBOLS
from data_service.store import ParquetStore

logger = logging.getLogger(__name__)

BYBIT_BASE = "https://api.bybit.com"
BYBIT_KLINE = "/v5/market/kline"
BYBIT_PER_PAGE = 1000  # max bars per request
BYBIT_REQ_PER_S = 8  # conservative: Bybit allows 10/s (50/5s)
DEFAULT_START_MS = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1000)

# Schema to write — matches what the signal pipeline expects
_KLINE_SCHEMA: dict[str, type] = {
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

SYMBOLS = [f"{s}USDT" for s in V10G_SYMBOLS]
DATA_DIR = Path("data-bybit")


async def _fetch_page(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int | None = None,
) -> pl.DataFrame:
    params: dict[str, str | int] = {
        "category": "linear",
        "symbol": symbol,
        "interval": "60",  # 1h
        "limit": BYBIT_PER_PAGE,
        "start": start_ms,
    }
    if end_ms is not None:
        params["end"] = end_ms

    resp = await client.get(f"{BYBIT_BASE}{BYBIT_KLINE}", params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        msg = data.get("retMsg", "unknown")
        # Empty result is not an error — we just reached end of data
        if msg == "ok":
            return _empty_df()
        raise RuntimeError(f"Bybit API error: {msg}")

    rows = data.get("result", {}).get("list", [])
    if not rows:
        return _empty_df()

    # Bybit returns newest first; reverse
    parsed = []
    for k in reversed(rows):
        parsed.append(
            {
                "open_time": datetime.fromtimestamp(int(k[0]) / 1000, tz=UTC),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": None,  # not provided by Bybit, must match Binance dtype
                "quote_volume": float(k[6]),  # Bybit turnover = quote volume
                "trades": 0,
                "taker_buy_volume": 0.0,
                "taker_buy_quote_volume": 0.0,
            }
        )
    # Force schema to match Binance exactly (polars concat sensitive to dtypes)
    df = pl.DataFrame(parsed)
    if not df.is_empty():
        df = df.with_columns(
            pl.col("close_time").cast(pl.Datetime("us", "UTC")),
            pl.col("trades").cast(pl.Int64),
        )
    return df


async def fetch_all_bybit_klines(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int = DEFAULT_START_MS,
) -> pl.DataFrame:
    """Paginate Bybit API to get all 1h klines for a symbol."""
    frames: list[pl.DataFrame] = []
    cursor = start_ms

    while True:
        df = await _fetch_page(client, symbol, cursor)
        if df.is_empty():
            break

        frames.append(df)
        last_ts = df["open_time"].max()
        if not isinstance(last_ts, datetime):
            break
        cursor = int(last_ts.timestamp() * 1000) + 3600000  # +1h

        if len(df) < BYBIT_PER_PAGE:
            break

        await asyncio.sleep(1.0 / BYBIT_REQ_PER_S)

    if not frames:
        return _empty_df()

    result = pl.concat(frames).unique(subset=["open_time"]).sort("open_time")
    logger.info("Fetched %d bars for %s", len(result), symbol)
    return result


async def download_all(limit: int | None = None) -> None:
    """Download all V10G_SYMBOLS 1h klines to data-bybit/."""
    store = ParquetStore(DATA_DIR)
    symbols_to_fetch = SYMBOLS[:limit] if limit else SYMBOLS

    async with httpx.AsyncClient(timeout=30) as client:
        for i, sym in enumerate(symbols_to_fetch):
            existing = store.read(sym, "1h")
            start = DEFAULT_START_MS
            if not existing.is_empty():
                last = store.latest_timestamp(sym, "1h")
                if last is not None:
                    start = int(last.timestamp() * 1000)
                    logger.info(
                        "%s: %d bars cached, resuming from %s",
                        sym,
                        len(existing),
                        last.strftime("%Y-%m-%d"),
                    )

            print(f"[{i + 1}/{len(symbols_to_fetch)}] {sym} ...")
            try:
                df = await fetch_all_bybit_klines(client, sym, start_ms=start)
                if not df.is_empty():
                    n = store.write(sym, "1h", df)
                    latest = df["open_time"].max()
                    print(f"       {n} bars, latest {latest}")
                else:
                    print("       no data")
            except Exception as e:
                logger.error("%s: %s", sym, e)
                # Continue with next symbol — don't lose progress
                continue


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "open_time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "quote_volume": pl.Float64,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(download_all(limit=limit))
