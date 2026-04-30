"""Live market-data cache refresh helpers."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
from data_service.fetcher import fetch_all_klines, fetch_klines
from data_service.store import ParquetStore

logger = logging.getLogger(__name__)


async def fetch_live_bars(
    *,
    store: ParquetStore,
    symbols: list[str],
    interval: str,
    timeframe_hours: int,
    min_bars: int = 200,
) -> dict[str, pl.DataFrame]:
    """Refresh local bar cache and return latest bars by live symbol."""

    async def fetch_symbol(
        client: httpx.AsyncClient, symbol: str
    ) -> tuple[str, pl.DataFrame | None]:
        """Fetch one symbol, returning ``(symbol, df)`` or ``(symbol, None)``."""
        pair = f"{symbol}USDT"
        try:
            latest = store.latest_timestamp(pair, interval)
            cached = store.read(pair, interval)
            cached_bars = len(cached)
            if latest is not None:
                age_hours = (datetime.now(tz=UTC) - latest).total_seconds() / 3600
                need_fetch = age_hours > timeframe_hours or cached_bars < min_bars
            else:
                need_fetch = True

            if need_fetch:
                new_bars = await fetch_needed_bars(
                    client,
                    pair=pair,
                    interval=interval,
                    latest=latest,
                    cached_bars=cached_bars,
                    min_bars=min_bars,
                    timeframe_hours=timeframe_hours,
                )
                if not new_bars.is_empty():
                    store.write(pair, interval, new_bars)
                    logger.info("Stored %d new bars for %s", len(new_bars), pair)

            df = store.read(pair, interval)
            if df.is_empty():
                logger.warning("No data available for %s", pair)
                return symbol, None

            if len(df) > min_bars:
                df = df.tail(min_bars)

            return symbol, df

        except Exception:
            logger.exception("Error fetching %s", pair)
            return symbol, None

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(*(fetch_symbol(client, sym) for sym in symbols))

    return {symbol: df for symbol, df in results if df is not None}


async def fetch_needed_bars(
    client: httpx.AsyncClient,
    *,
    pair: str,
    interval: str,
    latest: datetime | None,
    cached_bars: int,
    min_bars: int,
    timeframe_hours: int,
) -> pl.DataFrame:
    """Fetch the missing cache range for one symbol/interval."""
    # Re-fetch the latest stored open_time so a previously cached partial
    # candle can be replaced by the closed bar.
    if min_bars > 1000 and cached_bars < min_bars:
        # Binance returns at most 1000 klines per request. For fresh or
        # underfilled live caches, paginate from a bounded lookback large enough
        # to satisfy target-strategy warmups.
        lookback_bars = min_bars + max(10, min_bars // 20)
        start = datetime.now(tz=UTC) - timedelta(hours=timeframe_hours * lookback_bars)
        return await fetch_all_klines(
            client,
            symbol=pair,
            interval=interval,
            start_ms=int(start.timestamp() * 1000),
        )

    start_ms = int(latest.timestamp() * 1000) if latest else None
    return await fetch_klines(
        client,
        symbol=pair,
        interval=interval,
        start_ms=start_ms,
        limit=1000,
    )
