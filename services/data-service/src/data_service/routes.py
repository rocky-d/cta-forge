"""REST API routes for data-service."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import polars as pl
from fastapi import APIRouter, Query, Request

from . import fetcher

if TYPE_CHECKING:
    from .store import ParquetStore

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_store(request: Request) -> ParquetStore:
    return request.app.state.store


@router.get("/symbols")
async def list_symbols(request: Request) -> dict[str, list[str]]:
    """List symbols that have local data, plus fetch available from Binance."""
    store = _get_store(request)
    local = store.symbols()
    return {"local": local}


@router.get("/symbols/remote")
async def list_remote_symbols() -> dict[str, list[str]]:
    """Fetch all active USDT perpetual symbols from Binance."""
    async with httpx.AsyncClient(timeout=30) as client:
        symbols = await fetcher.fetch_symbols(client)
    return {"symbols": symbols}


@router.get("/bars/{symbol}")
async def get_bars(
    request: Request,
    symbol: str,
    tf: str = Query(default="6h", description="Timeframe interval"),
    start: str | None = Query(default=None, description="Start datetime (ISO 8601)"),
    end: str | None = Query(default=None, description="End datetime (ISO 8601)"),
) -> dict:
    """Get kline bars for a symbol from local store."""
    store = _get_store(request)
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    df = store.read(symbol.upper(), tf, start=start_dt, end=end_dt)

    if df.is_empty():
        return {"symbol": symbol, "interval": tf, "bars": 0, "data": []}

    # Return as list of dicts for JSON serialization
    records = df.with_columns(
        pl.col("open_time").dt.to_string("%Y-%m-%dT%H:%M:%SZ"),
        pl.col("close_time").dt.to_string("%Y-%m-%dT%H:%M:%SZ"),
    ).to_dicts()

    return {"symbol": symbol, "interval": tf, "bars": len(records), "data": records}


@router.post("/sync")
async def sync_data(
    request: Request,
    symbols: list[str] | None = None,
    tf: str = Query(default="6h"),
    start: str | None = Query(default=None),
) -> dict:
    """Sync (download/update) kline data for given symbols.

    If symbols is None, syncs all locally stored symbols.
    """
    store = _get_store(request)

    if symbols is None:
        symbols = store.symbols()
        if not symbols:
            return {
                "synced": 0,
                "message": "No symbols to sync. Provide a symbol list.",
            }

    start_ms = None
    if start:
        start_ms = int(
            datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000
        )

    results = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for symbol in symbols:
            sym = symbol.upper()

            # Resume from last stored bar
            effective_start = start_ms
            if effective_start is None:
                latest = store.latest_timestamp(sym, tf)
                if latest is not None:
                    # Re-fetch the latest stored open_time so an older partial
                    # candle can be overwritten once the closed bar is available.
                    effective_start = int(latest.timestamp() * 1000)

            df = await fetcher.fetch_all_klines(
                client,
                symbol=sym,
                interval=tf,
                start_ms=effective_start,
            )
            total = store.write(sym, tf, df)
            results[sym] = {"new_bars": len(df), "total_bars": total}

    return {"synced": len(results), "results": results}


@router.get("/status")
async def data_status(request: Request, tf: str = Query(default="6h")) -> dict:
    """Get data coverage summary."""
    store = _get_store(request)
    symbols = store.symbols()
    coverage = [store.coverage(s, tf) for s in symbols]
    return {
        "total_symbols": len(symbols),
        "interval": tf,
        "coverage": coverage,
    }
