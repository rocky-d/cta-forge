"""Shared historical data and signal pipeline utilities.

These helpers are used by both the standard v10g backtest and research/live
portfolio profiles. They deliberately stop at bars, indicators, timelines, and
signals; account simulation stays in the caller-specific backtest modules.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import polars as pl
from alpha_service.factors.v10g_composite import (
    V10GCompositeFactor,
    V10GCompositeParams,
    _compute_atr,
)
from core.constants import V10G_SYMBOLS
from data_service.fetcher import fetch_all_klines
from data_service.store import ParquetStore

from .decision import V10GStrategyParams

logger = logging.getLogger(__name__)

# Binance USDS-M Futures launched Sep 2019
DEFAULT_START_TS = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1000)

# Default symbols for backtest/research (Binance USDT-M pairs)
DEFAULT_SYMBOLS = [f"{symbol}USDT" for symbol in V10G_SYMBOLS]

WARMUP_BARS = 150  # symbols need at least this many bars before trading


async def fetch_bars(
    data_dir: str,
    symbols: list[str] | None = None,
    timeframe: str = "6h",
    start_ms: int | None = None,
    min_bars: int = 500,
) -> dict[str, pl.DataFrame]:
    """Load bars from parquet cache, incrementally fetching from Binance."""
    store = ParquetStore(data_dir)
    symbols = symbols or DEFAULT_SYMBOLS
    start_ms = start_ms or DEFAULT_START_TS
    bars: dict[str, pl.DataFrame] = {}

    async with httpx.AsyncClient(timeout=30) as client:
        for symbol in symbols:
            local = store.read(symbol, timeframe)
            if not local.is_empty() and len(local) >= min_bars:
                latest = store.latest_timestamp(symbol, timeframe)
                if latest is not None:
                    # Re-fetch the latest stored open_time so a previously
                    # cached partial candle can be replaced by the closed bar.
                    new_start = int(latest.timestamp() * 1000)
                    new_bars = await fetch_all_klines(
                        client,
                        symbol=symbol,
                        interval=timeframe,
                        start_ms=new_start,
                    )
                    if not new_bars.is_empty():
                        store.write(symbol, timeframe, new_bars)
            else:
                df = await fetch_all_klines(
                    client,
                    symbol=symbol,
                    interval=timeframe,
                    start_ms=start_ms,
                )
                if not df.is_empty():
                    store.write(symbol, timeframe, df)

            df = store.read(symbol, timeframe)
            if not df.is_empty() and len(df) >= min_bars:
                bars[symbol] = df
                logger.info(
                    "%s: %d bars (%s -> %s)",
                    symbol,
                    len(df),
                    df["open_time"][0].strftime("%Y-%m-%d"),
                    df["open_time"][-1].strftime("%Y-%m-%d"),
                )
            else:
                logger.info(
                    "%s: skipped (%d bars)",
                    symbol,
                    len(df) if not df.is_empty() else 0,
                )

    return bars


def precompute(
    bars_dict: dict[str, pl.DataFrame], params: V10GStrategyParams
) -> dict[str, dict]:
    """Precompute indicators per symbol using V10GCompositeFactor."""
    data: dict[str, dict] = {}
    factor_params = V10GCompositeParams(
        timeframe_hours=params.timeframe_hours,
        mom_lookbacks=params.mom_lookbacks,
        adx_threshold=params.adx_threshold,
        adx_ensemble=params.adx_ensemble,
        signal_persistence=params.signal_persistence,
        donchian_period=params.donchian_period,
        rvol_lookback=params.rvol_lookback,
        rvol_median_lookback=params.rvol_median_lookback,
        vol_filter_lookback=params.vol_filter_lookback,
        btc_filter_lookback=params.btc_filter_lookback,
    )
    factor = V10GCompositeFactor(params=factor_params)

    for symbol, df in bars_dict.items():
        indicators: dict[str, Any] = dict(factor.precompute(df))
        indicators["atr"] = _compute_atr(
            indicators["high"], indicators["low"], indicators["close"]
        )
        indicators["start_idx"] = 0
        indicators["length"] = len(indicators["close"])
        data[symbol] = indicators
    return data


def build_timeline(
    bars_dict: dict[str, pl.DataFrame],
) -> tuple[list[datetime], dict[datetime, int]]:
    """Build a unified timeline from all symbols' timestamps."""
    all_ts: set[datetime] = set()
    for df in bars_dict.values():
        all_ts.update(df["open_time"].to_list())
    timeline = sorted(all_ts)
    ts_to_idx = {ts: i for i, ts in enumerate(timeline)}
    return timeline, ts_to_idx


def align_data(
    bars_dict: dict[str, pl.DataFrame],
    data: dict[str, dict],
    ts_to_idx: dict[datetime, int],
) -> None:
    """Map each symbol's data to global timeline indices."""
    for symbol, df in bars_dict.items():
        timestamps = df["open_time"].to_list()
        global_indices = [ts_to_idx[ts] for ts in timestamps]
        data[symbol]["start_idx"] = global_indices[0]
        data[symbol]["global_indices"] = global_indices


def compute_signals(
    data: dict[str, dict],
    timeline: list[datetime],
    params: V10GStrategyParams,
    *,
    btc_filter: bool = True,
) -> dict[str, np.ndarray]:
    """Compute signals on a global timeline using V10GCompositeFactor."""
    n_global = len(timeline)
    factor_params = V10GCompositeParams(
        timeframe_hours=params.timeframe_hours,
        mom_lookbacks=params.mom_lookbacks,
        adx_threshold=params.adx_threshold,
        adx_ensemble=params.adx_ensemble,
        signal_persistence=params.signal_persistence,
        donchian_period=params.donchian_period,
        rvol_lookback=params.rvol_lookback,
        rvol_median_lookback=params.rvol_median_lookback,
        vol_filter_lookback=params.vol_filter_lookback,
        btc_filter_lookback=params.btc_filter_lookback,
    )
    factor = V10GCompositeFactor(params=factor_params)

    btc_ind = None
    if btc_filter and "BTCUSDT" in data:
        btc_ind = data["BTCUSDT"]

    signals = {symbol: np.zeros(n_global) for symbol in data}

    for symbol, symbol_data in data.items():
        start = symbol_data["start_idx"]
        n = symbol_data["length"]
        btc_ref = btc_ind if (btc_filter and symbol != "BTCUSDT") else None

        if btc_ref is not None and btc_ref["start_idx"] != start:
            aligned_btc = align_reference_indicators(
                reference=btc_ref,
                target_start=start,
                target_length=n,
                target_global_indices=symbol_data.get("global_indices"),
            )
            local_signals = factor.compute_signal_array(symbol_data, aligned_btc)
        else:
            local_signals = factor.compute_signal_array(symbol_data, btc_ref)

        global_indices = symbol_data.get("global_indices")
        if global_indices is None:
            global_indices = range(start, start + n)

        for local_idx, global_idx in enumerate(global_indices):
            if global_idx < n_global:
                signals[symbol][global_idx] = local_signals[local_idx]

    return signals


def align_reference_indicators(
    *,
    reference: dict,
    target_start: int,
    target_length: int,
    target_global_indices: list[int] | None = None,
) -> dict:
    """Align a reference symbol's indicator arrays to another symbol window."""
    reference_start = reference["start_idx"]
    reference_length = reference["length"]
    reference_indices = reference.get("global_indices")
    if reference_indices is None:
        reference_indices = range(reference_start, reference_start + reference_length)
    reference_by_global_idx = {
        global_idx: local_idx for local_idx, global_idx in enumerate(reference_indices)
    }
    target_indices = target_global_indices
    if target_indices is None:
        target_indices = range(target_start, target_start + target_length)

    aligned_reference: dict = {}
    for key, arr in reference.items():
        if key in ("start_idx", "length", "global_indices"):
            continue
        aligned = np.zeros(target_length)
        for local_idx, global_idx in enumerate(target_indices):
            reference_local_idx = reference_by_global_idx.get(global_idx)
            if reference_local_idx is not None:
                aligned[local_idx] = arr[reference_local_idx]
        aligned_reference[key] = aligned
    return aligned_reference
