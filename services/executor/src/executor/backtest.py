"""Backtest engine using V10GDecisionEngine.

Delegates all trading decisions to the same engine used in live trading,
ensuring exact parity between backtest and live results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

from .decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    V10GDecisionEngine,
    V10GStrategyParams,
)

logger = logging.getLogger(__name__)

# Binance USDS-M Futures launched Sep 2019
_DEFAULT_START_TS = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1000)

# Default symbols for backtest (derived from constants, Binance USDT-M pairs)
DEFAULT_SYMBOLS = [f"{s}USDT" for s in V10G_SYMBOLS]

WARMUP_BARS = 150  # symbols need at least this many bars before trading


@dataclass
class BacktestResult:
    """Complete backtest result with all data needed for reporting."""

    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    bars: dict[str, pl.DataFrame] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    days: int = 0
    start_date: str = ""
    end_date: str = ""
    initial_equity: float = 10_000.0


# ── Data loading ─────────────────────────────────────────────────


async def fetch_bars(
    data_dir: str,
    symbols: list[str] | None = None,
    timeframe: str = "6h",
    start_ms: int | None = None,
    min_bars: int = 500,
) -> dict[str, pl.DataFrame]:
    """Load bars from parquet cache, incrementally fetch from Binance."""
    store = ParquetStore(data_dir)
    symbols = symbols or DEFAULT_SYMBOLS
    start_ms = start_ms or _DEFAULT_START_TS
    bars: dict[str, pl.DataFrame] = {}

    async with httpx.AsyncClient(timeout=30) as client:
        for sym in symbols:
            # Check local cache
            local = store.read(sym, timeframe)
            if not local.is_empty() and len(local) >= min_bars:
                # Incremental update
                latest = store.latest_timestamp(sym, timeframe)
                if latest is not None:
                    new_start = int(latest.timestamp() * 1000) + 1
                    new_bars = await fetch_all_klines(
                        client, symbol=sym, interval=timeframe, start_ms=new_start
                    )
                    if not new_bars.is_empty():
                        store.write(sym, timeframe, new_bars)
            else:
                # Full fetch
                df = await fetch_all_klines(
                    client, symbol=sym, interval=timeframe, start_ms=start_ms
                )
                if not df.is_empty():
                    store.write(sym, timeframe, df)

            # Read final data
            df = store.read(sym, timeframe)
            if not df.is_empty() and len(df) >= min_bars:
                bars[sym] = df
                logger.info(
                    "%s: %d bars (%s -> %s)",
                    sym,
                    len(df),
                    df["open_time"][0].strftime("%Y-%m-%d"),
                    df["open_time"][-1].strftime("%Y-%m-%d"),
                )
            else:
                logger.info(
                    "%s: skipped (%d bars)", sym, len(df) if not df.is_empty() else 0
                )

    return bars


# ── Signal computation ───────────────────────────────────────────


def precompute(
    bars_dict: dict[str, pl.DataFrame], params: V10GStrategyParams
) -> dict[str, dict]:
    """Precompute indicators per symbol using V10GCompositeFactor."""
    data: dict[str, dict] = {}
    # Extract factor params from strategy params
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

    for sym, df in bars_dict.items():
        ind: dict[str, Any] = dict(factor.precompute(df))
        ind["atr"] = _compute_atr(ind["high"], ind["low"], ind["close"])
        ind["start_idx"] = 0
        ind["length"] = len(ind["close"])
        data[sym] = ind
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
    for sym, df in bars_dict.items():
        timestamps = df["open_time"].to_list()
        data[sym]["start_idx"] = ts_to_idx[timestamps[0]]


def compute_signals(
    data: dict[str, dict],
    timeline: list[datetime],
    params: V10GStrategyParams,
    *,
    btc_filter: bool = True,
) -> dict[str, np.ndarray]:
    """Compute signals on global timeline using V10GCompositeFactor."""
    n_global = len(timeline)

    # Extract factor params from strategy params
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

    sigs = {sym: np.zeros(n_global) for sym in data}

    for sym, d in data.items():
        start = d["start_idx"]
        n = d["length"]

        btc_ref = btc_ind if (btc_filter and sym != "BTCUSDT") else None

        if btc_ref is not None and btc_ref["start_idx"] != start:
            btc_start = btc_ref["start_idx"]
            btc_len = btc_ref["length"]
            aligned_btc: dict = {}
            for key in btc_ref:
                if key in ("start_idx", "length"):
                    continue
                arr = btc_ref[key]
                aligned = np.zeros(n)
                for li in range(n):
                    gi = start + li
                    btc_li = gi - btc_start
                    if 0 <= btc_li < btc_len:
                        aligned[li] = arr[btc_li]
                aligned_btc[key] = aligned
            local_sigs = factor.compute_signal_array(d, aligned_btc)
        else:
            local_sigs = factor.compute_signal_array(d, btc_ref)

        for li in range(n):
            gi = start + li
            if gi < n_global:
                sigs[sym][gi] = local_sigs[li]

    return sigs


# ── Core backtest loop ───────────────────────────────────────────


def run_backtest(
    data: dict[str, dict],
    sigs: dict[str, np.ndarray],
    timeline: list[datetime],
    start_idx: int,
    end_idx: int,
    *,
    initial_equity: float = 10_000.0,
    params: V10GStrategyParams | None = None,
) -> tuple[list[tuple[datetime, float]], list[dict[str, Any]]]:
    """Run backtest using V10GDecisionEngine for all decisions.

    Important: V10GDecisionEngine.tick() mutates state internally -- it deletes
    closed positions and adjusts partial qty. Callers must snapshot positions
    before tick() to settle cash flows correctly.

    Returns:
        (equity_curve, trades) where equity_curve is [(datetime, equity), ...]
    """
    if params is None:
        params = V10GStrategyParams(
            # Backtest: disable hard max_drawdown (use dd_circuit_breaker only)
            max_drawdown=1.0,
        )

    engine = V10GDecisionEngine(params)
    state = EngineState(
        initial_equity=initial_equity,
        peak_equity=initial_equity,
    )

    cash = initial_equity
    curve: list[tuple[datetime, float]] = []
    trades: list[dict[str, Any]] = []
    commission = params.commission

    def get_val(sym: str, gt: int, key: str) -> float | None:
        li = gt - data[sym]["start_idx"]
        if 0 <= li < data[sym]["length"]:
            return float(data[sym][key][li])
        return None

    for t in range(start_idx, end_idx):
        # Build snapshots
        snapshots: dict[str, BarSnapshot] = {}
        for sym in data:
            li = t - data[sym]["start_idx"]
            if li < WARMUP_BARS or li >= data[sym]["length"]:
                continue
            close = get_val(sym, t, "close")
            atr = get_val(sym, t, "atr")
            rvol = get_val(sym, t, "rvol")
            signal = sigs[sym][t]
            if close is None or atr is None:
                continue
            snapshots[sym] = BarSnapshot(
                close=close,
                atr=atr,
                rvol=rvol if rvol is not None else 0.01,
                signal=signal,
            )

        # Mark to market
        equity = cash
        for sym, pos in state.positions.items():
            snap = snapshots.get(sym)
            if snap is None:
                equity += abs(pos.qty) * pos.entry_price
            else:
                equity += abs(pos.qty) * pos.entry_price + pos.qty * (
                    snap.close - pos.entry_price
                )

        # Update recent returns for vol scaling
        if curve:
            prev_eq = curve[-1][1]
            if prev_eq > 0:
                ret = (equity - prev_eq) / prev_eq
                state.recent_returns.append(ret)

        # Snapshot positions BEFORE tick: tick() deletes closed positions
        positions_before = {sym: pos for sym, pos in state.positions.items()}

        # Get actions from decision engine
        actions = engine.tick(state, equity, snapshots)

        # Execute actions
        for action in actions:
            sym = action.symbol
            snap = snapshots.get(sym)

            if action.kind in (ActionKind.CLOSE, ActionKind.FLATTEN_ALL):
                pos = positions_before.get(sym)
                if pos is None:
                    continue
                price = snap.close if snap else pos.entry_price
                pnl = pos.qty * (price - pos.entry_price)
                pnl -= abs(pos.qty) * price * commission
                cash += abs(pos.qty) * pos.entry_price + pnl
                trades.append({"pnl": pnl})

            elif action.kind == ActionKind.PARTIAL_CLOSE:
                pos_before = positions_before.get(sym)
                if pos_before is None:
                    continue
                price = snap.close if snap else pos_before.entry_price
                close_qty = action.qty if pos_before.qty > 0 else -action.qty
                pnl = close_qty * (price - pos_before.entry_price)
                pnl -= abs(close_qty) * price * commission
                cash += abs(close_qty) * pos_before.entry_price + pnl
                trades.append({"pnl": pnl})

            elif action.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT):
                price = snap.close if snap else 0.0
                if price <= 0:
                    continue
                qty = action.qty if action.kind == ActionKind.OPEN_LONG else -action.qty
                cost = abs(qty) * price + abs(qty) * price * commission
                cash -= cost
                state.positions[sym] = PositionState(
                    symbol=sym,
                    qty=qty,
                    entry_price=price,
                    entry_bar=state.bar_count,
                    best_price=price,
                )

        # Record equity after trades
        equity = cash
        for sym, pos in state.positions.items():
            snap = snapshots.get(sym)
            if snap is None:
                equity += abs(pos.qty) * pos.entry_price
            else:
                equity += abs(pos.qty) * pos.entry_price + pos.qty * (
                    snap.close - pos.entry_price
                )
        curve.append((timeline[t], equity))

    # Close remaining positions at end
    t_e = end_idx - 1
    for sym, pos in list(state.positions.items()):
        li = t_e - data[sym]["start_idx"]
        if 0 <= li < data[sym]["length"]:
            price = float(data[sym]["close"][li])
        else:
            price = pos.entry_price
        pnl = pos.qty * (price - pos.entry_price)
        pnl -= abs(pos.qty) * price * commission
        trades.append({"pnl": pnl})

    return curve, trades


def calc_ulcer(curve: list[tuple[datetime, float]]) -> float:
    """Calculate Ulcer Index from equity curve."""
    eq = np.array([e[1] for e in curve])
    if len(eq) < 10:
        return 999.0
    rm = np.maximum.accumulate(eq)
    dd = (rm - eq) / rm
    return float(np.sqrt(np.mean(dd**2)))


# ── High-level orchestration ─────────────────────────────────────


async def run_full_backtest(
    data_dir: str,
    symbols: list[str] | None = None,
    initial_equity: float = 10_000.0,
    warmup: int = 200,
    params: V10GStrategyParams | None = None,
) -> BacktestResult:
    """Run complete backtest pipeline: fetch → precompute → simulate → return.

    This is the single entry point for backtest execution, used by both
    the REST API and scripts.
    """
    params = params or V10GStrategyParams(max_drawdown=1.0)
    tf = params.timeframe_str

    # 1. Fetch data
    bars = await fetch_bars(data_dir, symbols, tf)
    if not bars:
        return BacktestResult()

    # 2. Precompute indicators
    # Use composite params from strategy params if available
    # For now we use global _factor defaults, but this is the hook for profile-based signals
    data = precompute(bars, params)

    timeline, ts_to_idx = build_timeline(bars)
    align_data(bars, data, ts_to_idx)

    # 3. Compute signals
    sigs = compute_signals(data, timeline, params, btc_filter=True)

    # 4. Run backtest
    start = warmup
    end = len(timeline)
    days = (timeline[end - 1] - timeline[start]).days

    curve, trades = run_backtest(
        data,
        sigs,
        timeline,
        start,
        end,
        initial_equity=initial_equity,
        params=params,
    )

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        bars=bars,
        symbols=sorted(bars.keys()),
        days=days,
        start_date=timeline[start].strftime("%Y-%m-%d"),
        end_date=timeline[end - 1].strftime("%Y-%m-%d"),
        initial_equity=initial_equity,
    )
