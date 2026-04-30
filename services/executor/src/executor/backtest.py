"""Backtest engine using V10GDecisionEngine.

Delegates all trading decisions to the same engine used in live trading,
ensuring exact parity between backtest and live results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from .signal_pipeline import (
    DEFAULT_SYMBOLS as DEFAULT_SYMBOLS,
    WARMUP_BARS,
    align_data,
    build_timeline,
    compute_signals,
    fetch_bars,
    precompute,
)
from .decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    V10GDecisionEngine,
    V10GStrategyParams,
)

logger = logging.getLogger(__name__)


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
