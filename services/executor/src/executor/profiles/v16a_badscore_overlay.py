"""v16a Badscore Overlay portfolio profile.

This module contains the reusable target-weight implementation behind the
research script. It deliberately emits portfolio target weights rather than
exchange orders so backtest and live execution can share the same strategy
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from data_service.store import ParquetStore
from executor.backtest import (
    DEFAULT_SYMBOLS,
    align_data,
    build_timeline,
    compute_signals,
    precompute,
)
from executor.decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    V10GDecisionEngine,
    V10GStrategyParams,
)
from executor.portfolio_backtest import calculate_hourly_metrics
from executor.targeting import PortfolioTarget, StrategyProfile, normalize_gross

INITIAL_EQUITY = 10_000.0
EPS = 1e-12

V16A_PROFILE = StrategyProfile(
    slug="v16a-badscore-overlay",
    name="v16a Badscore Overlay",
    description=(
        "50% shifted v10g-engine-6h core + 50% 1h fast-exit top2 overlay, "
        "scaled by fixed badscore2_050 regime gate."
    ),
    timeframe_hours=1,
)


@dataclass(frozen=True)
class V16aTargetSet:
    """Target-weight matrix and supporting market data for v16a."""

    timeline: list[datetime]
    symbols: list[str]
    returns: np.ndarray
    target_weights: np.ndarray
    v10g_weights: np.ndarray
    overlay_weights: np.ndarray
    gate: np.ndarray


DATA_DIR = Path("data")


class V16aHistoricalStrategy:
    """Historical target provider backed by a precomputed v16a target matrix."""

    profile = V16A_PROFILE

    def __init__(self, target_set: V16aTargetSet) -> None:
        self._target_set = target_set
        self._index = {ts: i for i, ts in enumerate(target_set.timeline)}

    def target(self, timestamp: datetime) -> PortfolioTarget:
        idx = self._index.get(timestamp)
        if idx is None:
            raise KeyError(f"No v16a target for timestamp: {timestamp!r}")
        return PortfolioTarget(
            timestamp=timestamp,
            weights={
                symbol: float(self._target_set.target_weights[idx, i])
                for i, symbol in enumerate(self._target_set.symbols)
            },
            gross_cap=1.0,
        ).capped()


def v10g_params() -> V10GStrategyParams:
    return V10GStrategyParams(
        timeframe_str="6h",
        timeframe_hours=6,
        signal_threshold=0.40,
        min_hold_bars=16,
        max_hold_bars=100,
        max_drawdown=1.0,
    )


def overlay_params() -> V10GStrategyParams:
    return V10GStrategyParams(
        timeframe_str="1h",
        timeframe_hours=1,
        signal_threshold=0.55,
        min_hold_bars=8,
        max_hold_bars=48,
        rebalance_every=4,
        atr_stop_mult=4.0,
        risk_per_trade=0.010,
        max_positions=2,
        partial_take_profit=2.6,
        target_vol=0.08,
        tighten_stop_after_atr=1.6,
        tightened_stop_mult=2.3,
        signal_reversal_threshold=0.22,
        max_drawdown=1.0,
        commission=0.0004,
        mom_lookbacks=[24, 72, 168],
        adx_ensemble=[20, 28, 36],
        adx_threshold=20.0,
        signal_persistence=3,
        donchian_period=48,
        rvol_lookback=24,
        rvol_median_lookback=168,
        vol_filter_lookback=24,
        btc_filter_lookback=72,
    )


def load_bars(tf: str, min_bars: int = 500) -> dict[str, pl.DataFrame]:
    store = ParquetStore(DATA_DIR)
    bars = {}
    for sym in DEFAULT_SYMBOLS:
        df = store.read(sym, tf)
        if not df.is_empty() and len(df) >= min_bars:
            bars[sym] = df
    return bars


def rolling_mean_prev(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x)
    vals = np.where(valid, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cc = np.concatenate([[0], np.cumsum(valid.astype(int))])
    out = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        n = cc[i] - cc[i - window]
        if n:
            out[i] = (cs[i] - cs[i - window]) / n
    return out


def rolling_std_prev(x: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean_prev(x, window)
    mean2 = rolling_mean_prev(x * x, window)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def zret(close: np.ndarray, ret1: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[lookback:] = close[lookback:] / close[:-lookback] - 1
    vol = rolling_std_prev(ret1, max(24, lookback)) * np.sqrt(lookback)
    return out / (vol + EPS)


def local_overlay_signal(df, data_sym) -> np.ndarray:
    """Build a native 1h trend signal with micro pullback risk filtering.

    The signal is CTA-like: medium-horizon trend gives direction; 1h micro
    reversal / VWAP / close-location features are used to avoid chasing crowded
    spikes, not as standalone mean reversion alpha.
    """
    close = np.asarray(data_sym["close"], dtype=float)
    high = np.asarray(data_sym["high"], dtype=float)
    low = np.asarray(data_sym["low"], dtype=float)
    volume = np.asarray(data_sym["volume"], dtype=float)
    open_ = np.asarray(df["open"].to_numpy(), dtype=float)[: len(close)]
    quote = np.asarray(df["quote_volume"].to_numpy(), dtype=float)[: len(close)]

    ret1 = np.zeros(len(close))
    ret1[1:] = close[1:] / close[:-1] - 1
    intrabar = np.where(open_ > 0, close / open_ - 1, ret1)
    rv24 = rolling_std_prev(ret1, 24)
    vol_mean = rolling_mean_prev(quote, 168)
    vol_ratio = quote / (vol_mean + EPS)
    vwap = np.where(volume > 0, quote / np.maximum(volume, EPS), close)
    rng = np.maximum(high - low, EPS)
    clv = (2 * close - high - low) / rng

    trend = (
        0.42 * zret(close, ret1, 24)
        + 0.35 * zret(close, ret1, 72)
        + 0.23 * zret(close, ret1, 168)
    )
    trend_dir = np.sign(trend)
    trend_strength = np.clip(np.abs(trend) / 2.4, 0, 1)

    vwap_dev = (close - vwap) / (np.asarray(data_sym["atr"], dtype=float) + EPS)
    micro = (
        0.45 * (-clv) * np.sqrt(np.clip(vol_ratio, 0, 3))
        + 0.30 * (-(intrabar / (rv24 + EPS)))
        + 0.25 * (-vwap_dev)
    )
    micro = np.tanh(micro / 2.0)

    no_jump = np.abs(intrabar) < 2.8 * np.nan_to_num(rv24, nan=np.inf)
    same_side_spike = (np.sign(intrabar) == trend_dir) & (
        np.abs(intrabar) > 1.5 * np.nan_to_num(rv24, nan=np.inf)
    )
    bad_micro = (np.sign(micro) == -trend_dir) & (np.abs(micro) > 0.35)

    # Soft hour-of-day risk adjustment from the research harness. It is a
    # deterministic UTC-hour prior, not fitted per split.
    favorable_hours = {6, 13, 14, 17, 20}
    avoid_hours = {15, 16}
    hours = np.array([ts.hour for ts in df["open_time"].to_list()[: len(close)]])
    soft_hour = np.array([1.0 if h in favorable_hours else 0.75 for h in hours])
    soft_hour = np.where(np.isin(hours, list(avoid_hours)), 0.55, soft_hour)

    active = (np.abs(trend) > 0.80) & ~bad_micro & ~same_side_spike & no_jump
    return np.where(active, trend_dir * trend_strength * soft_hour, 0.0)


def map_local_to_global(
    local: np.ndarray, start: int, n_global: int, shift: int = 1
) -> np.ndarray:
    out = np.zeros(n_global)
    start_i = start + shift
    if start_i >= n_global:
        return out
    end_i = min(n_global, start_i + len(local) - shift)
    out[start_i:end_i] = np.nan_to_num(local[: end_i - start_i], nan=0.0)
    return np.clip(out, -1.0, 1.0)


def top_n_signals(signals: dict[str, np.ndarray], top_n: int) -> dict[str, np.ndarray]:
    syms = list(signals)
    n = len(next(iter(signals.values())))
    out = {sym: np.zeros(n) for sym in syms}
    for t in range(n):
        candidates = [
            (sym, signals[sym][t])
            for sym in syms
            if np.isfinite(signals[sym][t]) and signals[sym][t] != 0
        ]
        for sym, sig in sorted(candidates, key=lambda x: abs(x[1]), reverse=True)[
            :top_n
        ]:
            out[sym][t] = sig
    return out


def build_overlay_signals(bars, data, timeline) -> dict[str, np.ndarray]:
    signals = {}
    for sym, data_sym in data.items():
        local = local_overlay_signal(bars[sym], data_sym)
        signals[sym] = map_local_to_global(
            local, int(data_sym["start_idx"]), len(timeline), shift=1
        )
    return top_n_signals(signals, top_n=2)


def get_val(data, sym: str, t: int, key: str) -> float | None:
    li = t - int(data[sym]["start_idx"])
    if li < 0 or li >= int(data[sym]["length"]):
        return None
    val = float(data[sym][key][li])
    return val if np.isfinite(val) else None


def run_engine_positions(data, sigs, timeline, warmup: int, params: V10GStrategyParams):
    engine = V10GDecisionEngine(params)
    state = EngineState(initial_equity=INITIAL_EQUITY, peak_equity=INITIAL_EQUITY)
    cash = INITIAL_EQUITY
    syms = list(data)
    weights = np.zeros((len(timeline), len(syms)))
    curve = []
    trades = []

    for t in range(warmup, len(timeline)):
        snapshots = {}
        for sym in data:
            li = t - int(data[sym]["start_idx"])
            if li < warmup or li >= int(data[sym]["length"]):
                continue
            close = get_val(data, sym, t, "close")
            atr = get_val(data, sym, t, "atr")
            rvol = get_val(data, sym, t, "rvol")
            if close is None or atr is None:
                continue
            snapshots[sym] = BarSnapshot(
                close=close,
                atr=atr,
                rvol=rvol if rvol is not None else 0.01,
                signal=float(sigs.get(sym, np.zeros(len(timeline)))[t]),
            )

        equity = cash
        for sym, pos in state.positions.items():
            snap = snapshots.get(sym)
            if snap is None:
                equity += abs(pos.qty) * pos.entry_price
            else:
                equity += abs(pos.qty) * pos.entry_price + pos.qty * (
                    snap.close - pos.entry_price
                )

        if curve:
            prev_eq = curve[-1][1]
            if prev_eq > 0:
                state.recent_returns.append((equity - prev_eq) / prev_eq)

        positions_before = {sym: pos for sym, pos in state.positions.items()}
        actions = engine.tick(state, equity, snapshots)

        for action in actions:
            sym = action.symbol
            snap = snapshots.get(sym)
            if action.kind in (ActionKind.CLOSE, ActionKind.FLATTEN_ALL):
                pos = positions_before.get(sym)
                if pos is None:
                    continue
                price = snap.close if snap else pos.entry_price
                pnl = (
                    pos.qty * (price - pos.entry_price)
                    - abs(pos.qty) * price * params.commission
                )
                cash += abs(pos.qty) * pos.entry_price + pnl
                trades.append({"pnl": pnl})
            elif action.kind == ActionKind.PARTIAL_CLOSE:
                pos = positions_before.get(sym)
                if pos is None:
                    continue
                price = snap.close if snap else pos.entry_price
                close_qty = action.qty if pos.qty > 0 else -action.qty
                pnl = (
                    close_qty * (price - pos.entry_price)
                    - abs(close_qty) * price * params.commission
                )
                cash += abs(close_qty) * pos.entry_price + pnl
                trades.append({"pnl": pnl})
            elif action.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT):
                price = snap.close if snap else 0.0
                if price <= 0:
                    continue
                qty = action.qty if action.kind == ActionKind.OPEN_LONG else -action.qty
                cash -= abs(qty) * price * (1.0 + params.commission)
                state.positions[sym] = PositionState(
                    sym, qty, price, state.bar_count, price
                )

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

        if equity > 0:
            for j, sym in enumerate(syms):
                pos = state.positions.get(sym)
                snap = snapshots.get(sym)
                if pos is not None and snap is not None:
                    weights[t, j] = pos.qty * snap.close / equity

    return syms, weights, curve, trades


def build_v10g_sleeve():
    params = v10g_params()
    bars = load_bars("6h")
    data = precompute(bars, params)
    timeline, ts_to_idx = build_timeline(bars)
    align_data(bars, data, ts_to_idx)
    sigs = compute_signals(data, timeline, params, btc_filter=True)
    shifted = {
        sym: np.concatenate([[0.0], np.asarray(sig[:-1], dtype=float)])
        for sym, sig in sigs.items()
    }
    return (*run_engine_positions(data, shifted, timeline, 200, params), timeline)


def build_overlay_sleeve():
    params = overlay_params()
    bars = load_bars("1h", min_bars=5_000)
    data = precompute(bars, params)
    timeline, ts_to_idx = build_timeline(bars)
    align_data(bars, data, ts_to_idx)
    sigs = build_overlay_signals(bars, data, timeline)
    return (*run_engine_positions(data, sigs, timeline, 24 * 45, params), timeline)


def load_hourly_returns(timeline, syms):
    store = ParquetStore(DATA_DIR)
    idx = {ts: i for i, ts in enumerate(timeline)}
    close = np.full((len(timeline), len(syms)), np.nan)
    for j, sym in enumerate(syms):
        df = store.read(sym, "1h")
        for row in df.iter_rows(named=True):
            i = idx.get(row["open_time"])
            if i is not None:
                close[i, j] = float(row["close"])
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    prev = np.vstack([np.full(close.shape[1], np.nan), close[:-1]])
    ret[~np.isfinite(close) | ~np.isfinite(prev)] = np.nan
    return ret


def align_weights(
    src_timeline, src_syms, src_weights, dst_timeline, dst_syms, *, forward_fill: bool
) -> np.ndarray:
    out = np.zeros((len(dst_timeline), len(dst_syms)))
    src_sym_idx = {sym: i for i, sym in enumerate(src_syms)}
    if forward_fill:
        j = 0
        cur = np.zeros(len(src_syms))
        for i, ts in enumerate(dst_timeline):
            while j < len(src_timeline) and src_timeline[j] <= ts:
                cur = src_weights[j]
                j += 1
            for k, sym in enumerate(dst_syms):
                si = src_sym_idx.get(sym)
                if si is not None:
                    out[i, k] = cur[si]
    else:
        src_ts_idx = {ts: i for i, ts in enumerate(src_timeline)}
        for i, ts in enumerate(dst_timeline):
            j = src_ts_idx.get(ts)
            if j is None:
                continue
            for k, sym in enumerate(dst_syms):
                si = src_sym_idx.get(sym)
                if si is not None:
                    out[i, k] = src_weights[j, si]
    return out


def daily_symbol_returns(days):
    store = ParquetStore(DATA_DIR)
    day_idx = {day: i for i, day in enumerate(days)}
    close = np.full((len(days), len(DEFAULT_SYMBOLS)), np.nan)
    for j, sym in enumerate(DEFAULT_SYMBOLS):
        df = store.read(sym, "1h")
        last = {}
        for row in df.iter_rows(named=True):
            day = row["open_time"].date().isoformat()
            if day in day_idx:
                last[day] = float(row["close"])
        for day, px in last.items():
            close[day_idx[day], j] = px
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    prev = np.vstack([np.full(close.shape[1], np.nan), close[:-1]])
    ret[~np.isfinite(close) | ~np.isfinite(prev)] = np.nan
    return ret


def expanding_badscore_gate(timeline) -> np.ndarray:
    days = sorted({ts.date().isoformat() for ts in timeline})
    sym_ret = daily_symbol_returns(days)
    valid_counts = np.sum(np.isfinite(sym_ret), axis=1)
    market = np.divide(
        np.nansum(sym_ret, axis=1),
        valid_counts,
        out=np.full(sym_ret.shape[0], np.nan),
        where=valid_counts > 0,
    )
    vol20 = np.full(len(days), np.nan)
    eff60 = np.full(len(days), np.nan)
    corr60 = np.full(len(days), np.nan)

    for i in range(len(days)):
        if i >= 20:
            vol20[i] = np.nanstd(market[i - 20 : i]) * np.sqrt(365)
        if i >= 60:
            win = market[i - 60 : i]
            valid = win[np.isfinite(win)]
            if len(valid) >= 48:
                eff60[i] = abs(np.prod(1 + valid) - 1) / max(np.sum(np.abs(valid)), EPS)
            cols = []
            for j in range(sym_ret.shape[1]):
                col = sym_ret[i - 60 : i, j]
                if np.isfinite(col).sum() >= 45:
                    cols.append(np.nan_to_num(col, nan=np.nanmean(col)))
            if len(cols) >= 8:
                corr = np.corrcoef(np.vstack(cols))
                iu = np.triu_indices_from(corr, k=1)
                corr60[i] = np.nanmean(corr[iu])

    scale_by_day = {}
    hist_vol, hist_eff, hist_corr = [], [], []
    for day, vol, eff, corr in zip(days, vol20, eff60, corr60):
        bad = 0
        if len(hist_vol) >= 252 and np.isfinite(vol) and vol > np.nanmedian(hist_vol):
            bad += 1
        if len(hist_eff) >= 252 and np.isfinite(eff) and eff < np.nanmedian(hist_eff):
            bad += 1
        if (
            len(hist_corr) >= 252
            and np.isfinite(corr)
            and corr > np.nanquantile(hist_corr, 0.66)
        ):
            bad += 1
        scale_by_day[day] = 0.5 if bad >= 2 else 1.0
        if np.isfinite(vol):
            hist_vol.append(float(vol))
        if np.isfinite(eff):
            hist_eff.append(float(eff))
        if np.isfinite(corr):
            hist_corr.append(float(corr))

    return np.array(
        [scale_by_day[ts.date().isoformat()] for ts in timeline], dtype=float
    )


def split_metrics(timeline, pnl):
    days = np.array([ts.date().isoformat() for ts in timeline])
    out = {}
    for name, lo, hi in [
        ("2020_2021", "2020-01-01", "2021-12-31"),
        ("2022_2023", "2022-01-01", "2023-12-31"),
        ("2024_2026", "2024-01-01", "2026-12-31"),
    ]:
        mask = (days >= lo) & (days <= hi)
        m = calculate_hourly_metrics(pnl[mask], initial_equity=INITIAL_EQUITY)
        out[name] = {k: v for k, v in m.items() if k not in {"equity", "drawdown"}}
    return out


def build_v16a_target_set(
    data_dir: str | Path,
    *,
    v10g_allocation: float = 0.5,
    overlay_allocation: float = 0.5,
    gross_cap: float = 1.0,
) -> V16aTargetSet:
    """Build the full historical v16a target-weight matrix from local data."""
    global DATA_DIR
    DATA_DIR = Path(data_dir)

    v_syms, v_weights, v_curve, _v_trades, v_timeline = build_v10g_sleeve()
    o_syms, o_weights, o_curve, _o_trades, o_timeline = build_overlay_sleeve()

    start = max(v_curve[0][0], o_curve[0][0])
    end = min(v_curve[-1][0], o_curve[-1][0])
    timeline = [ts for ts in o_timeline if start <= ts <= end]
    symbols = sorted(set(v_syms) | set(o_syms))

    v_aligned = align_weights(
        v_timeline, v_syms, v_weights, timeline, symbols, forward_fill=True
    )
    o_aligned = align_weights(
        o_timeline, o_syms, o_weights, timeline, symbols, forward_fill=False
    )
    returns = load_hourly_returns(timeline, symbols)
    gate = expanding_badscore_gate(timeline)

    target = v10g_allocation * v_aligned + overlay_allocation * o_aligned
    target *= gate[:, None]
    for i in range(target.shape[0]):
        capped = normalize_gross(
            {symbol: target[i, j] for j, symbol in enumerate(symbols)},
            gross_cap=gross_cap,
        )
        target[i] = np.array([capped.get(symbol, 0.0) for symbol in symbols])

    return V16aTargetSet(
        timeline=timeline,
        symbols=symbols,
        returns=returns,
        target_weights=target,
        v10g_weights=v_aligned,
        overlay_weights=o_aligned,
        gate=gate,
    )
