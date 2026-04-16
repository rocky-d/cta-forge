"""CTA-Forge v10g — Max-range backtest using V10GDecisionEngine.

This version delegates all trading decisions to V10GDecisionEngine,
ensuring exact parity between backtest and live trading logic.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from alpha_service.factors.v10g_composite import (
    V10GCompositeFactor,
    _compute_atr,
)
from core.metrics import calculate_metrics
from data_service.fetcher import fetch_all_klines
from data_service.store import ParquetStore
from executor.decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    V10GDecisionEngine,
    V10GStrategyParams,
)

BINANCE_URL = "https://fapi.binance.com"
OUT_DIR = Path(__file__).resolve().parents[2] / "backtest-results"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# All 19 symbols — each will start from its own earliest available data
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ADAUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "INJUSDT",
    "TIAUSDT",
    "SEIUSDT",
]
TIMEFRAME = "6h"
INITIAL_EQUITY = 10_000.0
COMMISSION = 0.0004

# Binance USDS-M Futures launched Sep 2019
START_TS = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1000)


async def fetch_all():
    """Load bars from local parquet cache, fetch from Binance if missing."""
    store = ParquetStore(DATA_DIR)
    bars = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for sym in SYMBOLS:
            print(f"  {sym}...", end=" ", flush=True)

            # Check local cache
            local = store.read(sym, TIMEFRAME)
            if not local.is_empty() and len(local) >= 500:
                # Incremental update: fetch from last stored bar
                latest = store.latest_timestamp(sym, TIMEFRAME)
                if latest is not None:
                    start_ms = int(latest.timestamp() * 1000) + 1
                    new_bars = await fetch_all_klines(
                        client, symbol=sym, interval=TIMEFRAME, start_ms=start_ms
                    )
                    if not new_bars.is_empty():
                        store.write(sym, TIMEFRAME, new_bars)
                        print(f"+{len(new_bars)} new", end=" ", flush=True)
            else:
                # Full fetch from start
                df = await fetch_all_klines(
                    client, symbol=sym, interval=TIMEFRAME, start_ms=START_TS
                )
                if not df.is_empty():
                    store.write(sym, TIMEFRAME, df)

            # Read final data from store
            df = store.read(sym, TIMEFRAME)
            if not df.is_empty() and len(df) >= 500:
                first = df["open_time"][0].strftime("%Y-%m-%d")
                last = df["open_time"][-1].strftime("%Y-%m-%d")
                bars[sym] = df
                print(f"✓ {len(df)} bars ({first} → {last})", flush=True)
            else:
                n = len(df) if not df.is_empty() else 0
                print(f"✗ skipped ({n} bars)", flush=True)
            await asyncio.sleep(0.15)
    return bars


# Shared factor instance
_v10g_factor = V10GCompositeFactor()


def precompute(bars_dict):
    """Precompute indicators per symbol using V10GCompositeFactor."""
    data = {}
    for sym, df in bars_dict.items():
        ind = _v10g_factor.precompute(df)
        ind["atr"] = _compute_atr(ind["high"], ind["low"], ind["close"])
        ind["start_idx"] = 0
        ind["length"] = len(ind["close"])
        data[sym] = ind
    return data


def compute_signals(data, timeline, params):
    """Compute signals on global timeline using V10GCompositeFactor."""
    n_global = len(timeline)
    use_btc = params.get("btc_filter", False)

    btc_ind = None
    if use_btc and "BTCUSDT" in data:
        btc_ind = data["BTCUSDT"]

    sigs = {sym: np.zeros(n_global) for sym in data}

    for sym, d in data.items():
        start = d["start_idx"]
        n = d["length"]

        btc_ref = btc_ind if (use_btc and sym != "BTCUSDT") else None

        if btc_ref is not None and btc_ref["start_idx"] != start:
            btc_start = btc_ref["start_idx"]
            btc_len = btc_ref["length"]
            aligned_btc = {}
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
            local_sigs = _v10g_factor.compute_signal_array(d, aligned_btc)
        else:
            local_sigs = _v10g_factor.compute_signal_array(d, btc_ref)

        for li in range(n):
            gi = start + li
            if gi < n_global:
                sigs[sym][gi] = local_sigs[li]

    return sigs


def build_timeline(bars_dict):
    """Build a unified timeline from all symbols' timestamps."""
    all_ts = set()
    for df in bars_dict.values():
        all_ts.update(df["open_time"].to_list())
    timeline = sorted(all_ts)
    ts_to_idx = {ts: i for i, ts in enumerate(timeline)}
    return timeline, ts_to_idx


def align_data(bars_dict, data, ts_to_idx):
    """Map each symbol's data to global timeline indices."""
    for sym, df in bars_dict.items():
        timestamps = df["open_time"].to_list()
        global_start = ts_to_idx[timestamps[0]]
        data[sym]["start_idx"] = global_start


def run_backtest(data, sigs, timeline, start_idx, end_idx, params):
    """Run backtest using V10GDecisionEngine for all decisions."""
    # Build strategy params from backtest params dict
    strategy_params = V10GStrategyParams(
        signal_threshold=params["signal_threshold"],
        min_hold_bars=params["min_hold_bars"],
        atr_stop_mult=params["atr_stop_mult"],
        risk_per_trade=params["risk_per_trade"],
        max_positions=params["max_positions"],
        rebalance_every=params["rebalance_every"],
        partial_take_profit=params.get("partial_take_profit", 0),
        target_vol=params.get("target_vol", 0),
        max_hold_bars=params.get("max_hold_bars", 0),
        tighten_stop_after_atr=params.get("tighten_stop_after_atr", 0),
        tightened_stop_mult=params.get("tightened_stop_mult", 3.0),
        dd_circuit_breaker=params.get("dd_circuit_breaker", 0),
        # Backtest: disable hard max_drawdown stop (use dd_circuit_breaker soft stop only)
        # Original backtest never had this feature - set to 1.0 to effectively disable
        max_drawdown=params.get("max_drawdown", 1.0),
        risk_parity=params.get("risk_parity", True),
        signal_reversal_threshold=0.15,
        max_single_position_pct=0.15,
        commission=COMMISSION,
    )

    engine = V10GDecisionEngine(strategy_params)
    state = EngineState(
        initial_equity=INITIAL_EQUITY,
        peak_equity=INITIAL_EQUITY,
    )

    cash = INITIAL_EQUITY
    curve = []
    trades = []

    # Helper: get data at global timeline index
    def get_val(sym, gt, key):
        li = gt - data[sym]["start_idx"]
        if 0 <= li < data[sym]["length"]:
            return data[sym][key][li]
        return None

    # Warmup threshold: symbols need at least 150 bars of history
    WARMUP_BARS = 150

    for t in range(start_idx, end_idx):
        # Build snapshots for all symbols with sufficient data
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
                # Use entry price if no current data
                equity += abs(pos.qty) * pos.entry_price
            else:
                # Position value = notional + unrealized PnL
                equity += abs(pos.qty) * pos.entry_price + pos.qty * (
                    snap.close - pos.entry_price
                )

        # Update recent returns for vol scaling
        if curve:
            prev_eq = curve[-1][1]
            if prev_eq > 0:
                ret = (equity - prev_eq) / prev_eq
                state.recent_returns.append(ret)

        # Snapshot positions BEFORE tick: tick() deletes closed positions from
        # state internally (so opens can reuse the slot), so we need the
        # pre-tick snapshot to settle the trade cash flows.
        positions_before = {sym: pos for sym, pos in state.positions.items()}

        # Get actions from decision engine
        actions = engine.tick(state, equity, snapshots)

        # Execute actions
        for action in actions:
            sym = action.symbol
            snap = snapshots.get(sym)

            if action.kind == ActionKind.CLOSE or action.kind == ActionKind.FLATTEN_ALL:
                # Position was already removed from state by tick(); use snapshot
                pos = positions_before.get(sym)
                if pos is None:
                    continue
                price = snap.close if snap else pos.entry_price
                pnl = pos.qty * (price - pos.entry_price)
                pnl -= abs(pos.qty) * price * COMMISSION
                cash += abs(pos.qty) * pos.entry_price + pnl
                trades.append({"pnl": pnl})

            elif action.kind == ActionKind.PARTIAL_CLOSE:
                # tick() already adjusted pos.qty in state; use state for current qty
                pos_before = positions_before.get(sym)
                if pos_before is None:
                    continue
                price = snap.close if snap else pos_before.entry_price
                # action.qty is the half-qty to close
                close_qty = action.qty if pos_before.qty > 0 else -action.qty
                pnl = close_qty * (price - pos_before.entry_price)
                pnl -= abs(close_qty) * price * COMMISSION
                cash += abs(close_qty) * pos_before.entry_price + pnl
                trades.append({"pnl": pnl})

            elif action.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT):
                price = snap.close if snap else 0.0
                if price <= 0:
                    continue
                qty = action.qty if action.kind == ActionKind.OPEN_LONG else -action.qty
                cost = abs(qty) * price + abs(qty) * price * COMMISSION
                cash -= cost
                state.positions[sym] = PositionState(
                    symbol=sym,
                    qty=qty,
                    entry_price=price,
                    entry_bar=state.bar_count,
                    best_price=price,
                )

        # Record equity (recalculate after trades)
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

        if t % 2000 == 0:
            print(f"    step {t}/{end_idx}: equity=${equity:,.0f}", flush=True)

    # Close remaining positions at end
    t_e = end_idx - 1
    for sym, pos in list(state.positions.items()):
        li = t_e - data[sym]["start_idx"]
        if 0 <= li < data[sym]["length"]:
            price = data[sym]["close"][li]
        else:
            price = pos.entry_price
        pnl = pos.qty * (price - pos.entry_price)
        pnl -= abs(pos.qty) * price * COMMISSION
        trades.append({"pnl": pnl})

    return curve, trades


def calc_ulcer(curve):
    eq = np.array([e[1] for e in curve])
    if len(eq) < 10:
        return 999
    rm = np.maximum.accumulate(eq)
    dd = (rm - eq) / rm
    return np.sqrt(np.mean(dd**2))


async def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print(
        "CTA-Forge v10g — Max-Range Backtest (V10GDecisionEngine)",
        flush=True,
    )
    print("=" * 60, flush=True)

    print("\nFetching from 2019-09-01...", flush=True)
    bars = await fetch_all()
    print(f"\n→ {len(bars)} symbols loaded\n", flush=True)

    # Show data availability
    print("Data availability:", flush=True)
    for sym in sorted(
        bars.keys(),
        key=lambda s: bars[s]["open_time"][0],
    ):
        df = bars[sym]
        first = df["open_time"][0].strftime("%Y-%m-%d")
        last = df["open_time"][-1].strftime("%Y-%m-%d")
        print(
            f"  {sym:12s} {first} → {last}  ({len(df)} bars)",
            flush=True,
        )

    # Build unified timeline
    timeline, ts_to_idx = build_timeline(bars)
    print(
        f"\nUnified timeline: {len(timeline)} bars "
        f"({timeline[0].strftime('%Y-%m-%d')} → "
        f"{timeline[-1].strftime('%Y-%m-%d')})",
        flush=True,
    )

    # Precompute indicators
    data = precompute(bars)
    align_data(bars, data, ts_to_idx)

    params = {
        "mom_lookbacks": [20, 60, 120],
        "adx_threshold": 25,
        "adx_ensemble": [22, 27, 32],
        "signal_threshold": 0.40,
        "min_hold_bars": 16,
        "atr_stop_mult": 5.0,
        "risk_per_trade": 0.015,
        "max_positions": 5,
        "rebalance_every": 4,
        "partial_take_profit": 2.5,
        "target_vol": 0.12,
        "btc_filter": True,
        "max_hold_bars": 100,
        "tighten_stop_after_atr": 2.0,
        "tightened_stop_mult": 3.0,
        "dd_circuit_breaker": 0.08,
        "risk_parity": True,
        "signal_persistence": 2,
        "adaptive_lookback": True,
    }

    print("\nComputing signals...", flush=True)
    sigs = compute_signals(data, timeline, params)

    # Start from index 200 to ensure warmup
    start = 200
    end = len(timeline)
    days = (timeline[end - 1] - timeline[start]).days
    print(
        f"Backtesting: {timeline[start].strftime('%Y-%m-%d')} → "
        f"{timeline[end - 1].strftime('%Y-%m-%d')} ({days} days)\n",
        flush=True,
    )

    curve, trades = run_backtest(data, sigs, timeline, start, end, params)
    m = calculate_metrics(curve, trades)
    ulcer = calc_ulcer(curve)

    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULTS ({days} days, {len(bars)} symbols)", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(
        f"  Return: {m.total_return * 100:+.1f}%  "
        f"Ann: {m.annualized_return * 100:+.1f}%",
        flush=True,
    )
    print(
        f"  Sharpe: {m.sharpe_ratio:.2f}  Sortino: {m.sortino_ratio:.2f}",
        flush=True,
    )
    print(
        f"  MaxDD:  {m.max_drawdown * 100:.1f}%  Calmar: {m.calmar_ratio:.2f}",
        flush=True,
    )
    print(
        f"  PF: {m.profit_factor:.2f}  "
        f"Win: {m.win_rate * 100:.1f}%  "
        f"Trades: {m.num_trades}",
        flush=True,
    )
    print(f"  Ulcer: {ulcer:.4f}", flush=True)

    # Yearly breakdown
    yearly = {}
    for ts, eq in curve:
        yr = ts.year
        if yr not in yearly:
            yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq

    print("\nYearly returns:", flush=True)
    for yr in sorted(yearly):
        yr_ret = (yearly[yr]["last"] - yearly[yr]["first"]) / yearly[yr]["first"] * 100
        print(f"  {yr}: {yr_ret:+.1f}%", flush=True)

    # === Charts (3-panel: equity + BTC/ETH overlay, drawdown, monthly) ===
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_b = [e[0] for e in curve]
    eq_b = [e[1] for e in curve]
    curve_start_ts = ts_b[0]

    # BTC / ETH price series for overlay (from bars already in memory)
    def _price_series(sym: str) -> list[tuple]:
        df = bars.get(sym)
        if df is None or df.is_empty():
            return []
        return [
            (t, p)
            for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
            if t >= curve_start_ts
        ]

    btc_prices = _price_series("BTCUSDT")
    eth_prices = _price_series("ETHUSDT")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(24, 12),
        height_ratios=[3.5, 1, 1.5],
        gridspec_kw={"hspace": 0.15},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, hspace=0.15)

    # === Top panel: Equity (left Y) + BTC/ETH indexed price (right Y) ===
    ax = axes[0]
    ax2 = ax.twinx()

    btc_base = btc_prices[0][1] if btc_prices else 1
    eth_base = eth_prices[0][1] if eth_prices else 1
    ln_btc = ax2.plot(
        [p[0] for p in btc_prices],
        [p[1] / btc_base * 100 for p in btc_prices],
        linewidth=0.6,
        color="#F7931A",
        alpha=0.4,
        label="BTC (indexed)",
    )
    ln_eth = ax2.plot(
        [p[0] for p in eth_prices],
        [p[1] / eth_base * 100 for p in eth_prices],
        linewidth=0.6,
        color="#627EEA",
        alpha=0.4,
        label="ETH (indexed)",
    )
    ax2.set_ylabel("BTC / ETH Index (start = 100)", fontsize=9, color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888", labelsize=8)

    ln_eq = ax.plot(
        ts_b,
        eq_b,
        linewidth=1.8,
        color="#2ecc71",
        label="CTA-Forge v10g",
        zorder=10,
    )
    ax.axhline(
        y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.4
    )
    ax.fill_between(
        ts_b,
        INITIAL_EQUITY,
        eq_b,
        where=[e >= INITIAL_EQUITY for e in eq_b],
        alpha=0.1,
        color="#2ecc71",
        zorder=5,
    )
    ax.fill_between(
        ts_b,
        INITIAL_EQUITY,
        eq_b,
        where=[e < INITIAL_EQUITY for e in eq_b],
        alpha=0.1,
        color="#e74c3c",
        zorder=5,
    )

    # Keep equity curve on top of BTC/ETH lines
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_title(
        f"CTA-Forge v10g — Max-Range Backtest (V10GDecisionEngine)\n"
        f"{len(bars)} symbols · 6h · $10K start · {days} days · vs BTC & ETH buy-and-hold",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"${x:,.0f}"))

    lns = ln_eq + ln_btc + ln_eth
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    # Bottom xlabel: metrics summary
    yr_str = "  ".join(
        [
            f"{yr}: {(yearly[yr]['last'] - yearly[yr]['first']) / yearly[yr]['first'] * 100:+.1f}%"
            for yr in sorted(yearly)
        ]
    )
    btc_ret = (btc_prices[-1][1] / btc_prices[0][1] - 1) * 100 if btc_prices else 0
    eth_ret = (eth_prices[-1][1] / eth_prices[0][1] - 1) * 100 if eth_prices else 0
    line1 = (
        f"Return: {m.total_return * 100:+.1f}%   Ann: {m.annualized_return * 100:+.1f}%   "
        f"Sharpe: {m.sharpe_ratio:.2f}   Sortino: {m.sortino_ratio:.2f}   "
        f"MaxDD: {m.max_drawdown * 100:.1f}%   Calmar: {m.calmar_ratio:.2f}   "
        f"PF: {m.profit_factor:.2f}   Win: {m.win_rate * 100:.1f}%   "
        f"Trades: {m.num_trades}   Ulcer: {ulcer:.4f}   |   "
        f"BTC B&H: {btc_ret:+.0f}%   ETH B&H: {eth_ret:+.0f}%\n{yr_str}"
    )
    axes[2].set_xlabel(
        line1, fontsize=7.5, family="monospace", color="#2c3e50", labelpad=6
    )

    # === Middle: Drawdown ===
    eq_arr = np.array(eq_b)
    rm = np.maximum.accumulate(eq_arr)
    dd_pct = (rm - eq_arr) / rm * 100
    axes[1].fill_between(ts_b, 0, -dd_pct, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.15)

    # === Bottom: Monthly returns ===
    monthly = {}
    for ii in range(1, len(eq_b)):
        key = ts_b[ii].strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"start": eq_b[ii - 1]}
        monthly[key]["end"] = eq_b[ii]
    mos = list(monthly.keys())
    rets = [
        (monthly[m_]["end"] - monthly[m_]["start"]) / monthly[m_]["start"] * 100
        for m_ in mos
    ]
    axes[2].bar(
        range(len(mos)),
        rets,
        color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rets],
        alpha=0.8,
        width=0.8,
    )
    step = max(1, len(mos) // 24)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels(
        [mos[ii] for ii in range(0, len(mos), step)],
        rotation=45,
        ha="right",
        fontsize=6,
    )
    axes[2].set_ylabel("Monthly %")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.15, axis="y")

    pos_months = sum(1 for r in rets if r > 0)
    axes[2].text(
        0.02,
        0.95,
        f"Positive: {pos_months}/{len(rets)} months ({pos_months / len(rets) * 100:.0f}%)",
        transform=axes[2].transAxes,
        fontsize=8,
        va="top",
    )

    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    fig.savefig(
        OUT_DIR / "backtest_v10g_engine.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # Save metrics
    (OUT_DIR / "metrics_v10g_engine.json").write_text(
        json.dumps(
            {
                "period": (
                    f"{timeline[start].strftime('%Y-%m-%d')} → "
                    f"{timeline[end - 1].strftime('%Y-%m-%d')}"
                ),
                "days": days,
                "symbols": len(bars),
                "symbol_list": sorted(bars.keys()),
                "sharpe": m.sharpe_ratio,
                "sortino": m.sortino_ratio,
                "return": m.total_return,
                "ann_return": m.annualized_return,
                "max_dd": m.max_drawdown,
                "calmar": m.calmar_ratio,
                "pf": m.profit_factor,
                "win_rate": m.win_rate,
                "trades": m.num_trades,
                "ulcer": ulcer,
                "yearly": {
                    str(yr): (yearly[yr]["last"] - yearly[yr]["first"])
                    / yearly[yr]["first"]
                    * 100
                    for yr in sorted(yearly)
                },
                "engine": "V10GDecisionEngine",
            },
            indent=2,
        )
    )

    print(f"\n✅ Done in {time.time() - t0:.0f}s", flush=True)
    print(f"Charts: {OUT_DIR / 'backtest_v10g_engine.png'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
