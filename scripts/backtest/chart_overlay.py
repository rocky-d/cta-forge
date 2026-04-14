"""Re-chart v10g with BTC & ETH price overlay. Runs full backtest to get curve.

Usage: uv run python scripts/backtest/chart_overlay.py
"""
from __future__ import annotations
import asyncio, json, time
from datetime import UTC, datetime
from pathlib import Path
import httpx, numpy as np, polars as pl

from reporter_service.metrics import calculate_metrics

from v10g_maxrange import (
    SYMBOLS, TIMEFRAME, INITIAL_EQUITY, COMMISSION, START_TS,
    BINANCE_URL, fetch_klines_from, precompute, build_timeline,
    align_data, compute_signals, run_backtest, calc_ulcer
)

OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")


async def fetch_price_series(client, symbol):
    """Fetch full price series for overlay."""
    all_rows = []
    start_time = START_TS
    while len(all_rows) < 20000:
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 1500, "startTime": start_time}
        try:
            resp = await client.get(f"{BINANCE_URL}/fapi/v1/klines", params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5); continue
            if resp.status_code != 200: break
            raw = resp.json()
            if not raw: break
            for k in raw:
                all_rows.append((datetime.fromtimestamp(k[0]/1000, tz=UTC), float(k[4])))
            if len(raw) < 1500: break
            start_time = raw[-1][0] + 1
            await asyncio.sleep(0.1)
        except:
            break
    return all_rows


async def main():
    t0 = time.time()
    print("Fetching klines for all symbols + BTC/ETH overlay...", flush=True)

    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch bars for backtest
        bars = {}
        for sym in SYMBOLS:
            print(f"  {sym}...", end=" ", flush=True)
            df = await fetch_klines_from(client, sym, START_TS)
            if df is not None and len(df) >= 500:
                bars[sym] = df
                print(f"✓ {len(df)}", flush=True)
            else:
                print("✗", flush=True)
            await asyncio.sleep(0.15)

        # Fetch BTC and ETH price for overlay
        print("  BTC overlay...", flush=True)
        btc_prices = await fetch_price_series(client, "BTCUSDT")
        print("  ETH overlay...", flush=True)
        eth_prices = await fetch_price_series(client, "ETHUSDT")

    print(f"\n{len(bars)} symbols loaded", flush=True)

    # Run backtest
    timeline, ts_to_idx = build_timeline(bars)
    data = precompute(bars)
    align_data(bars, data, ts_to_idx)

    params = {
        "mom_lookbacks": [20, 60, 120],
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

    print("Computing signals...", flush=True)
    sigs = compute_signals(data, timeline, params)
    start, end = 200, len(timeline)
    print("Running backtest...", flush=True)
    curve, trades = run_backtest(data, sigs, timeline, start, end, params)
    m = calculate_metrics(curve, trades)
    ulcer = calc_ulcer(curve)
    days = (timeline[end-1] - timeline[start]).days

    yearly = {}
    for ts, eq in curve:
        yr = ts.year
        if yr not in yearly: yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq

    # === CHART ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ts_eq = [e[0] for e in curve]
    eq_vals = [e[1] for e in curve]

    # Normalize BTC and ETH to start at same value as equity at curve start
    curve_start_ts = ts_eq[0]

    # Find BTC/ETH price at curve start
    btc_ts = [p[0] for p in btc_prices]
    btc_px = [p[1] for p in btc_prices]
    eth_ts = [p[0] for p in eth_prices]
    eth_px = [p[1] for p in eth_prices]

    # Filter to only overlap with equity curve timerange
    btc_filt = [(t, p) for t, p in btc_prices if t >= curve_start_ts]
    eth_filt = [(t, p) for t, p in eth_prices if t >= curve_start_ts]

    # Create chart — phone landscape with notch margin (~2:1)
    fig, axes = plt.subplots(3, 1, figsize=(24, 12), height_ratios=[3.5, 1, 1.5],
                              gridspec_kw={"hspace": 0.15})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, hspace=0.15)

    # === Top panel: Equity (left Y) + BTC/ETH price (right Y) ===
    ax = axes[0]
    ax2 = ax.twinx()

    # Right axis: BTC and ETH normalized returns (base=100)
    btc_base = btc_filt[0][1]
    eth_base = eth_filt[0][1]
    ln_btc = ax2.plot([p[0] for p in btc_filt], [p[1] / btc_base * 100 for p in btc_filt],
                       linewidth=0.6, color="#F7931A", alpha=0.4, label="BTC (indexed)")
    ln_eth = ax2.plot([p[0] for p in eth_filt], [p[1] / eth_base * 100 for p in eth_filt],
                       linewidth=0.6, color="#627EEA", alpha=0.4, label="ETH (indexed)")
    ax2.set_ylabel("BTC / ETH Index (start = 100)", fontsize=9, color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888", labelsize=8)

    # Left axis: Equity curve (on top, higher zorder)
    ln_eq = ax.plot(ts_eq, eq_vals, linewidth=1.8, color="#2ecc71", label="CTA-Forge v10g", zorder=10)
    ax.axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.4)

    # Fill under equity
    ax.fill_between(ts_eq, INITIAL_EQUITY, eq_vals,
                     where=[e >= INITIAL_EQUITY for e in eq_vals], alpha=0.1, color="#2ecc71", zorder=5)
    ax.fill_between(ts_eq, INITIAL_EQUITY, eq_vals,
                     where=[e < INITIAL_EQUITY for e in eq_vals], alpha=0.1, color="#e74c3c", zorder=5)

    # Make left axis draw on top of right
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_title(
        f"CTA-Forge v10g — Max-Range Backtest ({timeline[start].strftime('%Y-%m')} → {timeline[end-1].strftime('%Y-%m')})\n"
        f"{len(bars)} symbols · 6h · $10K start · {days} days · vs BTC & ETH buy-and-hold",
        fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    # Combined legend
    lns = ln_eq + ln_btc + ln_eth
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    # Info as xlabel of bottom chart (no extra whitespace)
    yr_str = "  ".join([f"{yr}: {(yearly[yr]['last']-yearly[yr]['first'])/yearly[yr]['first']*100:+.1f}%"
                         for yr in sorted(yearly)])
    btc_ret = (btc_filt[-1][1] / btc_filt[0][1] - 1) * 100 if btc_filt else 0
    eth_ret = (eth_filt[-1][1] / eth_filt[0][1] - 1) * 100 if eth_filt else 0
    line1 = (f"Return: {m.total_return*100:+.1f}%   Ann: {m.annualized_return*100:+.1f}%   "
             f"Sharpe: {m.sharpe_ratio:.2f}   Sortino: {m.sortino_ratio:.2f}   "
             f"MaxDD: {m.max_drawdown*100:.1f}%   Calmar: {m.calmar_ratio:.2f}   "
             f"PF: {m.profit_factor:.2f}   Win: {m.win_rate*100:.1f}%   "
             f"Trades: {m.num_trades}   Ulcer: {ulcer:.4f}   │   "
             f"BTC B&H: {btc_ret:+.0f}%   ETH B&H: {eth_ret:+.0f}%\n{yr_str}")
    axes[2].set_xlabel(line1, fontsize=7.5, family="monospace", color="#2c3e50", labelpad=6)

    # === Middle: Drawdown ===
    eq_arr = np.array(eq_vals)
    rm = np.maximum.accumulate(eq_arr)
    dd_pct = (rm - eq_arr)/rm*100
    axes[1].fill_between(ts_eq, 0, -dd_pct, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.15)

    # === Bottom: Monthly returns ===
    monthly = {}
    for ii in range(1, len(eq_vals)):
        key = ts_eq[ii].strftime("%Y-%m")
        if key not in monthly: monthly[key] = {"start": eq_vals[ii-1]}
        monthly[key]["end"] = eq_vals[ii]
    mos = list(monthly.keys())
    rets = [(monthly[m_]["end"]-monthly[m_]["start"])/monthly[m_]["start"]*100 for m_ in mos]
    axes[2].bar(range(len(mos)), rets,
                color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rets], alpha=0.8, width=0.8)
    step = max(1, len(mos)//24)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels([mos[ii] for ii in range(0, len(mos), step)], rotation=45, ha="right", fontsize=6)
    axes[2].set_ylabel("Monthly %")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.15, axis="y")
    pos_months = sum(1 for r in rets if r > 0)
    axes[2].text(0.02, 0.95, f"Positive: {pos_months}/{len(rets)} months ({pos_months/len(rets)*100:.0f}%)",
                 transform=axes[2].transAxes, fontsize=8, va="top")

    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    out_path = OUT_DIR / "backtest_v10g_maxrange_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✅ Done in {time.time()-t0:.0f}s → {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
