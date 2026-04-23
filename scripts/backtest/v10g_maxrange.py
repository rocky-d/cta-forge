"""CTA-Forge v10g — Max-range backtest CLI.

Thin wrapper around executor.backtest (same engine as live trading)
with report_service.plot for chart generation.

Usage: uv run python scripts/backtest/v10g_maxrange.py --profile 6h_default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from core.metrics import calculate_metrics
from executor.backtest import calc_ulcer, run_full_backtest
from executor.decision import V10GStrategyParams
from report_service.plot import plot_backtest

OUT_DIR = Path(__file__).resolve().parents[2] / "backtest-results"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INITIAL_EQUITY = 10_000.0

# ── Profiles ─────────────────────────────────────────────────────

PROFILES = {
    "6h_default": V10GStrategyParams(
        timeframe_str="6h",
        timeframe_hours=6,
        signal_threshold=0.40,
        min_hold_bars=16,
        max_hold_bars=100,
    ),
    "1h_scaled": V10GStrategyParams(
        timeframe_str="1h",
        timeframe_hours=1,
        signal_threshold=0.40,
        min_hold_bars=16 * 6,  # scale bars to match same physical time
        max_hold_bars=100 * 6,
        rebalance_every=4 * 6,
        mom_lookbacks=[20 * 6, 60 * 6, 120 * 6],
        donchian_period=20 * 6,
        rvol_lookback=20 * 6,
        rvol_median_lookback=120 * 6,
        vol_filter_lookback=20 * 6,
        btc_filter_lookback=60 * 6,
    ),
}


def _price_series(bars, sym, start_ts):
    """Extract (datetime, close) series from bars for chart overlay."""
    df = bars.get(sym)
    if df is None or df.is_empty():
        return []
    return [
        (t, p)
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if t >= start_ts
    ]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="6h_default",
        help="Strategy profile to run",
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    t0 = time.time()
    print("=" * 60)
    print(f"CTA-Forge v10g — Max-Range Backtest [{args.profile}]")
    print("=" * 60)

    print(f"\nRunning backtest with timeframe: {profile.timeframe_str}...")
    result = await run_full_backtest(
        data_dir=str(DATA_DIR),
        initial_equity=INITIAL_EQUITY,
        params=profile,
    )

    if not result.equity_curve:
        print("No data available.")
        return

    # Metrics
    m = calculate_metrics(result.equity_curve, result.trades)
    ulcer = calc_ulcer(result.equity_curve)

    # Yearly breakdown
    yearly: dict[int, dict[str, float]] = {}
    for ts, eq in result.equity_curve:
        yr = ts.year
        if yr not in yearly:
            yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({result.days} days, {len(result.symbols)} symbols)")
    print(f"{'=' * 60}")
    print(f"  Profile: {args.profile}")
    print(f"  Period: {result.start_date} -> {result.end_date}")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(
        f"  Return: {m.total_return * 100:+.1f}%  "
        f"Ann: {m.annualized_return * 100:+.1f}%"
    )
    print(f"  Sharpe: {m.sharpe_ratio:.2f}  Sortino: {m.sortino_ratio:.2f}")
    print(f"  MaxDD:  {m.max_drawdown * 100:.1f}%  Calmar: {m.calmar_ratio:.2f}")
    print(
        f"  PF: {m.profit_factor:.2f}  "
        f"Win: {m.win_rate * 100:.1f}%  "
        f"Trades: {m.num_trades}"
    )
    print(f"  Ulcer: {ulcer:.4f}")

    print("\nYearly returns:")
    for yr in sorted(yearly):
        yr_ret = (yearly[yr]["last"] - yearly[yr]["first"]) / yearly[yr]["first"] * 100
        print(f"  {yr}: {yr_ret:+.1f}%")

    # Generate chart using report_service.plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_start_ts = result.equity_curve[0][0]
    btc_prices = _price_series(result.bars, "BTCUSDT", curve_start_ts)
    eth_prices = _price_series(result.bars, "ETHUSDT", curve_start_ts)

    yearly_pct = {
        str(yr): round(
            (yearly[yr]["last"] - yearly[yr]["first"]) / yearly[yr]["first"] * 100, 1
        )
        for yr in sorted(yearly)
    }

    img = plot_backtest(
        equity_curve=result.equity_curve,
        btc_prices=btc_prices,
        eth_prices=eth_prices,
        metrics={
            "total_return": m.total_return,
            "annualized_return": m.annualized_return,
            "sharpe_ratio": m.sharpe_ratio,
            "sortino_ratio": m.sortino_ratio,
            "max_drawdown": m.max_drawdown,
            "calmar_ratio": m.calmar_ratio,
            "profit_factor": m.profit_factor,
            "win_rate": m.win_rate,
            "num_trades": m.num_trades,
            "ulcer_index": ulcer,
        },
        yearly=yearly_pct,
        title_extra=(
            f"{len(result.symbols)} symbols · {profile.timeframe_str} · "
            f"${INITIAL_EQUITY:,.0f} start · {result.days} days · "
            f"vs BTC & ETH buy-and-hold"
        ),
        initial_equity=INITIAL_EQUITY,
    )
    chart_path = OUT_DIR / f"backtest_v10g_{args.profile}.png"
    chart_path.write_bytes(img)

    # Save metrics JSON
    (OUT_DIR / f"metrics_v10g_{args.profile}.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "timeframe": profile.timeframe_str,
                "period": f"{result.start_date} -> {result.end_date}",
                "days": result.days,
                "symbols": len(result.symbols),
                "symbol_list": result.symbols,
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
                "yearly": yearly_pct,
                "engine": "V10GDecisionEngine",
            },
            indent=2,
        )
    )

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    asyncio.run(main())
