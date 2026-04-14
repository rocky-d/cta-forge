"""Walk-forward validation for target_vol.

Instead of reimplementing the backtest, we modify v10g's params and 
filter the equity curve by date range. Much simpler, no replication bugs.
"""
import asyncio, sys, json, copy
from datetime import UTC, datetime
from pathlib import Path
import numpy as np


from v10g_maxrange import (
    SYMBOLS, START_TS, INITIAL_EQUITY,
    fetch_klines_from, precompute, build_timeline,
    align_data, compute_signals, run_backtest
)

OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")

FOLDS = [
    {"name": "Fold1", "is_start": "2019-10-28", "is_end": "2022-03-31",
     "oos_start": "2022-04-01", "oos_end": "2023-06-30"},
    {"name": "Fold2", "is_start": "2020-10-01", "is_end": "2023-06-30",
     "oos_start": "2023-07-01", "oos_end": "2024-09-30"},
    {"name": "Fold3", "is_start": "2021-10-01", "is_end": "2024-09-30",
     "oos_start": "2024-10-01", "oos_end": "2026-04-12"},
]

TARGET_VOLS = [0.0, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

BASE_PARAMS = {
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


def metrics_for_segment(curve, start_date, end_date):
    """Extract metrics from a full equity curve for a date segment."""
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)

    segment = [(ts, eq) for ts, eq in curve if start_dt <= ts <= end_dt]
    if len(segment) < 20:
        return None

    # Returns
    returns = []
    for i in range(1, len(segment)):
        prev_v = segment[i - 1][1]
        if prev_v > 0:
            returns.append((segment[i][1] - prev_v) / prev_v)

    returns = np.array(returns)
    if len(returns) < 10 or np.std(returns) == 0:
        return None

    ann_factor = np.sqrt(4 * 365)
    sharpe = float(np.mean(returns) / np.std(returns) * ann_factor)

    # Max drawdown within segment
    peak = segment[0][1]
    max_dd = 0
    for _, v in segment:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    total_ret = (segment[-1][1] - segment[0][1]) / segment[0][1]
    days = (len(segment) * 6) / 24
    ann_ret = (1 + total_ret) ** (365 / max(days, 1)) - 1 if days > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        "total_return": round(total_ret * 100, 1),
        "ann_return": round(ann_ret * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
    }


async def main():
    import httpx

    print("Fetching data...", flush=True)
    async with httpx.AsyncClient(timeout=30) as client:
        bars = {}
        for sym in SYMBOLS:
            df = await fetch_klines_from(client, sym, START_TS)
            if df is not None and len(df) >= 500:
                bars[sym] = df
            await asyncio.sleep(0.15)

    print(f"→ {len(bars)} symbols loaded\n", flush=True)

    timeline, ts_to_idx = build_timeline(bars)
    data = precompute(bars)
    align_data(bars, data, ts_to_idx)

    results = {}

    for tv in TARGET_VOLS:
        label = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"
        print(f"Running full backtest with {label}...", flush=True)

        params = copy.deepcopy(BASE_PARAMS)
        params["target_vol"] = tv

        sigs = compute_signals(data, timeline, params)

        # Need start_idx (first bar after warm-up)
        start_idx = 200  # same as v10g
        end_idx = len(timeline) - 1
        curve, trades = run_backtest(data, sigs, timeline, start_idx, end_idx, params)

        print(f"  curve length: {len(curve)}, final equity: ${curve[-1][1]:,.0f}, trades: {len(trades)}", flush=True)

        for fold in FOLDS:
            is_m = metrics_for_segment(curve, fold["is_start"], fold["is_end"])
            oos_m = metrics_for_segment(curve, fold["oos_start"], fold["oos_end"])

            key = f"{fold['name']}_{label}"
            results[key] = {"is": is_m, "oos": oos_m}

        print(flush=True)

    # Print results table
    for fold in FOLDS:
        print(f"=== {fold['name']} ===", flush=True)
        print(f"  IS:  {fold['is_start']} → {fold['is_end']}", flush=True)
        print(f"  OOS: {fold['oos_start']} → {fold['oos_end']}", flush=True)

        for tv in TARGET_VOLS:
            label = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"
            key = f"{fold['name']}_{label}"
            is_r = results[key]["is"]
            oos_r = results[key]["oos"]

            is_s = f"Sharpe {is_r['sharpe']:5.2f}  Ret {is_r['total_return']:+7.1f}%  DD {is_r['max_dd']:5.1f}%" if is_r else "N/A"
            oos_s = f"Sharpe {oos_r['sharpe']:5.2f}  Ret {oos_r['total_return']:+7.1f}%  DD {oos_r['max_dd']:5.1f}%" if oos_r else "N/A"
            print(f"  {label:10s} IS: {is_s}  |  OOS: {oos_s}", flush=True)

        print(flush=True)

    # Summary
    print("=" * 70, flush=True)
    print("SUMMARY: Average OOS metrics by target_vol", flush=True)
    print("=" * 70, flush=True)

    for tv in TARGET_VOLS:
        label = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"
        oos_sharpes = []
        oos_rets = []
        oos_dds = []
        for fold in FOLDS:
            key = f"{fold['name']}_{label}"
            r = results[key]["oos"]
            if r:
                oos_sharpes.append(r["sharpe"])
                oos_rets.append(r["total_return"])
                oos_dds.append(r["max_dd"])

        if oos_sharpes:
            avg_s = np.mean(oos_sharpes)
            avg_r = np.mean(oos_rets)
            avg_dd = np.mean(oos_dds)
            print(f"  {label:10s}  OOS avg Sharpe: {avg_s:5.2f}  avg Ret: {avg_r:+6.1f}%  avg DD: {avg_dd:5.1f}%", flush=True)

    out_path = OUT_DIR / "walk_forward_target_vol.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
