"""Walk-forward validation chart for target_vol comparison."""
import asyncio, sys, json, copy
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/libs/cta-core/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/services/reporter/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/scripts/backtest")

from v10g_maxrange import (
    SYMBOLS, START_TS, INITIAL_EQUITY,
    fetch_klines_from, precompute, build_timeline,
    align_data, compute_signals, run_backtest
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

OUT = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")

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

VOLS_TO_PLOT = [
    (0.0, "vol=OFF", "#95a5a6", "--"),
    (0.12, "vol=0.12 (v10g)", "#2ecc71", "-"),
    (0.20, "vol=0.20 (v15d)", "#e74c3c", "-"),
    (0.30, "vol=0.30", "#f39c12", "-."),
]

FOLDS = [
    {"is": ("2019-10-28", "2022-03-31"), "oos": ("2022-04-01", "2023-06-30")},
    {"is": ("2020-10-01", "2023-06-30"), "oos": ("2023-07-01", "2024-09-30")},
    {"is": ("2021-10-01", "2024-09-30"), "oos": ("2024-10-01", "2026-04-12")},
]


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

    curves = {}
    for tv, label, _, _ in VOLS_TO_PLOT:
        params = copy.deepcopy(BASE_PARAMS)
        params["target_vol"] = tv
        sigs = compute_signals(data, timeline, params)
        curve, trades = run_backtest(data, sigs, timeline, 200, len(timeline) - 1, params)
        curves[label] = curve
        final = curve[-1][1]
        print(f"  {label}: final=${final:,.0f}, trades={len(trades)}", flush=True)

    # ── Chart ──
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(24, 12), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.15}
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93)

    # Top: equity curves
    for tv, label, color, ls in VOLS_TO_PLOT:
        c = curves[label]
        dates = [ts for ts, _ in c]
        eqs = [eq for _, eq in c]
        ax1.plot(dates, eqs, color=color, linewidth=1.8 if "v10g" in label else 1.2,
                 linestyle=ls, label=label, alpha=0.9)

    # Shade OOS regions for Fold 3 (most recent)
    oos3_start = datetime.fromisoformat("2024-10-01").replace(tzinfo=UTC)
    oos3_end = datetime.fromisoformat("2026-04-12").replace(tzinfo=UTC)
    ax1.axvspan(oos3_start, oos3_end, alpha=0.08, color="red", label="Fold 3 OOS")

    ax1.set_title("Walk-Forward Validation: target_vol Comparison", fontsize=14, pad=10)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Bottom: bar chart of OOS Sharpe by fold
    TARGET_VOLS_BAR = [0.0, 0.12, 0.20, 0.30]
    labels_bar = ["vol=OFF", "vol=0.12", "vol=0.20", "vol=0.30"]
    colors_bar = ["#95a5a6", "#2ecc71", "#e74c3c", "#f39c12"]

    # Load walk-forward results
    wf = json.loads((OUT / "walk_forward_target_vol.json").read_text())

    x = np.arange(3)  # 3 folds
    width = 0.18
    for i, (tv, lbl, col) in enumerate(zip(TARGET_VOLS_BAR, labels_bar, colors_bar)):
        key_part = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"
        sharpes = []
        for fi in range(3):
            k = f"Fold{fi+1}_{key_part}"
            r = wf.get(k, {}).get("oos")
            sharpes.append(r["sharpe"] if r else 0)
        offset = (i - 1.5) * width
        bars = ax2.bar(x + offset, sharpes, width, label=lbl, color=col, alpha=0.85)
        for bar, s in zip(bars, sharpes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f"{s:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([
        "Fold 1 OOS\n(2022-04 → 2023-06)",
        "Fold 2 OOS\n(2023-07 → 2024-09)",
        "Fold 3 OOS\n(2024-10 → 2026-04)"
    ], fontsize=9)
    ax2.set_ylabel("OOS Sharpe", fontsize=10)
    ax2.set_title("Out-of-Sample Sharpe by Fold", fontsize=12, pad=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # Info strip
    info = ("Walk-Forward Validation  │  3 Folds (expanding window)  │  "
            "OOS avg Sharpe: vol=OFF 1.36  vol=0.12 1.15  vol=0.20 1.14  vol=0.30 1.21  │  "
            "Conclusion: vol=0.20 fails Fold 3 OOS (Sharpe -0.21), v10g vol=0.12 is robust")
    ax2.set_xlabel(info, fontsize=7.5, family="monospace", color="#2c3e50", labelpad=8)

    out_path = OUT / "walk_forward_target_vol.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✅ Chart: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
