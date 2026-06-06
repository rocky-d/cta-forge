#!/usr/bin/env python3
"""Backtest comparison: 19-symbol baseline vs 20-symbol (adding ZEC).

Uses the EXACT same live config parameters as the current mainnet-pilot:
- core_phase_hours=2, gross_cap=4.0
- 50/50 v10g core / overlay allocation
- v16a badscore gate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/executor/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs/core/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/data-service/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/report-service/src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "backtest-results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── now import the backtest modules ──
from executor.signal_pipeline import DEFAULT_SYMBOLS  # noqa: E402

print(f"\nOriginal DEFAULT_SYMBOLS ({len(DEFAULT_SYMBOLS)}): {DEFAULT_SYMBOLS}")

# ── add ZEC to DEFAULT_SYMBOLS for the 20-symbol run ──
_SYMBOLS_19 = list(DEFAULT_SYMBOLS)  # snapshot
DEFAULT_SYMBOLS.append("ZECUSDT")
_SYMBOLS_20 = list(DEFAULT_SYMBOLS)
print(f"20-symbol list ({len(_SYMBOLS_20)}): {_SYMBOLS_20}")

# ── import build_v16a_target_set AFTER modifying DEFAULT_SYMBOLS ──
# (the module-level import in v16a_badscore_overlay already happened since
#  signal_pipeline was imported, but the badscore gate uses DEFAULT_SYMBOLS
#  at call time, and load_bars uses it too)
import executor.profiles.v16a_badscore_overlay as v16a_profile  # noqa: E402
from executor.portfolio_backtest import (  # noqa: E402
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (  # noqa: E402
    INITIAL_EQUITY,
    build_v16a_target_set,
)

# Also need to update DEFAULT_SYMBOLS in v16a_profile module if it has its own copy
v16a_profile.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS


# ── run both backtests ──
def run_backtest(label: str, symbol_count: int, use_zec: bool) -> dict:
    """Run backtest and return metrics dict."""
    if not use_zec:
        # Restore 19-symbol list
        DEFAULT_SYMBOLS.clear()
        DEFAULT_SYMBOLS.extend(_SYMBOLS_19)
        v16a_profile.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS

    print(f"\n=== Running {label} ({symbol_count} symbols) ===")
    t0 = time.time()

    target_set = build_v16a_target_set(
        str(DATA_DIR),
        core_phase_hours=2,
        gross_cap=4.0,
    )

    result = run_target_weight_backtest(
        target_set.timeline,
        target_set.returns,
        target_set.target_weights,
        initial_equity=INITIAL_EQUITY,
    )
    metrics = calculate_hourly_metrics(result.returns, initial_equity=INITIAL_EQUITY)

    downside = result.returns[result.returns < 0]
    downside_vol = np.std(downside) * np.sqrt(365 * 24) if len(downside) else 0.0
    sortino = metrics["ann_return"] / downside_vol if downside_vol > 1e-12 else 0.0
    calmar = (
        metrics["ann_return"] / metrics["max_dd"] if metrics["max_dd"] > 1e-12 else 0.0
    )

    equity_curve = list(zip(target_set.timeline, metrics["equity"].tolist()))
    avg_gross = np.mean(np.sum(np.abs(target_set.target_weights), axis=1))
    avg_turnover = np.mean(result.turnover)
    days = (target_set.timeline[-1] - target_set.timeline[0]).days

    print(f"  Symbols: {sorted(set(target_set.symbols))}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Ann Return: {metrics['ann_return']*100:.1f}%")
    print(f"  Total Return: {metrics['return']*100:.1f}%")
    print(f"  MaxDD: {metrics['max_dd']*100:.2f}%")
    print(f"  Sortino: {sortino:.3f}")
    print(f"  Calmar: {calmar:.3f}")
    print(f"  Avg gross: {avg_gross:.3f}")
    print(f"  Avg turnover/h: {avg_turnover:.4f}")
    print(f"  Period: {target_set.timeline[0].date()} -> {target_set.timeline[-1].date()} ({days}d)")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return {
        "label": label,
        "symbol_count": symbol_count,
        "symbols": sorted(set(target_set.symbols)),
        "timeline": target_set.timeline,
        "equity_curve": equity_curve,
        "returns": result.returns,
        "target_weights": target_set.target_weights,
        "metrics": {
            "sharpe": metrics["sharpe"],
            "total_return": metrics["return"],
            "ann_return": metrics["ann_return"],
            "max_dd": metrics["max_dd"],
            "sortino": sortino,
            "calmar": calmar,
            "avg_gross": avg_gross,
            "avg_turnover_per_hour": avg_turnover,
            "days": days,
        },
    }


# Run 19-symbol (baseline)
baseline = run_backtest("19-symbol Baseline", 19, use_zec=False)

# Restore 20-symbol list and run
DEFAULT_SYMBOLS.clear()
DEFAULT_SYMBOLS.extend(_SYMBOLS_20)
v16a_profile.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS

with_zec = run_backtest("20-symbol (+ZEC)", 20, use_zec=True)

# ── save metrics ──
summary = {
    "live_config": {
        "core_phase_hours": 2,
        "gross_cap": 4.0,
        "v10g_allocation": 0.5,
        "overlay_allocation": 0.5,
        "initial_equity": INITIAL_EQUITY,
    },
    "baseline_19": baseline["metrics"],
    "with_zec_20": with_zec["metrics"],
    "delta": {
        key: with_zec["metrics"][key] - baseline["metrics"][key]
        for key in ["sharpe", "total_return", "ann_return", "max_dd", "sortino", "calmar"]
    },
}

metrics_path = OUT_DIR / "metrics_v16a_zec_comparison.json"
with open(metrics_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nMetrics saved to {metrics_path}")

# ── plot comparison chart ──
print("\n=== Plotting comparison chart ===")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import io  # noqa: E402


def _monthly_bars(timeline, equity):
    """Return (month_labels, monthly_returns_pct)."""
    monthly: dict[str, dict[str, float]] = {}
    for i in range(1, len(equity)):
        key = timeline[i].strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"start": equity[i - 1]}
        monthly[key]["end"] = equity[i]
    labels = list(monthly.keys())
    rets = [
        (monthly[m]["end"] - monthly[m]["start"]) / monthly[m]["start"] * 100
        for m in labels
    ]
    return labels, rets


def _drawdown(eq):
    rm = np.maximum.accumulate(eq)
    return (rm - eq) / rm * 100


def _yearly_percentages(timeline, equity):
    result: dict[str, float] = {}
    years = sorted({t.year for t in timeline})
    for y in years:
        indices = [i for i, t in enumerate(timeline) if t.year == y]
        if not indices:
            continue
        start_val = equity[indices[0]]
        if start_val is None or start_val == 0:
            continue
        result[str(y)] = float((equity[indices[-1]] / start_val - 1) * 100)
    return result


# Common timeline (use the shorter one for alignment)
b_ts = [e[0] for e in baseline["equity_curve"]]
b_eq = np.array([e[1] for e in baseline["equity_curve"]])
z_ts = [e[0] for e in with_zec["equity_curve"]]
z_eq = np.array([e[1] for e in with_zec["equity_curve"]])

# Align to common timeline
common_start = max(b_ts[0], z_ts[0])
common_end = min(b_ts[-1], z_ts[-1])
b_start_idx = next(i for i, ts in enumerate(b_ts) if ts >= common_start)
z_start_idx = next(i for i, ts in enumerate(z_ts) if ts >= common_start)

# Find end indices (inclusive of common_end)
try:
    b_end_idx = next(i for i, ts in enumerate(b_ts) if ts > common_end)
except StopIteration:
    b_end_idx = len(b_ts)
try:
    z_end_idx = next(i for i, ts in enumerate(z_ts) if ts > common_end)
except StopIteration:
    z_end_idx = len(z_ts)

b_ts_c = b_ts[b_start_idx:b_end_idx]
b_eq_c = b_eq[b_start_idx:b_end_idx]
z_ts_c = z_ts[z_start_idx:z_end_idx]
z_eq_c = z_eq[z_start_idx:z_end_idx]

bm = baseline["metrics"]
zm = with_zec["metrics"]

fig, axes = plt.subplots(
    3, 1,
    figsize=(24, 12),
    height_ratios=[3.5, 1, 1.5],
    gridspec_kw={"hspace": 0.15},
)
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.06, right=0.94, top=0.93, hspace=0.15)

# ── Panel 1: Equity ──
ax1 = axes[0]
ax1.plot(b_ts_c, b_eq_c, linewidth=1.5, color="#3498db", label=f"19 symbols (Sharpe {bm['sharpe']:.2f})")
ax1.plot(z_ts_c, z_eq_c, linewidth=1.5, color="#e67e22", label=f"20 symbols +ZEC (Sharpe {zm['sharpe']:.2f})")
ax1.axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
ax1.grid(True, alpha=0.15)
ax1.set_title(
    "v16a Badscore Overlay — 19 vs 20 Symbols (+ZEC)\n"
    f"Live config: core_phase=2h, gross_cap=4.0, 50/50 v10g/overlay, ${INITIAL_EQUITY:,.0f} initial",
    fontsize=12,
    fontweight="bold",
    pad=10,
)

# ── Panel 2: Drawdown ──
ax2 = axes[1]
b_dd = _drawdown(b_eq_c)
z_dd = _drawdown(z_eq_c)
ax2.fill_between(b_ts_c, 0, -b_dd, color="#3498db", alpha=0.3, label=f"19 symbols (MaxDD {bm['max_dd']*100:.1f}%)")
ax2.plot(z_ts_c, -z_dd, linewidth=1.2, color="#e67e22", alpha=0.8, label=f"20 symbols +ZEC (MaxDD {zm['max_dd']*100:.1f}%)")
ax2.set_ylabel("DD %", fontsize=9)
ax2.legend(loc="lower left", fontsize=8, framealpha=0.8)
ax2.grid(True, alpha=0.15)

# ── Panel 3: Monthly P&L ──
ax3 = axes[2]
b_months, b_rets = _monthly_bars(b_ts_c, b_eq_c)
z_months, z_rets = _monthly_bars(z_ts_c, z_eq_c)

x = np.arange(len(b_months))
width = 0.4
b_colors = ["#27ae60" if r >= 0 else "#c0392b" for r in b_rets]
z_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in z_rets]
ax3.bar(x - width / 2, b_rets, width, color=b_colors, alpha=0.7, label="19 symbols")
ax3.bar(x + width / 2, z_rets, width, color=z_colors, alpha=0.7, label="20 symbols +ZEC")
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
ax3.grid(True, alpha=0.15, axis="y")

step = max(1, len(b_months) // 24)
ax3.set_xticks(range(0, len(b_months), step))
ax3.set_xticklabels([b_months[i] for i in range(0, len(b_months), step)], rotation=45, ha="right", fontsize=6)

pos_19 = sum(1 for r in b_rets if r > 0)
pos_20 = sum(1 for r in z_rets if r > 0)
ax3.text(
    0.02, 0.95,
    f"Positive months: 19s → {pos_19}/{len(b_rets)} ({pos_19/len(b_rets)*100:.0f}%)   |   20s+ZEC → {pos_20}/{len(z_rets)} ({pos_20/len(z_rets)*100:.0f}%)",
    transform=ax3.transAxes, fontsize=8, va="top",
)

for a in axes[:2]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate()

# ── Summary text under chart ──
delta_lines = [
    f"Δ Sharpe: {summary['delta']['sharpe']:+.3f}",
    f"Δ Ann Return: {summary['delta']['ann_return']*100:+.2f}%",
    f"Δ MaxDD: {summary['delta']['max_dd']*100:+.2f}%",
    f"Δ Sortino: {summary['delta']['sortino']:+.3f}",
    f"Δ Calmar: {summary['delta']['calmar']:+.3f}",
]
summary_text = " | ".join(delta_lines)
fig.text(0.5, 0.01, summary_text, ha="center", fontsize=8, family="monospace", color="#555")

# ── X-axis metrics label ──
label_parts = []
for label, m in [("19s", bm), ("20s+ZEC", zm)]:
    parts = [
        f"{label}: Return {m['total_return']*100:+.1f}%",
        f"Ann {m['ann_return']*100:+.1f}%",
        f"Sharpe {m['sharpe']:.2f}",
        f"Sortino {m['sortino']:.2f}",
        f"MaxDD {m['max_dd']*100:.1f}%",
        f"Calmar {m['calmar']:.2f}",
        f"Gross {m['avg_gross']:.3f}",
        f"Turnover/h {m['avg_turnover_per_hour']:.4f}",
        f"{m['days']}d",
    ]
    label_parts.append("   ".join(parts))
full_label = "   |   ".join(label_parts)
yr_19 = _yearly_percentages(b_ts_c, b_eq_c)
yr_20 = _yearly_percentages(z_ts_c, z_eq_c)
yr_str = "   |   ".join([
    "Yearly 19s: " + "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yr_19.items())),
    "Yearly 20s: " + "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yr_20.items())),
])
axes[2].set_xlabel(
    f"{full_label}\n{yr_str}",
    fontsize=6.5, family="monospace", color="#2c3e50", labelpad=4,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.99])
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
buf.seek(0)

chart_path = OUT_DIR / "backtest_v16a_19vs20_zec_comparison.png"
chart_path.write_bytes(buf.read())
print(f"Chart saved to {chart_path}")

# ── print final summary ──
print("\n" + "=" * 70)
print("SUMMARY: 19-symbol vs 20-symbol (+ZEC)")
print("=" * 70)
for key in ["sharpe", "total_return", "ann_return", "max_dd", "sortino", "calmar"]:
    bv = baseline["metrics"][key]
    zv = with_zec["metrics"][key]
    dv = summary["delta"][key]
    if "return" in key or "max_dd" in key:
        print(f"  {key:20s}: 19s={bv*100:+.2f}%   20s={zv*100:+.2f}%   Δ={dv*100:+.2f}%")
    else:
        print(f"  {key:20s}: 19s={bv:.3f}       20s={zv:.3f}       Δ={dv:+.3f}")
print(f"  {'avg_gross':20s}: 19s={baseline['metrics']['avg_gross']:.3f}     20s={with_zec['metrics']['avg_gross']:.3f}     Δ={with_zec['metrics']['avg_gross']-baseline['metrics']['avg_gross']:+.3f}")
print(f"  {'turnover/h':20s}: 19s={baseline['metrics']['avg_turnover_per_hour']:.4f}   20s={with_zec['metrics']['avg_turnover_per_hour']:.4f}   Δ={with_zec['metrics']['avg_turnover_per_hour']-baseline['metrics']['avg_turnover_per_hour']:+.4f}")
print(f"\nChart: {chart_path}")
print(f"Metrics: {metrics_path}")
