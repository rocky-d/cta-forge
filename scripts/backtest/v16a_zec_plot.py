#!/usr/bin/env python3
"""Plot v16a 19-vs-20 symbol comparison chart from cached data (no backfill)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/executor/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs/core/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/data-service/src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / "data")
OUT_DIR = PROJECT_ROOT / "backtest-results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from executor.signal_pipeline import DEFAULT_SYMBOLS
import executor.profiles.v16a_badscore_overlay as v16a_profile
from executor.portfolio_backtest import calculate_hourly_metrics, run_target_weight_backtest
from executor.profiles.v16a_badscore_overlay import INITIAL_EQUITY, build_v16a_target_set

_SYMBOLS_19 = list(DEFAULT_SYMBOLS)

print("=== 19-symbol baseline (no backfill) ===")
target19 = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0, backfill=False)
result19 = run_target_weight_backtest(target19.timeline, target19.returns, target19.target_weights, initial_equity=INITIAL_EQUITY)
metrics19 = calculate_hourly_metrics(result19.returns, initial_equity=INITIAL_EQUITY)

DEFAULT_SYMBOLS.append("ZECUSDT")
v16a_profile.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS

print("=== 20-symbol +ZEC (no backfill) ===")
target20 = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0, backfill=False)
result20 = run_target_weight_backtest(target20.timeline, target20.returns, target20.target_weights, initial_equity=INITIAL_EQUITY)
metrics20 = calculate_hourly_metrics(result20.returns, initial_equity=INITIAL_EQUITY)

b_ts = [e[0] for e in list(zip(target19.timeline, metrics19["equity"].tolist()))]
b_eq = np.array([e[1] for e in list(zip(target19.timeline, metrics19["equity"].tolist()))])
z_ts = [e[0] for e in list(zip(target20.timeline, metrics20["equity"].tolist()))]
z_eq = np.array([e[1] for e in list(zip(target20.timeline, metrics20["equity"].tolist()))])

# Align
common_start = max(b_ts[0], z_ts[0])
common_end = min(b_ts[-1], z_ts[-1])
b_start_idx = next(i for i, ts in enumerate(b_ts) if ts >= common_start)
z_start_idx = next(i for i, ts in enumerate(z_ts) if ts >= common_start)
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

# Compute derived metrics
def _compute_metrics(returns, initial_equity):
    m = calculate_hourly_metrics(returns, initial_equity)
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(365 * 24) if len(downside) else 0.0
    m["sortino"] = m["ann_return"] / downside_vol if downside_vol > 1e-12 else 0.0
    m["calmar"] = m["ann_return"] / m["max_dd"] if m["max_dd"] > 1e-12 else 0.0
    return m

def _drawdown(eq):
    rm = np.maximum.accumulate(eq)
    return (rm - eq) / rm * 100

def _monthly_bars(timeline, equity):
    monthly: dict = {}
    for i in range(1, len(equity)):
        key = timeline[i].strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"start": equity[i - 1]}
        monthly[key]["end"] = equity[i]
    labels = list(monthly.keys())
    rets = [(monthly[m]["end"] - monthly[m]["start"]) / monthly[m]["start"] * 100 for m in labels]
    return labels, rets

def _yearly(timeline, equity):
    result: dict = {}
    years = sorted({t.year for t in timeline})
    for y in years:
        indices = [i for i, t in enumerate(timeline) if t.year == y]
        if not indices:
            continue
        result[str(y)] = float((equity[indices[-1]] / equity[indices[0]] - 1) * 100)
    return result

bm = _compute_metrics(result19.returns, INITIAL_EQUITY)
zm = _compute_metrics(result20.returns, INITIAL_EQUITY)

# ── Plot ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(24, 12), height_ratios=[3.5, 1, 1.5],
                         gridspec_kw={"hspace": 0.15})
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.06, right=0.94, top=0.93, hspace=0.15)

# Panel 1: Equity
ax1 = axes[0]
ax1.plot(b_ts_c, b_eq_c, linewidth=1.5, color="#3498db",
         label=f"19 symbols  (Sharpe {bm['sharpe']:.2f}, AnnRet {bm['ann_return']*100:+.1f}%, MaxDD {bm['max_dd']*100:.1f}%)")
ax1.plot(z_ts_c, z_eq_c, linewidth=1.5, color="#e67e22",
         label=f"20 symbols +ZEC  (Sharpe {zm['sharpe']:.2f}, AnnRet {zm['ann_return']*100:+.1f}%, MaxDD {zm['max_dd']*100:.1f}%)")
ax1.axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
ax1.grid(True, alpha=0.15)
ax1.set_title(
    "v16a Badscore Overlay — 19 vs 20 Symbols (+ZEC)\n"
    f"Live config: core_phase=2h, gross_cap=4.0, 50/50 v10g/overlay, ${INITIAL_EQUITY:,.0f} initial",
    fontsize=12, fontweight="bold", pad=10)

# Panel 2: Drawdown
ax2 = axes[1]
b_dd = _drawdown(b_eq_c)
z_dd = _drawdown(z_eq_c)
ax2.fill_between(b_ts_c, 0, -b_dd, color="#3498db", alpha=0.3,
                 label=f"19 symbols (MaxDD {bm['max_dd']*100:.1f}%)")
ax2.plot(z_ts_c, -z_dd, linewidth=1.2, color="#e67e22", alpha=0.8,
         label=f"20 symbols +ZEC (MaxDD {zm['max_dd']*100:.1f}%)")
ax2.set_ylabel("DD %", fontsize=9)
ax2.legend(loc="lower left", fontsize=8, framealpha=0.8)
ax2.grid(True, alpha=0.15)

# Panel 3: Monthly P&L
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
ax3.set_xticklabels([b_months[i] for i in range(0, len(b_months), step)],
                    rotation=45, ha="right", fontsize=6)
pos_19 = sum(1 for r in b_rets if r > 0)
pos_20 = sum(1 for r in z_rets if r > 0)
ax3.text(0.02, 0.95,
         f"Positive months: 19s→{pos_19}/{len(b_rets)} ({pos_19/len(b_rets)*100:.0f}%)  |  20s+ZEC→{pos_20}/{len(z_rets)} ({pos_20/len(z_rets)*100:.0f}%)",
         transform=ax3.transAxes, fontsize=8, va="top")

for a in axes[:2]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate()

# Delta summary
delta_sharpe = zm["sharpe"] - bm["sharpe"]
delta_ann = zm["ann_return"] - bm["ann_return"]
delta_dd = zm["max_dd"] - bm["max_dd"]
delta_sortino = zm["sortino"] - bm["sortino"]
delta_calmar = zm["calmar"] - bm["calmar"]
delta_lines = [
    f"Δ Sharpe: {delta_sharpe:+.3f}",
    f"Δ Ann Return: {delta_ann*100:+.2f}%",
    f"Δ MaxDD: {delta_dd*100:+.2f}%",
    f"Δ Sortino: {delta_sortino:+.3f}",
    f"Δ Calmar: {delta_calmar:+.3f}",
]
fig.text(0.5, 0.01, "  |  ".join(delta_lines), ha="center", fontsize=8,
         family="monospace", color="#555")

# Metrics labels
label_parts = []
for label, m in [("19s", bm), ("20s+ZEC", zm)]:
    parts = [
        f"{label}: Return {m['total_return']*100:+.1f}%",
        f"Ann {m['ann_return']*100:+.1f}%",
        f"Sharpe {m['sharpe']:.2f}",
        f"Sortino {m['sortino']:.2f}",
        f"MaxDD {m['max_dd']*100:.1f}%",
        f"Calmar {m['calmar']:.2f}",
    ]
    label_parts.append("   ".join(parts))
yr_19 = _yearly(b_ts_c, b_eq_c)
yr_20 = _yearly(z_ts_c, z_eq_c)
yr_str = "   |   ".join([
    "Yearly 19s: " + "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yr_19.items())),
    "Yearly 20s: " + "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yr_20.items())),
])
axes[2].set_xlabel(
    f"{'   |   '.join(label_parts)}\n{yr_str}",
    fontsize=6.5, family="monospace", color="#2c3e50", labelpad=4)

plt.tight_layout(rect=[0, 0.04, 1, 0.99])
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
buf.seek(0)

chart_path = OUT_DIR / "backtest_v16a_19vs20_zec_comparison.png"
chart_path.write_bytes(buf.read())
print(f"\nChart saved: {chart_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: 19 vs 20 symbols (+ZEC)")
print("=" * 70)
for key, label in [("sharpe","Sharpe"), ("total_return","Total Ret"), ("ann_return","Ann Ret"),
                    ("max_dd","MaxDD"), ("sortino","Sortino"), ("calmar","Calmar")]:
    bv = bm[key]
    zv = zm[key]
    dv = zv - bv
    if "return" in key or "max_dd" in key:
        print(f"  {label:12s}: 19s={bv*100:+.2f}%    20s={zv*100:+.2f}%     Δ={dv*100:+.2f}%")
    else:
        print(f"  {label:12s}: 19s={bv:.3f}       20s={zv:.3f}        Δ={dv:+.3f}")
