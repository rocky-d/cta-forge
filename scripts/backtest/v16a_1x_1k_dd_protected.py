#!/usr/bin/env python3
"""v16a 1x unlevered backtest with DD protection (soft=15%, hard=30%).

Matches live strategy config: core_phase_hours=2, gross_cap=1.0, 19 symbols.
Applies L1 soft scaling (dd_scale linear [15%,30%] → gross_cap reduction)
and L2 hard stop (flatten + suppress at DD ≥ 30%).
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "sans-serif",
    "grid.alpha": 0.12,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "services/executor/src"))
sys.path.insert(0, str(PROJECT_ROOT / "libs/core/src"))
sys.path.insert(0, str(PROJECT_ROOT / "services/data-service/src"))

from executor.targeting import normalize_gross
from executor.profiles.v16a_badscore_overlay import build_v16a_target_set

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "backtest-results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── config (matches live: core_phase=2h, 1x gross_cap=1.0) ──
INITIAL_EQUITY = 1000.0
GROSS_CAP = 1.0
CORE_PHASE_HOURS = 2
SOFT_DD_LIMIT = 0.03  # scaled from 0.15 for 1x (gross_cap=1.0 vs live 4.0)
HARD_DD_LIMIT = 0.06  # scaled from 0.30 for 1x
FEE = 0.000432  # HL taker fee
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

print(f"=== v16a 1x unlevered backtest with DD protection ===")
print(f"  Initial equity: ${INITIAL_EQUITY:,.0f}")
print(f"  Gross cap: {GROSS_CAP}")
print(f"  Soft DD limit: {SOFT_DD_LIMIT*100:.0f}%")
print(f"  Hard DD limit: {HARD_DD_LIMIT*100:.0f}%")
print(f"  Core phase: {CORE_PHASE_HOURS}h")
print(f"  Run tag: {RUN_TAG}")

t0 = time.time()

target_set = build_v16a_target_set(
    str(DATA_DIR),
    core_phase_hours=CORE_PHASE_HOURS,
    gross_cap=GROSS_CAP,
)

# ── DD-aware backtest ──
returns = target_set.returns
raw_weights = target_set.target_weights
timeline = target_set.timeline
symbols = target_set.symbols
n_steps = len(timeline)

equity = INITIAL_EQUITY
peak = INITIAL_EQUITY
equity_curve: list[float] = [equity]
pnl = np.zeros(n_steps - 1, dtype=float)
turnover = np.zeros(n_steps - 1, dtype=float)
previous_weights = np.zeros(len(symbols), dtype=float)
dd_scale_history: list[float] = []
hard_stop_events: list[str] = []
soft_active_steps: int = 0

for t in range(n_steps - 1):
    # Compute DD
    dd = (peak - equity) / peak if peak > 0 else 0.0

    # DD scaling
    if dd >= HARD_DD_LIMIT:
        # Hard stop: flatten, suppress new exposure
        if np.any(np.abs(previous_weights) > 1e-12):
            hard_stop_events.append(
                f"{timeline[t]} DD={dd*100:.1f}% → flatten"
            )
        current_weights = np.zeros(len(symbols), dtype=float)
        dd_scale = 0.0
    elif dd >= SOFT_DD_LIMIT and HARD_DD_LIMIT > SOFT_DD_LIMIT:
        dd_scale = max(
            0.0,
            1.0 - (dd - SOFT_DD_LIMIT) / (HARD_DD_LIMIT - SOFT_DD_LIMIT),
        )
        soft_active_steps += 1
        effective_cap = GROSS_CAP * dd_scale
        raw = np.nan_to_num(raw_weights[t], nan=0.0)
        capped = normalize_gross(
            {sym: raw[i] for i, sym in enumerate(symbols)},
            gross_cap=effective_cap,
        )
        current_weights = np.array(
            [capped.get(sym, 0.0) for sym in symbols]
        )
    else:
        dd_scale = 1.0
        current_weights = np.nan_to_num(raw_weights[t], nan=0.0)

    dd_scale_history.append(dd_scale)

    # Turnover and PnL
    turnover[t] = float(np.sum(np.abs(current_weights - previous_weights)))
    pnl[t] = float(
        np.sum(current_weights * np.nan_to_num(returns[t + 1], nan=0.0))
        - turnover[t] * FEE
    )

    equity *= (1.0 + pnl[t])
    peak = max(peak, equity)
    equity_curve.append(equity)
    previous_weights = current_weights.copy()

eq = np.array(equity_curve)
per_step_returns = pnl

# ── metrics ──
total_ret = (eq[-1] / INITIAL_EQUITY) - 1.0
log_ret = np.log(eq[-1] / eq[0]) if eq[-1] > 0 and eq[0] > 0 else 0.0
days = (timeline[-1] - timeline[0]).days
ann_ret = float((1.0 + total_ret) ** (365.25 / days) - 1.0) if days > 0 else 0.0
ann_vol = float(np.std(per_step_returns) * np.sqrt(365 * 24))
sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

max_dd_val = 0.0
peak_val = INITIAL_EQUITY
for v in eq:
    peak_val = max(peak_val, v)
    dd_val = (peak_val - v) / peak_val if peak_val > 0 else 0.0
    max_dd_val = max(max_dd_val, dd_val)

downside = per_step_returns[per_step_returns < 0]
down_vol = float(np.std(downside) * np.sqrt(365 * 24)) if len(downside) > 0 else 0.0
sortino = ann_ret / down_vol if down_vol > 1e-12 else 0.0
calmar = ann_ret / max_dd_val if max_dd_val > 1e-12 else 0.0

avg_gross = float(np.mean(np.sum(np.abs(raw_weights), axis=1)))
avg_eff_gross = float(np.mean(
    [np.sum(np.abs(np.nan_to_num(raw_weights[t], nan=0.0))) * s
     for t, s in enumerate(dd_scale_history)]
))
avg_turnover = float(np.mean(turnover))
avg_dd_scale = float(np.mean(dd_scale_history))

start_date = timeline[0].strftime("%Y-%m-%d")
end_date = timeline[-1].strftime("%Y-%m-%d")

print(f"\n  Symbols: {sorted(symbols)} ({len(symbols)})")
print(f"  Period: {start_date} → {end_date} ({days}d)")
print(f"  Sharpe:         {sharpe:.4f}")
print(f"  Ann Return:     {ann_ret*100:.2f}%")
print(f"  Total Return:   {total_ret*100:.2f}%")
print(f"  Max DD:         {max_dd_val*100:.2f}%")
print(f"  Sortino:        {sortino:.4f}")
print(f"  Calmar:         {calmar:.4f}")
print(f"  Volatility:     {ann_vol*100:.2f}%")
print(f"  Avg Raw Gross:  {avg_gross:.4f}")
print(f"  Avg Eff Gross:  {avg_eff_gross:.4f}")
print(f"  Avg Turnover/h: {avg_turnover:.6f}")
print(f"  Avg dd_scale:   {avg_dd_scale:.6f}")
print(f"  Soft active:    {soft_active_steps} steps")
print(f"  Hard stops:     {len(hard_stop_events)}")
if hard_stop_events:
    for e in hard_stop_events:
        print(f"    {e}")
print(f"  Elapsed:        {time.time() - t0:.1f}s")

# ── save metrics JSON ──
metrics_json = {
    "run_tag": RUN_TAG,
    "config": {
        "initial_equity": INITIAL_EQUITY,
        "gross_cap": GROSS_CAP,
        "core_phase_hours": CORE_PHASE_HOURS,
        "leverage": "1x",
        "soft_dd_limit": SOFT_DD_LIMIT,
        "hard_dd_limit": HARD_DD_LIMIT,
        "fee": FEE,
    },
    "period": {"start": start_date, "end": end_date, "days": days},
    "symbols": sorted(symbols),
    "symbol_count": len(symbols),
    "dd_protection": {
        "soft_active_steps": soft_active_steps,
        "hard_stop_events": hard_stop_events,
        "avg_dd_scale": avg_dd_scale,
    },
    "metrics": {
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "total_return": total_ret,
        "max_dd": max_dd_val,
        "sortino": sortino,
        "calmar": calmar,
        "volatility": ann_vol,
        "avg_raw_gross": avg_gross,
        "avg_eff_gross": avg_eff_gross,
        "avg_turnover_per_hour": avg_turnover,
    },
}
metrics_path = OUT_DIR / f"v16a_1x_1k_dd_protected_{RUN_TAG}.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_json, f, indent=2)

# ── Three-panel chart ──
FIG_W, FIG_H = 16, 9
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
gs = fig.add_gridspec(
    3, 1, height_ratios=[3.5, 0.9, 1.6],
    hspace=0.08,
    top=0.93, bottom=0.10, left=0.06, right=0.97,
)

dd_pct = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq) * 100

# Monthly P&L
mo_start: dict[str, float] = {}
mo_end: dict[str, float] = {}
for i in range(1, len(eq)):
    k = timeline[i - 1].strftime("%Y-%m")
    if k not in mo_start:
        mo_start[k] = eq[i - 1]
    mo_end[k] = eq[i]
mo_labels = list(mo_start.keys())
mo_rets = [(mo_end[m] - mo_start[m]) / mo_start[m] * 100 for m in mo_labels]

pos_months = sum(1 for r in mo_rets if r > 0)
avg_win = float(np.mean([r for r in mo_rets if r > 0]))
avg_loss = float(np.mean([r for r in mo_rets if r < 0]))

# Yearly
yearly: dict[str, float] = {}
for y in sorted({t.year for t in timeline}):
    idxs = [i for i, t in enumerate(timeline) if t.year == y]
    if idxs and eq[idxs[0]] > 0:
        yearly[str(y)] = float((eq[idxs[-1]] / eq[idxs[0]] - 1) * 100)

# Panel 1: Equity
ax1 = fig.add_subplot(gs[0])
norm_eq = eq / INITIAL_EQUITY
ax1.plot(timeline[:len(eq)], norm_eq, linewidth=1.4, color="#1a5276")
ax1.axhline(y=1.0, color="#95a5a6", linestyle="--", linewidth=0.7, alpha=0.5)
ax1.set_ylabel("Equity / Initial", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
ax1.grid(True, alpha=0.12)
ax1.set_title(
    f"v16a — 1× Unlevered + DD Protection  |  {start_date} → {end_date}  |  "
    f"${INITIAL_EQUITY:,.0f} initial  |  {len(symbols)} symbols",
    fontsize=13, fontweight="bold", pad=8,
)
box = (
    f"Sharpe {sharpe:.2f}    Ann {ann_ret*100:.1f}%    "
    f"Total {total_ret*100:.1f}%    MaxDD {max_dd_val*100:.1f}%    "
    f"Sortino {sortino:.2f}    Calmar {calmar:.2f}"
)
ax1.text(
    0.02, 0.96, box, transform=ax1.transAxes,
    fontsize=9, va="top", fontfamily="monospace", color="#2c3e50",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.92),
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# Panel 2: Drawdown
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(timeline[:len(eq)], 0, -dd_pct, color="#c0392b", alpha=0.2, linewidth=0)
ax2.plot(timeline[:len(eq)], -dd_pct, linewidth=0.7, color="#c0392b")
ax2.set_ylabel("DD %", fontsize=9)
ax2.grid(True, alpha=0.12)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

# Panel 3: Monthly P&L
ax3 = fig.add_subplot(gs[2])
x = np.arange(len(mo_labels))
colors = ["#27ae60" if r >= 0 else "#c0392b" for r in mo_rets]
ax3.bar(x, mo_rets, width=0.75, color=colors, alpha=0.78, linewidth=0)
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9)
ax3.grid(True, alpha=0.12, axis="y")

step = max(1, len(mo_labels) // 20)
tick_pos = list(range(0, len(mo_labels), step))
ax3.set_xticks(tick_pos)
ax3.set_xticklabels(
    [mo_labels[i] for i in tick_pos], rotation=0, ha="center", fontsize=7,
)
ax3.text(
    0.015, 0.94,
    f"Win {pos_months}/{len(mo_labels)} ({pos_months/len(mo_labels)*100:.0f}%)    "
    f"Avg +{avg_win:.2f}%    Avg −{abs(avg_loss):.2f}%",
    transform=ax3.transAxes, fontsize=8.5, va="top", color="#555",
)

# X-axis info bar
yr_str = "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yearly.items()))
info = (
    f"Vol {ann_vol*100:.1f}%    Gross {avg_gross:.3f} (eff {avg_eff_gross:.3f})    "
    f"Turnover/h {avg_turnover:.5f}    {days}d    "
    f"dd_scale avg {avg_dd_scale:.3f}    "
    f"Soft⇩{soft_active_steps}    Hard⇩{len(hard_stop_events)}    ║    "
    f"{yr_str}"
)
ax3.set_xlabel(info, fontsize=7, fontfamily="monospace", color="#555", labelpad=2)
fig.autofmt_xdate()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
plt.close(fig)
buf.seek(0)

chart_path = OUT_DIR / f"v16a_1x_1k_dd_protected_{RUN_TAG}.png"
chart_path.write_bytes(buf.read())

print(f"\n  Chart: {chart_path.name}  ({chart_path.stat().st_size/1024:.0f}KB  {FIG_W*150}×{FIG_H*150})")
print(f"  Metrics: {metrics_path.name}")
print(f"Done. Total elapsed: {time.time() - t0:.1f}s")
