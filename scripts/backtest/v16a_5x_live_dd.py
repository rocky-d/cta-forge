#!/usr/bin/env python3
"""v16a 5x levered backtest — exact live config with DD protection.

target_scale=5.0, gross_cap=4.0, DD soft=15% hard=30%.
Matches live: raw weights × target_scale → cap to gross_cap → DD-scale.
"""

from __future__ import annotations

import io, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.family": "sans-serif", "grid.alpha": 0.12,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "services/executor/src"))
sys.path.insert(0, str(PROJECT_ROOT / "libs/core/src"))
sys.path.insert(0, str(PROJECT_ROOT / "services/data-service/src"))

from executor.targeting import normalize_gross  # noqa: E402
from executor.profiles.v16a_badscore_overlay import build_v16a_target_set  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "backtest-results"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── exact live config ──
INITIAL_EQUITY = 1000.0
GROSS_CAP = 4.0         # TARGET_GROSS_CAP
TARGET_SCALE = 5.0       # TARGET_SCALE
SOFT_DD = 0.15
HARD_DD = 0.30
CORE_PHASE = 2
FEE = 0.0004
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

print(f"=== v16a 5x levered (exact live) ===")
print(f"  equity=${INITIAL_EQUITY:,.0f}  cap={GROSS_CAP}  scale={TARGET_SCALE}")
print(f"  DD soft={SOFT_DD*100:.0f}% hard={HARD_DD*100:.0f}%  phase={CORE_PHASE}h")
t0 = time.time()

target_set = build_v16a_target_set(str(DATA_DIR), core_phase_hours=CORE_PHASE, gross_cap=1.0)
returns = target_set.returns
raw_w = target_set.target_weights
tl = target_set.timeline
syms = target_set.symbols
N = len(tl)

eq = INITIAL_EQUITY
peak = INITIAL_EQUITY
eq_hist = [eq]
pnl = np.zeros(N - 1)
to_hist = np.zeros(N - 1)
prev_w = np.zeros(len(syms))
dd_scales: list[float] = []
soft_n = 0
hard_events: list[str] = []

def _scale_and_cap(raw_step: np.ndarray, cap: float) -> np.ndarray:
    scaled = raw_step * TARGET_SCALE
    d = normalize_gross({syms[i]: float(scaled[i]) for i in range(len(syms))}, gross_cap=cap)
    return np.array([d.get(s, 0.0) for s in syms])

for t in range(N - 1):
    dd = (peak - eq) / peak if peak > 0 else 0.0

    if dd >= HARD_DD:
        if np.any(np.abs(prev_w) > 1e-12):
            hard_events.append(f"{tl[t]} DD={dd*100:.1f}% → flatten")
        cur_w = np.zeros(len(syms))
        ds = 0.0
    else:
        # Normal: scale → cap to gross_cap
        cur_w = _scale_and_cap(np.nan_to_num(raw_w[t], nan=0.0), GROSS_CAP)

        if dd >= SOFT_DD and HARD_DD > SOFT_DD:
            ds = max(0.0, 1.0 - (dd - SOFT_DD) / (HARD_DD - SOFT_DD))
            soft_n += 1
            eff_cap = GROSS_CAP * ds
            cur_w = _scale_and_cap(np.nan_to_num(raw_w[t], nan=0.0), eff_cap)
        else:
            ds = 1.0

    dd_scales.append(ds)
    to_hist[t] = float(np.sum(np.abs(cur_w - prev_w)))
    pnl[t] = float(np.sum(cur_w * np.nan_to_num(returns[t + 1], nan=0.0)) - to_hist[t] * FEE)
    eq *= (1.0 + pnl[t])
    peak = max(peak, eq)
    eq_hist.append(eq)
    prev_w = cur_w.copy()

eq_a = np.array(eq_hist)

# ── metrics ──
total_ret = eq_a[-1] / INITIAL_EQUITY - 1.0
days = (tl[-1] - tl[0]).days
ann_ret = float((1.0 + total_ret) ** (365.25 / days) - 1.0) if days > 0 else 0.0
ann_vol = float(np.std(pnl) * np.sqrt(365 * 24))
sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
max_dd = 0.0; pv = INITIAL_EQUITY
for v in eq_a:
    pv = max(pv, v)
    max_dd = max(max_dd, (pv - v) / pv if pv > 0 else 0.0)
down = pnl[pnl < 0]
down_vol = float(np.std(down) * np.sqrt(365 * 24)) if len(down) > 0 else 0.0
sortino = ann_ret / down_vol if down_vol > 1e-12 else 0.0
calmar = ann_ret / max_dd if max_dd > 1e-12 else 0.0
avg_raw_gross = float(np.mean(np.sum(np.abs(raw_w), axis=1)))
eff_grosses = []
for t in range(N - 1):
    dd = (np.max(eq_hist[:t+1]) - eq_hist[t]) / np.max(eq_hist[:t+1]) if np.max(eq_hist[:t+1]) > 0 else 0.0
    rw = np.nan_to_num(raw_w[t], nan=0.0)
    sw = rw * TARGET_SCALE
    if dd >= HARD_DD:
        eg = 0.0
    elif dd >= SOFT_DD:
        ds = max(0.0, 1.0 - (dd - SOFT_DD) / (HARD_DD - SOFT_DD))
        d = normalize_gross({syms[i]: float(sw[i]) for i in range(len(syms))}, gross_cap=GROSS_CAP * ds)
        eg = float(np.sum(np.abs([d.get(s, 0.0) for s in syms])))
    else:
        d = normalize_gross({syms[i]: float(sw[i]) for i in range(len(syms))}, gross_cap=GROSS_CAP)
        eg = float(np.sum(np.abs([d.get(s, 0.0) for s in syms])))
    eff_grosses.append(eg)
avg_eff_gross = float(np.mean(eff_grosses))
avg_turnover = float(np.mean(to_hist))
avg_ds = float(np.mean(dd_scales))

sd, ed = tl[0].strftime("%Y-%m-%d"), tl[-1].strftime("%Y-%m-%d")
ann_ret_pct = ann_ret * 100; total_ret_pct = total_ret * 100
max_dd_pct = max_dd * 100; ann_vol_pct = ann_vol * 100

print(f"\n  {len(syms)} symbols  {sd} → {ed} ({days}d)")
print(f"  Sharpe {sharpe:.4f}  Ann {ann_ret_pct:.2f}%  Total {total_ret_pct:.2f}%")
print(f"  MaxDD {max_dd_pct:.2f}%  Sortino {sortino:.4f}  Calmar {calmar:.4f}")
print(f"  Vol {ann_vol_pct:.2f}%  Gross raw {avg_raw_gross:.3f} eff {avg_eff_gross:.3f}")
print(f"  Turnover/h {avg_turnover:.5f}  DD scale avg {avg_ds:.4f}")
print(f"  Soft⇩{soft_n}/{N-1}  Hard⇩{len(hard_events)}")
if hard_events:
    for e in hard_events[:5]: print(f"    {e}")
    if len(hard_events) > 5: print(f"    ... {len(hard_events)-5} more")
print(f"  Time: {time.time()-t0:.1f}s")

# ── save JSON ──
json.dump({
    "run_tag": RUN_TAG,
    "config": {"initial_equity": INITIAL_EQUITY, "gross_cap": GROSS_CAP,
               "target_scale": TARGET_SCALE, "core_phase_hours": CORE_PHASE,
               "leverage": "5x", "soft_dd_limit": SOFT_DD, "hard_dd_limit": HARD_DD, "fee": FEE},
    "period": {"start": sd, "end": ed, "days": days},
    "symbols": sorted(syms), "symbol_count": len(syms),
    "dd_protection": {"soft_active_steps": soft_n, "hard_stop_events": hard_events, "avg_dd_scale": avg_ds},
    "metrics": {"sharpe": sharpe, "ann_return": ann_ret, "total_return": total_ret,
                "max_dd": max_dd, "sortino": sortino, "calmar": calmar,
                "volatility": ann_vol, "avg_raw_gross": avg_raw_gross,
                "avg_eff_gross": avg_eff_gross, "avg_turnover_per_hour": avg_turnover},
}, open(OUT_DIR / f"v16a_5x_live_dd_{RUN_TAG}.json", "w"), indent=2)

# ── three-panel chart 16:9 ──
FW, FH = 16, 9
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 0.9, 1.6],
                      hspace=0.08, top=0.93, bottom=0.10, left=0.06, right=0.97)

dd_pct = (np.maximum.accumulate(eq_a) - eq_a) / np.maximum.accumulate(eq_a) * 100

# Monthly P&L
ms, me = {}, {}
for i in range(1, len(eq_a)):
    k = tl[i-1].strftime("%Y-%m")
    if k not in ms: ms[k] = eq_a[i-1]
    me[k] = eq_a[i]
ml = list(ms.keys())
mr = [(me[m] - ms[m]) / ms[m] * 100 for m in ml]
pos = sum(1 for r in mr if r > 0)
aw = float(np.mean([r for r in mr if r > 0]))
al = float(np.mean([r for r in mr if r < 0]))

yearly = {}
for y in sorted({t.year for t in tl}):
    idxs = [i for i, t in enumerate(tl) if t.year == y]
    if idxs and eq_a[idxs[0]] > 0:
        yearly[str(y)] = float((eq_a[idxs[-1]] / eq_a[idxs[0]] - 1) * 100)

ax1 = fig.add_subplot(gs[0])
ax1.plot(tl[:len(eq_a)], eq_a, linewidth=1.2, color="#1a5276")
ax1.axhline(y=INITIAL_EQUITY, color="#95a5a6", linestyle="--", linewidth=0.7, alpha=0.5)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.1f}K" if abs(x)>=1000 else f"${x:.0f}"))
ax1.grid(True, alpha=0.12)
ax1.set_title(f"v16a — 5× Levered (exact live config)  |  {sd} → {ed}  |  ${INITIAL_EQUITY:,.0f} initial  |  {len(syms)} symbols",
              fontsize=13, fontweight="bold", pad=8)
box = (f"Sharpe {sharpe:.2f}    Ann {ann_ret_pct:.1f}%    Total {total_ret_pct:.1f}%    "
       f"MaxDD {max_dd_pct:.1f}%    Sortino {sortino:.2f}    Calmar {calmar:.2f}")
ax1.text(0.02, 0.96, box, transform=ax1.transAxes, fontsize=9, va="top",
         fontfamily="monospace", color="#2c3e50",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.92))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(tl[:len(eq_a)], 0, -dd_pct, color="#c0392b", alpha=0.2, linewidth=0)
ax2.plot(tl[:len(eq_a)], -dd_pct, linewidth=0.7, color="#c0392b")
ax2.axhline(y=-SOFT_DD*100, color="#f39c12", linestyle="--", linewidth=0.6, alpha=0.5, label=f"Soft {SOFT_DD*100:.0f}%")
ax2.axhline(y=-HARD_DD*100, color="#c0392b", linestyle="--", linewidth=0.6, alpha=0.5, label=f"Hard {HARD_DD*100:.0f}%")
ax2.legend(loc="lower left", fontsize=7, framealpha=0.7)
ax2.set_ylabel("DD %", fontsize=9); ax2.grid(True, alpha=0.12)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

ax3 = fig.add_subplot(gs[2])
x = np.arange(len(ml))
ax3.bar(x, mr, width=0.75, color=["#27ae60" if r>=0 else "#c0392b" for r in mr], alpha=0.78, linewidth=0)
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9); ax3.grid(True, alpha=0.12, axis="y")
sp = max(1, len(ml)//20)
ax3.set_xticks(range(0, len(ml), sp))
ax3.set_xticklabels([ml[i] for i in range(0, len(ml), sp)], rotation=0, ha="center", fontsize=7)
ax3.text(0.015, 0.94, f"Win {pos}/{len(ml)} ({pos/len(ml)*100:.0f}%)    Avg +{aw:.2f}%    Avg −{abs(al):.2f}%",
         transform=ax3.transAxes, fontsize=8.5, va="top", color="#555")

yr_str = "  ".join(f"{y}: {r:+.1f}%" for y, r in sorted(yearly.items()))
info = (f"Vol {ann_vol_pct:.1f}%    Gross raw {avg_raw_gross:.3f} eff {avg_eff_gross:.3f}    "
        f"Turnover/h {avg_turnover:.5f}    {days}d    DD scale {avg_ds:.3f}    "
        f"Soft⇩{soft_n} Hard⇩{len(hard_events)}    ║    {yr_str}")
ax3.set_xlabel(info, fontsize=7, fontfamily="monospace", color="#555", labelpad=2)
fig.autofmt_xdate()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
plt.close(fig); buf.seek(0)

cp = OUT_DIR / f"v16a_5x_live_dd_{RUN_TAG}.png"
cp.write_bytes(buf.read())
print(f"\n  Chart: {cp.name} ({cp.stat().st_size/1024:.0f}KB)")
print(f"  Equiv unlevered: total_ret={((eq_a[-1]/INITIAL_EQUITY)**(1/5)-1)*100:.1f}%")
print(f"Done. ({time.time()-t0:.1f}s)")
