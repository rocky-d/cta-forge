#!/usr/bin/env python3
"""v16a 19 vs 20 symbols (+ZEC) backtest comparison — BUILD + RUN + CHART."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "services/executor/src"))
sys.path.insert(0, str(PROJ / "libs/core/src"))
sys.path.insert(0, str(PROJ / "services/data-service/src"))
sys.path.insert(0, str(PROJ / "services/report-service/src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = str(PROJ / "data")
OUT_DIR = PROJ / "backtest-results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from executor.signal_pipeline import DEFAULT_SYMBOLS
import executor.profiles.v16a_badscore_overlay as v16a
from executor.portfolio_backtest import calculate_hourly_metrics, run_target_weight_backtest
from executor.profiles.v16a_badscore_overlay import INITIAL_EQUITY, build_v16a_target_set

_SYMBOLS_19 = list(DEFAULT_SYMBOLS)

def run(label, use_zec):
    if not use_zec:
        DEFAULT_SYMBOLS.clear()
        DEFAULT_SYMBOLS.extend(_SYMBOLS_19)
        v16a.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS
    print(f"\n>>> {label}")
    ts = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0)
    res = run_target_weight_backtest(ts.timeline, ts.returns, ts.target_weights, initial_equity=INITIAL_EQUITY)
    m = calculate_hourly_metrics(res.returns, initial_equity=INITIAL_EQUITY)
    down = res.returns[res.returns < 0]
    dvol = np.std(down) * np.sqrt(365 * 24) if len(down) else 0.0
    m["sortino"] = m["ann_return"] / dvol if dvol > 1e-12 else 0.0
    m["calmar"] = m["ann_return"] / m["max_dd"] if m["max_dd"] > 1e-12 else 0.0
    m["total_return"] = m["return"]  # alias
    m["avg_gross"] = float(np.mean(np.sum(np.abs(ts.target_weights), axis=1)))
    m["avg_turnover"] = float(np.mean(res.turnover))
    m["days"] = (ts.timeline[-1] - ts.timeline[0]).days
    curve = list(zip(ts.timeline, m["equity"].tolist()))
    print(f"  Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_return']*100:+.1f}%  MaxDD={m['max_dd']*100:.1f}%  Sortino={m['sortino']:.3f}  Calmar={m['calmar']:.3f}")
    return {"label": label, "curve": curve, "equity": m["equity"], "returns": res.returns, "metrics": m}

t0 = time.time()
baseline = run("19-symbol Baseline", False)

DEFAULT_SYMBOLS.clear()
DEFAULT_SYMBOLS.extend(_SYMBOLS_19 + ["ZECUSDT"])
v16a.DEFAULT_SYMBOLS = DEFAULT_SYMBOLS
with_zec = run("20-symbol +ZEC", True)

b_ts = [e[0] for e in baseline["curve"]]
b_eq = np.array([e[1] for e in baseline["curve"]])
z_ts = [e[0] for e in with_zec["curve"]]
z_eq = np.array([e[1] for e in with_zec["curve"]])

cs = max(b_ts[0], z_ts[0])
ce = min(b_ts[-1], z_ts[-1])
bs = next(i for i, t in enumerate(b_ts) if t >= cs)
zs = next(i for i, t in enumerate(z_ts) if t >= cs)
try: be = next(i for i, t in enumerate(b_ts) if t > ce)
except StopIteration: be = len(b_ts)
try: ze = next(i for i, t in enumerate(z_ts) if t > ce)
except StopIteration: ze = len(z_ts)

b_t, b_e = b_ts[bs:be], b_eq[bs:be]
z_t, z_e = z_ts[zs:ze], z_eq[zs:ze]
bm, zm = baseline["metrics"], with_zec["metrics"]

def dd(eq):
    rm = np.maximum.accumulate(eq)
    return (rm - eq) / rm * 100

def mb(tl, eq):
    mo = {}
    for i in range(1, len(eq)):
        k = tl[i].strftime("%Y-%m")
        if k not in mo: mo[k] = {"s": eq[i-1]}
        mo[k]["e"] = eq[i]
    lbs = list(mo.keys())
    rts = [(mo[m]["e"] - mo[m]["s"]) / mo[m]["s"] * 100 for m in lbs]
    return lbs, rts

def yr(tl, eq):
    r = {}
    for y in sorted({t.year for t in tl}):
        ii = [i for i, t in enumerate(tl) if t.year == y]
        if ii: r[str(y)] = float((eq[ii[-1]] / eq[ii[0]] - 1) * 100)
    return r

# ── CHART ──
fig, axes = plt.subplots(3, 1, figsize=(24, 12), height_ratios=[3.5, 1, 1.5],
                         gridspec_kw={"hspace": 0.15})
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.06, right=0.94, top=0.93, hspace=0.15)

ax1 = axes[0]
ax1.plot(b_t, b_e, lw=1.5, color="#3498db",
         label=f"19 symbols  (Sharpe {bm['sharpe']:.2f}  AnnRet {bm['ann_return']*100:+.1f}%  MaxDD {bm['max_dd']*100:.1f}%)")
ax1.plot(z_t, z_e, lw=1.5, color="#e67e22",
         label=f"20 symbols +ZEC  (Sharpe {zm['sharpe']:.2f}  AnnRet {zm['ann_return']*100:+.1f}%  MaxDD {zm['max_dd']*100:.1f}%)")
ax1.axhline(y=INITIAL_EQUITY, color="#7f8c8d", ls="--", lw=0.5, alpha=0.4)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
ax1.grid(True, alpha=0.15)
ax1.set_title("v16a Badscore Overlay — 19 vs 20 Symbols (+ZEC)\n"
              f"Live config: core_phase=2h, gross_cap=4.0, 50/50 v10g/overlay, ${INITIAL_EQUITY:,.0f} initial",
              fontsize=12, fontweight="bold", pad=10)

ax2 = axes[1]
b_dd, z_dd = dd(b_e), dd(z_e)
ax2.fill_between(b_t, 0, -b_dd, color="#3498db", alpha=0.3, label=f"19s MaxDD {bm['max_dd']*100:.1f}%")
ax2.plot(z_t, -z_dd, lw=1.2, color="#e67e22", alpha=0.8, label=f"20s+ZEC MaxDD {zm['max_dd']*100:.1f}%")
ax2.set_ylabel("DD %", fontsize=9)
ax2.legend(loc="lower left", fontsize=8, framealpha=0.8)
ax2.grid(True, alpha=0.15)

ax3 = axes[2]
b_mo, b_ret = mb(b_t, b_e)
z_mo, z_ret = mb(z_t, z_e)
x = np.arange(len(b_mo))
w = 0.4
bc = ["#27ae60" if r >= 0 else "#c0392b" for r in b_ret]
zc = ["#2ecc71" if r >= 0 else "#e74c3c" for r in z_ret]
ax3.bar(x - w/2, b_ret, w, color=bc, alpha=0.7, label="19 symbols")
ax3.bar(x + w/2, z_ret, w, color=zc, alpha=0.7, label="20 symbols +ZEC")
ax3.axhline(y=0, color="black", lw=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
ax3.grid(True, alpha=0.15, axis="y")
st = max(1, len(b_mo) // 24)
ax3.set_xticks(range(0, len(b_mo), st))
ax3.set_xticklabels([b_mo[i] for i in range(0, len(b_mo), st)], rotation=45, ha="right", fontsize=6)
p19 = sum(1 for r in b_ret if r > 0)
p20 = sum(1 for r in z_ret if r > 0)
ax3.text(0.02, 0.95, f"Positive: 19s→{p19}/{len(b_ret)} ({p19/len(b_ret)*100:.0f}%)  |  20s+ZEC→{p20}/{len(z_ret)} ({p20/len(z_ret)*100:.0f}%)",
         transform=ax3.transAxes, fontsize=8, va="top")

for a in axes[:2]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate()

# Delta footer
ds = zm["sharpe"] - bm["sharpe"]
da = zm["ann_return"] - bm["ann_return"]
dd_ = zm["max_dd"] - bm["max_dd"]
dso = zm["sortino"] - bm["sortino"]
dc = zm["calmar"] - bm["calmar"]
fig.text(0.5, 0.01, f"Δ Sharpe: {ds:+.3f}  |  Δ AnnRet: {da*100:+.2f}%  |  Δ MaxDD: {dd_*100:+.2f}%  |  Δ Sortino: {dso:+.3f}  |  Δ Calmar: {dc:+.3f}",
         ha="center", fontsize=8, family="monospace", color="#555")

# Metrics xlabel
parts = []
for lbl, m in [("19s", bm), ("20s+ZEC", zm)]:
    parts.append(f"{lbl}: Ret {m['total_return']*100:+.1f}%  Ann {m['ann_return']*100:+.1f}%  Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  MaxDD {m['max_dd']*100:.1f}%  Calmar {m['calmar']:.2f}  Gross {m['avg_gross']:.3f}  Turn/h {m['avg_turnover']:.4f}  {m['days']}d")
yr19 = yr(b_t, b_e)
yr20 = yr(z_t, z_e)
yrs = f"19s yearly: {'  '.join(f'{y}: {r:+.1f}%' for y,r in sorted(yr19.items()))}   |   20s yearly: {'  '.join(f'{y}: {r:+.1f}%' for y,r in sorted(yr20.items()))}"
axes[2].set_xlabel(f"{parts[0]}\n{parts[1]}\n{yrs}", fontsize=6, family="monospace", color="#2c3e50", labelpad=4)

plt.tight_layout(rect=[0, 0.04, 1, 0.99])
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
buf.seek(0)

png = OUT_DIR / "backtest_v16a_19vs20_zec.png"
png.write_bytes(buf.read())
print(f"\nChart: {png}")

# JSON
summary = {
    "live_config": {"core_phase_hours": 2, "gross_cap": 4.0, "v10g_alloc": 0.5, "overlay_alloc": 0.5, "initial_equity": INITIAL_EQUITY},
    "19_symbols": {k: bm[k] for k in ["sharpe","total_return","ann_return","max_dd","sortino","calmar","avg_gross","avg_turnover","days"]},
    "20_symbols_zec": {k: zm[k] for k in ["sharpe","total_return","ann_return","max_dd","sortino","calmar","avg_gross","avg_turnover","days"]},
    "delta": {"sharpe": ds, "ann_return": da, "max_dd": dd_, "sortino": dso, "calmar": dc},
}
with open(OUT_DIR / "metrics_v16a_zec_comparison.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# Print summary
print("\n" + "=" * 65)
print("SUMMARY: 19 vs 20 symbols (+ZEC)")
print("=" * 65)
for k, lbl in [("sharpe","Sharpe"),("total_return","Total"),("ann_return","Ann"),
               ("max_dd","MaxDD"),("sortino","Sortino"),("calmar","Calmar")]:
    bv, zv = bm[k], zm[k]
    dv = zv - bv
    fmt = f"  {lbl:10s}  19s={bv*100:+.2f}%   20s={zv*100:+.2f}%   Δ={dv*100:+.2f}%" if "return" in k or "max_dd" in k else f"  {lbl:10s}  19s={bv:.3f}     20s={zv:.3f}      Δ={dv:+.3f}"
    print(fmt)
print(f"  {'Gross':10s}  19s={bm['avg_gross']:.3f}     20s={zm['avg_gross']:.3f}      Δ={zm['avg_gross']-bm['avg_gross']:+.3f}")
print(f"  {'Turn/h':10s}  19s={bm['avg_turnover']:.4f}   20s={zm['avg_turnover']:.4f}    Δ={zm['avg_turnover']-bm['avg_turnover']:+.4f}")
print(f"\nDone in {time.time()-t0:.0f}s")
