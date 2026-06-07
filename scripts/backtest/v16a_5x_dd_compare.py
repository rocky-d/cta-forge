#!/usr/bin/env python3
"""v16a 5x — DD protection vs No protection comparison."""

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
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
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

INITIAL_EQUITY = 1000.0
GROSS_CAP = 4.0; TARGET_SCALE = 5.0
SOFT_DD = 0.15; HARD_DD = 0.30
CORE_PHASE = 2; FEE = 0.0004
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

print(f"=== v16a 5x DD protection vs No protection ===")
print(f"  equity=${INITIAL_EQUITY:,.0f} cap={GROSS_CAP} scale={TARGET_SCALE}")
print(f"  DD soft={SOFT_DD*100:.0f}% hard={HARD_DD*100:.0f}%")

target_set = build_v16a_target_set(str(DATA_DIR), core_phase_hours=CORE_PHASE, gross_cap=1.0)
returns = target_set.returns; raw_w = target_set.target_weights
tl = target_set.timeline; syms = target_set.symbols; N = len(tl)

def _scale_and_cap(raw_step, cap, dd_s=1.0):
    s = np.nan_to_num(raw_step, nan=0.0) * TARGET_SCALE * dd_s
    d = normalize_gross({syms[i]: float(s[i]) for i in range(len(syms))}, gross_cap=cap)
    return np.array([d.get(s, 0.0) for s in syms])

def run(name, use_dd):
    eq = INITIAL_EQUITY; peak = INITIAL_EQUITY
    eq_hist = [eq]; prev = np.zeros(len(syms))
    soft_n = 0; hard_n = 0
    to_vals = np.zeros(N - 1)
    pnl_vals = np.zeros(N - 1)
    dd_scales = []

    for t in range(N - 1):
        dd = (peak - eq) / peak if peak > 0 else 0.0

        if use_dd and dd >= HARD_DD:
            cur = np.zeros(len(syms)); ds = 0.0; hard_n += 1
        else:
            ds = 1.0
            if use_dd and dd >= SOFT_DD and HARD_DD > SOFT_DD:
                ds = max(0.0, 1.0 - (dd - SOFT_DD) / (HARD_DD - SOFT_DD))
                soft_n += 1
            cur = _scale_and_cap(raw_w[t], GROSS_CAP, dd_s=ds)

        dd_scales.append(ds)
        to_vals[t] = float(np.sum(np.abs(cur - prev)))
        pnl_vals[t] = float(np.sum(cur * np.nan_to_num(returns[t + 1], nan=0.0)) - to_vals[t] * FEE)
        eq *= (1.0 + pnl_vals[t])
        peak = max(peak, eq)
        eq_hist.append(eq)
        prev = cur.copy()

    eq_a = np.array(eq_hist)
    total_ret = eq_a[-1] / INITIAL_EQUITY - 1.0
    days = (tl[-1] - tl[0]).days
    ann_ret = float((1.0 + total_ret) ** (365.25 / days) - 1.0) if days > 0 else 0.0
    ann_vol = float(np.std(pnl_vals) * np.sqrt(365 * 24))
    shp = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    mdd = 0.0; pv = INITIAL_EQUITY
    for v in eq_a:
        pv = max(pv, v); mdd = max(mdd, (pv - v) / pv if pv > 0 else 0.0)
    dn = pnl_vals[pnl_vals < 0]
    dn_vol = float(np.std(dn) * np.sqrt(365 * 24)) if len(dn) > 0 else 0.0
    sor = ann_ret / dn_vol if dn_vol > 1e-12 else 0.0
    cal = ann_ret / mdd if mdd > 1e-12 else 0.0
    avg_to = float(np.mean(to_vals))
    avg_ds = float(np.mean(dd_scales)) if dd_scales else 1.0

    print(f"\n  [{name}]")
    print(f"    Sharpe {shp:.4f}  Ann {ann_ret*100:.2f}%  Total {total_ret*100:.2f}%")
    print(f"    MaxDD {mdd*100:.2f}%  Sortino {sor:.4f}  Calmar {cal:.4f}")
    print(f"    Vol {ann_vol*100:.2f}%  Turnover/h {avg_to:.5f}")
    if use_dd:
        print(f"    Soft⇩{soft_n} Hard⇩{hard_n}  DD scale avg {avg_ds:.4f}")

    return {
        "equity": eq_a, "pnl": pnl_vals, "sharpe": shp,
        "ann_ret": ann_ret, "total_ret": total_ret, "max_dd": mdd,
        "sortino": sor, "calmar": cal, "volatility": ann_vol,
        "avg_turnover": avg_to, "soft_n": soft_n, "hard_n": hard_n,
        "avg_dd_scale": avg_ds,
    }

t0 = time.time()
r_dd = run("DD protection [soft=15% hard=30%]", use_dd=True)
r_no = run("No protection", use_dd=False)

print(f"\n  Δ metrics (DD − No DD):")
for k in ["sharpe", "ann_ret", "total_ret", "max_dd", "sortino", "calmar"]:
    dv = r_dd[k] - r_no[k]
    pct = f"{dv*100:+.2f}%" if k in ("ann_ret", "total_ret", "max_dd") else f"{dv:+.4f}"
    print(f"    {k}: {pct}")
print(f"  Time: {time.time()-t0:.1f}s")

# ── Three-panel comparison chart 16:9 ──
FW, FH = 16, 9
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 0.9, 1.6],
                      hspace=0.08, top=0.93, bottom=0.10, left=0.06, right=0.97)

eq_dd = r_dd["equity"]; eq_no = r_no["equity"]
dd_dd = (np.maximum.accumulate(eq_dd) - eq_dd) / np.maximum.accumulate(eq_dd) * 100
dd_no = (np.maximum.accumulate(eq_no) - eq_no) / np.maximum.accumulate(eq_no) * 100

# Monthly P&L (use DD-protected for labels — same timeline)
ms, me = {}, {}
for i in range(1, len(eq_dd)):
    k = tl[i-1].strftime("%Y-%m")
    if k not in ms: ms[k] = eq_dd[i-1]
    me[k] = eq_dd[i]
ml = list(ms.keys())
mr_dd = [(me[m] - ms[m]) / ms[m] * 100 for m in ml]

ms2, me2 = {}, {}
for i in range(1, len(eq_no)):
    k = tl[i-1].strftime("%Y-%m")
    if k not in ms2: ms2[k] = eq_no[i-1]
    me2[k] = eq_no[i]
mr_no = [(me2[m] - ms2[m]) / ms2[m] * 100 for m in ml]

# Panel 1: Equity
ax1 = fig.add_subplot(gs[0])
ax1.plot(tl[:len(eq_no)], eq_no, linewidth=1.0, color="#95a5a6", alpha=0.7,
         label=f"No DD (Sharpe {r_no['sharpe']:.2f})")
ax1.plot(tl[:len(eq_dd)], eq_dd, linewidth=1.4, color="#1a5276",
         label=f"DD protection (Sharpe {r_dd['sharpe']:.2f})")
ax1.axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.4)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.1f}K" if abs(x)>=1000 else f"${x:.0f}"))
ax1.grid(True, alpha=0.10)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
sd = tl[0].strftime("%Y-%m-%d"); ed = tl[-1].strftime("%Y-%m-%d"); days = (tl[-1]-tl[0]).days
ax1.set_title(f"v16a 5× Levered — DD protection vs No protection  |  {sd} → {ed}  |  ${INITIAL_EQUITY:,.0f} initial  |  {len(syms)} symbols",
              fontsize=13, fontweight="bold", pad=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# Panel 2: Drawdown
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(tl[:len(eq_no)], 0, -dd_no, color="#95a5a6", alpha=0.15, linewidth=0)
ax2.plot(tl[:len(eq_no)], -dd_no, linewidth=0.6, color="#95a5a6", alpha=0.7,
         label=f"No DD (MaxDD {r_no['max_dd']*100:.1f}%)")
ax2.fill_between(tl[:len(eq_dd)], 0, -dd_dd, color="#c0392b", alpha=0.18, linewidth=0)
ax2.plot(tl[:len(eq_dd)], -dd_dd, linewidth=0.9, color="#c0392b",
         label=f"DD protection (MaxDD {r_dd['max_dd']*100:.1f}%)")
ax2.axhline(y=-SOFT_DD*100, color="#f39c12", linestyle="--", linewidth=0.6, alpha=0.5,
            label=f"Soft {SOFT_DD*100:.0f}%")
ax2.axhline(y=-HARD_DD*100, color="#c0392b", linestyle=":", linewidth=0.6, alpha=0.5,
            label=f"Hard {HARD_DD*100:.0f}%")
ax2.legend(loc="lower left", fontsize=7, framealpha=0.7, ncol=2)
ax2.set_ylabel("DD %", fontsize=9); ax2.grid(True, alpha=0.10)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

# Panel 3: Monthly P&L side-by-side
ax3 = fig.add_subplot(gs[2])
x = np.arange(len(ml)); w = 0.35
ax3.bar(x - w/2, mr_no, w, color=["#27ae60" if r>=0 else "#c0392b" for r in mr_no],
        alpha=0.45, linewidth=0, label="No DD")
ax3.bar(x + w/2, mr_dd, w, color=["#2ecc71" if r>=0 else "#e74c3c" for r in mr_dd],
        alpha=0.85, linewidth=0, label="DD protection")
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9); ax3.grid(True, alpha=0.10, axis="y")
ax3.legend(loc="upper right", fontsize=8, framealpha=0.85)
sp = max(1, len(ml)//20)
ax3.set_xticks(range(0, len(ml), sp))
ax3.set_xticklabels([ml[i] for i in range(0, len(ml), sp)], rotation=0, ha="center", fontsize=7)

# Bottom info bar
yr_str = ""
for y in sorted({t.year for t in tl}):
    for label, eq in [("NoDD", eq_no), ("DD", eq_dd)]:
        idxs = [i for i, t in enumerate(tl) if t.year == y and i < len(eq)]
        if idxs and eq[idxs[0]] > 0:
            r = (eq[min(idxs[-1], len(eq)-1)] / eq[idxs[0]] - 1) * 100
            yr_str += f" {y} {r:+.0f}%"
            break

delta_lines = []
for k, label in [("sharpe", "ΔSharpe"), ("total_ret", "ΔTotalRet"),
                 ("max_dd", "ΔMaxDD"), ("sortino", "ΔSortino"), ("calmar", "ΔCalmar")]:
    dv = r_dd[k] - r_no[k]
    if k in ("total_ret", "max_dd"):
        delta_lines.append(f"{label} {dv*100:+.2f}%")
    else:
        delta_lines.append(f"{label} {dv:+.3f}")

info = ("DD: Vol {:.1f}% | To/h {:.5f} | {} | NoDD: Vol {:.1f}% | To/h {:.5f} || {}".format(
    r_dd["volatility"]*100, r_dd["avg_turnover"],
    " | ".join(delta_lines),
    r_no["volatility"]*100, r_no["avg_turnover"], yr_str.strip()))
ax3.set_xlabel(info, fontsize=6.5, fontfamily="monospace", color="#555", labelpad=2)
fig.autofmt_xdate()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
plt.close(fig); buf.seek(0)
cp = OUT_DIR / f"v16a_5x_dd_vs_nodd_{RUN_TAG}.png"
cp.write_bytes(buf.read())
print(f"\n  Chart: {cp.name} ({cp.stat().st_size/1024:.0f}KB)")
print(f"Done. ({time.time()-t0:.1f}s)")
