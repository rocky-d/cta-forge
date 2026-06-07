#!/usr/bin/env python3
"""v16a 5x — No DD vs DD soft=15% vs DD soft=20% (all hard=30%)."""

from __future__ import annotations

import io, sys, time
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
    "font.family": "sans-serif", "grid.alpha": 0.10,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "services/executor/src"))
sys.path.insert(0, str(PROJECT_ROOT / "libs/core/src"))
sys.path.insert(0, str(PROJECT_ROOT / "services/data-service/src"))

from executor.targeting import normalize_gross  # noqa: E402
from executor.profiles.v16a_badscore_overlay import build_v16a_target_set  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "backtest-results"; OUT_DIR.mkdir(parents=True, exist_ok=True)

EQ = 1000.0; GC = 4.0; TS = 5.0; HD = 0.30; CP = 2; FEE = 0.0004
RUN = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

print(f"=== v16a 5x — No DD vs Soft=15% vs Soft=20% (all Hard=30%) ===")

target_set = build_v16a_target_set(str(DATA_DIR), core_phase_hours=CP, gross_cap=1.0)
ret = target_set.returns; rw = target_set.target_weights
tl = target_set.timeline; syms = target_set.symbols; N = len(tl)

def _sc(raw_step, dd_s=1.0):
    s = np.nan_to_num(raw_step, nan=0.0) * TS * dd_s
    d = normalize_gross({syms[i]: float(s[i]) for i in range(len(syms))}, gross_cap=GC)
    return np.array([d.get(s, 0.0) for s in syms])

def run_backtest(label, soft, hard):
    eq = EQ; peak = EQ; eq_h = [eq]; prev = np.zeros(len(syms))
    pnl_arr = np.zeros(N - 1); to_arr = np.zeros(N - 1)
    sn = 0; hn = 0; ds_vals = []

    for t in range(N - 1):
        dd = (peak - eq) / peak if peak > 0 else 0.0

        if hard > 0 and dd >= hard:
            cur = np.zeros(len(syms)); ds = 0.0; hn += 1
        else:
            ds = 1.0
            if soft < hard and dd >= soft:
                ds = max(0.0, 1.0 - (dd - soft) / (hard - soft)); sn += 1
            cur = _sc(rw[t], dd_s=ds)

        ds_vals.append(ds)
        to_arr[t] = float(np.sum(np.abs(cur - prev)))
        pnl_arr[t] = float(np.sum(cur * np.nan_to_num(ret[t + 1], nan=0.0)) - to_arr[t] * FEE)
        eq *= (1.0 + pnl_arr[t])
        peak = max(peak, eq)
        eq_h.append(eq)
        prev = cur.copy()

    eq_a = np.array(eq_h)
    tr = eq_a[-1] / EQ - 1.0
    days = (tl[-1] - tl[0]).days
    ar = float((1.0 + tr) ** (365.25 / days) - 1.0) if days > 0 else 0.0
    av = float(np.std(pnl_arr) * np.sqrt(365 * 24))
    sh = ar / av if av > 1e-12 else 0.0
    md = 0.0; pv = EQ
    for v in eq_a:
        pv = max(pv, v); md = max(md, (pv - v) / pv if pv > 0 else 0.0)
    dn = pnl_arr[pnl_arr < 0]
    dv = float(np.std(dn) * np.sqrt(365 * 24)) if len(dn) > 0 else 0.0
    so = ar / dv if dv > 1e-12 else 0.0
    ca = ar / md if md > 1e-12 else 0.0
    a_to = float(np.mean(to_arr))
    ads = float(np.mean(ds_vals)) if ds_vals else 1.0

    print(f"  [{label}] Sharpe {sh:.4f} Ann {ar*100:.2f}% Total {tr*100:.2f}% MaxDD {md*100:.2f}% "
          f"Sortino {so:.4f} Calmar {ca:.4f} Vol {av*100:.2f}% To/h {a_to:.5f} "
          + (f"Soft⇩{sn} Hard⇩{hn} DSavg {ads:.4f}" if soft < hard else ""))

    return dict(equity=eq_a, sharpe=sh, ann_ret=ar, total_ret=tr, max_dd=md,
                sortino=so, calmar=ca, volatility=av, avg_turnover=a_to,
                soft_n=sn, hard_n=hn, avg_ds=ads, label=label)

t0 = time.time()
r_none = run_backtest("No DD", soft=1.0, hard=2.0)
r_15   = run_backtest("Soft=15% Hard=30%", soft=0.15, hard=0.30)
r_20   = run_backtest("Soft=20% Hard=30%", soft=0.20, hard=0.30)
results = [r_none, r_15, r_20]

print(f"\n  Time: {time.time()-t0:.1f}s")

# ── Three-panel chart 16:9 ──
FW, FH = 16, 9
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 0.9, 1.6],
                      hspace=0.08, top=0.93, bottom=0.10, left=0.06, right=0.97)

COLORS = {"No DD": "#bdc3c7", "Soft=15% Hard=30%": "#1a5276", "Soft=20% Hard=30%": "#27ae60"}
STYLES = {"No DD": (0.9, "dashed"), "Soft=15% Hard=30%": (1.4, "solid"), "Soft=20% Hard=30%": (1.4, "solid")}

# Panel 1: Equity
ax1 = fig.add_subplot(gs[0])
for r in results:
    lw, ls = STYLES[r["label"]]
    ax1.plot(tl[:len(r["equity"])], r["equity"], linewidth=lw, linestyle=ls,
             color=COLORS[r["label"]],
             label=f'{r["label"]} (Sh {r["sharpe"]:.2f})')
ax1.axhline(y=EQ, color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.4)
ax1.set_ylabel("Equity ($)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.1f}K" if abs(x)>=1000 else f"${x:.0f}"))
ax1.grid(True, alpha=0.08); ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
sd = tl[0].strftime("%Y-%m-%d"); ed = tl[-1].strftime("%Y-%m-%d")
ax1.set_title(f"v16a 5× Levered — DD Protection Variants  |  {sd} → {ed}  |  ${EQ:,.0f} initial  |  {len(syms)} symbols",
              fontsize=13, fontweight="bold", pad=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# Panel 2: Drawdown
ax2 = fig.add_subplot(gs[1], sharex=ax1)
for r in results:
    dd_a = (np.maximum.accumulate(r["equity"]) - r["equity"]) / np.maximum.accumulate(r["equity"]) * 100
    lw, ls = STYLES[r["label"]]
    ax2.plot(tl[:len(r["equity"])], -dd_a, linewidth=lw, linestyle=ls,
             color=COLORS[r["label"]], alpha=0.85,
             label=f'{r["label"]} (MaxDD {r["max_dd"]*100:.1f}%)')
ax2.axhline(y=-0.15*100, color="#f39c12", linestyle="--", linewidth=0.6, alpha=0.4, label="Soft 15%")
ax2.axhline(y=-0.20*100, color="#27ae60", linestyle="--", linewidth=0.6, alpha=0.4, label="Soft 20%")
ax2.axhline(y=-0.30*100, color="#c0392b", linestyle=":", linewidth=0.6, alpha=0.4, label="Hard 30%")
ax2.legend(loc="lower left", fontsize=7, framealpha=0.7, ncol=2)
ax2.set_ylabel("DD %", fontsize=9); ax2.grid(True, alpha=0.08)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

# Panel 3: Monthly P&L
ax3 = fig.add_subplot(gs[2])
ml = []
for i in range(1, len(r_none["equity"])):
    k = tl[i-1].strftime("%Y-%m")
    if k not in [m[0] for m in ml]:
        ml.append((k, r_none["equity"][i-1], r_none["equity"][i]))
    else:
        idx = [j for j, m in enumerate(ml) if m[0]==k][0]
        ml[idx] = (k, ml[idx][1], r_none["equity"][i])

x = np.arange(len(ml)); w = 0.25
for idx, r in enumerate(results):
    mrs = []; m_end = {}
    for i in range(1, len(r["equity"])):
        k = tl[i-1].strftime("%Y-%m"); m_end[k] = r["equity"][i]
    m_start = {}
    for i in range(1, len(r["equity"])):
        k = tl[i-1].strftime("%Y-%m")
        if k not in m_start: m_start[k] = r["equity"][i-1]
    for m_lab, _, _ in ml:
        if m_lab in m_start:
            mrs.append((m_end.get(m_lab, m_start[m_lab]) - m_start[m_lab]) / m_start[m_lab] * 100)
        else:
            mrs.append(0.0)

    bc = ["#bdc3c7" if rv>=0 else "#95a5a6" for rv in mrs] if idx==0 else \
         ["#1a5276" if rv>=0 else "#c0392b" for rv in mrs] if idx==1 else \
         ["#27ae60" if rv>=0 else "#c0392b" for rv in mrs]
    ax3.bar(x + (idx-1)*w, mrs, w, color=bc, alpha=0.55 if idx==0 else 0.85, linewidth=0)

ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Monthly Return %", fontsize=9); ax3.grid(True, alpha=0.08, axis="y")
sp = max(1, len(ml)//20)
ax3.set_xticks(range(0, len(ml), sp))
ax3.set_xticklabels([m[0] for m in ml][::sp], rotation=0, ha="center", fontsize=7)

parts = []
for r in results:
    parts.append(f'{r["label"]}: Sh{r["sharpe"]:.2f} Ann{r["ann_ret"]*100:.1f}% '
                 f'MDD{r["max_dd"]*100:.1f}% Ca{r["calmar"]:.2f} Vol{r["volatility"]*100:.1f}%')
ax3.set_xlabel(" | ".join(parts), fontsize=6.5, fontfamily="monospace", color="#555", labelpad=2)
fig.autofmt_xdate()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
plt.close(fig); buf.seek(0)
cp = OUT_DIR / f"v16a_5x_dd_3way_{RUN}.png"
cp.write_bytes(buf.read())
print(f"\n  Chart: {cp.name} ({cp.stat().st_size/1024:.0f}KB)  Done.")
