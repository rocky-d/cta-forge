"""19-symbol-only three-panel chart — wide, high-res."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from executor.profiles.v16a_badscore_overlay import INITIAL_EQUITY, build_v16a_target_set

DATA_DIR = Path("data")
OUT_DIR = Path("backtest-results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building target set...")
    ts = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0)
    timeline = ts.timeline
    symbols = ts.symbols
    n, s = len(timeline), len(symbols)

    w = np.nan_to_num(ts.target_weights, nan=0.0)
    r = np.nan_to_num(ts.returns, nan=0.0)
    contrib = np.zeros((n, s))
    contrib[1:, :] = w[:-1] * r[1:]
    eq_symbols = np.exp(np.cumsum(np.log1p(contrib), axis=0)) * INITIAL_EQUITY

    colours = plt.cm.tab20(np.linspace(0, 1, s))

    # Wide layout: 5x standard width, 16:6 aspect per panel
    fig, axes = plt.subplots(3, 1, figsize=(32, 18), dpi=200)
    fig.patch.set_facecolor("#fafafa")
    alpha_line = 0.75
    lw = 0.6

    # Panel 1: Equity
    ax = axes[0]; ax.set_facecolor("#fafafa")
    for j in range(s):
        ax.plot(timeline, eq_symbols[:, j], color=colours[j], alpha=alpha_line, lw=lw, label=symbols[j][:-4])
    ax.axhline(INITIAL_EQUITY, color="#aaa", lw=0.5, ls="--")
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.set_title(f"v16a Live Config (phase 2 · cap 4x) — {s} Individual Symbol Equity Curves", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.15, lw=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(labelsize=8)
    ax.legend(
        ncol=4, loc="upper left", fontsize=7, framealpha=0.3,
        columnspacing=0.6, handlelength=1.2, handletextpad=0.3,
    )

    # Panel 2: Drawdown
    ax = axes[1]; ax.set_facecolor("#fafafa")
    for j in range(s):
        p = np.maximum.accumulate(eq_symbols[:, j])
        dd = (eq_symbols[:, j] / p - 1) * 100
        ax.fill_between(timeline, dd, 0, color=colours[j], alpha=0.08, lw=0)
        ax.plot(timeline, dd, color=colours[j], alpha=alpha_line, lw=lw)
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title(f"Individual Symbol Drawdown", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.15, lw=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(labelsize=8)

    # Panel 3: Monthly Returns
    ax = axes[2]; ax.set_facecolor("#fafafa")
    months = sorted({f"{t.year}-{t.month:02d}" for t in timeline}, key=lambda m: tuple(map(int, m.split("-"))))
    # Vectorized monthly: group by month
    monthly_returns = {}
    for i, t in enumerate(timeline):
        key = f"{t.year}-{t.month:02d}"
        if key not in monthly_returns:
            monthly_returns[key] = {"start": i, "end": i}
        monthly_returns[key]["end"] = i
    data = np.zeros((len(months), s))
    for mi, m in enumerate(months):
        a, b = monthly_returns[m]["start"], monthly_returns[m]["end"]
        if b > a:
            data[mi] = (eq_symbols[b] / eq_symbols[a - 1] - 1) * 100

    x = np.arange(len(months))
    bw = 0.85 / s
    for j in range(s):
        offset = (j - s / 2) * bw
        ax.bar(x + offset, data[:, j], bw, color=colours[j], alpha=0.65, lw=0)

    step = max(1, len(months) // 24)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Return (%)", fontsize=10)
    ax.set_title(f"Monthly Returns per Symbol", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.15, lw=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(0, color="#888", lw=0.5)
    ax.tick_params(labelsize=8)

    # Legend for monthly panel
    handles = [plt.Rectangle((0,0),1,1, color=colours[j], alpha=0.7) for j in range(s)]
    ax.legend(handles, [sym[:-4] for sym in symbols], ncol=4, fontsize=6.5, loc="upper right", framealpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    out = OUT_DIR / "backtest_v16a_19symbols_only_20260528.png"
    out.write_bytes(buf.getvalue())
    print(f"Saved {out}  ({len(buf.getvalue())/1024:.0f} KB)")


if __name__ == "__main__":
    main()
