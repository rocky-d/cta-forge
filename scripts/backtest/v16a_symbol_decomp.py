"""19-symbol decomposition + total portfolio three-panel chart."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from executor.portfolio_backtest import run_target_weight_backtest
from executor.profiles.v16a_badscore_overlay import (
    INITIAL_EQUITY,
    build_v16a_target_set,
)

DATA_DIR = Path("data")
OUT_DIR = Path("backtest-results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building v16a target set...")
    target_set = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0)

    timeline = target_set.timeline
    symbols = target_set.symbols
    weights = target_set.target_weights  # (T, S)
    returns = target_set.returns  # (T, S)
    n, s = len(timeline), len(symbols)

    # Total via the canonical backtest
    print("Running canonical backtest...")
    result = run_target_weight_backtest(timeline, returns, weights, initial_equity=INITIAL_EQUITY, fee=0.000432)
    total_equity = np.array([e[1] for e in result.equity_curve])

    # Per-symbol: same lag as run_target_weight_backtest
    # weights[t] * returns[t+1] → pnl at t+1
    print(f"Computing per-symbol equity for {s} symbols...")
    symbol_eq = np.ones((n, s)) * INITIAL_EQUITY
    for t in range(n - 1):
        w = np.nan_to_num(weights[t], nan=0.0)
        r_turnover = np.nan_to_num(returns[t + 1], nan=0.0) - np.sum(np.abs(w - np.nan_to_num(
            weights[t - 1] if t > 0 else np.zeros(s), nan=0.0
        ))) * 0.0004 / s
        for j in range(s):
            pnl = w[j] * np.nan_to_num(returns[t + 1, j], nan=0.0) - (
                np.abs(w[j] - (np.nan_to_num(weights[t - 1, j], nan=0.0) if t > 0 else 0.0)) * 0.0004
            )
            symbol_eq[t + 1, j] = symbol_eq[t, j] * (1.0 + pnl)

    # Total drawdown
    total_peak = np.maximum.accumulate(total_equity)
    total_dd = (total_equity / total_peak - 1) * 100

    # Metrics from canonical
    total_ret = (total_equity[-1] / INITIAL_EQUITY - 1) * 100
    pnl = np.diff(np.log(total_equity))
    ann_ret = np.mean(pnl) * 365 * 24 * 100
    ann_vol = np.std(pnl) * np.sqrt(365 * 24)
    sharpe = ann_ret / 100 / ann_vol if ann_vol > 0 else 0
    max_dd = np.min(total_dd)

    # Monthly returns: per-symbol contribution
    monthly = {}
    for t in range(n):
        dt = timeline[t]
        key = f"{dt.year}-{dt.month:02d}"
        if key not in monthly:
            monthly[key] = {"idx_start": t, "idx_end": t}
        monthly[key]["idx_end"] = t
    for key in monthly:
        a, b = monthly[key]["idx_start"], monthly[key]["idx_end"]
        if b > a:
            monthly[key]["Total"] = (total_equity[b] / total_equity[a - 1] - 1) * 100
            for j in range(s):
                monthly[key][symbols[j]] = (symbol_eq[b, j] / symbol_eq[a - 1, j] - 1) * 100
        else:
            monthly[key]["Total"] = 0.0
            for j in range(s):
                monthly[key][symbols[j]] = 0.0

    print(f"  Sharpe={sharpe:.3f}  Return={total_ret:.1f}%  MaxDD={max_dd:.1f}%")

    # ---- Plot ----
    W = 3  # widen factor
    colours = plt.cm.tab20(np.linspace(0, 1, s + 1))
    TOTAL = "#111111"

    fig, axes = plt.subplots(3, 1, figsize=(6.4 * W, 3.6 * W * 1.5))
    fig.patch.set_facecolor("#fafafa")

    # Panel 1: Equity
    ax = axes[0]; ax.set_facecolor("#ffffff")
    for j in range(s):
        ax.plot(timeline, symbol_eq[:, j], color=colours[j], alpha=0.45, lw=0.5, label=symbols[j])
    ax.plot(timeline, total_equity, color=TOTAL, lw=2.0, label="Total")
    ax.axhline(INITIAL_EQUITY, color="#888", lw=0.5, ls="--")
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"v16a Live Config — {s} Symbols + Total")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(ncol=2, loc="upper left", fontsize=5.5, framealpha=0.5, columnspacing=0.5)

    # Panel 2: Drawdown
    ax = axes[1]; ax.set_facecolor("#ffffff")
    for j in range(s):
        p = np.maximum.accumulate(symbol_eq[:, j])
        dd = (symbol_eq[:, j] / p - 1) * 100
        ax.fill_between(timeline, dd, 0, color=colours[j], alpha=0.10, lw=0)
        ax.plot(timeline, dd, color=colours[j], alpha=0.30, lw=0.3)
    ax.fill_between(timeline, total_dd, 0, color=TOTAL, alpha=0.20, lw=0)
    ax.plot(timeline, total_dd, color=TOTAL, lw=1.5)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Panel 3: Monthly Returns
    ax = axes[2]; ax.set_facecolor("#ffffff")
    months = sorted(monthly.keys())
    x = np.arange(len(months))
    bw = 0.9 / (s + 1)
    for j in range(s):
        vals = [monthly[m].get(symbols[j], 0.0) for m in months]
        offset = (j - s / 2) * bw
        ax.bar(x + offset, vals, bw, color=colours[j], alpha=0.55, lw=0)
    total_vals = [monthly[m]["Total"] for m in months]
    offset = (s - s / 2) * bw + bw
    ax.bar(x + offset, total_vals, bw, color=TOTAL, alpha=0.85, lw=0, label="Total")

    step = max(1, len(months) // 24)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Monthly Returns ({s} symbols + Total)")
    ax.grid(True, alpha=0.2, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(0, color="#888", lw=0.5)
    ax.legend(fontsize=6, loc="upper right")

    fig.text(
        0.5, 0.005,
        f"v16a Badscore Overlay · cap 4x · phase 2 · 50/50 core/overlay · "
        f"Sharpe {sharpe:.2f} · Return {total_ret:.1f}% · MaxDD {max_dd:.1f}% · "
        f"{timeline[0].date()} → {timeline[-1].date()}",
        ha="center", fontsize=6.5, color="#666",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.99])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    out = OUT_DIR / "backtest_v16a_19symbols_decomposed_20260528.png"
    out.write_bytes(buf.getvalue())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
