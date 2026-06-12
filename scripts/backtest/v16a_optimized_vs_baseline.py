"""Baseline vs Optimized v16a comparison backtest.

Optimized: signal_persistence 3→1, badscore gate scale 0.5→0.25
"""

from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from dataclasses import replace
from executor.profiles.v16a_badscore_overlay import (
    INITIAL_EQUITY,
    build_v16a_target_set,
)

DATA_DIR = Path("data")
OUT_DIR = Path("backtest-results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_COLORS: dict[str, str] = {}


def _price_series(sym: str, start, end):
    from data_service.store import ParquetStore
    df = ParquetStore(DATA_DIR).read(sym, "1h")
    if df.is_empty():
        return []
    return [
        (t, float(p))
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if start <= t <= end
    ]


def _yearly_pct(timeline, equity):
    r: dict[str, float] = {}
    for y in sorted({t.year for t in timeline}):
        idx = [i for i, t in enumerate(timeline) if t.year == y]
        if idx:
            r[str(y)] = float((equity[idx[-1]] / equity[idx[0]] - 1) * 100)
    return r


def run_one(label: str, *, gate_scale: float, persistence: int) -> dict:
    """Build target set with custom gate scale (via monkey-patch) and specified
    v10g_params, then run backtest."""
    import executor.profiles.v16a_badscore_overlay as mod

    # Patch v10g_params to use target persistence
    orig_params = mod.v10g_params

    def patched_params(**overrides):
        p = orig_params(**overrides)
        return replace(p, signal_persistence=persistence)

    # Patch gate scale
    orig_gate_fn = mod.expanding_badscore_gate

    def patched_gate(timeline):
        with patch.object(mod, "expanding_badscore_gate", orig_gate_fn):
            gate = orig_gate_fn(timeline)
            # Replace 0.5 with target scale in the gate array
            # gate is an ndarray where values are either 1.0 or 0.5
            result = gate.copy()
            result[np.isclose(result, 0.5)] = gate_scale
            return result

    with (
        patch.object(mod, "v10g_params", patched_params),
        patch.object(mod, "expanding_badscore_gate", patched_gate),
    ):
        ts = build_v16a_target_set(DATA_DIR, core_phase_hours=2, gross_cap=4.0)

    result = run_target_weight_backtest(
        ts.timeline, ts.returns, ts.target_weights,
        initial_equity=INITIAL_EQUITY, fee=0.000432,
    )
    metrics = calculate_hourly_metrics(result.returns, initial_equity=INITIAL_EQUITY)

    return {
        "label": label,
        "timeline": ts.timeline,
        "equity": metrics["equity"],
        "drawdown": metrics["drawdown"],
        "returns": result.returns,
        "sharpe": metrics["sharpe"],
        "ann_return": metrics["ann_return"],
        "total_return": metrics["return"],
        "max_dd": metrics["max_dd"],
        "avg_gross": float(np.mean(np.sum(np.abs(ts.target_weights), axis=1))),
        "avg_turnover": float(np.mean(result.turnover)),
        "symbols": ts.symbols,
    }


def main():
    print("Running baseline...")
    baseline = run_one("Baseline", gate_scale=0.5, persistence=3)

    print("Running optimized...")
    optimized = run_one("Optimized", gate_scale=0.25, persistence=1)

    for r in [baseline, optimized]:
        print(
            f"  {r['label']:12s}  Sharpe={r['sharpe']:.3f}"
            f"  Return={r['total_return']*100:.1f}%"
            f"  MaxDD={r['max_dd']*100:.2f}%"
            f"  AnnRet={r['ann_return']*100:.1f}%"
            f"  Gross={r['avg_gross']:.4f}"
            f"  Turn/h={r['avg_turnover']:.4f}"
        )

    # ---- Plot ----
    t_b = baseline["timeline"]
    e_b = baseline["equity"]
    dd_b = baseline["drawdown"]
    e_o = optimized["equity"]
    dd_o = optimized["drawdown"]
    symbols = baseline["symbols"]

    btc_p = _price_series("BTCUSDT", t_b[0], t_b[-1])
    eth_p = _price_series("ETHUSDT", t_b[0], t_b[-1])
    yearly_b = _yearly_pct(t_b, e_b)
    yearly_o = _yearly_pct(t_b, e_o)

    fig = plt.figure(figsize=(20, 14), dpi=200)
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(3, 1, figure=fig, hspace=0.35)

    B_COLOR = "#2E86AB"
    O_COLOR = "#A23B72"

    # Panel 1: Equity
    ax = fig.add_subplot(gs[0]); ax.set_facecolor("#fafafa")
    ax.plot(t_b, e_b, color=B_COLOR, lw=1.2, label="Baseline (persistence=3, gate=0.50)")
    ax.plot(t_b, e_o, color=O_COLOR, lw=1.2, label="Optimized (persistence=1, gate=0.25)")
    ax.axhline(INITIAL_EQUITY, color="#aaa", lw=0.5, ls="--")
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.set_title(
        f"v16a Baseline vs Optimized — Equity Curves\n"
        f"B: Sharpe {baseline['sharpe']:.2f} · Return {baseline['total_return']*100:.1f}% · "
        f"MaxDD {baseline['max_dd']*100:.1f}%  |  "
        f"O: Sharpe {optimized['sharpe']:.2f} · Return {optimized['total_return']*100:.1f}% · "
        f"MaxDD {optimized['max_dd']*100:.1f}%",
        fontsize=10, weight="bold",
    )
    ax.grid(True, alpha=0.15, lw=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, framealpha=0.4)
    ax.tick_params(labelsize=8)

    # Panel 2: Drawdown
    ax = fig.add_subplot(gs[1]); ax.set_facecolor("#fafafa")
    dd_b_neg = (e_b / np.maximum.accumulate(e_b) - 1) * 100
    dd_o_neg = (e_o / np.maximum.accumulate(e_o) - 1) * 100
    ax.fill_between(t_b, dd_b_neg, 0, color=B_COLOR, alpha=0.15, lw=0)
    ax.fill_between(t_b, dd_o_neg, 0, color=O_COLOR, alpha=0.15, lw=0)
    ax.plot(t_b, dd_b_neg, color=B_COLOR, lw=1.0, label="Baseline")
    ax.plot(t_b, dd_o_neg, color=O_COLOR, lw=1.0, label="Optimized")
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title("Underwater Drawdown", fontsize=10, weight="bold")
    ax.grid(True, alpha=0.15, lw=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=8, framealpha=0.4)
    ax.tick_params(labelsize=8)

    # Panel 3: Monthly Returns (bar chart)
    ax = fig.add_subplot(gs[2]); ax.set_facecolor("#fafafa")
    months = sorted(set((t.year, t.month) for t in t_b))
    mon_labels = [f"{y}-{m:02d}" for y, m in months]
    # Compute monthly returns from equity curves
    b_ret = []
    o_ret = []
    idx = 0
    for y, m in months:
        indices = [i for i, t in enumerate(t_b) if t.year == y and t.month == m]
        if indices and len(indices) >= 2:
            b_ret.append((e_b[indices[-1]] / e_b[indices[0]] - 1) * 100)
            o_ret.append((e_o[indices[-1]] / e_o[indices[0]] - 1) * 100)
        else:
            b_ret.append(0)
            o_ret.append(0)

    x = np.arange(len(months))
    bw = 0.35
    ax.bar(x - bw / 2, b_ret, bw, color=B_COLOR, alpha=0.7, lw=0, label="Baseline")
    ax.bar(x + bw / 2, o_ret, bw, color=O_COLOR, alpha=0.7, lw=0, label="Optimized")
    step = max(1, len(months) // 18)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([mon_labels[i] for i in range(0, len(months), step)], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Return (%)", fontsize=10)
    ax.set_title("Monthly Returns", fontsize=10, weight="bold")
    ax.grid(True, alpha=0.15, lw=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(0, color="#888", lw=0.5)
    ax.legend(fontsize=8, framealpha=0.4)
    ax.tick_params(labelsize=8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = OUT_DIR / "backtest_v16a_baseline_vs_optimized_20260528.png"
    out.write_bytes(buf.getvalue())
    print(f"Saved {out}  ({len(buf.getvalue())/1024:.0f} KB)")

    # Metrics JSON
    import json
    result = {
        "baseline": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in baseline.items() if k not in ("timeline", "equity", "drawdown", "returns", "symbols")},
        "optimized": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in optimized.items() if k not in ("timeline", "equity", "drawdown", "returns", "symbols")},
    }
    # Convert numpy types
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        return o
    with open(OUT_DIR / "metrics_v16a_baseline_vs_optimized_20260528.json", "w") as f:
        json.dump(conv(result), f, indent=2, default=str)


if __name__ == "__main__":
    main()
