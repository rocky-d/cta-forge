"""v16a 1x vs 5x leverage comparison backtest.

Two configs share the same v16a strategy logic (profile, allocations, gate,
phase hours, factor params).  They differ only in target_scale and gross_cap:

  Config A "1x":          target_scale = 1.0,  gross_cap = 1.0
  Config B "5x live eq":  target_scale = 5.0,  gross_cap = 4.0

DD breaker (SOFT_DD_LIMIT / HARD_DD_LIMIT) and dd_circuit_breaker are NOT
applied — this backtest isolates the strategy signal from risk management.

Run:
    uv run python scripts/backtest/v16a_1x_vs_5x.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from executor.profiles.v16a_badscore_overlay import (
    build_v16a_target_set,
    normalize_gross,
)
from backtest import (
    BacktestMetrics,
    ChartSeries,
    create_comparison_figure,
    compute_drawdown_series,
    compute_metrics,
    compute_monthly_returns,
    run_execution_backtest,
    save_figure,
)
from core import HL_TAKER_FEE

OUT_DIR = ROOT / "backtest-results"
DATA_DIR = ROOT / "data"

# ── Config definitions ───────────────────────────────────────────

CONFIGS = [
    {
        "label": "v16a 1x (scale=1.0 cap=1.0)",
        "color": "#2f81f7",
        "target_scale": 1.0,
        "gross_cap": 1.0,
    },
    {
        "label": "v16a 5x live eq (scale=5.0 cap=4.0)",
        "color": "#f0883e",
        "target_scale": 5.0,
        "gross_cap": 4.0,
    },
]

EXECUTION = {
    "initial_equity": 10_000.0,
    "fee": HL_TAKER_FEE,
    "slippage": 0.0001,
    "min_order_notional": 10.0,
}


def scale_and_cap(
    weights: np.ndarray,
    symbols: list[str],
    *,
    target_scale: float,
    gross_cap: float,
) -> np.ndarray:
    """Multiply raw target weights by scale, then cap each row at gross_cap."""
    scaled = weights * target_scale
    result = np.zeros_like(scaled)
    for i in range(scaled.shape[0]):
        capped = normalize_gross(
            {sym: float(scaled[i, j]) for j, sym in enumerate(symbols)},
            gross_cap=gross_cap,
        )
        for j, sym in enumerate(symbols):
            result[i, j] = capped.get(sym, 0.0)
    return result


def main() -> None:
    t0 = time.monotonic()

    # ── 1. Build raw v16a target set (once, uncapped) ────────────
    print(f"[1/5] Building v16a target set from {DATA_DIR}...")
    ts = build_v16a_target_set(
        str(DATA_DIR),
        gross_cap=999.0,  # uncapped — we scale + cap per config
    )
    print(
        f"       {len(ts.timeline)} hourly bars, {len(ts.symbols)} symbols, "
        f"{ts.timeline[0].date()} → {ts.timeline[-1].date()}"
    )

    chart_series: list[ChartSeries] = []

    for cfg in CONFIGS:
        label = cfg["label"]
        color = cfg["color"]
        scale = cfg["target_scale"]
        cap = cfg["gross_cap"]

        # ── 2. Scale + cap weights ───────────────────────────────
        print(
            f"\n[2/5] {label}: scale={scale}, cap={cap}"
        )
        weights = scale_and_cap(
            ts.target_weights,
            ts.symbols,
            target_scale=scale,
            gross_cap=cap,
        )
        gross = np.sum(np.abs(weights), axis=1)
        print(f"       mean gross exposure: {np.mean(gross):.3f}")

        # ── 3. Run execution backtest ────────────────────────────
        print(f"[3/5] Running execution backtest...")
        result = run_execution_backtest(
            timeline=ts.timeline,
            returns=ts.returns,
            target_weights=weights,
            initial_equity=EXECUTION["initial_equity"],
            fee=EXECUTION["fee"],
            slippage=EXECUTION["slippage"],
            min_order_notional=EXECUTION["min_order_notional"],
        )
        print(
            f"       total turnover: {np.sum(result.turnover):.2f}, "
            f"avg orders/bar: {np.mean(result.order_counts[result.order_counts > 0]):.1f}"
        )

        # ── 4. Compute metrics ───────────────────────────────────
        print(f"[4/5] Computing metrics...")
        m = compute_metrics(
            result.returns,
            initial_equity=EXECUTION["initial_equity"],
            weights=result.realized_weights,
        )
        print(
            f"       Return {m.total_return * 100:.1f}%  "
            f"Ann {m.annualized_return * 100:.1f}%  "
            f"Sharpe {m.sharpe_ratio:.2f}  "
            f"MaxDD {m.max_drawdown * 100:.1f}%  "
            f"Calmar {m.calmar_ratio:.2f}"
        )

        # ── 5. Build ChartSeries ─────────────────────────────────
        equity = np.array([e for _, e in result.equity_curve])
        equity_norm = equity / equity[0]
        dd = compute_drawdown_series(equity_norm)
        monthly = compute_monthly_returns(ts.timeline, equity_norm)

        chart_series.append(
            ChartSeries(
                label=label,
                color=color,
                equity=equity_norm,
                drawdown=dd,
                monthly_returns=monthly,
                metrics=m,
                timestamps=ts.timeline,
            )
        )

    # ── Generate chart ───────────────────────────────────────────
    print(f"\n[5/5] Generating comparison chart...")
    fig = create_comparison_figure(
        chart_series,
        title="v16a: 1x vs 5x Leverage Comparison (same strategy, perp-only)",
    )
    chart_path = OUT_DIR / "v16a_1x_vs_5x.png"
    save_figure(fig, chart_path)
    print(f"       saved to {chart_path}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
