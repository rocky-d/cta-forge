"""v16a 1x: gross_cap 1.0 vs 0.8 comparison.

Same strategy, same target_scale=1.0.  Only gross_cap differs.
Run: uv run python scripts/backtest/v16a_1x_cap_compare.py
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

TARGET_SCALE = 1.0
INITIAL_EQUITY = 10_000.0
FEE = HL_TAKER_FEE
SLIPPAGE = 0.0001
MIN_ORDER_NOTIONAL = 10.0

CONFIGS = [
    {"label": "v16a 1x  cap=1.0", "color": "#2f81f7", "gross_cap": 1.0},
    {"label": "v16a 1x  cap=0.8", "color": "#f0883e", "gross_cap": 0.8},
]


def scale_and_cap(weights, symbols, *, target_scale, gross_cap):
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


def main():
    t0 = time.monotonic()

    print("[1/3] Building v16a target set (once, uncapped)...")
    ts = build_v16a_target_set(str(DATA_DIR), gross_cap=999.0)
    print(
        f"       {len(ts.timeline)} bars, {len(ts.symbols)} symbols, "
        f"{ts.timeline[0].date()} → {ts.timeline[-1].date()}"
    )

    chart_series: list[ChartSeries] = []
    for cfg in CONFIGS:
        label = cfg["label"]
        color = cfg["color"]
        cap = cfg["gross_cap"]

        print(f"\n[2/3] {label}...")
        weights = scale_and_cap(
            ts.target_weights, ts.symbols, target_scale=TARGET_SCALE, gross_cap=cap
        )
        gross = np.sum(np.abs(weights), axis=1)
        print(f"       mean gross: {np.mean(gross):.3f}")

        result = run_execution_backtest(
            timeline=ts.timeline,
            returns=ts.returns,
            target_weights=weights,
            initial_equity=INITIAL_EQUITY,
            fee=FEE,
            slippage=SLIPPAGE,
            min_order_notional=MIN_ORDER_NOTIONAL,
        )

        m = compute_metrics(
            result.returns,
            initial_equity=INITIAL_EQUITY,
            weights=result.realized_weights,
        )
        print(
            f"       Ann {m.annualized_return*100:.1f}%  "
            f"Sharpe {m.sharpe_ratio:.2f}  "
            f"MaxDD {m.max_drawdown*100:.1f}%  "
            f"Calmar {m.calmar_ratio:.2f}"
        )

        equity = np.array([e for _, e in result.equity_curve])
        equity_norm = equity / equity[0]
        dd = compute_drawdown_series(equity_norm)
        monthly = compute_monthly_returns(ts.timeline, equity_norm)

        chart_series.append(
            ChartSeries(
                label=f"{label}  Ann {m.annualized_return*100:.1f}%  "
                f"Sharpe {m.sharpe_ratio:.2f}  MaxDD {m.max_drawdown*100:.1f}%",
                color=color,
                equity=equity_norm,
                drawdown=dd,
                monthly_returns=monthly,
                metrics=m,
                timestamps=ts.timeline,
            )
        )

    print(f"\n[3/3] Generating chart...")
    fig = create_comparison_figure(
        chart_series,
        title="v16a 1x: gross_cap 1.0 vs 0.8",
    )
    chart_path = OUT_DIR / "v16a_1x_cap_compare.png"
    save_figure(fig, chart_path)
    print(f"       saved to {chart_path}")
    print(f"\nDone in {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
