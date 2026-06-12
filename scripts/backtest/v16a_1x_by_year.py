"""v16a 1x baseline — per-year backtests from each start year to present.

One target-set build, filtered by start year, one 3-panel chart per year.
Run: uv run python scripts/backtest/v16a_1x_by_year.py
"""

from __future__ import annotations

from datetime import datetime, timezone
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

DATA_DIR = "data"
OUT_DIR = ROOT / "backtest-results"
FEE = HL_TAKER_FEE
SLIPPAGE = 0.0001
MIN_ORDER = 10.0
EQUITY = 10_000.0
TARGET_SCALE = 1.0
GROSS_CAP = 1.0

YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def main():
    t0 = time.monotonic()

    print("[1/2] Building v16a target set (full history, uncapped)...")
    ts = build_v16a_target_set(DATA_DIR, gross_cap=999.0)
    print(
        f"       {len(ts.timeline)} bars, {len(ts.symbols)} symbols, "
        f"{ts.timeline[0].date()} → {ts.timeline[-1].date()}"
    )

    for year in YEARS:
        start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        # Find slice index
        idx = next(
            (i for i, t in enumerate(ts.timeline) if t >= start_dt), None
        )
        if idx is None:
            print(f"\n[y{year}] No data at or after {start_dt.date()} — skip")
            continue

        timeline_y = ts.timeline[idx:]
        returns_y = ts.returns[idx:]
        target_y = ts.target_weights[idx:]
        print(f"\n[y{year}] {timeline_y[0]} → {timeline_y[-1]} ({len(timeline_y)} bars)")

        # Scale + cap
        scaled = target_y * TARGET_SCALE
        weights = np.zeros_like(scaled)
        for i in range(scaled.shape[0]):
            capped = normalize_gross(
                {s: float(scaled[i, j]) for j, s in enumerate(ts.symbols)},
                gross_cap=GROSS_CAP,
            )
            for j in range(weights.shape[1]):
                weights[i, j] = capped.get(ts.symbols[j], 0.0)

        gross = np.sum(np.abs(weights), axis=1)
        result = run_execution_backtest(
            timeline_y, returns_y, weights,
            initial_equity=EQUITY, fee=FEE, slippage=SLIPPAGE, min_order_notional=MIN_ORDER,
        )

        m = compute_metrics(result.returns, initial_equity=EQUITY, weights=result.realized_weights)
        ann = m.annualized_return * 100
        dd_pct = m.max_drawdown * 100
        tot = m.total_return * 100
        print(
            f"       total {tot:.1f}%  ann {ann:.1f}%  Sharpe {m.sharpe_ratio:.2f}  "
            f"MaxDD {dd_pct:.1f}%  Calmar {m.calmar_ratio:.2f}  avg gross {np.mean(gross):.3f}"
        )

        eq = np.array([e for _, e in result.equity_curve])
        eq_n = eq / eq[0]
        dd = compute_drawdown_series(eq_n)
        monthly = compute_monthly_returns(timeline_y, eq_n)

        label = (
            f"v16a 1x {year}→  "
            f"Ann {ann:.1f}%  Sharpe {m.sharpe_ratio:.2f}  MaxDD {dd_pct:.1f}%"
        )
        cs = ChartSeries(
            label=label, color="#2f81f7",
            equity=eq_n, drawdown=dd, monthly_returns=monthly,
            metrics=m, timestamps=timeline_y,
        )
        fig = create_comparison_figure(
            [cs],
            title=f"v16a 1x — {year}-01-01 → present (perp-only, rolling 3yr gate, phase 2)",
            drawdown_colors=["#f85149"],
        )
        path = OUT_DIR / f"v16a_1x_{year}.png"
        save_figure(fig, path)
        print(f"       saved {path}")

    print(f"\nDone in {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
