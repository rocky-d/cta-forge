"""v16a 1x leverage backtest — single config, 3-panel chart.

Same v16a strategy logic as live (profile, allocations, rolling 3yr gate,
phase 2).  DD breaker not applied.

Run:
    uv run python scripts/backtest/v16a_1x.py
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

# ── Config ───────────────────────────────────────────────────────

TARGET_SCALE = 1.0
GROSS_CAP = 1.0
INITIAL_EQUITY = 10_000.0
FEE = HL_TAKER_FEE  # 0.000432
SLIPPAGE = 0.0001
MIN_ORDER_NOTIONAL = 10.0
LABEL = "v16a 1x"


def scale_and_cap(
    weights: np.ndarray,
    symbols: list[str],
    *,
    target_scale: float,
    gross_cap: float,
) -> np.ndarray:
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

    print(f"[1/4] Building v16a target set (rolling 3yr gate, phase 2)...")
    ts = build_v16a_target_set(
        str(DATA_DIR),
        gross_cap=999.0,
    )
    print(
        f"       {len(ts.timeline)} bars, {len(ts.symbols)} symbols, "
        f"{ts.timeline[0].date()} → {ts.timeline[-1].date()}"
    )

    print(f"[2/4] Scaling weights (scale={TARGET_SCALE}, cap={GROSS_CAP})...")
    weights = scale_and_cap(
        ts.target_weights,
        ts.symbols,
        target_scale=TARGET_SCALE,
        gross_cap=GROSS_CAP,
    )
    gross = np.sum(np.abs(weights), axis=1)
    print(f"       mean gross exposure: {np.mean(gross):.3f}")

    print(f"[3/4] Running execution backtest...")
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
        f"       Return {m.total_return * 100:.1f}%  "
        f"Ann {m.annualized_return * 100:.1f}%  "
        f"Vol {m.volatility * 100:.1f}%  "
        f"Sharpe {m.sharpe_ratio:.2f}  "
        f"Sortino {m.sortino_ratio:.2f}  "
        f"MaxDD {m.max_drawdown * 100:.1f}%  "
        f"Calmar {m.calmar_ratio:.2f}"
    )

    print(f"[4/4] Generating chart...")
    equity = np.array([e for _, e in result.equity_curve])
    equity_norm = equity / equity[0]
    dd = compute_drawdown_series(equity_norm)
    monthly = compute_monthly_returns(ts.timeline, equity_norm)

    cs = ChartSeries(
        label=f"{LABEL}  |  Ann {m.annualized_return*100:.1f}%  "
        f"Sharpe {m.sharpe_ratio:.2f}  MaxDD {m.max_drawdown*100:.1f}%",
        color="#2f81f7",
        equity=equity_norm,
        drawdown=dd,
        monthly_returns=monthly,
        metrics=m,
        timestamps=ts.timeline,
    )

    fig = create_comparison_figure(
        [cs],
        title="v16a 1x leverage — perp-only, rolling 3yr gate, phase 2",
    )
    chart_path = OUT_DIR / "v16a_1x.png"
    save_figure(fig, chart_path)
    print(f"       saved to {chart_path}")

    print(f"\nDone in {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
