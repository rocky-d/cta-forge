"""Research backtest for v16a Badscore Overlay.

Candidate: v16a Badscore Overlay (`joint-v10g-fast-overlay-badscore-v1`).

This script is intentionally thin: strategy construction lives in
`executor.profiles.v16a_badscore_overlay`, and target-weight simulation lives in
`executor.portfolio_backtest` so later live integration can share the same code.

Run:
    uv run python scripts/backtest/joint_badscore_research.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data_service.store import ParquetStore
from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_execution_backtest,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (
    INITIAL_EQUITY,
    build_v16a_target_set,
    split_metrics,
)
from report_service.plot import plot_backtest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "backtest-results"


def price_series(sym: str, start_ts, end_ts) -> list[tuple[object, float]]:
    df = ParquetStore(DATA_DIR).read(sym, "1h")
    if df.is_empty():
        return []
    return [
        (t, float(p))
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if start_ts <= t <= end_ts
    ]


def yearly_percentages(timeline, equity: np.ndarray) -> dict[str, float]:
    yearly: dict[int, dict[str, float]] = {}
    for ts, value in zip(timeline, equity):
        yearly.setdefault(ts.year, {"first": float(value)})
        yearly[ts.year]["last"] = float(value)
    return {
        str(year): round((values["last"] / values["first"] - 1) * 100, 1)
        for year, values in sorted(yearly.items())
    }


def plot_result(
    timeline,
    returns: np.ndarray,
    metrics: dict,
    symbols: list[str],
    targets: np.ndarray,
    turnover: np.ndarray,
) -> None:
    equity_curve = list(zip(timeline, metrics["equity"].tolist()))
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(365 * 24) if len(downside) else 0.0
    sortino = metrics["ann_return"] / downside_vol if downside_vol > 1e-12 else 0.0
    calmar = (
        metrics["ann_return"] / abs(metrics["max_dd"])
        if abs(metrics["max_dd"]) > 1e-12
        else 0.0
    )

    img = plot_backtest(
        equity_curve=equity_curve,
        btc_prices=price_series("BTCUSDT", timeline[0], timeline[-1]),
        eth_prices=price_series("ETHUSDT", timeline[0], timeline[-1]),
        metrics={
            "total_return": metrics["return"],
            "annualized_return": metrics["ann_return"],
            "sharpe_ratio": metrics["sharpe"],
            "sortino_ratio": sortino,
            "max_drawdown": metrics["max_dd"],
            "calmar_ratio": calmar,
            "profit_factor": None,
            "win_rate": None,
            "num_trades": None,
            "ulcer_index": metrics["ulcer"],
        },
        yearly=yearly_percentages(timeline, metrics["equity"]),
        title_extra=(
            "v16a Badscore Overlay · shifted v10g 6h core + "
            "1h fast-exit top2 overlay · 50/50 · badscore2_050 · "
            f"{len(symbols)} symbols · ${INITIAL_EQUITY:,.0f} start · "
            f"{(timeline[-1] - timeline[0]).days} days · "
            f"avg gross {np.mean(np.sum(np.abs(targets), axis=1)):.3f} · "
            f"turn/h {np.mean(turnover):.4f} · vs BTC & ETH buy-and-hold"
        ),
        initial_equity=INITIAL_EQUITY,
        strategy_label="v16a Badscore Overlay",
    )
    (OUT_DIR / "backtest_joint_badscore_research.png").write_bytes(img)


def metric_payload(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in {"equity", "drawdown"}}


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building v16a target set...")
    target_set = build_v16a_target_set(DATA_DIR)

    print("Simulating target-weight portfolios...")
    v10g = run_target_weight_backtest(
        target_set.timeline,
        target_set.returns,
        target_set.v10g_weights,
        initial_equity=INITIAL_EQUITY,
    )
    overlay = run_target_weight_backtest(
        target_set.timeline,
        target_set.returns,
        target_set.overlay_weights,
        initial_equity=INITIAL_EQUITY,
    )
    joint = run_target_weight_backtest(
        target_set.timeline,
        target_set.returns,
        target_set.target_weights,
        initial_equity=INITIAL_EQUITY,
    )
    realistic = run_execution_backtest(
        target_set.timeline,
        target_set.returns,
        target_set.target_weights,
        initial_equity=INITIAL_EQUITY,
        fee=0.0004,
        slippage=0.0001,
        min_order_notional=10.0,
    )

    metrics = {
        "shifted_v10g": calculate_hourly_metrics(
            v10g.returns, initial_equity=INITIAL_EQUITY
        ),
        "fast_exit_top2_overlay": calculate_hourly_metrics(
            overlay.returns, initial_equity=INITIAL_EQUITY
        ),
        "v16a_badscore_overlay": calculate_hourly_metrics(
            joint.returns, initial_equity=INITIAL_EQUITY
        ),
        "v16a_execution_realistic": calculate_hourly_metrics(
            realistic.returns, initial_equity=INITIAL_EQUITY
        ),
    }
    plot_result(
        target_set.timeline,
        joint.returns,
        metrics["v16a_badscore_overlay"],
        target_set.symbols,
        target_set.target_weights,
        joint.turnover,
    )

    result = {
        "profile": "v16a-badscore-overlay",
        "research_name": "joint-v10g-fast-overlay-badscore-v1",
        "period": f"{target_set.timeline[0].date()} -> {target_set.timeline[-1].date()}",
        "symbols": target_set.symbols,
        "gross_cap": 1.0,
        "commission": 0.0004,
        "execution_realistic_assumptions": {
            "commission": 0.0004,
            "slippage": 0.0001,
            "min_order_notional": 10.0,
            "funding_rates": "not included yet",
            "notes": [
                "target orders are constrained by minimum notional",
                "sign flips are split into close-to-flat and open-new-side legs",
                "realized weights can differ from target weights when an order is below the notional threshold",
            ],
        },
        "allocation": {"shifted_v10g": 0.5, "fast_exit_top2_overlay": 0.5},
        "badscore_gate": {
            "scale_when_active": 0.5,
            "conditions_required": 2,
            "conditions": [
                "market_volatility_above_expanding_median",
                "market_trend_efficiency_below_expanding_median",
                "mean_cross_asset_correlation_above_expanding_66pct_quantile",
            ],
        },
        "metrics": {name: metric_payload(values) for name, values in metrics.items()},
        "splits": {
            "shifted_v10g": split_metrics(target_set.timeline, v10g.returns),
            "fast_exit_top2_overlay": split_metrics(
                target_set.timeline, overlay.returns
            ),
            "v16a_badscore_overlay": split_metrics(target_set.timeline, joint.returns),
            "v16a_execution_realistic": split_metrics(
                target_set.timeline, realistic.returns
            ),
        },
        "avg_gross": float(np.mean(np.sum(np.abs(target_set.target_weights), axis=1))),
        "avg_turnover_per_hour": float(np.mean(joint.turnover)),
        "execution_realistic": {
            "avg_realized_gross": float(
                np.mean(np.sum(np.abs(realistic.realized_weights), axis=1))
            ),
            "avg_turnover_per_hour": float(np.mean(realistic.turnover)),
            "avg_orders_per_hour": float(np.mean(realistic.order_counts)),
            "avg_ignored_gross": float(np.mean(realistic.ignored_gross)),
            "max_ignored_gross": float(np.max(realistic.ignored_gross)),
        },
        "validation_note": "Target-weight research backtest plus simple execution-realistic variant; funding, liquidity depth, and exchange margin are still approximations.",
    }
    (OUT_DIR / "metrics_joint_badscore_research.json").write_text(
        json.dumps(result, indent=2)
    )

    print("\nRESULTS")
    for label, values in metrics.items():
        print(
            f"  {label}: Sharpe={values['sharpe']:.2f} "
            f"Return={values['return'] * 100:+.1f}% MaxDD={values['max_dd'] * 100:.1f}%"
        )
    print(f"  Avg gross: {result['avg_gross']:.3f}")
    print(f"  Avg turnover/hour: {result['avg_turnover_per_hour']:.4f}")
    print(
        "  Execution realistic: "
        f"avg gross {result['execution_realistic']['avg_realized_gross']:.3f}, "
        f"turn/h {result['execution_realistic']['avg_turnover_per_hour']:.4f}, "
        f"orders/h {result['execution_realistic']['avg_orders_per_hour']:.4f}"
    )
    print(f"\nMetrics: {OUT_DIR / 'metrics_joint_badscore_research.json'}")
    print(f"Chart:   {OUT_DIR / 'backtest_joint_badscore_research.png'}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
