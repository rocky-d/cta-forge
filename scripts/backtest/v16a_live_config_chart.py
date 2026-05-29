"""Live-config backtest chart for v16a mainnet-pilot production settings.

Matches: core_phase_hours=2, gross_cap=4.0
Equivalent to: joint_badscore_research.py with live parameters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (
    INITIAL_EQUITY,
    build_v16a_target_set,
)
from report_service.plot import plot_backtest

DATA_DIR = Path("data")
OUT_DIR = Path("backtest-results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def price_series(
    symbol: str, start: object, end: object
) -> list[tuple[object, float]]:
    from data_service.store import ParquetStore
    store = ParquetStore(DATA_DIR)
    df = store.read(symbol, "1h")
    if df.is_empty():
        return []
    return [
        (t, float(p))
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if start <= t <= end
    ]


def yearly_percentages(
    timeline: list, equity: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    years = sorted({t.year for t in timeline})
    for y in years:
        indices = [i for i, t in enumerate(timeline) if t.year == y]
        if not indices:
            continue
        start_val = equity[indices[0]]
        if start_val is None or start_val == 0:
            continue
        end_val = equity[indices[-1]]
        result[str(y)] = float((end_val / start_val - 1) * 100)
    return result


def main() -> None:
    t0 = time.time()
    print("Building v16a target set (core_phase_hours=2, gross_cap=4.0)...")
    target_set = build_v16a_target_set(
        DATA_DIR,
        core_phase_hours=2,
        gross_cap=4.0,
    )

    timeline = target_set.timeline
    symbols = target_set.symbols
    targets = target_set.target_weights

    print("Simulating target-weight portfolio...")
    result = run_target_weight_backtest(
        timeline,
        target_set.returns,
        targets,
        initial_equity=INITIAL_EQUITY,
    )
    metrics = calculate_hourly_metrics(result.returns, initial_equity=INITIAL_EQUITY)

    downside = result.returns[result.returns < 0]
    downside_vol = np.std(downside) * np.sqrt(365 * 24) if len(downside) else 0.0
    sortino = metrics["ann_return"] / downside_vol if downside_vol > 1e-12 else 0.0
    calmar = (
        metrics["ann_return"] / metrics["max_dd"] if metrics["max_dd"] > 1e-12 else 0.0
    )

    equity_curve = list(zip(timeline, metrics["equity"].tolist()))
    avg_gross = np.mean(np.sum(np.abs(targets), axis=1))
    avg_turnover = np.mean(result.turnover)
    days = (timeline[-1] - timeline[0]).days

    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Return: {metrics['return']*100:.1f}%")
    print(f"  MaxDD:  {metrics['max_dd']*100:.2f}%")
    print(f"  Avg gross: {avg_gross:.3f}")
    print(f"  Avg turnover/h: {avg_turnover:.4f}")

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
            "ulcer_index": metrics.get("ulcer"),
        },
        yearly=yearly_percentages(timeline, metrics["equity"]),
        title_extra=(
            "v16a Badscore Overlay · LIVE config (cap 4x · phase 2) · "
            "shifted v10g 6h core + 1h fast-exit top2 overlay · "
            f"50/50 · {len(symbols)} symbols · "
            f"${INITIAL_EQUITY:,.0f} · {days}d · "
            f"avg gross {avg_gross:.3f} · turn/h {avg_turnover:.4f}"
        ),
        initial_equity=INITIAL_EQUITY,
        strategy_label="v16a Live Config",
    )
    (OUT_DIR / "backtest_v16a_live_config_20260528.png").write_bytes(img)

    result_json = {
        "profile": "v16a-mainnet-pilot (live config)",
        "core_phase_hours": 2,
        "gross_cap": 4.0,
        "period": f"{timeline[0].date()} -> {timeline[-1].date()}",
        "symbols": symbols,
        "metrics": {
            "sharpe": metrics["sharpe"],
            "total_return": metrics["return"],
            "ann_return": metrics["ann_return"],
            "max_dd": metrics["max_dd"],
            "sortino": sortino,
            "calmar": calmar,
        },
        "avg_gross": avg_gross,
        "avg_turnover_per_hour": avg_turnover,
    }
    with open(OUT_DIR / "metrics_v16a_live_config_20260528.json", "w") as f:
        json.dump(result_json, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
