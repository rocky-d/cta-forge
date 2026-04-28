"""Target-weight portfolio backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class TargetBacktestResult:
    """Result of a target-weight mark-to-market simulation."""

    equity_curve: list[tuple[datetime, float]]
    returns: np.ndarray
    turnover: np.ndarray
    target_weights: np.ndarray


def run_target_weight_backtest(
    timeline: list[datetime],
    returns: np.ndarray,
    target_weights: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    fee: float = 0.0004,
) -> TargetBacktestResult:
    """Simulate next-bar PnL from signed target weights.

    ``target_weights[t]`` is held from ``timeline[t]`` to ``timeline[t + 1]``.
    Turnover fees are charged when moving from previous target to current target.
    """
    if returns.shape != target_weights.shape:
        raise ValueError("returns and target_weights must have the same shape")
    if len(timeline) != returns.shape[0]:
        raise ValueError("timeline length must match returns rows")

    pnl = np.zeros(returns.shape[0], dtype=float)
    turnover = np.zeros(returns.shape[0], dtype=float)
    previous = np.zeros(returns.shape[1], dtype=float)

    for t in range(returns.shape[0] - 1):
        current = np.nan_to_num(target_weights[t], nan=0.0)
        turnover[t] = float(np.sum(np.abs(current - previous)))
        pnl[t + 1] = float(
            np.sum(current * np.nan_to_num(returns[t + 1], nan=0.0)) - turnover[t] * fee
        )
        previous = current

    equity = initial_equity * np.cumprod(1.0 + pnl)
    curve = list(zip(timeline, equity.tolist()))
    return TargetBacktestResult(
        equity_curve=curve,
        returns=pnl,
        turnover=turnover,
        target_weights=target_weights,
    )


def calculate_hourly_metrics(
    returns: np.ndarray, *, initial_equity: float = 10_000.0
) -> dict[str, float | np.ndarray]:
    """Calculate simple hourly annualized metrics for a target backtest."""
    equity = initial_equity * np.cumprod(1.0 + returns)
    total = float(equity[-1] / equity[0] - 1.0)
    ann_return = float((1.0 + total) ** ((365 * 24) / max(len(equity), 1)) - 1.0)
    volatility = float(np.std(returns) * np.sqrt(365 * 24))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "return": total,
        "ann_return": ann_return,
        "volatility": volatility,
        "sharpe": ann_return / volatility if volatility > EPS else 0.0,
        "max_dd": float(drawdown.min()),
        "ulcer": float(np.sqrt(np.mean(drawdown * drawdown))),
        "equity": equity,
        "drawdown": drawdown,
    }
