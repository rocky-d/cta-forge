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


@dataclass(frozen=True)
class ExecutionBacktestResult:
    """Result of a simple order-aware target execution simulation."""

    equity_curve: list[tuple[datetime, float]]
    returns: np.ndarray
    turnover: np.ndarray
    target_weights: np.ndarray
    realized_weights: np.ndarray
    order_counts: np.ndarray
    ignored_turnover: np.ndarray
    ignored_gross: np.ndarray


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


def _apply_target_orders(
    current: np.ndarray,
    target: np.ndarray,
    *,
    equity: float,
    min_order_notional: float,
) -> tuple[np.ndarray, float, int, float]:
    """Move current weights toward target using live-like order constraints.

    Sign flips are split into close-to-flat then open-new-side legs. Each leg is
    independently subject to the minimum notional threshold, matching the live
    target-order reconciliation path more closely than a direct weight jump.
    """
    next_weights = current.copy()
    turnover = 0.0
    order_count = 0
    ignored_turnover = 0.0

    for idx, desired in enumerate(target):
        cur = float(next_weights[idx])
        tgt = float(desired)

        legs: list[float]
        if abs(cur) > EPS and cur * tgt < 0:
            legs = [-cur, tgt]
        else:
            legs = [tgt - cur]

        for leg in legs:
            if abs(leg) <= EPS:
                continue
            notional = abs(leg) * equity
            if notional < min_order_notional:
                ignored_turnover += abs(leg)
                continue
            next_weights[idx] += leg
            turnover += abs(leg)
            order_count += 1

    return next_weights, turnover, order_count, ignored_turnover


def run_execution_backtest(
    timeline: list[datetime],
    returns: np.ndarray,
    target_weights: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    fee: float = 0.0004,
    slippage: float = 0.0001,
    min_order_notional: float = 10.0,
    funding_rates: np.ndarray | None = None,
) -> ExecutionBacktestResult:
    """Simulate order-aware target execution with simple frictions.

    ``target_weights[t]`` is reconciled into realized weights at ``timeline[t]``
    and held to ``timeline[t + 1]``. The simulator is intentionally small and
    deterministic: it models live-like minimum notional handling, sign-flip leg
    splitting, turnover fees, optional slippage, and optional per-period funding.

    Funding convention: positive ``funding_rates`` are paid by longs and earned
    by shorts, so funding PnL is ``-weight * funding_rate``.
    """
    if returns.shape != target_weights.shape:
        raise ValueError("returns and target_weights must have the same shape")
    if len(timeline) != returns.shape[0]:
        raise ValueError("timeline length must match returns rows")
    if funding_rates is not None and funding_rates.shape != returns.shape:
        raise ValueError("funding_rates must match returns shape")
    if min_order_notional < 0:
        raise ValueError("min_order_notional must be non-negative")

    pnl = np.zeros(returns.shape[0], dtype=float)
    turnover = np.zeros(returns.shape[0], dtype=float)
    order_counts = np.zeros(returns.shape[0], dtype=int)
    ignored_turnover = np.zeros(returns.shape[0], dtype=float)
    ignored_gross = np.zeros(returns.shape[0], dtype=float)
    realized = np.zeros_like(target_weights, dtype=float)

    current = np.zeros(returns.shape[1], dtype=float)
    equity = float(initial_equity)

    for t in range(returns.shape[0] - 1):
        target = np.nan_to_num(target_weights[t], nan=0.0)
        current, turnover[t], order_counts[t], ignored_turnover[t] = (
            _apply_target_orders(
                current,
                target,
                equity=equity,
                min_order_notional=min_order_notional,
            )
        )
        ignored_gross[t] = float(np.sum(np.abs(target - current)))
        realized[t] = current

        market_pnl = float(np.sum(current * np.nan_to_num(returns[t + 1], nan=0.0)))
        funding_pnl = 0.0
        if funding_rates is not None:
            funding_pnl = -float(
                np.sum(current * np.nan_to_num(funding_rates[t + 1], nan=0.0))
            )
        cost = turnover[t] * (fee + slippage)
        pnl[t + 1] = market_pnl + funding_pnl - cost
        equity *= 1.0 + pnl[t + 1]

    realized[-1] = current
    equity = initial_equity * np.cumprod(1.0 + pnl)
    curve = list(zip(timeline, equity.tolist()))
    return ExecutionBacktestResult(
        equity_curve=curve,
        returns=pnl,
        turnover=turnover,
        target_weights=target_weights,
        realized_weights=realized,
        order_counts=order_counts,
        ignored_turnover=ignored_turnover,
        ignored_gross=ignored_gross,
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
