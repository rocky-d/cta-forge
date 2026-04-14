"""Performance metrics calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    num_trades: int = 0


def calculate_metrics(
    equity_curve: list[tuple],
    trades: list[dict],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365 * 4,  # 6h bars
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        equity_curve: list of (timestamp, equity) tuples.
        trades: list of trade dicts with 'pnl' key.
        risk_free_rate: annual risk-free rate.
        periods_per_year: number of data periods per year.

    Returns:
        PerformanceMetrics dataclass.
    """
    if len(equity_curve) < 2:
        return PerformanceMetrics()

    equities = np.array([e[1] for e in equity_curve])
    returns = np.diff(equities) / equities[:-1]

    # Basic return metrics
    total_return = (equities[-1] - equities[0]) / equities[0]
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0

    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 1e-10 else 0.0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(periods_per_year)
        if len(downside_returns) > 1
        else 0.0
    )
    sortino = excess_return / downside_std if downside_std > 1e-10 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Calmar ratio
    calmar = annualized_return / max_dd if max_dd > 1e-10 else 0.0

    # Trade statistics
    pnls = [t.get("pnl", 0.0) for t in trades]
    num_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
    avg_pnl = np.mean(pnls) if pnls else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 1e-10
        else float("inf")
        if gross_profit > 0
        else 0.0
    )

    return PerformanceMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        volatility=float(volatility),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        calmar_ratio=float(calmar),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        avg_trade_pnl=float(avg_pnl),
        num_trades=num_trades,
    )
