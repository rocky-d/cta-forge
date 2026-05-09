"""Performance metrics calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

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


@dataclass
class LivePerformanceMetrics:
    """Performance metrics for timestamped live journals.

    Live journals can have irregular cadence because of deploys, outages, or
    profile changes. These metrics therefore annualize returns from real
    elapsed wall-clock time instead of assuming one equity record is one bar.
    """

    metrics: PerformanceMetrics
    annualized_return_raw: float | None = None
    annualized_status: str = "insufficient"
    elapsed_days: float = 0.0
    cadence_median_hours: float | None = None
    record_count: int = 0


SECONDS_PER_YEAR = 365 * 24 * 60 * 60
UNSTABLE_ANNUALIZATION_DAYS = 7.0
SHORT_SAMPLE_ANNUALIZATION_DAYS = 30.0


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


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

    # Max drawdown (positive magnitude from peak, not negative underwater value)
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


def calculate_live_metrics(
    equity_curve: list[tuple],
    trades: list[dict],
    risk_free_rate: float = 0.0,
) -> LivePerformanceMetrics:
    """Calculate live-journal metrics using actual elapsed time.

    The returned ``metrics.annualized_return`` is suppressed to ``0.0`` until
    the sample reaches the stable threshold to avoid presenting a noisy
    extrapolation as a reliable figure. The unsuppressed timestamp-based value
    remains available in ``annualized_return_raw`` for diagnostics.
    """
    if len(equity_curve) < 2:
        return LivePerformanceMetrics(
            metrics=PerformanceMetrics(num_trades=len(trades)),
            record_count=len(equity_curve),
        )

    curve = sorted((_coerce_datetime(ts), float(eq)) for ts, eq in equity_curve)
    timestamps = [ts for ts, _eq in curve]
    equities = np.array([eq for _ts, eq in curve])
    returns = np.diff(equities) / equities[:-1]
    elapsed_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    elapsed_days = elapsed_seconds / (24 * 60 * 60) if elapsed_seconds > 0 else 0.0
    deltas = [
        (right - left).total_seconds() / (60 * 60)
        for left, right in zip(timestamps, timestamps[1:])
        if (right - left).total_seconds() > 0
    ]
    cadence_median_hours = float(np.median(deltas)) if deltas else None

    total_return = float(equities[-1] / equities[0] - 1.0) if equities[0] > 0 else 0.0
    annualized_return_raw: float | None = None
    if elapsed_seconds > 0 and equities[0] > 0 and equities[-1] > 0:
        annualized_return_raw = float(
            (equities[-1] / equities[0]) ** (SECONDS_PER_YEAR / elapsed_seconds) - 1.0
        )

    if elapsed_days <= 0 or annualized_return_raw is None:
        annualized_status = "insufficient"
        reported_annualized = 0.0
    elif elapsed_days < UNSTABLE_ANNUALIZATION_DAYS:
        annualized_status = "unstable"
        reported_annualized = 0.0
    elif elapsed_days < SHORT_SAMPLE_ANNUALIZATION_DAYS:
        annualized_status = "short_sample"
        reported_annualized = 0.0
    else:
        annualized_status = "ok"
        reported_annualized = annualized_return_raw

    periods_per_year = (
        365 * 24 / cadence_median_hours if cadence_median_hours else 365 * 24
    )
    volatility = (
        float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    )
    excess_return = reported_annualized - risk_free_rate
    sharpe = excess_return / volatility if volatility > 1e-10 else 0.0

    downside_returns = returns[returns < 0]
    downside_std = (
        float(downside_returns.std() * np.sqrt(periods_per_year))
        if len(downside_returns) > 1
        else 0.0
    )
    sortino = excess_return / downside_std if downside_std > 1e-10 else 0.0

    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    calmar = reported_annualized / max_dd if max_dd > 1e-10 else 0.0

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

    return LivePerformanceMetrics(
        metrics=PerformanceMetrics(
            total_return=total_return,
            annualized_return=reported_annualized,
            volatility=volatility,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=max_dd,
            calmar_ratio=float(calmar),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_trade_pnl=float(avg_pnl),
            num_trades=num_trades,
        ),
        annualized_return_raw=annualized_return_raw,
        annualized_status=annualized_status,
        elapsed_days=elapsed_days,
        cadence_median_hours=cadence_median_hours,
        record_count=len(curve),
    )
