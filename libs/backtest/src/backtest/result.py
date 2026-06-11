"""Backtest result types and metrics dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


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


@dataclass
class BacktestMetrics:
    """Unified performance metrics computed from a backtest result.

    All rates are decimal fractions (e.g. 0.15 = 15%), not percentages.
    Drawdown values use the project convention: positive magnitudes from peak.
    """

    # Returns
    total_return: float
    annualized_return: float
    volatility: float

    # Risk
    max_drawdown: float
    max_dd_duration_bars: int = 0
    avg_drawdown: float = 0.0
    ulcer_index: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    tail_ratio: float = 0.0

    # Portfolio (weight-driven)
    avg_gross_exposure: float | None = None
    avg_net_exposure: float | None = None
    avg_turnover: float | None = None

    # Trade-level (trade-driven, optional)
    num_trades: int | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_trade_pnl: float | None = None

    # Reference data (for downstream charting / storage)
    equity_curve: list[float] | None = None
    drawdown_curve: list[float] | None = None
    monthly_returns: dict[str, float] | None = None


@dataclass
class ChartSeries:
    """Data for one line/series in a comparison chart.

    The chart module consumes this without knowing anything about strategies,
    profiles, or data sources.
    """

    label: str
    color: str
    equity: np.ndarray  # normalized (start = 1.0)
    drawdown: np.ndarray  # positive fraction from peak
    monthly_returns: dict[str, float]  # "YYYY-MM" → return fraction
    metrics: BacktestMetrics | None = None
    timestamps: list[datetime] | None = None
