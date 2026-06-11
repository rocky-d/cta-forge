"""Backtest shared library for cta-forge.

Provides simulation engines, unified metrics, and chart generation
as pure functions with no dependency on service-layer code.
"""

from backtest.engine import run_execution_backtest, run_target_weight_backtest
from backtest.metrics import calculate_hourly_metrics, compute_metrics
from backtest.result import (
    BacktestMetrics,
    ChartSeries,
    ExecutionBacktestResult,
    TargetBacktestResult,
)

__all__ = [
    "BacktestMetrics",
    "ChartSeries",
    "ExecutionBacktestResult",
    "TargetBacktestResult",
    "calculate_hourly_metrics",
    "compute_metrics",
    "run_execution_backtest",
    "run_target_weight_backtest",
]
