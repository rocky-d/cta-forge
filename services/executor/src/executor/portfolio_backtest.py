"""Target-weight portfolio backtest utilities.

This module is a backward-compatibility shim.  New code should import
directly from ``backtest``:

    from backtest import run_target_weight_backtest, compute_metrics
"""

from __future__ import annotations

from backtest.engine import (
    _apply_target_orders,
    run_execution_backtest,
    run_target_weight_backtest,
)
from backtest.metrics import calculate_hourly_metrics
from backtest.result import ExecutionBacktestResult, TargetBacktestResult

__all__ = [
    "_apply_target_orders",
    "ExecutionBacktestResult",
    "TargetBacktestResult",
    "calculate_hourly_metrics",
    "run_execution_backtest",
    "run_target_weight_backtest",
]
