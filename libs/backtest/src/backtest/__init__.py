"""Backtest shared library for cta-forge.

Provides simulation engines, unified metrics, chart generation,
and experiment orchestration as pure functions with no dependency
on service-layer strategy code.
"""

from backtest.chart import (
    DEFAULT_COLORS,
    DEFAULT_FIGSIZE,
    DEFAULT_PANELS,
    MAX_CONFIGS,
    PanelSpec,
    create_comparison_figure,
    save_figure,
)
from backtest.engine import run_execution_backtest, run_target_weight_backtest
from backtest.experiment import (
    AccountConfig,
    BacktestExperiment,
    DataConfig,
    OutputConfig,
    StrategyConfig,
    TimeRangeConfig,
    compute_drawdown_series,
    compute_monthly_returns,
    run_backtest,
    save_experiment_artifacts,
)
from backtest.metrics import calculate_hourly_metrics, compute_metrics
from backtest.result import (
    BacktestMetrics,
    ChartSeries,
    ExecutionBacktestResult,
    TargetBacktestResult,
)

__all__ = [
    # Engine
    "run_target_weight_backtest",
    "run_execution_backtest",
    # Metrics
    "compute_metrics",
    "calculate_hourly_metrics",
    # Result types
    "BacktestMetrics",
    "ChartSeries",
    "ExecutionBacktestResult",
    "TargetBacktestResult",
    # Chart
    "create_comparison_figure",
    "save_figure",
    "PanelSpec",
    "DEFAULT_COLORS",
    "DEFAULT_FIGSIZE",
    "DEFAULT_PANELS",
    "MAX_CONFIGS",
    # Experiment
    "BacktestExperiment",
    "DataConfig",
    "TimeRangeConfig",
    "AccountConfig",
    "StrategyConfig",
    "OutputConfig",
    "run_backtest",
    "save_experiment_artifacts",
    "compute_drawdown_series",
    "compute_monthly_returns",
]
