"""Experiment definition and execution for backtest research.

Provides declarative config dataclasses and a thin orchestration function.
No strategy knowledge — this module delegates to engine.py and metrics.py.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .engine import run_target_weight_backtest
from .metrics import compute_metrics
from .result import BacktestMetrics, TargetBacktestResult


# ── Config dataclasses (pure data, serializable) ─────────────────


@dataclass
class DataConfig:
    """Data source specification."""

    path: str | Path = "data"
    timeframe: str = "1h"
    min_bars_per_symbol: int = 500


@dataclass
class TimeRangeConfig:
    """Time range for a backtest run."""

    start: datetime | None = None
    end: datetime | None = None
    warmup_bars: int = 150


@dataclass
class AccountConfig:
    """Account and cost parameters."""

    initial_equity: float = 10_000.0
    fee: float = 0.000432  # HL taker fee
    slippage: float = 0.0001
    min_order_notional: float = 10.0


@dataclass
class StrategyConfig:
    """Strategy parameter overrides.

    Any field left as None uses the profile default.
    """

    gross_cap: float | None = None
    target_scale: float | None = None
    dd_circuit_breaker: float | None = None
    core_phase_hours: int | None = None
    gate_rolling_years: float | None = None
    v10g_allocation: float | None = None
    overlay_allocation: float | None = None


@dataclass
class OutputConfig:
    """Output and artifact settings."""

    base_dir: str | Path = "backtest-results"
    subdir: str | None = None  # None → auto "{name}_{timestamp}"


# ── Experiment definition ────────────────────────────────────────


@dataclass
class BacktestExperiment:
    """A single backtest experiment configuration.

    Pure data — no side effects. Call ``run_backtest()`` to execute.
    """

    name: str
    symbols: list[str]
    timeline: list[datetime]
    returns: np.ndarray  # T×S matrix
    target_weights: np.ndarray  # T×S matrix
    data: DataConfig = field(default_factory=DataConfig)
    time_range: TimeRangeConfig = field(default_factory=TimeRangeConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    meta: dict[str, Any] = field(default_factory=dict)


# ── Execution ────────────────────────────────────────────────────


def run_backtest(
    experiment: BacktestExperiment,
) -> tuple[TargetBacktestResult, BacktestMetrics]:
    """Run a backtest experiment through the target-weight engine.

    Returns:
        (result, metrics)
    """
    result = run_target_weight_backtest(
        timeline=experiment.timeline,
        returns=experiment.returns,
        target_weights=experiment.target_weights,
        initial_equity=experiment.account.initial_equity,
        fee=experiment.account.fee,
    )
    metrics = compute_metrics(
        result.returns,
        initial_equity=experiment.account.initial_equity,
        periods_per_year=365 * 24,
        weights=result.target_weights,
    )
    return result, metrics


def compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Compute drawdown as a fraction of running peak.

    Returns a numpy array the same length as ``equity``, where each value is
    the drawdown fraction from the current running maximum (0.0 = at peak,
    0.05 = 5% below peak). Positive magnitudes, consistent with
    ``BacktestMetrics.max_drawdown`` and ``ChartSeries.drawdown``.
    """
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / running_max
    dd[dd < 0] = 0
    return dd


def compute_monthly_returns(
    timeline: list[datetime], equity: np.ndarray
) -> dict[str, float]:
    """Compute monthly PnL from equity curve (fraction, NOT percent)."""
    result: dict[str, float] = {}
    if len(timeline) < 2:
        return result
    months = sorted({(t.year, t.month) for t in timeline})
    for y, m in months:
        indices = [i for i, t in enumerate(timeline) if (t.year, t.month) == (y, m)]
        if len(indices) < 2:
            continue
        start_val = equity[indices[0]]
        if start_val is None or start_val == 0:
            continue
        key = f"{y}-{m:02d}"
        result[key] = float(equity[indices[-1]] / start_val - 1)
    return result


def save_experiment_artifacts(
    experiment: BacktestExperiment,
    metrics: BacktestMetrics,
    *,
    chart_path: str | Path | None = None,
) -> Path:
    """Save metrics.json and experiment.json to the output directory.

    Returns:
        The output directory path.
    """
    subdir = experiment.output.subdir or (
        f"{experiment.name.replace(' ', '_')}_"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = Path(experiment.output.base_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "name": experiment.name,
        "symbols": experiment.symbols,
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "volatility": metrics.volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "calmar_ratio": metrics.calmar_ratio,
        "ulcer_index": metrics.ulcer_index,
        "tail_ratio": metrics.tail_ratio,
        "avg_gross_exposure": metrics.avg_gross_exposure,
        "avg_turnover": metrics.avg_turnover,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    git_commit = _git_commit_hash()

    config_payload = {
        "name": experiment.name,
        "symbols": experiment.symbols,
        "data": asdict(experiment.data),
        "time_range": _serialize_time_range(experiment.time_range),
        "account": asdict(experiment.account),
        "strategy": asdict(experiment.strategy),
        "meta": experiment.meta,
    }
    if git_commit:
        config_payload["git_commit"] = git_commit
    (out_dir / "experiment.json").write_text(
        json.dumps(config_payload, indent=2, default=str)
    )

    return out_dir


def _serialize_time_range(tr: TimeRangeConfig) -> dict[str, Any]:
    return {
        "start": tr.start.isoformat() if tr.start else None,
        "end": tr.end.isoformat() if tr.end else None,
        "warmup_bars": tr.warmup_bars,
    }


def _git_commit_hash() -> str | None:
    """Return short commit hash if available, None on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
