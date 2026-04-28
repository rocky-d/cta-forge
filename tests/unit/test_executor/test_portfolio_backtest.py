"""Tests for target-weight portfolio backtest utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_target_weight_backtest,
)


def test_target_weight_backtest_uses_next_bar_returns_and_fees() -> None:
    timeline = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(4)]
    returns = np.array(
        [
            [0.0],
            [0.10],
            [-0.05],
            [0.02],
        ]
    )
    targets = np.array(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ]
    )

    result = run_target_weight_backtest(
        timeline, returns, targets, initial_equity=100.0, fee=0.01
    )

    # t0 target is held into t1: +10%, less 1% initial turnover fee.
    assert result.returns[1] == pytest.approx(0.09)
    # t1 target is held into t2: -5%, no turnover.
    assert result.returns[2] == pytest.approx(-0.05)
    # t2 exits to flat and pays 1% turnover, then no market exposure.
    assert result.returns[3] == pytest.approx(-0.01)
    assert result.equity_curve[-1][1] == pytest.approx(100.0 * 1.09 * 0.95 * 0.99)


def test_target_weight_backtest_validates_shapes() -> None:
    timeline = [datetime(2024, 1, 1, tzinfo=UTC)]
    returns = np.zeros((1, 2))
    targets = np.zeros((1, 1))

    with pytest.raises(ValueError, match="same shape"):
        run_target_weight_backtest(timeline, returns, targets)


def test_calculate_hourly_metrics_reports_drawdown() -> None:
    metrics = calculate_hourly_metrics(
        np.array([0.0, 0.10, -0.05]), initial_equity=100.0
    )

    assert metrics["return"] == pytest.approx(0.045)
    assert metrics["max_dd"] == pytest.approx(-0.05)
    assert metrics["sharpe"] > 0
