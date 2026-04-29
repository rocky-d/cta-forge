"""Tests for target-weight portfolio backtest utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_execution_backtest,
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


def test_execution_backtest_applies_min_notional_and_slippage() -> None:
    timeline = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(4)]
    returns = np.array([[0.0], [0.10], [0.10], [0.10]])
    targets = np.array([[0.05], [0.20], [-0.10], [-0.10]])

    result = run_execution_backtest(
        timeline,
        returns,
        targets,
        initial_equity=100.0,
        fee=0.0,
        slippage=0.01,
        min_order_notional=10.0,
    )

    # Initial $5 target is ignored, then the later $20 target is executed.
    assert result.realized_weights[0, 0] == pytest.approx(0.0)
    assert result.ignored_gross[0] == pytest.approx(0.05)
    assert result.realized_weights[1, 0] == pytest.approx(0.20)
    assert result.returns[2] == pytest.approx(0.20 * 0.10 - 0.20 * 0.01)
    # Sign flip is split into a close leg plus an open-short leg.
    assert result.order_counts[2] == 2
    assert result.realized_weights[2, 0] == pytest.approx(-0.10)


def test_execution_backtest_applies_funding_rates() -> None:
    timeline = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(2)]
    returns = np.zeros((2, 2))
    targets = np.array([[0.5, -0.25], [0.5, -0.25]])
    funding = np.array([[0.0, 0.0], [0.001, 0.002]])

    result = run_execution_backtest(
        timeline,
        returns,
        targets,
        initial_equity=100.0,
        fee=0.0,
        slippage=0.0,
        min_order_notional=0.0,
        funding_rates=funding,
    )

    # Long pays 0.5 * 0.001, short earns 0.25 * 0.002; net zero.
    assert result.returns[1] == pytest.approx(0.0)


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
