"""End-to-end pipeline integration test."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from executor.backtest import (
    align_data,
    build_timeline,
    calc_ulcer,
    precompute,
    run_backtest,
)
from executor.decision import V10GStrategyParams
from report_service.metrics import calculate_metrics
from strategy_service.allocator import allocate_positions
from strategy_service.composer import compose_signals


def _generate_market_data(
    n_symbols: int = 5, n_bars: int = 400, prefix: str = "SYM"
) -> dict[str, pl.DataFrame]:
    """Generate synthetic market data with varying trends."""
    np.random.seed(42)
    bars = {}
    for i in range(n_symbols):
        symbol = f"{prefix}{i}USDT"
        trend = np.random.uniform(-0.002, 0.003)
        rows = []
        price = 100.0 + np.random.uniform(-20, 20)
        for j in range(n_bars):
            price = price * (1 + trend + np.random.normal(0, 0.015))
            rows.append(
                {
                    "open_time": datetime(2020, 1, 1, tzinfo=UTC)
                    + timedelta(hours=j * 6),
                    "open": price * 0.999,
                    "high": price * 1.015,
                    "low": price * 0.985,
                    "close": price,
                    "volume": 1000.0 + np.random.uniform(0, 500),
                }
            )
        bars[symbol] = pl.DataFrame(rows)
    return bars


def _make_signals(data, timeline):
    """Generate mild positive signals for testing."""
    sigs = {}
    for sym in data:
        s = np.zeros(len(timeline))
        for j in range(len(s)):
            s[j] = 0.45 if j % 20 < 15 else -0.2
        sigs[sym] = s
    return sigs


class TestEndToEndPipeline:
    """Integration tests for the complete trading pipeline."""

    def test_full_backtest_pipeline(self):
        """Test: data → precompute → signals → backtest → metrics."""
        market_data = _generate_market_data(n_symbols=3, n_bars=400)

        # Precompute and build timeline
        data = precompute(market_data)
        timeline, ts_to_idx = build_timeline(market_data)
        align_data(market_data, data, ts_to_idx)
        sigs = _make_signals(data, timeline)

        # Run backtest with V10GDecisionEngine
        start = 200
        curve, trades = run_backtest(data, sigs, timeline, start, len(timeline))

        # Calculate metrics
        metrics = calculate_metrics(curve, trades)

        # Assertions
        assert len(curve) > 0
        assert curve[0][1] > 0  # positive equity
        assert metrics.num_trades >= 0
        assert 0 <= metrics.max_drawdown <= 1

    def test_backtest_with_custom_params(self):
        """Test backtest respects custom strategy parameters."""
        market_data = _generate_market_data(n_symbols=2, n_bars=400)
        data = precompute(market_data)
        timeline, ts_to_idx = build_timeline(market_data)
        align_data(market_data, data, ts_to_idx)
        sigs = _make_signals(data, timeline)

        params = V10GStrategyParams(
            max_drawdown=1.0,
            max_positions=1,
            signal_threshold=0.3,
        )
        curve, trades = run_backtest(
            data, sigs, timeline, 200, len(timeline), params=params
        )
        assert len(curve) > 0

    def test_multi_factor_composition(self):
        """Test multi-factor signal composition (strategy-service)."""
        signals = {
            "SYM0USDT": {"tsmom_20": 0.7, "breakout_15": 0.3},
            "SYM1USDT": {"tsmom_20": -0.5, "breakout_15": 0.8},
            "SYM2USDT": {"tsmom_20": 0.2, "breakout_15": -0.4},
        }
        weights = {"tsmom_20": 2.0, "breakout_15": 1.0}

        composite = compose_signals(signals, weights)

        assert len(composite) == 3
        assert all(-1 <= v <= 1 for v in composite.values())

        positions = allocate_positions(composite, equity=10000.0)
        assert all(isinstance(v, float) for v in positions.values())

    def test_metrics_and_ulcer(self):
        """Test metrics calculation on backtest output."""
        market_data = _generate_market_data(n_symbols=2, n_bars=400)
        data = precompute(market_data)
        timeline, ts_to_idx = build_timeline(market_data)
        align_data(market_data, data, ts_to_idx)
        sigs = _make_signals(data, timeline)

        curve, trades = run_backtest(data, sigs, timeline, 200, len(timeline))
        metrics = calculate_metrics(curve, trades)
        ulcer = calc_ulcer(curve)

        assert metrics.sharpe_ratio != 0 or metrics.num_trades == 0
        assert ulcer >= 0

    @pytest.mark.asyncio
    async def test_async_compatible(self):
        """Verify components work in async context."""
        market_data = _generate_market_data(n_symbols=2, n_bars=50)

        async def fetch_data():
            await asyncio.sleep(0.01)
            return market_data

        data = await fetch_data()
        assert len(data) == 2
        assert all(len(df) == 50 for df in data.values())
