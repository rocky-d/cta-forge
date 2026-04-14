"""End-to-end pipeline integration test."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from alpha_service.factors.momentum import TSMOMFactor
from engine_service.backtest import BacktestEngine
from reporter_service.metrics import calculate_metrics
from strategy_service.allocator import allocate_positions
from strategy_service.composer import compose_signals


def _generate_market_data(
    n_symbols: int = 5, n_bars: int = 200
) -> dict[str, pl.DataFrame]:
    """Generate synthetic market data with varying trends."""
    np.random.seed(42)
    bars = {}
    for i in range(n_symbols):
        symbol = f"SYM{i}USDT"
        trend = np.random.uniform(-0.002, 0.003)  # Random trend per symbol
        rows = []
        price = 100.0 + np.random.uniform(-20, 20)
        for j in range(n_bars):
            price = price * (1 + trend + np.random.normal(0, 0.015))
            rows.append(
                {
                    "open_time": datetime(2024, 1, 1, tzinfo=UTC)
                    + timedelta(hours=j * 6),
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0 + np.random.uniform(0, 500),
                }
            )
        bars[symbol] = pl.DataFrame(rows)
    return bars


class TestEndToEndPipeline:
    """Integration tests for the complete trading pipeline."""

    def test_full_backtest_pipeline(self):
        """Test: data → alpha → strategy → backtest → report."""
        # 1. Generate data
        market_data = _generate_market_data(n_symbols=5, n_bars=150)

        # 2. Setup factor
        tsmom = TSMOMFactor(lookback=20)

        # 3. Run backtest with integrated pipeline
        engine = BacktestEngine(initial_equity=10000.0)

        def compute_signal(symbol: str, bars: pl.DataFrame) -> float:
            result = tsmom.compute(bars)
            if result.is_empty():
                return 0.0
            return float(result["signal"][-1])

        def allocate(signals: dict[str, float], equity: float) -> dict[str, float]:
            return allocate_positions(signals, equity, max_position_pct=0.2)

        result = engine.run(
            bars=market_data,
            compute_signals=compute_signal,
            allocate=allocate,
        )

        # 4. Calculate metrics
        metrics = calculate_metrics(result.equity_curve, result.trades)

        # Assertions
        assert result.final_equity > 0
        assert len(result.equity_curve) == 150
        assert metrics.num_trades >= 0
        assert 0 <= metrics.max_drawdown <= 1

    def test_multi_factor_composition(self):
        """Test multi-factor signal composition."""
        signals = {
            "SYM0USDT": {"tsmom_20": 0.7, "breakout_15": 0.3},
            "SYM1USDT": {"tsmom_20": -0.5, "breakout_15": 0.8},
            "SYM2USDT": {"tsmom_20": 0.2, "breakout_15": -0.4},
        }
        weights = {"tsmom_20": 2.0, "breakout_15": 1.0}

        composite = compose_signals(signals, weights)

        assert len(composite) == 3
        assert all(-1 <= v <= 1 for v in composite.values())

        # Allocate based on composite
        positions = allocate_positions(composite, equity=10000.0)
        assert all(isinstance(v, float) for v in positions.values())

    def test_backtest_with_declining_market(self):
        """Test backtest handles losing scenarios gracefully."""
        # Generate strongly declining market
        np.random.seed(123)
        bars = {}
        for i in range(3):
            symbol = f"BEAR{i}USDT"
            rows = []
            price = 100.0
            for j in range(100):
                price = price * (1 - 0.005 + np.random.normal(0, 0.01))
                rows.append(
                    {
                        "open_time": datetime(2024, 1, 1, tzinfo=UTC)
                        + timedelta(hours=j * 6),
                        "open": price * 1.001,
                        "high": price * 1.02,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )
            bars[symbol] = pl.DataFrame(rows)

        engine = BacktestEngine(initial_equity=10000.0)
        result = engine.run(
            bars=bars,
            compute_signals=lambda s, b: 0.5,  # Always bullish (wrong in this market)
            allocate=lambda sigs, eq: {s: sigs[s] * eq * 0.2 for s in sigs},
        )

        # Should still complete
        assert result.final_equity > 0
        assert result.max_drawdown > 0.1  # Should have significant drawdown

    @pytest.mark.asyncio
    async def test_async_compatible(self):
        """Verify components work in async context."""
        market_data = _generate_market_data(n_symbols=2, n_bars=50)

        # Simulate async data fetch
        async def fetch_data():
            await asyncio.sleep(0.01)
            return market_data

        data = await fetch_data()
        assert len(data) == 2
        assert all(len(df) == 50 for df in data.values())
