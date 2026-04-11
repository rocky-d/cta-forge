"""Tests for backtest engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from engine.backtest import BacktestEngine


def _make_bars(n: int = 100, base: float = 100.0, trend: float = 0.001) -> pl.DataFrame:
    np.random.seed(42)
    rows = []
    price = base
    for i in range(n):
        price = price * (1 + trend + np.random.normal(0, 0.01))
        rows.append(
            {
                "open_time": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6),
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


class TestBacktestEngine:
    def test_basic_run(self):
        bars = {"BTCUSDT": _make_bars(100, trend=0.002)}
        engine = BacktestEngine(initial_equity=10000.0)

        result = engine.run(
            bars=bars,
            compute_signals=lambda sym, b: 0.5 if len(b) > 30 else 0.0,
            allocate=lambda sigs, eq: {s: sigs[s] * eq * 0.5 for s in sigs},
        )

        assert result.final_equity > 0
        assert len(result.equity_curve) == 100

    def test_empty_bars(self):
        engine = BacktestEngine()
        result = engine.run(bars={}, compute_signals=lambda s, b: 0, allocate=lambda s, e: {})
        assert result.final_equity == 0.0

    def test_commission_applied(self):
        bars = {"BTCUSDT": _make_bars(50)}
        engine = BacktestEngine(initial_equity=10000.0, commission_rate=0.001)

        trade_count = [0]

        def allocate(sigs, eq):
            trade_count[0] += 1
            return {"BTCUSDT": 5000.0 if trade_count[0] % 2 == 0 else 0.0}

        result = engine.run(
            bars=bars,
            compute_signals=lambda s, b: 0.5,
            allocate=allocate,
        )

        # Should have some trades
        assert result.final_equity != 10000.0

    def test_max_drawdown_calculated(self):
        # Create declining bars
        bars = {"BTCUSDT": _make_bars(100, trend=-0.002)}
        engine = BacktestEngine(initial_equity=10000.0)

        result = engine.run(
            bars=bars,
            compute_signals=lambda s, b: 0.5,
            allocate=lambda sigs, eq: {s: sigs[s] * eq * 0.3 for s in sigs},
        )

        assert result.max_drawdown > 0  # Should have drawdown in declining market
