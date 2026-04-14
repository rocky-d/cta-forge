"""Tests for alpha factors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from alpha_service.factors.breakout import DonchianBreakoutFactor
from alpha_service.factors.carry import FundingRateCarryFactor
from alpha_service.factors.momentum import TSMOMFactor
from alpha_service.factors.volatility import VolatilityRegimeFactor


def _make_trending_bars(n: int = 100, direction: float = 1.0) -> pl.DataFrame:
    """Create bars with a clear trend."""
    base = 100.0
    rows = []
    for i in range(n):
        price = base + direction * i * 0.5 + np.random.normal(0, 0.1)
        rows.append(
            {
                "open_time": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6),
                "open": price,
                "high": price + abs(np.random.normal(0, 0.5)),
                "low": price - abs(np.random.normal(0, 0.5)),
                "close": price + direction * 0.1,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def _make_bars_with_timestamps(n: int = 100) -> pl.DataFrame:
    """Create bars with proper timestamps."""
    base = 100.0
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC)]
    for i in range(1, n):
        timestamps.append(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6))

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n)
    closes = base * np.cumprod(1 + returns)

    return pl.DataFrame(
        {
            "open_time": timestamps,
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.full(n, 1000.0),
        }
    )


class TestTSMOM:
    def test_basic_output_shape(self):
        bars = _make_bars_with_timestamps(100)
        factor = TSMOMFactor(lookback=30)
        result = factor.compute(bars)

        assert "open_time" in result.columns
        assert "signal" in result.columns
        assert len(result) == 70  # 100 - 30

    def test_signal_range(self):
        bars = _make_bars_with_timestamps(200)
        factor = TSMOMFactor(lookback=30)
        result = factor.compute(bars)

        signals = result["signal"].to_numpy()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_uptrend_positive_signal(self):
        bars = _make_trending_bars(100, direction=1.0)
        factor = TSMOMFactor(lookback=20)
        result = factor.compute(bars)

        # Most signals should be positive in an uptrend
        signals = result["signal"].to_numpy()
        assert np.mean(signals > 0) > 0.7

    def test_too_few_bars(self):
        bars = _make_bars_with_timestamps(10)
        factor = TSMOMFactor(lookback=30)
        result = factor.compute(bars)
        assert result.is_empty()

    def test_name(self):
        factor = TSMOMFactor(lookback=30)
        assert factor.name == "tsmom_30"


class TestDonchianBreakout:
    def test_basic_output(self):
        bars = _make_bars_with_timestamps(100)
        factor = DonchianBreakoutFactor(period=15, adx_threshold=0)
        result = factor.compute(bars)

        assert "signal" in result.columns
        signals = result["signal"].to_numpy()
        assert set(np.unique(signals)).issubset({-1.0, 0.0, 1.0})

    def test_name(self):
        factor = DonchianBreakoutFactor(period=15)
        assert factor.name == "breakout_15"

    def test_too_few_bars(self):
        bars = _make_bars_with_timestamps(5)
        factor = DonchianBreakoutFactor(period=15)
        result = factor.compute(bars)
        assert result.is_empty()


class TestFundingRateCarry:
    def test_no_funding_column(self):
        bars = _make_bars_with_timestamps(50)
        factor = FundingRateCarryFactor()
        result = factor.compute(bars)

        # Should return zero signals
        assert len(result) == 50
        assert (result["signal"] == 0.0).all()

    def test_with_funding_rate(self):
        bars = _make_bars_with_timestamps(50)
        # Add positive funding rate (longs pay shorts)
        bars = bars.with_columns(pl.lit(0.001).alias("funding_rate"))
        factor = FundingRateCarryFactor(lookback=8)
        result = factor.compute(bars)

        # Positive funding → contrarian → should be negative signal
        signals = result["signal"].to_numpy()
        assert np.all(signals[8:] <= 0)

    def test_name(self):
        assert FundingRateCarryFactor(lookback=8).name == "carry_8"


class TestVolatilityRegime:
    def test_basic_output(self):
        bars = _make_bars_with_timestamps(100)
        factor = VolatilityRegimeFactor(short_window=10, long_window=50)
        result = factor.compute(bars)

        assert "signal" in result.columns
        signals = result["signal"].to_numpy()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_name(self):
        factor = VolatilityRegimeFactor(short_window=10, long_window=50)
        assert factor.name == "vol_regime_10_50"

    def test_too_few_bars(self):
        bars = _make_bars_with_timestamps(20)
        factor = VolatilityRegimeFactor(long_window=50)
        result = factor.compute(bars)
        assert result.is_empty()
