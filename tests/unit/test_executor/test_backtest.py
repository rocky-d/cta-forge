"""Tests for backtest engine (V10GDecisionEngine-based)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from executor.backtest import (
    BacktestResult,
    align_data,
    build_timeline,
    calc_ulcer,
    precompute,
    run_backtest,
)
from executor.decision import V10GStrategyParams


def _make_bars(
    n: int = 300,
    base: float = 100.0,
    trend: float = 0.001,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV bars."""
    np.random.seed(seed)
    rows = []
    price = base
    for i in range(n):
        price = price * (1 + trend + np.random.normal(0, 0.02))
        rows.append(
            {
                "open_time": datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6),
                "open": price * 0.999,
                "high": price * 1.015,
                "low": price * 0.985,
                "close": price,
                "volume": 1000.0 + np.random.uniform(0, 500),
            }
        )
    return pl.DataFrame(rows)


class TestBuildTimeline:
    def test_single_symbol(self):
        bars = {"BTCUSDT": _make_bars(50)}
        timeline, ts_to_idx = build_timeline(bars)
        assert len(timeline) == 50
        assert len(ts_to_idx) == 50

    def test_multi_symbol_union(self):
        bars1 = _make_bars(50, seed=1)
        bars2 = _make_bars(50, seed=1)  # same timestamps
        timeline, _ = build_timeline({"A": bars1, "B": bars2})
        assert len(timeline) == 50

    def test_timeline_sorted(self):
        bars = {"X": _make_bars(100)}
        timeline, _ = build_timeline(bars)
        for i in range(1, len(timeline)):
            assert timeline[i] > timeline[i - 1]


class TestPrecompute:
    def test_produces_required_keys(self):
        bars = {"BTCUSDT": _make_bars(200)}
        data = precompute(bars)
        assert "BTCUSDT" in data
        d = data["BTCUSDT"]
        assert "close" in d
        assert "atr" in d
        assert "start_idx" in d
        assert "length" in d
        assert d["length"] == 200


class TestRunBacktest:
    def _setup(self, n: int = 400, n_symbols: int = 2):
        """Create bars, precompute, build timeline, compute dummy signals."""
        bars = {}
        for i in range(n_symbols):
            sym = f"SYM{i}USDT"
            bars[sym] = _make_bars(n, base=100 + i * 50, seed=42 + i)

        data = precompute(bars)
        timeline, ts_to_idx = build_timeline(bars)
        align_data(bars, data, ts_to_idx)

        # Simple signals: alternating positive/negative
        sigs = {}
        for sym in data:
            s = np.zeros(len(timeline))
            for j in range(len(s)):
                # Mild positive signal for trending bars
                s[j] = 0.5 if j % 20 < 15 else -0.3
            sigs[sym] = s

        return data, sigs, timeline

    def test_basic_run_returns_curve_and_trades(self):
        data, sigs, timeline = self._setup(400, 2)
        curve, trades = run_backtest(data, sigs, timeline, 200, len(timeline))
        assert len(curve) > 0
        assert isinstance(curve[0], tuple)
        assert isinstance(curve[0][0], datetime)
        assert isinstance(curve[0][1], float)

    def test_equity_starts_near_initial(self):
        data, sigs, timeline = self._setup(400, 1)
        curve, _ = run_backtest(
            data, sigs, timeline, 200, len(timeline), initial_equity=50_000.0
        )
        # First bar equity should be close to initial (no trades yet on first bar)
        assert abs(curve[0][1] - 50_000.0) < 5_000.0

    def test_custom_params(self):
        data, sigs, timeline = self._setup(400, 1)
        params = V10GStrategyParams(
            max_drawdown=1.0,
            max_positions=2,
            signal_threshold=0.3,
        )
        curve, trades = run_backtest(
            data, sigs, timeline, 200, len(timeline), params=params
        )
        assert len(curve) > 0

    def test_empty_data_returns_empty(self):
        curve, trades = run_backtest({}, {}, [], 0, 0)
        assert curve == []
        assert trades == []


class TestCalcUlcer:
    def test_flat_curve(self):
        now = datetime(2024, 1, 1, tzinfo=UTC)
        curve = [(now + timedelta(hours=i), 10000.0) for i in range(50)]
        assert calc_ulcer(curve) == pytest.approx(0.0, abs=1e-10)

    def test_declining_curve(self):
        now = datetime(2024, 1, 1, tzinfo=UTC)
        curve = [(now + timedelta(hours=i), 10000.0 - i * 50) for i in range(50)]
        assert calc_ulcer(curve) > 0

    def test_short_curve(self):
        now = datetime(2024, 1, 1, tzinfo=UTC)
        curve = [(now, 100.0)]
        assert calc_ulcer(curve) == 999.0


class TestBacktestResult:
    def test_defaults(self):
        r = BacktestResult()
        assert r.equity_curve == []
        assert r.trades == []
        assert r.initial_equity == 10_000.0
