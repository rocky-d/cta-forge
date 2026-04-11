"""Tests for strategy-server modules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from fastapi.testclient import TestClient
from strategy_server.allocator import allocate_positions
from strategy_server.app import app
from strategy_server.composer import compose_signals
from strategy_server.risk import apply_trailing_stops, check_drawdown, compute_atr
from strategy_server.selector import select_assets


def _make_bars(n: int = 100, base: float = 100.0, trend: float = 0.0) -> pl.DataFrame:
    rows = []
    price = base
    for i in range(n):
        price = price * (1 + trend + np.random.normal(0, 0.01))
        rows.append(
            {
                "open_time": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6),
                "open": price * 0.999,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1000.0 + np.random.normal(0, 100),
            }
        )
    return pl.DataFrame(rows)


class TestComposer:
    def test_basic_composition(self):
        signals = {
            "BTCUSDT": {"momentum": 0.8, "breakout": 0.6},
            "ETHUSDT": {"momentum": -0.3, "breakout": 0.5},
        }
        weights = {"momentum": 2.0, "breakout": 1.0}
        result = compose_signals(signals, weights)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert -1.0 <= result["BTCUSDT"] <= 1.0
        assert -1.0 <= result["ETHUSDT"] <= 1.0

    def test_empty_signals(self):
        result = compose_signals({}, {"momentum": 1.0})
        assert result == {}

    def test_missing_factor(self):
        signals = {"BTCUSDT": {"momentum": 0.5}}
        weights = {"momentum": 1.0, "breakout": 1.0}
        result = compose_signals(signals, weights)
        # breakout defaults to 0.0
        assert "BTCUSDT" in result


class TestSelector:
    def test_select_top_n(self):
        np.random.seed(42)
        universe = {
            "BTCUSDT": _make_bars(100, trend=0.005),
            "ETHUSDT": _make_bars(100, trend=0.003),
            "SOLUSDT": _make_bars(100, trend=-0.002),
            "DOGEUSDT": _make_bars(100, trend=0.0),
        }
        selected = select_assets(universe, top_n=2, lookback=90)
        assert len(selected) <= 2

    def test_too_few_bars(self):
        universe = {"BTCUSDT": _make_bars(10)}
        selected = select_assets(universe, lookback=90)
        assert selected == []


class TestAllocator:
    def test_basic_allocation(self):
        signals = {"BTCUSDT": 0.8, "ETHUSDT": -0.5}
        result = allocate_positions(signals, equity=10000.0)
        assert result["BTCUSDT"] > 0  # long
        assert result["ETHUSDT"] < 0  # short

    def test_max_position_cap(self):
        signals = {"BTCUSDT": 1.0}
        result = allocate_positions(signals, equity=10000.0, max_position_pct=0.05)
        assert abs(result["BTCUSDT"]) <= 500.0 + 0.01  # 5% of 10000

    def test_empty_signals(self):
        result = allocate_positions({}, equity=10000.0)
        assert result == {}

    def test_asymmetric_allocation(self):
        signals = {"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5}
        result = allocate_positions(signals, equity=10000.0, long_ratio=0.7, short_ratio=0.3, max_position_pct=0.5)
        total_long = sum(v for v in result.values() if v > 0)
        total_short = abs(sum(v for v in result.values() if v < 0))
        assert total_long > total_short  # 70/30 split


class TestRisk:
    def test_compute_atr(self):
        bars = _make_bars(50)
        atr = compute_atr(bars)
        assert atr > 0

    def test_trailing_stop_long(self):
        bars = _make_bars(50, base=100)
        last_price = float(bars["close"][-1])
        atr = compute_atr(bars)

        # Entry well above current → stop will trigger
        positions = {"BTCUSDT": 1000.0}
        entry_prices = {"BTCUSDT": last_price + atr * 5}
        result = apply_trailing_stops(positions, {"BTCUSDT": bars}, entry_prices, atr_mult=2.0)
        assert result["BTCUSDT"] == 0.0  # stopped out

    def test_drawdown_within_limits(self):
        assert check_drawdown(equity=9500, peak_equity=10000, max_drawdown=0.15) is True

    def test_drawdown_exceeded(self):
        assert check_drawdown(equity=8000, peak_equity=10000, max_drawdown=0.15) is False


class TestRoutes:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health(self):
        assert self.client.get("/health").status_code == 200

    def test_compose_endpoint(self):
        resp = self.client.post(
            "/compose",
            json={
                "signals": {"BTC": {"m": 0.5, "b": 0.3}},
                "weights": {"m": 2.0, "b": 1.0},
            },
        )
        assert resp.status_code == 200
        assert "composite_signals" in resp.json()

    def test_allocate_endpoint(self):
        resp = self.client.post(
            "/allocate",
            json={
                "signals": {"BTC": 0.8, "ETH": -0.5},
                "equity": 10000,
            },
        )
        assert resp.status_code == 200
        assert "positions" in resp.json()

    def test_drawdown_endpoint(self):
        resp = self.client.post(
            "/check-drawdown",
            json={
                "equity": 9000,
                "peak_equity": 10000,
                "max_drawdown": 0.15,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["within_limits"] is True
        assert abs(data["current_drawdown"] - 0.1) < 0.001

    def test_config_endpoint(self):
        resp = self.client.get("/config")
        assert resp.status_code == 200
        assert "long_ratio" in resp.json()
