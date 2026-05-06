"""Tests for alpha-service registry and routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alpha_service.app import app
from alpha_service.registry import FactorRegistry, registry
from fastapi.testclient import TestClient


def _make_bars(n: int = 160, *, trend: float = 0.002) -> list[dict]:
    rows: list[dict] = []
    price = 100.0
    for i in range(n):
        price *= 1 + trend
        rows.append(
            {
                "open_time": (
                    datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6)
                ).isoformat(),
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1000.0,
            }
        )
    return rows


class TestRegistry:
    def test_auto_discover(self):
        reg = FactorRegistry()
        reg.auto_discover()
        factors = reg.list_factors()
        assert len(factors) >= 4  # momentum, breakout, carry, volatility
        assert "tsmom_30" in factors
        assert "breakout_15" in factors

    def test_register_and_get(self):
        reg = FactorRegistry()

        class FakeFactor:
            @property
            def name(self) -> str:
                return "fake"

            def compute(self, bars):
                return bars

        reg.register(FakeFactor())
        assert reg.get("fake") is not None
        assert reg.get("nonexistent") is None


class TestRoutes:
    def setup_method(self):
        self.client = TestClient(app, raise_server_exceptions=True)
        # Ensure factors are discovered (lifespan may not run in all TestClient modes)
        if not registry.list_factors():
            registry.auto_discover()

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_list_factors(self):
        resp = self.client.get("/factors")
        assert resp.status_code == 200
        factors = resp.json()["factors"]
        assert len(factors) >= 4

    def test_factor_config(self):
        resp = self.client.get("/factors/tsmom_30/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "tsmom_30"
        assert "lookback" in data["config"]

    def test_factor_not_found(self):
        resp = self.client.get("/factors/nonexistent/config")
        assert resp.status_code == 404

    def test_v10g_compute_accepts_btc_reference_bars(self):
        bars = _make_bars(trend=0.002)
        btc_bars = _make_bars(trend=-0.004)

        resp = self.client.post(
            "/compute",
            json={
                "symbol": "ETHUSDT",
                "bars": bars,
                "btc_bars": btc_bars,
                "factors": ["v10g_composite"],
            },
        )

        assert resp.status_code == 200
        rows = resp.json()["signals"]["v10g_composite"]
        assert isinstance(rows, list)
        assert rows

    def test_v10g_batch_uses_btc_symbol_as_reference(self):
        btc_bars = _make_bars(trend=-0.004)
        alt_bars = _make_bars(trend=0.002)

        batch = self.client.post(
            "/compute/batch",
            json={
                "symbols": {"BTCUSDT": btc_bars, "ETHUSDT": alt_bars},
                "factors": ["v10g_composite"],
            },
        )
        single = self.client.post(
            "/compute",
            json={
                "symbol": "ETHUSDT",
                "bars": alt_bars,
                "btc_bars": btc_bars,
                "factors": ["v10g_composite"],
            },
        )

        assert batch.status_code == 200
        assert single.status_code == 200
        assert (
            batch.json()["signals"]["ETHUSDT"]["v10g_composite"]
            == single.json()["signals"]["v10g_composite"]
        )
