"""Tests for alpha-server registry and routes."""

from __future__ import annotations

from alpha.app import app
from alpha.registry import FactorRegistry, registry
from fastapi.testclient import TestClient


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
