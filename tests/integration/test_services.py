"""Service-level integration tests.

Tests each FastAPI service via httpx TestClient, verifying HTTP endpoints
return correct responses. No external dependencies (Binance, Hyperliquid).
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from alpha_service.app import app as alpha_app
from alpha_service.registry import registry as alpha_registry
from data_service.app import app as data_app
from data_service.store import ParquetStore
from report_service.app import app as report_app
from strategy_service.app import app as strategy_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bars(n: int = 150, seed: int = 42) -> list[dict]:
    """Generate synthetic OHLCV bars as list[dict] for JSON payloads."""
    rng = np.random.default_rng(seed)
    price = 100.0
    bars = []
    for i in range(n):
        ret = rng.normal(0.001, 0.02)
        price *= 1 + ret
        bars.append(
            {
                "open_time": f"2024-01-01T{(i * 6) % 24:02d}:00:00Z",
                "open": round(price * 0.999, 2),
                "high": round(price * 1.01, 2),
                "low": round(price * 0.99, 2),
                "close": round(price, 2),
                "volume": round(1000 + rng.uniform(0, 500), 2),
            }
        )
    return bars


BARS_150 = _make_bars(150)
BARS_50 = _make_bars(50, seed=99)


# ---------------------------------------------------------------------------
# Alpha-service
# ---------------------------------------------------------------------------


class TestAlphaService:
    """Integration tests for alpha-service HTTP endpoints."""

    @pytest.fixture(autouse=True)
    def _discover_factors(self):
        """Ensure factors are registered (normally done in lifespan)."""
        alpha_registry.auto_discover()

    @pytest.fixture
    def client(self):
        return AsyncClient(
            transport=ASGITransport(app=alpha_app), base_url="http://test"
        )

    @pytest.mark.asyncio
    async def test_list_factors(self, client: AsyncClient):
        resp = await client.get("/factors")
        assert resp.status_code == 200
        data = resp.json()
        assert "factors" in data
        assert isinstance(data["factors"], list)
        assert len(data["factors"]) > 0

    @pytest.mark.asyncio
    async def test_factor_config(self, client: AsyncClient):
        # Get first available factor
        resp = await client.get("/factors")
        factor_name = resp.json()["factors"][0]

        resp = await client.get(f"/factors/{factor_name}/config")
        assert resp.status_code == 200
        assert "name" in resp.json()
        assert "config" in resp.json()

    @pytest.mark.asyncio
    async def test_factor_config_not_found(self, client: AsyncClient):
        resp = await client.get("/factors/nonexistent_factor/config")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_compute_single(self, client: AsyncClient):
        resp = await client.post(
            "/compute",
            json={"symbol": "BTCUSDT", "bars": BARS_150},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTCUSDT"
        assert "signals" in data
        # Should have at least one factor result
        assert len(data["signals"]) > 0

    @pytest.mark.asyncio
    async def test_compute_specific_factors(self, client: AsyncClient):
        resp = await client.get("/factors")
        factors = resp.json()["factors"]

        resp = await client.post(
            "/compute",
            json={
                "symbol": "ETHUSDT",
                "bars": BARS_150,
                "factors": [factors[0]],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["signals"]) == 1

    @pytest.mark.asyncio
    async def test_compute_batch(self, client: AsyncClient):
        resp = await client.post(
            "/compute/batch",
            json={
                "symbols": {
                    "BTCUSDT": BARS_150,
                    "ETHUSDT": BARS_50,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "BTCUSDT" in data["signals"]
        assert "ETHUSDT" in data["signals"]

    @pytest.mark.asyncio
    async def test_compute_unknown_factor(self, client: AsyncClient):
        resp = await client.post(
            "/compute",
            json={
                "symbol": "BTCUSDT",
                "bars": BARS_150,
                "factors": ["nonexistent"],
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Strategy-service
# ---------------------------------------------------------------------------


class TestStrategyService:
    """Integration tests for strategy-service HTTP endpoints."""

    @pytest.fixture
    def client(self):
        return AsyncClient(
            transport=ASGITransport(app=strategy_app), base_url="http://test"
        )

    @pytest.mark.asyncio
    async def test_compose(self, client: AsyncClient):
        resp = await client.post(
            "/signal/compose",
            json={
                "signals": {
                    "BTCUSDT": {"tsmom_30": 0.7, "breakout_15": 0.3},
                    "ETHUSDT": {"tsmom_30": -0.5, "breakout_15": 0.8},
                },
                "weights": {"tsmom_30": 2.0, "breakout_15": 1.0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "composite_signals" in data
        assert "BTCUSDT" in data["composite_signals"]
        assert "ETHUSDT" in data["composite_signals"]
        # Signals clipped to [-1, 1]
        for v in data["composite_signals"].values():
            assert -1.0 <= v <= 1.0

    @pytest.mark.asyncio
    async def test_allocate(self, client: AsyncClient):
        resp = await client.post(
            "/portfolio/allocate",
            json={
                "signals": {"BTCUSDT": 0.8, "ETHUSDT": -0.5, "SOLUSDT": 0.3},
                "equity": 10000.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data
        assert len(data["positions"]) == 3

    @pytest.mark.asyncio
    async def test_select(self, client: AsyncClient):
        resp = await client.post(
            "/portfolio/select",
            json={
                "universe": {
                    "BTCUSDT": BARS_150,
                    "ETHUSDT": BARS_50,
                },
                "top_n": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "selected" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_drawdown_within_limits(self, client: AsyncClient):
        resp = await client.post(
            "/risk/drawdown",
            json={"equity": 9500.0, "peak_equity": 10000.0, "max_drawdown": 0.15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["within_limits"] is True
        assert abs(data["current_drawdown"] - 0.05) < 0.001

    @pytest.mark.asyncio
    async def test_drawdown_breached(self, client: AsyncClient):
        resp = await client.post(
            "/risk/drawdown",
            json={"equity": 8000.0, "peak_equity": 10000.0, "max_drawdown": 0.15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["within_limits"] is False
        assert abs(data["current_drawdown"] - 0.2) < 0.001

    @pytest.mark.asyncio
    async def test_risk_check(self, client: AsyncClient):
        resp = await client.post(
            "/risk/check",
            json={
                "positions": {"BTCUSDT": 1000.0},
                "bars": {"BTCUSDT": BARS_150},
                "entry_prices": {"BTCUSDT": 100.0},
                "atr_mult": 2.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data
        assert "stopped_out" in data

    @pytest.mark.asyncio
    async def test_config(self, client: AsyncClient):
        resp = await client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "long_ratio" in data
        assert "short_ratio" in data
        assert "max_drawdown" in data
        assert "trailing_stop_atr_mult" in data


# ---------------------------------------------------------------------------
# Report-service
# ---------------------------------------------------------------------------


class TestReportService:
    """Integration tests for report-service HTTP endpoints."""

    @pytest.fixture
    def client(self):
        return AsyncClient(
            transport=ASGITransport(app=report_app), base_url="http://test"
        )

    @pytest.fixture
    def sample_curve(self) -> list[tuple[str, float]]:
        """Generate a simple equity curve."""
        rng = np.random.default_rng(42)
        equity = 10000.0
        curve = []
        for i in range(100):
            equity *= 1 + rng.normal(0.001, 0.01)
            curve.append(
                (f"2024-01-{(i // 4) + 1:02d}T{(i % 4) * 6:02d}:00:00", equity)
            )
        return curve

    @pytest.fixture
    def sample_trades(self) -> list[dict]:
        return [
            {"pnl": 50.0, "symbol": "BTC"},
            {"pnl": -20.0, "symbol": "ETH"},
            {"pnl": 30.0, "symbol": "BTC"},
            {"pnl": -10.0, "symbol": "SOL"},
            {"pnl": 80.0, "symbol": "BTC"},
        ]

    @pytest.mark.asyncio
    async def test_report(
        self,
        client: AsyncClient,
        sample_curve: list[tuple[str, float]],
        sample_trades: list[dict],
    ):
        resp = await client.post(
            "/report",
            json={"equity_curve": sample_curve, "trades": sample_trades},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "raw" in data
        assert data["raw"]["num_trades"] == 5
        assert data["raw"]["sharpe_ratio"] != 0

    @pytest.mark.asyncio
    async def test_report_empty_trades(
        self, client: AsyncClient, sample_curve: list[tuple[str, float]]
    ):
        resp = await client.post(
            "/report",
            json={"equity_curve": sample_curve, "trades": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw"]["num_trades"] == 0

    @pytest.mark.asyncio
    async def test_plot_equity(
        self, client: AsyncClient, sample_curve: list[tuple[str, float]]
    ):
        resp = await client.post(
            "/plot",
            json={"equity_curve": sample_curve, "chart_type": "equity"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100  # Non-trivial PNG

    @pytest.mark.asyncio
    async def test_plot_drawdown(
        self, client: AsyncClient, sample_curve: list[tuple[str, float]]
    ):
        resp = await client.post(
            "/plot",
            json={"equity_curve": sample_curve, "chart_type": "drawdown"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_plot_base64(
        self, client: AsyncClient, sample_curve: list[tuple[str, float]]
    ):
        resp = await client.post(
            "/plot/base64",
            json={"equity_curve": sample_curve, "chart_type": "equity"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image" in data
        assert data["media_type"] == "image/png"
        # Valid base64
        import base64

        decoded = base64.b64decode(data["image"])
        assert decoded[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Data-service (limited, no Binance calls)
# ---------------------------------------------------------------------------


class TestDataService:
    """Integration tests for data-service HTTP endpoints.

    Only tests endpoints that don't require external API calls.
    Uses a temporary empty data directory.
    """

    @pytest.fixture
    def client(self, tmp_path):
        # Override store with empty temp dir
        data_app.state.store = ParquetStore(str(tmp_path))
        return AsyncClient(
            transport=ASGITransport(app=data_app), base_url="http://test"
        )

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_symbols_empty_store(self, client: AsyncClient):
        resp = await client.get("/symbols")
        assert resp.status_code == 200
        data = resp.json()
        assert "local" in data

    @pytest.mark.asyncio
    async def test_bars_empty(self, client: AsyncClient):
        resp = await client.get("/bars/BTCUSDT?tf=6h")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bars"] == 0
