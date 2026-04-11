"""Tests for exchange-server FastAPI routes."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from exchange_server.adapter import AccountState, MarketSnapshot, OrderResult, Position
from exchange_server.app import app
from httpx import ASGITransport, AsyncClient


def _make_mock_adapter() -> AsyncMock:
    """Create a mock adapter with sensible defaults."""
    adapter = AsyncMock()
    adapter.get_account_state.return_value = AccountState(
        equity=Decimal("2000"),
        available_balance=Decimal("1500"),
        total_margin_used=Decimal("500"),
        positions=[
            Position(
                symbol="BTC",
                size=Decimal("0.01"),
                entry_price=Decimal("73000"),
                unrealized_pnl=Decimal("50"),
                leverage=3,
            )
        ],
    )
    adapter.get_market_snapshot.return_value = MarketSnapshot(
        symbol="BTC",
        mid_price=Decimal("73500"),
        best_bid=Decimal("73499"),
        best_ask=Decimal("73501"),
        mark_price=Decimal("73500"),
        funding_rate=Decimal("0.0001"),
        timestamp_ms=1000000,
    )
    adapter.place_market_order.return_value = OrderResult(
        order_id="test-123", success=True, message="filled", avg_price=73500.0
    )
    adapter.place_limit_order.return_value = OrderResult(order_id="test-456", success=True, message="resting")
    adapter.cancel_order.return_value = True
    adapter.set_leverage.return_value = True
    return adapter


@pytest.fixture
def mock_adapter() -> AsyncMock:
    return _make_mock_adapter()


@pytest.fixture
def client(mock_adapter: AsyncMock) -> AsyncClient:
    """Create test client with mocked adapter."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_get_account(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """GET /account returns account state."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.get("/account")
    assert resp.status_code == 200
    data = resp.json()
    assert data["equity"] == "2000"
    assert len(data["positions"]) == 1
    assert data["positions"][0]["symbol"] == "BTC"


@pytest.mark.asyncio
async def test_get_market(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """GET /market/{symbol} returns market snapshot."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.get("/market/BTC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTC"
    assert data["mid_price"] == "73500"


@pytest.mark.asyncio
async def test_place_market_order(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """POST /order (market) places a market order."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.post(
            "/order",
            json={
                "symbol": "BTC",
                "is_buy": True,
                "size": 0.01,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["order_id"] == "test-123"
    assert data["avg_price"] == 73500.0


@pytest.mark.asyncio
async def test_place_limit_order(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """POST /order (limit) places a limit order."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.post(
            "/order",
            json={
                "symbol": "ETH",
                "is_buy": False,
                "size": 0.5,
                "price": 2300.0,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["order_id"] == "test-456"


@pytest.mark.asyncio
async def test_cancel_order(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """POST /cancel cancels an order."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.post(
            "/cancel",
            json={
                "symbol": "BTC",
                "order_id": "test-123",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_set_leverage(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """POST /leverage sets leverage."""
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.post(
            "/leverage",
            json={
                "symbol": "BTC",
                "leverage": 5,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_order_failure_returns_400(client: AsyncClient, mock_adapter: AsyncMock) -> None:
    """POST /order returns 400 when order fails."""
    mock_adapter.place_market_order.return_value = OrderResult(
        order_id="", success=False, message="insufficient margin"
    )
    with patch("exchange_server.routes._get_adapter", return_value=mock_adapter):
        resp = await client.post(
            "/order",
            json={
                "symbol": "BTC",
                "is_buy": True,
                "size": 100.0,
            },
        )
    assert resp.status_code == 400
