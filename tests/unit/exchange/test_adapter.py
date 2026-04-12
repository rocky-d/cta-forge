"""Tests for exchange adapter protocol conformance."""

from __future__ import annotations

from decimal import Decimal

from exchange_server.adapter import (
    AccountState,
    ExchangeAdapter,
    MarketSnapshot,
    OrderResult,
    Position,
)


class MockExchangeAdapter:
    """Mock implementation of ExchangeAdapter for testing."""

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.positions: list[Position] = []
        self.equity = Decimal("10000")

    async def get_account_state(self) -> AccountState:
        return AccountState(
            equity=self.equity,
            available_balance=self.equity,
            total_margin_used=Decimal("0"),
            positions=self.positions,
        )

    async def get_position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            mid_price=Decimal("73000"),
            best_bid=Decimal("72999"),
            best_ask=Decimal("73001"),
            mark_price=Decimal("73000"),
            funding_rate=Decimal("0.0001"),
            timestamp_ms=1000000,
        )

    async def place_market_order(
        self, symbol: str, is_buy: bool, size: Decimal, *, reduce_only: bool = False
    ) -> OrderResult:
        self.orders.append({"symbol": symbol, "is_buy": is_buy, "size": size})
        return OrderResult(
            order_id="mock-123", success=True, message="filled", avg_price=73000.0
        )

    async def place_limit_order(
        self,
        symbol: str,
        is_buy: bool,
        size: Decimal,
        price: Decimal,
        *,
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResult:
        self.orders.append(
            {"symbol": symbol, "is_buy": is_buy, "size": size, "price": price}
        )
        return OrderResult(order_id="mock-456", success=True, message="resting")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        return 0

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return []

    async def set_leverage(
        self, symbol: str, leverage: int, cross: bool = True
    ) -> bool:
        return True

    async def transfer_to_perp(self, amount: Decimal) -> bool:
        return True

    async def close(self) -> None:
        pass


def test_mock_satisfies_protocol() -> None:
    """MockExchangeAdapter satisfies ExchangeAdapter protocol."""
    adapter = MockExchangeAdapter()
    assert isinstance(adapter, ExchangeAdapter)


async def test_mock_account_state() -> None:
    """Mock adapter returns correct account state."""
    adapter = MockExchangeAdapter()
    state = await adapter.get_account_state()
    assert state.equity == Decimal("10000")
    assert len(state.positions) == 0


async def test_mock_place_order() -> None:
    """Mock adapter records orders."""
    adapter = MockExchangeAdapter()
    result = await adapter.place_market_order("BTC", True, Decimal("0.01"))
    assert result.success
    assert len(adapter.orders) == 1
    assert adapter.orders[0]["symbol"] == "BTC"


async def test_mock_market_snapshot() -> None:
    """Mock adapter returns market data."""
    adapter = MockExchangeAdapter()
    snap = await adapter.get_market_snapshot("BTC")
    assert snap.mid_price == Decimal("73000")
    assert snap.symbol == "BTC"
