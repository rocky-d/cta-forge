"""Tests for live engine preflight and core logic."""

from __future__ import annotations

from decimal import Decimal

import pytest
from engine.live import LiveEngine
from exchange.adapter import AccountState, MarketSnapshot, OrderResult, Position


class FakeExchange:
    """Minimal fake exchange for testing preflight logic."""

    def __init__(
        self,
        equity: Decimal = Decimal("2000"),
        positions: list[Position] | None = None,
        open_orders: list[dict] | None = None,
    ) -> None:
        self._equity = equity
        self._positions = positions or []
        self._open_orders = open_orders or []
        self.closed_positions: list[str] = []
        self.cancelled_all: int = 0

    async def get_account_state(self) -> AccountState:
        return AccountState(
            equity=self._equity,
            available_balance=self._equity,
            total_margin_used=Decimal("0"),
            positions=self._positions,
        )

    async def get_position(self, symbol: str) -> Position | None:
        for p in self._positions:
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
        self.closed_positions.append(symbol)
        # Remove position after closing
        self._positions = [p for p in self._positions if p.symbol != symbol]
        return OrderResult(order_id="fake", success=True, message="filled")

    async def place_limit_order(self, symbol, is_buy, size, price, **kw) -> OrderResult:
        return OrderResult(order_id="fake", success=True)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        count = len(self._open_orders)
        self._open_orders = []
        self.cancelled_all += 1
        return count

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return self._open_orders

    async def set_leverage(
        self, symbol: str, leverage: int, cross: bool = True
    ) -> bool:
        return True

    async def transfer_to_perp(self, amount: Decimal) -> bool:
        return True

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_preflight_clean_account() -> None:
    """Preflight passes on clean account."""

    exchange = FakeExchange()
    engine = LiveEngine(exchange, dry_run=True)
    result = await engine._preflight()
    assert result is True


@pytest.mark.asyncio
async def test_preflight_low_equity() -> None:
    """Preflight fails on insufficient equity."""

    exchange = FakeExchange(equity=Decimal("50"))
    engine = LiveEngine(exchange, dry_run=True)
    result = await engine._preflight()
    assert result is False


@pytest.mark.asyncio
async def test_preflight_stale_positions() -> None:
    """Preflight closes stale positions."""

    positions = [
        Position(
            symbol="BTC",
            size=Decimal("0.01"),
            entry_price=Decimal("70000"),
            unrealized_pnl=Decimal("10"),
            leverage=1,
        ),
    ]
    exchange = FakeExchange(positions=positions)
    engine = LiveEngine(exchange, dry_run=False)
    result = await engine._preflight()
    assert result is True
    assert "BTC" in exchange.closed_positions


@pytest.mark.asyncio
async def test_preflight_stale_orders() -> None:
    """Preflight cancels stale open orders."""

    orders = [{"coin": "ETH", "oid": "123"}]
    exchange = FakeExchange(open_orders=orders)
    engine = LiveEngine(exchange, dry_run=False)
    result = await engine._preflight()
    assert result is True
    assert exchange.cancelled_all >= 1
