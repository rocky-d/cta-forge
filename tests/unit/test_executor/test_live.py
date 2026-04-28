"""Tests for live engine preflight and core logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pytest
from executor.live import LiveEngine
from executor.targeting import PortfolioTarget, StrategyProfile
from exchange.adapter import AccountState, MarketSnapshot, OrderResult, Position


class FakeExchange:
    """Minimal fake exchange for testing preflight logic."""

    def __init__(
        self,
        equity: Decimal = Decimal("2000"),
        positions: list[Position] | None = None,
        open_orders: list[dict] | None = None,
        market_price: Decimal = Decimal("73000"),
    ) -> None:
        self._equity = equity
        self._positions = positions or []
        self._open_orders = open_orders or []
        self._market_price = market_price
        self.closed_positions: list[str] = []
        self.orders: list[tuple[str, bool, Decimal, bool]] = []
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
            mid_price=self._market_price,
            best_bid=self._market_price - Decimal("1"),
            best_ask=self._market_price + Decimal("1"),
            mark_price=self._market_price,
            funding_rate=Decimal("0.0001"),
            timestamp_ms=1000000,
        )

    async def place_market_order(
        self, symbol: str, is_buy: bool, size: Decimal, *, reduce_only: bool = False
    ) -> OrderResult:
        self.orders.append((symbol, is_buy, size, reduce_only))
        if reduce_only:
            self.closed_positions.append(symbol)
            # Remove position after closing
            self._positions = [p for p in self._positions if p.symbol != symbol]
        return OrderResult(
            order_id="fake",
            success=True,
            message="filled",
            avg_price=float(self._market_price),
            filled_size=float(size),
        )

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
    """Preflight with clean_start closes stale positions."""

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
    engine = LiveEngine(exchange, dry_run=False, clean_start=True)
    result = await engine._preflight()
    assert result is True
    assert "BTC" in exchange.closed_positions


@pytest.mark.asyncio
async def test_preflight_positions_no_clean_start() -> None:
    """Preflight without clean_start does NOT close positions (reconcile later)."""

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
    engine = LiveEngine(exchange, dry_run=False, clean_start=False)
    result = await engine._preflight()
    assert result is True
    assert exchange.closed_positions == []


@pytest.mark.asyncio
async def test_preflight_stale_orders() -> None:
    """Preflight cancels stale open orders."""

    orders = [{"coin": "ETH", "oid": "123"}]
    exchange = FakeExchange(open_orders=orders)
    engine = LiveEngine(exchange, dry_run=False)
    result = await engine._preflight()
    assert result is True
    assert exchange.cancelled_all >= 1


@dataclass(frozen=True)
class StaticTargetStrategy:
    profile: StrategyProfile
    weights: dict[str, float]

    def target(self, timestamp: datetime) -> PortfolioTarget:
        return PortfolioTarget(timestamp=timestamp, weights=self.weights)


@pytest.mark.asyncio
async def test_target_tick_splits_sign_flip_reduce_first(tmp_path) -> None:
    """Target mode closes old side before opening the new side."""

    positions = [
        Position(
            symbol="BTC",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            leverage=1,
        ),
    ]
    exchange = FakeExchange(
        equity=Decimal("10000"),
        positions=positions,
        market_price=Decimal("50000"),
    )
    strategy = StaticTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={"BTCUSDT": -0.2},
    )
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=False,
        journal_dir=str(tmp_path / "journal"),
        target_strategy=strategy,
    )

    await engine._tick()

    assert exchange.orders == [
        ("BTC", False, Decimal("0.1"), True),
        ("BTC", False, Decimal("0.04"), False),
    ]
    assert engine._state.positions["BTC"].qty == pytest.approx(-0.04)
    targets = engine._journal.load_targets()
    assert len(targets) == 1
    assert targets[0]["profile"] == "test-target"
    assert targets[0]["normalized_gross"] == pytest.approx(0.2)
    assert targets[0]["ignored_weights"] == {}
    assert [order["reduce_only"] for order in targets[0]["orders"]] == [True, False]


@pytest.mark.asyncio
async def test_target_tick_skips_below_min_notional(tmp_path) -> None:
    """Target mode ignores dust deltas before sending exchange orders."""

    exchange = FakeExchange(equity=Decimal("10000"), market_price=Decimal("50000"))
    strategy = StaticTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={"BTCUSDT": 0.0005, "XRPUSDT": 0.05},
    )
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=False,
        journal_dir=str(tmp_path / "journal"),
        target_strategy=strategy,
        min_order_notional=10.0,
    )

    await engine._tick()

    assert exchange.orders == []
    assert engine._state.positions == {}
    targets = engine._journal.load_targets()
    assert len(targets) == 1
    assert targets[0]["orders"] == []
    assert targets[0]["ignored_weights"] == {"XRPUSDT": 0.05}


def test_unknown_live_profile_requires_target_provider() -> None:
    """Non-default profiles cannot silently fall back to v10g."""

    exchange = FakeExchange()
    with pytest.raises(ValueError, match="no live target provider"):
        LiveEngine(exchange, strategy_profile="v16a-badscore-overlay")
