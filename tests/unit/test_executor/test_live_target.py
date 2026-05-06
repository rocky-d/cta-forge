"""Tests for live target execution helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

import pytest
from executor.decision import EngineState, PositionState
from executor.journal import TradeJournal
from executor.live_target import (
    apply_target_fill,
    execute_target_order,
    execute_target_portfolio,
    normalize_target_weights,
    sync_target_state_from_account,
)
from executor.targeting import PortfolioTarget, StrategyProfile, TargetOrder
from exchange.adapter import AccountState, MarketSnapshot, OrderResult, Position


def _order(
    *,
    symbol: str = "BTC",
    side: Literal["buy", "sell"] = "buy",
    qty: float = 1.0,
    reduce_only: bool = False,
) -> TargetOrder:
    return TargetOrder(
        symbol=symbol,
        side=side,
        qty=qty,
        current_weight=0.0,
        target_weight=0.0,
        delta_weight=0.0,
        delta_notional=0.0,
        reduce_only=reduce_only,
    )


def test_normalize_target_weights_converts_symbols_and_tracks_ignored() -> None:
    weights, ignored = normalize_target_weights(
        {"BTCUSDT": 0.1, "BTC": 0.2, "XRPUSDT": 0.3},
        {"BTC"},
    )

    assert weights == {"BTC": pytest.approx(0.3)}
    assert ignored == {"XRPUSDT": pytest.approx(0.3)}


def test_sync_target_state_from_account_preserves_existing_position_metadata() -> None:
    state = EngineState(bar_count=10)
    state.positions["BTC"] = PositionState(
        symbol="BTC",
        qty=0.1,
        entry_price=50_000.0,
        entry_bar=3,
        best_price=55_000.0,
        partial_taken=True,
    )
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[
            Position(
                symbol="BTC",
                size=Decimal("0.2"),
                entry_price=Decimal("51000"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            ),
            Position(
                symbol="ETH",
                size=Decimal("1.5"),
                entry_price=Decimal("2500"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            ),
        ],
    )

    sync_target_state_from_account(state, account)

    btc = state.positions["BTC"]
    assert btc.qty == pytest.approx(0.2)
    assert btc.entry_price == pytest.approx(51_000.0)
    assert btc.entry_bar == 3
    assert btc.best_price == pytest.approx(55_000.0)
    assert btc.partial_taken is True
    eth = state.positions["ETH"]
    assert eth.entry_bar == 10
    assert eth.best_price == pytest.approx(2_500.0)


def test_apply_target_fill_reduce_only_close_removes_position() -> None:
    state = EngineState()
    state.positions["BTC"] = PositionState("BTC", 0.1, 50_000.0, 4, 55_000.0)

    apply_target_fill(state, _order(side="sell", qty=0.1, reduce_only=True), 51_000.0)

    assert "BTC" not in state.positions


def test_apply_target_fill_partial_reduce_keeps_entry_metadata() -> None:
    state = EngineState()
    state.positions["BTC"] = PositionState(
        "BTC", 0.2, 50_000.0, 4, 55_000.0, partial_taken=True
    )

    apply_target_fill(state, _order(side="sell", qty=0.05, reduce_only=True), 52_000.0)

    pos = state.positions["BTC"]
    assert pos.qty == pytest.approx(0.15)
    assert pos.entry_price == pytest.approx(50_000.0)
    assert pos.entry_bar == 4
    assert pos.best_price == pytest.approx(55_000.0)
    assert pos.partial_taken is True


def test_apply_target_fill_adds_with_weighted_average_entry() -> None:
    state = EngineState(bar_count=8)
    state.positions["BTC"] = PositionState("BTC", 0.1, 50_000.0, 4, 55_000.0)

    apply_target_fill(state, _order(side="buy", qty=0.1), 60_000.0)

    pos = state.positions["BTC"]
    assert pos.qty == pytest.approx(0.2)
    assert pos.entry_price == pytest.approx(55_000.0)
    assert pos.entry_bar == 4
    assert pos.best_price == pytest.approx(55_000.0)


def test_apply_target_fill_non_reduce_flip_resets_entry_metadata() -> None:
    state = EngineState(bar_count=8)
    state.positions["BTC"] = PositionState(
        "BTC", 0.1, 50_000.0, 4, 55_000.0, partial_taken=True
    )

    apply_target_fill(state, _order(side="sell", qty=0.3), 52_000.0)

    pos = state.positions["BTC"]
    assert pos.qty == pytest.approx(-0.2)
    assert pos.entry_price == pytest.approx(52_000.0)
    assert pos.entry_bar == 8
    assert pos.best_price == pytest.approx(52_000.0)
    assert pos.partial_taken is False


class PartialFillExchange:
    async def place_market_order(self, symbol, is_buy, size, *, reduce_only=False):
        return OrderResult(
            order_id="partial",
            success=True,
            message="filled",
            avg_price=51_000.0,
            filled_size=0.04,
        )


@pytest.mark.asyncio
async def test_execute_target_order_uses_actual_filled_size(tmp_path) -> None:
    state = EngineState(bar_count=3)
    state.positions["BTC"] = PositionState("BTC", 0.10, 50_000.0, 1, 51_000.0)
    journal = TradeJournal(tmp_path)

    ok = await execute_target_order(
        exchange=PartialFillExchange(),
        journal=journal,
        state=state,
        profile="test-profile",
        dry_run=False,
        order=_order(side="sell", qty=0.10, reduce_only=True),
        price=50_500.0,
    )

    assert ok is True
    assert state.positions["BTC"].qty == pytest.approx(0.06)
    trade = journal.load_trades()[-1]
    assert trade["qty"] == pytest.approx(0.04)
    assert trade["price"] == pytest.approx(51_000.0)


def test_sync_target_state_can_filter_to_managed_symbols() -> None:
    state = EngineState(bar_count=10)
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[
            Position(
                symbol="BTC",
                size=Decimal("0.2"),
                entry_price=Decimal("51000"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            ),
            Position(
                symbol="HYPE",
                size=Decimal("10"),
                entry_price=Decimal("20"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            ),
        ],
    )

    sync_target_state_from_account(state, account, {"BTC"})

    assert set(state.positions) == {"BTC"}


class FailingThenRecordingExchange:
    def __init__(self) -> None:
        self.orders: list[tuple[str, bool, Decimal, bool]] = []

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        prices = {"BTC": Decimal("50000"), "ETH": Decimal("2000")}
        price = prices[symbol]
        return MarketSnapshot(
            symbol=symbol,
            mid_price=price,
            best_bid=price,
            best_ask=price,
            mark_price=price,
            funding_rate=Decimal("0"),
            timestamp_ms=1,
        )

    async def place_market_order(
        self, symbol: str, is_buy: bool, size: Decimal, *, reduce_only: bool = False
    ) -> OrderResult:
        self.orders.append((symbol, is_buy, size, reduce_only))
        if reduce_only:
            return OrderResult(order_id="fail", success=False, message="rejected")
        return OrderResult(
            order_id="ok",
            success=True,
            message="filled",
            avg_price=2000.0,
            filled_size=float(size),
        )


class StaticTargetStrategy:
    profile = StrategyProfile(slug="test-target", name="Test Target")

    def target(self, timestamp):
        return PortfolioTarget(
            timestamp=timestamp,
            weights={"BTC": 0.0, "ETH": 0.1},
            gross_cap=1.0,
        )


@pytest.mark.asyncio
async def test_execute_target_portfolio_stops_after_failed_order(tmp_path) -> None:
    state = EngineState(bar_count=3)
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[
            Position(
                symbol="BTC",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            )
        ],
    )
    exchange = FailingThenRecordingExchange()

    orders = await execute_target_portfolio(
        exchange=exchange,
        journal=TradeJournal(tmp_path),
        state=state,
        account=account,
        equity=10_000.0,
        target_strategy=StaticTargetStrategy(),
        symbols=["BTC", "ETH"],
        profile="test-profile",
        dry_run=False,
        min_order_notional=10.0,
    )

    assert [order.symbol for order in orders] == ["BTC", "ETH"]
    assert [order[0] for order in exchange.orders] == ["BTC"]


@pytest.mark.asyncio
async def test_execute_target_portfolio_ignores_unmanaged_positions(tmp_path) -> None:
    state = EngineState(bar_count=3)
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[
            Position(
                symbol="HYPE",
                size=Decimal("10"),
                entry_price=Decimal("20"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            )
        ],
    )
    exchange = FailingThenRecordingExchange()

    orders = await execute_target_portfolio(
        exchange=exchange,
        journal=TradeJournal(tmp_path),
        state=state,
        account=account,
        equity=10_000.0,
        target_strategy=StaticTargetStrategy(),
        symbols=["BTC"],
        profile="test-profile",
        dry_run=False,
        min_order_notional=10.0,
    )

    assert orders == []
    assert exchange.orders == []
    assert state.positions == {}
