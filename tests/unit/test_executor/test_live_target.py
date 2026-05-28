"""Tests for live target execution helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, cast

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
from exchange.adapter import (
    AccountState,
    ExchangeAdapter,
    MarketSnapshot,
    OrderResult,
    Position,
)


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
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
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
    assert btc.qty == pytest.approx(0.1)
    assert btc.entry_price == pytest.approx(50_000.0)
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


class FilledSizeExchange:
    def __init__(self, filled_size: float, avg_price: float = 51_000.0) -> None:
        self.filled_size = filled_size
        self.avg_price = avg_price

    async def place_market_order(self, symbol, is_buy, size, *, reduce_only=False):
        return OrderResult(
            order_id="filled",
            success=True,
            message="filled",
            avg_price=self.avg_price,
            filled_size=self.filled_size,
        )


@pytest.mark.asyncio
async def test_execute_target_order_uses_actual_partial_fill_size(tmp_path) -> None:
    state = EngineState(bar_count=3)
    state.positions["BTC"] = PositionState("BTC", 0.10, 50_000.0, 1, 51_000.0)
    journal = TradeJournal(tmp_path)

    ok = await execute_target_order(
        exchange=cast(ExchangeAdapter, FilledSizeExchange(0.04)),
        journal=journal,
        state=state,
        profile="test-profile",
        dry_run=False,
        order=_order(side="sell", qty=0.10, reduce_only=True),
        price=50_500.0,
        bar=4,
    )

    assert ok is not None
    assert state.positions["BTC"].qty == pytest.approx(0.06)
    trade = journal.load_trades()[-1]
    assert trade["bar"] == 4
    assert trade["qty"] == pytest.approx(0.04)
    assert trade["price"] == pytest.approx(51_000.0)
    assert trade["exchange_order_id"] == "filled"


@pytest.mark.asyncio
async def test_execute_target_order_records_exchange_rounded_fill_size(
    tmp_path,
) -> None:
    state = EngineState(bar_count=3)
    journal = TradeJournal(tmp_path)

    ok = await execute_target_order(
        exchange=cast(ExchangeAdapter, FilledSizeExchange(13.9, avg_price=2.0777)),
        journal=journal,
        state=state,
        profile="test-profile",
        dry_run=False,
        order=_order(symbol="NEAR", side="buy", qty=13.85980505146829),
        price=2.0777,
        bar=4,
    )

    assert ok is not None
    assert state.positions["NEAR"].qty == pytest.approx(13.9)
    assert state.positions["NEAR"].entry_bar == 4
    trade = journal.load_trades()[-1]
    assert trade["qty"] == pytest.approx(13.9)
    assert trade["price"] == pytest.approx(2.0777)


def test_sync_target_state_resets_price_fields_when_exchange_position_changes() -> None:
    state = EngineState(bar_count=10)
    state.positions["NEAR"] = PositionState(
        symbol="NEAR",
        qty=13.9,
        entry_price=2.0,
        entry_bar=7,
        best_price=2.4,
    )
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[
            Position(
                symbol="NEAR",
                size=Decimal("6.6"),
                entry_price=Decimal("2.1"),
                unrealized_pnl=Decimal("0"),
                leverage=1,
            )
        ],
    )

    sync_target_state_from_account(state, account, {"NEAR"})

    position = state.positions["NEAR"]
    assert position.qty == pytest.approx(6.6)
    assert position.entry_price == pytest.approx(2.1)
    assert position.entry_bar == 10
    assert position.best_price == pytest.approx(2.1)


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
        exchange=cast(ExchangeAdapter, exchange),
        journal=TradeJournal(tmp_path),
        state=state,
        account=account,
        equity=10_000.0,
        target_strategy=StaticTargetStrategy(),
        symbols=["BTC", "ETH"],
        profile="test-profile",
        dry_run=False,
        min_order_notional=10.0,
        bar=4,
    )

    assert orders == []
    assert [order[0] for order in exchange.orders] == ["BTC"]
    target = TradeJournal(tmp_path).load_targets()[-1]
    assert target["bar"] == 4
    assert [order["symbol"] for order in target["planned_orders"]] == ["BTC", "ETH"]
    assert [order["symbol"] for order in target["submitted_orders"]] == ["BTC"]
    assert target["filled_trades"] == []
    assert target["failed_orders"] == [
        {
            "symbol": "BTC",
            "side": "sell",
            "qty": pytest.approx(0.1),
            "current_weight": pytest.approx(0.5),
            "target_weight": pytest.approx(0.0),
            "delta_weight": pytest.approx(-0.5),
            "delta_notional": pytest.approx(-5000.0),
            "reduce_only": True,
            "status": "failed",
            "reason": "rejected_or_unfilled",
        }
    ]


@pytest.mark.asyncio
async def test_execute_target_portfolio_records_target_and_fills_on_same_bar(
    tmp_path,
) -> None:
    state = EngineState(bar_count=3)
    account = AccountState(
        equity=Decimal("10000"),
        available_balance=Decimal("10000"),
        total_margin_used=Decimal("0"),
        positions=[],
    )
    exchange = FailingThenRecordingExchange()
    journal = TradeJournal(tmp_path)

    orders = await execute_target_portfolio(
        exchange=cast(ExchangeAdapter, exchange),
        journal=journal,
        state=state,
        account=account,
        equity=10_000.0,
        target_strategy=StaticTargetStrategy(),
        symbols=["ETH"],
        profile="test-profile",
        dry_run=False,
        min_order_notional=10.0,
        bar=4,
    )

    assert [order.symbol for order in orders] == ["ETH"]
    target = journal.load_targets()[-1]
    trade = journal.load_trades()[-1]
    assert target["bar"] == trade["bar"] == 4
    assert [order["symbol"] for order in target["planned_orders"]] == ["ETH"]
    assert [order["symbol"] for order in target["submitted_orders"]] == ["ETH"]
    assert [order["symbol"] for order in target["filled_trades"]] == ["ETH"]
    assert target["failed_orders"] == []
    assert trade["exchange_order_id"] == "ok"


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
        exchange=cast(ExchangeAdapter, exchange),
        journal=TradeJournal(tmp_path),
        state=state,
        account=account,
        equity=10_000.0,
        target_strategy=StaticTargetStrategy(),
        symbols=["BTC"],
        profile="test-profile",
        dry_run=False,
        min_order_notional=10.0,
        bar=4,
    )

    assert orders == []
    assert exchange.orders == []
    assert state.positions == {}


@pytest.mark.asyncio
async def test_apply_target_fill_bar_matches_journal_trade_bar(tmp_path) -> None:
    """The bar passed to apply_target_fill must match the bar in the trade record."""
    state = EngineState(bar_count=7)
    journal = TradeJournal(tmp_path)

    order = _order(side="buy", qty=0.5)
    ok = await execute_target_order(
        exchange=cast(ExchangeAdapter, FilledSizeExchange(0.5)),
        journal=journal,
        state=state,
        profile="test-profile",
        dry_run=False,
        order=order,
        price=50_000.0,
        bar=8,
    )

    assert ok is not None
    trade = journal.load_trades()[-1]
    assert trade["bar"] == 8
    pos = state.positions["BTC"]
    assert pos.entry_bar == 8

    target = TradeJournal(tmp_path).load_targets()
    assert len(target) == 0, "execute_target_order does not call record_target"


@pytest.mark.asyncio
async def test_dry_run_journals_without_exchange_placement(tmp_path) -> None:
    """Dry-run target orders write trade journal but skip exchange placement."""
    state = EngineState(bar_count=3)
    journal = TradeJournal(tmp_path)

    ok = await execute_target_order(
        exchange=cast(ExchangeAdapter, FilledSizeExchange(0.5)),
        journal=journal,
        state=state,
        profile="test-profile",
        dry_run=True,
        order=_order(side="buy", qty=0.5),
        price=50_000.0,
        bar=4,
    )

    assert ok is not None
    trade = journal.load_trades()[-1]
    assert trade["bar"] == 4
    assert trade["kind"] == "buy"
    assert trade["qty"] == 0.5
    assert trade["price"] == 50_000.0
    assert trade["reason"] == "target:test-profile"
    assert "exchange_order_id" not in trade
