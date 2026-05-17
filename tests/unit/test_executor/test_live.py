"""Tests for live engine preflight and core logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from executor.decision import PositionState
from executor.journal import TradeJournal
from executor.live import LiveEngine
from executor.state import JsonFileLiveStateStore
from executor.targeting import PortfolioTarget, StrategyProfile
from exchange.adapter import AccountState, MarketSnapshot, OrderResult, Position


class CaptureNotifier:
    """Collect notifications emitted by the live engine."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)


class FakeExchange:
    """Minimal fake exchange for testing preflight logic."""

    def __init__(
        self,
        equity: Decimal = Decimal("2000"),
        available_balance: Decimal | None = None,
        positions: list[Position] | None = None,
        open_orders: list[dict] | None = None,
        market_price: Decimal = Decimal("73000"),
        market_prices: dict[str, Decimal] | None = None,
        filled_size: Decimal | None = None,
        order_success: bool = True,
    ) -> None:
        self._equity = equity
        self._available_balance = (
            equity if available_balance is None else available_balance
        )
        self._positions = positions or []
        self._open_orders = open_orders or []
        self._market_price = market_price
        self._market_prices = market_prices or {}
        self._filled_size = filled_size
        self._order_success = order_success
        self.closed_positions: list[str] = []
        self.orders: list[tuple[str, bool, Decimal, bool]] = []
        self.cancelled_all: int = 0

    async def get_account_state(self) -> AccountState:
        return AccountState(
            equity=self._equity,
            available_balance=self._available_balance,
            total_margin_used=Decimal("0"),
            positions=self._positions,
        )

    async def get_position(self, symbol: str) -> Position | None:
        for p in self._positions:
            if p.symbol == symbol:
                return p
        return None

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        market_price = self._market_prices.get(symbol, self._market_price)
        return MarketSnapshot(
            symbol=symbol,
            mid_price=market_price,
            best_bid=market_price - Decimal("1"),
            best_ask=market_price + Decimal("1"),
            mark_price=market_price,
            funding_rate=Decimal("0.0001"),
            timestamp_ms=1000000,
        )

    async def place_market_order(
        self, symbol: str, is_buy: bool, size: Decimal, *, reduce_only: bool = False
    ) -> OrderResult:
        self.orders.append((symbol, is_buy, size, reduce_only))
        if not self._order_success:
            return OrderResult(order_id="", success=False, message="rejected")
        if reduce_only:
            self.closed_positions.append(symbol)
            # Remove position after closing
            self._positions = [p for p in self._positions if p.symbol != symbol]
        filled_size = self._filled_size if self._filled_size is not None else size
        return OrderResult(
            order_id="fake",
            success=True,
            message="filled",
            avg_price=float(self._market_price),
            filled_size=float(filled_size),
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


def test_live_engine_accepts_injected_journal_store(tmp_path) -> None:
    """LiveEngine can depend on a journal port supplied by callers."""

    exchange = FakeExchange()
    journal = TradeJournal(tmp_path / "journal")
    engine = LiveEngine(
        exchange,
        dry_run=True,
        journal=journal,
        live_instance_id="ignored-when-journal-is-injected",
    )

    engine._journal.record_tick(1, 100.0, 100.0, {})

    assert journal.load_equity()[0]["equity"] == 100.0
    assert "live_instance_id" not in journal.load_equity()[0]


def test_live_engine_accepts_injected_state_store(tmp_path) -> None:
    """LiveEngine keeps an injected state store for future persistence backends."""

    exchange = FakeExchange()
    state_store = JsonFileLiveStateStore(tmp_path / "state.json")
    engine = LiveEngine(exchange, dry_run=True, state_store=state_store)

    assert engine._state_store is state_store


def test_live_engine_records_runtime_identity_metadata(tmp_path) -> None:
    """Runtime identity passed to LiveEngine reaches additive journal metadata."""

    exchange = FakeExchange()
    engine = LiveEngine(
        exchange,
        dry_run=True,
        journal_dir=str(tmp_path / "journal"),
        live_instance_id="cta-forge-mainnet-pilot-01",
        run_id="20260514T221212Z-deadbeef",
        public_instance_slug="mainnet-pilot",
    )

    engine._journal.record_tick(1, 100.0, 100.0, {})

    record = engine._journal.load_equity()[0]
    assert record["live_instance_id"] == "cta-forge-mainnet-pilot-01"
    assert record["run_id"] == "20260514T221212Z-deadbeef"
    assert record["public_instance_slug"] == "mainnet-pilot"


@pytest.mark.asyncio
async def test_live_engine_position_snapshot_includes_exposure_weight() -> None:
    """Actual journal positions include signed raw exposure weights for collectors."""

    exchange = FakeExchange(
        market_prices={"BTC": Decimal("100000"), "ARB": Decimal("0.50")}
    )
    engine = LiveEngine(exchange, dry_run=True)
    engine._state.positions = {
        "BTC": PositionState(
            "BTC", qty=0.001, entry_price=90000, entry_bar=1, best_price=100000
        ),
        "ARB": PositionState(
            "ARB", qty=-200, entry_price=0.60, entry_bar=1, best_price=0.50
        ),
    }

    snapshot = await engine._position_snapshot(equity=2000.0)

    assert snapshot["BTC"] == {
        "side": "long",
        "qty": 0.001,
        "entry": 90000.0,
        "best": 100000.0,
        "exposure_weight": 0.05,
    }
    assert snapshot["ARB"] == {
        "side": "short",
        "qty": 200.0,
        "entry": 0.6,
        "best": 0.5,
        "exposure_weight": -0.05,
    }


@pytest.mark.asyncio
async def test_live_engine_persists_position_exposure_weight_without_breaking_journal(
    tmp_path,
) -> None:
    """Enriched position fields remain additive in the existing file journal."""

    exchange = FakeExchange(market_prices={"SOL": Decimal("150")})
    engine = LiveEngine(exchange, dry_run=True, journal_dir=str(tmp_path / "journal"))
    engine._state.positions = {
        "SOL": PositionState(
            "SOL", qty=0.2, entry_price=140, entry_bar=1, best_price=150
        )
    }

    positions = await engine._position_snapshot(equity=100.0)
    engine._journal.record_tick(1, 100.0, 100.0, positions)

    record = engine._journal.load_equity()[0]
    assert record["positions"]["SOL"]["side"] == "long"
    assert record["positions"]["SOL"]["qty"] == 0.2
    assert record["positions"]["SOL"]["exposure_weight"] == 0.3


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
async def test_preflight_allows_pilot_min_equity_override() -> None:
    """Small pilot accounts can lower the minimum equity gate explicitly."""

    exchange = FakeExchange(equity=Decimal("99.7"))
    engine = LiveEngine(exchange, dry_run=True, min_equity=50.0)
    result = await engine._preflight()
    assert result is True


@pytest.mark.asyncio
async def test_preflight_rejects_equity_above_cap() -> None:
    """Pilot accounts fail closed when they exceed the configured equity cap."""

    exchange = FakeExchange(equity=Decimal("250"))
    engine = LiveEngine(exchange, dry_run=True, max_equity=200.0)
    result = await engine._preflight()
    assert result is False


@pytest.mark.asyncio
async def test_preflight_rejects_low_available_balance() -> None:
    """Pilot accounts require usable perp collateral, not only total equity."""

    exchange = FakeExchange(equity=Decimal("99.7"), available_balance=Decimal("0"))
    engine = LiveEngine(
        exchange,
        dry_run=True,
        min_equity=50.0,
        min_available_balance=50.0,
    )
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
    required_timeframes = ()

    def target(self, timestamp: datetime) -> PortfolioTarget:
        return PortfolioTarget(timestamp=timestamp, weights=self.weights)


@dataclass(frozen=True)
class WarmupTargetStrategy(StaticTargetStrategy):
    required_timeframes = (("1h", 1, 60_000), ("6h", 6, 10_000))


class RefreshingTargetStrategy:
    required_timeframes = ()

    def __init__(self, profile: StrategyProfile, weights: dict[str, float]) -> None:
        self.profile = profile
        self.weights = weights
        self.refresh_forces: list[bool] = []

    def refresh(self, *, force: bool = False) -> object:
        self.refresh_forces.append(force)
        return object()

    def target(self, timestamp: datetime) -> PortfolioTarget:
        return PortfolioTarget(timestamp=timestamp, weights=self.weights)


def test_restore_equity_state_uses_journal_high_water_mark(tmp_path) -> None:
    """Restart recovery uses journal equity to repair stale persisted peaks."""

    exchange = FakeExchange(equity=Decimal("99"))
    engine = LiveEngine(exchange, dry_run=True, journal_dir=str(tmp_path))
    engine._state.peak_equity = 100.0
    engine._journal.record_tick(1, 101.0, 100.0, {})
    engine._journal.record_tick(2, 98.0, 101.0, {})

    engine._restore_equity_state_from_journal(99.0)

    assert engine._state.peak_equity == pytest.approx(101.0)
    assert engine._last_tick_equity == pytest.approx(98.0)


@pytest.mark.asyncio
async def test_target_tick_updates_peak_and_never_reports_negative_dd(tmp_path) -> None:
    """Target mode maintains peak/DD invariants outside decision.tick()."""

    exchange = FakeExchange(equity=Decimal("101"))
    strategy = StaticTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={},
    )
    notify = CaptureNotifier()
    engine = LiveEngine(
        exchange,
        dry_run=True,
        target_strategy=strategy,
        notify=notify,
        journal_dir=str(tmp_path),
    )
    engine._state.peak_equity = 100.0
    engine._last_tick_equity = 99.0

    await engine._tick()

    assert engine._state.peak_equity == pytest.approx(101.0)
    assert engine._state.recent_returns == [pytest.approx((101.0 - 99.0) / 99.0)]
    assert "DD 0.00%" in notify.messages[-1]
    equity_records = engine._journal.load_equity()
    assert equity_records[-1]["peak"] == pytest.approx(101.0)
    assert equity_records[-1]["dd_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_target_tick_forces_target_refresh_after_fetch(
    monkeypatch, tmp_path
) -> None:
    """Target mode does not reuse an old matrix after parquet refresh."""

    exchange = FakeExchange(equity=Decimal("101"))
    strategy = RefreshingTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={},
    )
    engine = LiveEngine(
        exchange,
        dry_run=True,
        target_strategy=strategy,
        journal_dir=str(tmp_path),
    )

    async def fake_fetch_target_data():
        return None

    monkeypatch.setattr(engine, "_fetch_target_data", fake_fetch_target_data)

    await engine._tick()

    assert strategy.refresh_forces == [True]


@pytest.mark.asyncio
async def test_target_strategy_can_request_warmup_cache_sizes(monkeypatch) -> None:
    """Target strategies can ask live fetch to backfill beyond one API page."""

    exchange = FakeExchange()
    strategy = WarmupTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={},
    )
    engine = LiveEngine(exchange, dry_run=True, target_strategy=strategy)
    calls: list[dict] = []

    async def fake_fetch_bars(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(engine, "_fetch_bars", fake_fetch_bars)

    await engine._fetch_target_data()

    assert calls == [
        {"interval": "1h", "timeframe_hours": 1, "min_bars": 60_000},
        {"interval": "6h", "timeframe_hours": 6, "min_bars": 10_000},
    ]


def _bars_frame(start: datetime, rows: int = 2) -> pl.DataFrame:
    times = [start + timedelta(hours=i) for i in range(rows)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1.0] * rows,
            "close_time": times,
            "quote_volume": [100.0 + i for i in range(rows)],
            "trades": [1] * rows,
            "taker_buy_volume": [0.5] * rows,
            "taker_buy_quote_volume": [50.0 + i for i in range(rows)],
            "ignore": [0.0] * rows,
        }
    )


@pytest.mark.asyncio
async def test_initial_large_warmup_fetch_uses_paginated_backfill(
    monkeypatch, tmp_path
) -> None:
    """Fresh target-mode caches use paginated fetch for warmup-sized history."""

    exchange = FakeExchange()
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=True,
        data_dir=str(tmp_path / "data"),
    )
    bars = _bars_frame(datetime(2024, 1, 1, tzinfo=UTC))
    paginated_calls: list[dict] = []

    async def fake_fetch_all_klines(client, **kwargs):
        paginated_calls.append(kwargs)
        return bars

    async def fail_single_page_fetch(*args, **kwargs):
        raise AssertionError("single-page fetch should not be used")

    monkeypatch.setattr("executor.live_data.fetch_all_klines", fake_fetch_all_klines)
    monkeypatch.setattr("executor.live_data.fetch_klines", fail_single_page_fetch)

    await engine._fetch_bars(interval="1h", timeframe_hours=1, min_bars=60_000)

    assert paginated_calls
    assert paginated_calls[0]["symbol"] == "BTCUSDT"
    assert paginated_calls[0]["interval"] == "1h"
    assert paginated_calls[0]["start_ms"] is not None
    assert len(engine._store.read("BTCUSDT", "1h")) == 2


@pytest.mark.asyncio
async def test_underfilled_large_warmup_cache_uses_paginated_backfill(
    monkeypatch, tmp_path
) -> None:
    """A recent but underfilled cache still needs warmup backfill."""

    exchange = FakeExchange()
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=True,
        data_dir=str(tmp_path / "data"),
    )
    cached = _bars_frame(datetime.now(tz=UTC) - timedelta(hours=2), rows=2)
    engine._store.write("BTCUSDT", "1h", cached)
    backfill = _bars_frame(datetime(2024, 1, 1, tzinfo=UTC), rows=3)
    paginated_calls: list[dict] = []

    async def fake_fetch_all_klines(client, **kwargs):
        paginated_calls.append(kwargs)
        return backfill

    async def fail_single_page_fetch(*args, **kwargs):
        raise AssertionError("single-page fetch should not be used")

    monkeypatch.setattr("executor.live_data.fetch_all_klines", fake_fetch_all_klines)
    monkeypatch.setattr("executor.live_data.fetch_klines", fail_single_page_fetch)

    await engine._fetch_bars(interval="1h", timeframe_hours=1, min_bars=60_000)

    assert paginated_calls
    assert len(engine._store.read("BTCUSDT", "1h")) == 5


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
    notifier = CaptureNotifier()
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=False,
        journal_dir=str(tmp_path / "journal"),
        target_strategy=strategy,
        notify=notifier,
    )

    await engine._tick()

    assert notifier.messages[-1] == (
        "⏰ Tick #1 | Eq $10000.0 | Avail $10000.0 | DD 0.00%\n"
        "Actions (2):\n"
        "- SELL BTC $5000 reduce\n"
        "- SELL BTC $2000\n"
        "Positions: BTC S"
    )
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
async def test_target_mode_max_drawdown_flattens_without_new_exposure(tmp_path) -> None:
    """Target mode shares the live max-DD hard stop instead of rebalancing."""

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
        equity=Decimal("8000"),
        positions=positions,
        market_price=Decimal("50000"),
    )
    strategy = StaticTargetStrategy(
        profile=StrategyProfile("test-target", "Test target", timeframe_hours=1),
        weights={"BTCUSDT": 0.5},
    )
    engine = LiveEngine(
        exchange,
        symbols=["BTC"],
        dry_run=False,
        journal_dir=str(tmp_path / "journal"),
        target_strategy=strategy,
    )
    engine._state.peak_equity = 10_000.0

    await engine._tick()

    assert exchange.orders == [("BTC", False, Decimal("0.1"), True)]
    assert engine._state.positions == {}
    assert engine._state.dd_breaker_active is True


@pytest.mark.asyncio
async def test_close_position_uses_actual_filled_size(tmp_path) -> None:
    """Live state and journal reflect partial IOC fills, not requested size."""

    exchange = FakeExchange(
        market_price=Decimal("51000"),
        filled_size=Decimal("0.04"),
    )
    engine = LiveEngine(
        exchange,
        dry_run=False,
        journal_dir=str(tmp_path / "journal"),
    )
    engine._state.positions["BTC"] = PositionState(
        symbol="BTC",
        qty=0.10,
        entry_price=50_000.0,
        entry_bar=0,
        best_price=51_000.0,
    )

    await engine._close_position("BTC", "test_partial_fill")

    assert engine._state.positions["BTC"].qty == pytest.approx(0.06)
    trade = engine._journal.load_trades()[-1]
    assert trade["qty"] == pytest.approx(0.04)
    assert trade["price"] == pytest.approx(51_000.0)


@pytest.mark.asyncio
async def test_close_failure_restores_pre_tick_position(tmp_path) -> None:
    """Failed reduce orders must not leave optimistic decision state persisted."""

    exchange = FakeExchange(order_success=False)
    engine = LiveEngine(exchange, dry_run=False, journal_dir=str(tmp_path / "journal"))
    pos = PositionState("BTC", 0.10, 50_000.0, 0, 51_000.0)
    engine._state.positions.pop("BTC", None)

    ok = await engine._close_position(
        "BTC",
        "failed_reduce",
        positions_before={"BTC": pos},
    )

    assert ok is False
    assert engine._state.positions["BTC"].qty == pytest.approx(0.10)
    assert engine._journal.load_trades() == []


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
