from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from executor.live import LiveState
from executor.live_persistence_dual import (
    DualLiveJournalStore,
    DualLiveStateStore,
    parse_persistence_backend,
)


@dataclass
class RecordingJournal:
    name: str
    events: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]]
    fail_ops: set[str] = field(default_factory=set)
    equity_rows: list[dict[str, Any]] = field(default_factory=lambda: [{"bar": 1}])
    trade_rows: list[dict[str, Any]] = field(
        default_factory=lambda: [{"symbol": "BTC"}]
    )
    signal_rows: list[dict[str, Any]] = field(default_factory=lambda: [{"signals": {}}])
    target_rows: list[dict[str, Any]] = field(
        default_factory=lambda: [{"target_gross": 1.0}]
    )

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
        *,
        dry_run: bool = False,
    ) -> None:
        self._record("record_tick", bar, equity, peak_equity, positions)

    def record_trade(
        self,
        bar: int,
        kind: str,
        symbol: str,
        qty: float,
        price: float,
        reason: str,
        *,
        side: str = "",
        entry_price: float = 0.0,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        held_bars: int = 0,
        exchange_order_id: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._record(
            "record_trade",
            bar,
            kind,
            symbol,
            qty,
            price,
            reason,
            side=side,
            entry_price=entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held_bars,
            exchange_order_id=exchange_order_id,
        )

    def record_signals(
        self, bar: int, signals: dict[str, float], *, dry_run: bool = False
    ) -> None:
        self._record("record_signals", bar, signals)

    def record_target(
        self,
        *,
        bar: int,
        profile: str,
        target_ts: str,
        staleness_seconds: float,
        target_gross: float,
        normalized_gross: float,
        weights: dict[str, float],
        orders: list[dict],
        ignored_weights: dict[str, float] | None = None,
        submitted_orders: list[dict] | None = None,
        filled_trades: list[dict] | None = None,
        failed_orders: list[dict] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._record(
            "record_target",
            bar=bar,
            profile=profile,
            target_ts=target_ts,
            staleness_seconds=staleness_seconds,
            target_gross=target_gross,
            normalized_gross=normalized_gross,
            weights=weights,
            orders=orders,
            ignored_weights=ignored_weights,
            submitted_orders=submitted_orders,
            filled_trades=filled_trades,
            failed_orders=failed_orders,
        )

    def load_equity(self) -> list[dict]:
        return self.equity_rows

    def load_trades(self) -> list[dict]:
        return self.trade_rows

    def load_signals(self) -> list[dict]:
        return self.signal_rows

    def load_targets(self) -> list[dict]:
        return self.target_rows

    def _record(self, operation: str, *args: Any, **kwargs: Any) -> None:
        self.events.append((self.name, operation, args, kwargs))
        if operation in self.fail_ops:
            msg = f"{self.name} failed {operation}"
            raise RuntimeError(msg)


@dataclass
class RecordingStateStore:
    name: str
    events: list[tuple[str, str, int | None]]
    state: LiveState | None = None
    fail_save: bool = False

    def load(self) -> LiveState | None:
        self.events.append((self.name, "load", None))
        return self.state

    def save(self, state: LiveState) -> None:
        self.events.append((self.name, "save", state.bar_count))
        if self.fail_save:
            msg = f"{self.name} failed save"
            raise RuntimeError(msg)
        self.state = state


def test_parse_persistence_backend_defaults_to_file() -> None:
    assert parse_persistence_backend(None) == "file"
    assert parse_persistence_backend("") == "file"
    assert parse_persistence_backend("  FILE ") == "file"
    assert parse_persistence_backend("dual") == "dual"
    assert parse_persistence_backend("postgres") == "postgres"


@pytest.mark.parametrize("value", ["", "sqlite", "postgresql", "db"])
def test_parse_persistence_backend_rejects_invalid_non_empty_values(value: str) -> None:
    if not value:
        return
    with pytest.raises(ValueError, match="invalid PERSISTENCE_BACKEND"):
        parse_persistence_backend(value)


def test_dual_journal_writes_primary_before_shadow_and_reads_primary() -> None:
    events: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
    primary = RecordingJournal("primary", events, equity_rows=[{"bar": 10}])
    shadow = RecordingJournal("shadow", events, equity_rows=[{"bar": 99}])
    store = DualLiveJournalStore(primary, shadow)

    store.record_tick(10, 100.0, 101.0, {"BTC": {"qty": 1.0}})
    store.record_trade(10, "target_buy", "BTC", 0.1, 100000.0, "target")
    store.record_signals(10, {"BTC": 0.5})
    store.record_target(
        bar=10,
        profile="v16a",
        target_ts="2026-05-15T00:00:00Z",
        staleness_seconds=120.0,
        target_gross=0.5,
        normalized_gross=0.5,
        weights={"BTC": 0.5},
        orders=[],
    )

    assert [event[:2] for event in events] == [
        ("primary", "record_tick"),
        ("shadow", "record_tick"),
        ("primary", "record_trade"),
        ("shadow", "record_trade"),
        ("primary", "record_signals"),
        ("shadow", "record_signals"),
        ("primary", "record_target"),
        ("shadow", "record_target"),
    ]
    assert store.load_equity() == [{"bar": 10}]
    assert store.load_trades() == primary.trade_rows
    assert store.load_signals() == primary.signal_rows
    assert store.load_targets() == primary.target_rows


def test_dual_journal_warn_policy_keeps_primary_success_when_shadow_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
    primary = RecordingJournal("primary", events)
    shadow = RecordingJournal("shadow", events, fail_ops={"record_signals"})
    store = DualLiveJournalStore(primary, shadow)

    store.record_signals(11, {"ETH": 0.2})

    assert [event[:2] for event in events] == [
        ("primary", "record_signals"),
        ("shadow", "record_signals"),
    ]
    assert "shadow live journal write failed: record_signals" in caplog.text


def test_dual_journal_raise_policy_raises_after_primary_success() -> None:
    events: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
    primary = RecordingJournal("primary", events)
    shadow = RecordingJournal("shadow", events, fail_ops={"record_tick"})
    store = DualLiveJournalStore(primary, shadow, shadow_failure_policy="raise")

    with pytest.raises(RuntimeError, match="shadow failed record_tick"):
        store.record_tick(12, 100.0, 100.0, {})

    assert [event[:2] for event in events] == [
        ("primary", "record_tick"),
        ("shadow", "record_tick"),
    ]


def test_dual_journal_does_not_shadow_when_primary_fails() -> None:
    events: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
    primary = RecordingJournal("primary", events, fail_ops={"record_trade"})
    shadow = RecordingJournal("shadow", events)
    store = DualLiveJournalStore(primary, shadow)

    with pytest.raises(RuntimeError, match="primary failed record_trade"):
        store.record_trade(13, "target_buy", "BTC", 0.1, 100000.0, "target")

    assert [event[:2] for event in events] == [("primary", "record_trade")]


def test_dual_state_store_reads_primary_and_saves_primary_before_shadow() -> None:
    events: list[tuple[str, str, int | None]] = []
    state = LiveState(bar_count=21, initial_equity=100.0, peak_equity=105.0)
    primary = RecordingStateStore("primary", events, state=state)
    shadow = RecordingStateStore("shadow", events)
    store = DualLiveStateStore(primary, shadow)

    assert store.load() is state
    updated = LiveState(bar_count=22, initial_equity=100.0, peak_equity=106.0)
    store.save(updated)

    assert events == [
        ("primary", "load", None),
        ("primary", "save", 22),
        ("shadow", "save", 22),
    ]
    assert primary.state is updated
    assert shadow.state is updated


def test_dual_state_store_warn_policy_keeps_primary_success_when_shadow_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[tuple[str, str, int | None]] = []
    primary = RecordingStateStore("primary", events)
    shadow = RecordingStateStore("shadow", events, fail_save=True)
    store = DualLiveStateStore(primary, shadow)

    state = LiveState(bar_count=23, initial_equity=100.0, peak_equity=100.0)
    store.save(state)

    assert events == [("primary", "save", 23), ("shadow", "save", 23)]
    assert primary.state is state
    assert "shadow live checkpoint write failed: save" in caplog.text


def test_dual_state_store_raise_policy_raises_after_primary_success() -> None:
    events: list[tuple[str, str, int | None]] = []
    primary = RecordingStateStore("primary", events)
    shadow = RecordingStateStore("shadow", events, fail_save=True)
    store = DualLiveStateStore(primary, shadow, shadow_failure_policy="raise")

    state = LiveState(bar_count=24, initial_equity=100.0, peak_equity=100.0)
    with pytest.raises(RuntimeError, match="shadow failed save"):
        store.save(state)

    assert events == [("primary", "save", 24), ("shadow", "save", 24)]
    assert primary.state is state
