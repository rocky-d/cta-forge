"""Tests for engine state persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from executor.live import LivePosition, LiveState
from executor.state import (
    JsonFileLiveStateStore,
    decode_state_payload,
    encode_state_payload,
    load_state,
    save_state,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_save_and_load_empty_state(tmp_path: Path) -> None:
    """Save and load state with no positions."""
    path = tmp_path / "state.json"
    state = LiveState(
        bar_count=5,
        initial_equity=10000.0,
        peak_equity=10500.0,
        recent_returns=[0.01, -0.02],
        last_tick_equity=10400.0,
    )

    save_state(state, path)
    assert path.exists()

    loaded = load_state(path)
    assert loaded is not None
    assert loaded.bar_count == 5
    assert loaded.initial_equity == 10000.0
    assert loaded.peak_equity == 10500.0
    assert loaded.recent_returns == [0.01, -0.02]
    assert loaded.last_tick_equity == 10400.0
    assert len(loaded.positions) == 0


def test_save_and_load_with_positions(tmp_path: Path) -> None:
    """Save and load state with positions."""
    path = tmp_path / "state.json"
    state = LiveState(
        bar_count=10,
        initial_equity=2000.0,
        peak_equity=2100.0,
        dd_breaker_active=True,
        last_signals={"BTC": 0.55, "ETH": 0.42},
    )
    state.positions["BTC"] = LivePosition(
        symbol="BTC",
        side="long",
        entry_price=73000.0,
        entry_bar=3,
        size=Decimal("0.005"),
        trailing_stop=69000.0,
        highest_pnl=0.02,
        bars_held=7,
    )
    state.positions["ETH"] = LivePosition(
        symbol="ETH",
        side="short",
        entry_price=2250.0,
        entry_bar=5,
        size=Decimal("0.15"),
        trailing_stop=2350.0,
        highest_pnl=0.01,
        bars_held=5,
    )

    save_state(state, path)
    loaded = load_state(path)

    assert loaded is not None
    assert loaded.bar_count == 10
    assert loaded.dd_breaker_active is True
    assert loaded.last_signals == {"BTC": 0.55, "ETH": 0.42}
    assert loaded.recent_returns == []
    assert loaded.last_tick_equity is None
    assert len(loaded.positions) == 2

    btc = loaded.positions["BTC"]
    assert btc.side == "long"
    assert btc.entry_price == 73000.0
    assert btc.size == Decimal("0.005")
    assert btc.trailing_stop == 69000.0
    assert btc.bars_held == 7

    eth = loaded.positions["ETH"]
    assert eth.side == "short"
    assert eth.size == Decimal("0.15")


def test_json_file_live_state_store_roundtrip(tmp_path: Path) -> None:
    """File-backed state store preserves the existing JSON state format."""
    path = tmp_path / "state.json"
    store = JsonFileLiveStateStore(path)
    state = LiveState(
        bar_count=7,
        initial_equity=100.123456789,
        peak_equity=105.987654321,
        recent_returns=[0.00123456789],
        last_tick_equity=104.555555555,
    )

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.bar_count == 7
    assert loaded.initial_equity == 100.123456789
    assert loaded.peak_equity == 105.987654321
    assert loaded.recent_returns == [0.00123456789]
    assert loaded.last_tick_equity == 104.555555555


def test_state_payload_codec_preserves_existing_checkpoint_shape() -> None:
    """Pure codec keeps the same persisted state payload shape for DB reuse."""
    state = LiveState(
        bar_count=12,
        initial_equity=100.123456789,
        peak_equity=111.987654321,
        dd_breaker_active=True,
        last_signals={"BTC": 0.25},
        recent_returns=[float(i) for i in range(130)],
        last_tick_equity=109.555555555,
    )
    state.positions["BTC"] = LivePosition(
        symbol="BTC",
        side="long",
        entry_price=73000.0,
        entry_bar=3,
        size=Decimal("0.000123456789"),
        trailing_stop=69000.0,
        highest_pnl=0.02,
        bars_held=7,
    )

    payload = encode_state_payload(
        state,
        saved_at=datetime(2026, 5, 14, 6, 3, tzinfo=UTC),
    )
    decoded = decode_state_payload(payload)

    assert payload["version"] == 1
    assert payload["saved_at"] == "2026-05-14T06:03:00+00:00"
    assert payload["recent_returns"] == [float(i) for i in range(10, 130)]
    assert payload["positions"]["BTC"]["size"] == "0.000123456789"
    assert decoded is not None
    assert decoded.bar_count == state.bar_count
    assert decoded.last_tick_equity == state.last_tick_equity
    assert decoded.positions["BTC"].size == Decimal("0.000123456789")


def test_decode_state_payload_rejects_wrong_version() -> None:
    assert decode_state_payload({"version": 99}) is None


def test_load_missing_file(tmp_path: Path) -> None:
    """Loading from non-existent file returns None."""
    result = load_state(tmp_path / "nope.json")
    assert result is None


def test_load_corrupted_file(tmp_path: Path) -> None:
    """Loading corrupted JSON returns None."""
    path = tmp_path / "state.json"
    path.write_text("not json at all")
    result = load_state(path)
    assert result is None


def test_load_wrong_version(tmp_path: Path) -> None:
    """Loading wrong version returns None."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99}))
    result = load_state(path)
    assert result is None


def test_save_atomic(tmp_path: Path) -> None:
    """Save should not leave .tmp files."""
    path = tmp_path / "state.json"
    state = LiveState(bar_count=1, initial_equity=1000.0, peak_equity=1000.0)
    save_state(state, path)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    """Nested state paths should fail less brittle runtime deployments."""
    path = tmp_path / "nested" / "state.json"
    state = LiveState(bar_count=1, initial_equity=1000.0, peak_equity=1000.0)

    save_state(state, path)

    assert path.exists()
