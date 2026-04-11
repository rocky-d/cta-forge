"""Tests for engine state persistence."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING

from engine.live import LivePosition, LiveState
from engine.state import load_state, save_state

if TYPE_CHECKING:
    from pathlib import Path


def test_save_and_load_empty_state(tmp_path: Path) -> None:
    """Save and load state with no positions."""
    path = tmp_path / "state.json"
    state = LiveState(bar_count=5, initial_equity=10000.0, peak_equity=10500.0)

    save_state(state, path)
    assert path.exists()

    loaded = load_state(path)
    assert loaded is not None
    assert loaded.bar_count == 5
    assert loaded.initial_equity == 10000.0
    assert loaded.peak_equity == 10500.0
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
