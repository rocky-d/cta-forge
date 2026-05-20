"""State persistence for the live trading engine.

Saves engine state to JSON on every tick so restarts can resume
without losing position tracking, equity history, or signal state.

File format: JSON with ISO timestamps for human readability.
Location: configurable, defaults to `{project}/engine-state.json`.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from typing import Any
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from .live import LivePosition, LiveState

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "engine-state.json"


class LiveStateStore(Protocol):
    """Persistence port for live engine checkpoints."""

    def load(self) -> LiveState | None:
        """Load the latest live engine state checkpoint."""
        ...

    def save(self, state: LiveState) -> None:
        """Persist the latest live engine state checkpoint."""
        ...


class JsonFileLiveStateStore:
    """File-backed live state store using the existing JSON format."""

    def __init__(self, path: str | Path = DEFAULT_STATE_FILE) -> None:
        self._path = Path(path)

    def load(self) -> LiveState | None:
        """Load the latest live engine state checkpoint."""
        return load_state(self._path)

    def save(self, state: LiveState) -> None:
        """Persist the latest live engine state checkpoint."""
        save_state(state, self._path)


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, o: object) -> object:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


def encode_state_payload(
    state: LiveState,
    *,
    saved_at: datetime | None = None,
) -> dict[str, Any]:
    """Encode live engine state into the persisted checkpoint payload."""

    saved_at = saved_at or datetime.now(tz=UTC)
    return {
        "version": 1,
        "saved_at": saved_at.isoformat(),
        "bar_count": state.bar_count,
        "initial_equity": state.initial_equity,
        "peak_equity": state.peak_equity,
        "dd_breaker_active": state.dd_breaker_active,
        "last_signals": state.last_signals,
        "recent_returns": [float(x) for x in state.recent_returns[-120:]],
        "last_tick_equity": state.last_tick_equity,
        "positions": {
            sym: {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "entry_bar": pos.entry_bar,
                "size": str(pos.size),
                "trailing_stop": pos.trailing_stop,
                "highest_pnl": pos.highest_pnl,
                "bars_held": pos.bars_held,
            }
            for sym, pos in state.positions.items()
        },
    }


def decode_state_payload(data: dict[str, Any]) -> LiveState | None:
    """Decode a persisted checkpoint payload into live engine state."""

    if data.get("version") != 1:
        logger.warning("Unknown state version %s — ignoring", data.get("version"))
        return None

    state = LiveState(
        bar_count=int(data["bar_count"]),
        initial_equity=float(data["initial_equity"]),
        peak_equity=float(data["peak_equity"]),
        dd_breaker_active=data.get("dd_breaker_active", False),
        last_signals=data.get("last_signals", {}),
        recent_returns=[float(x) for x in data.get("recent_returns", [])[-120:]],
        last_tick_equity=(
            float(data["last_tick_equity"])
            if data.get("last_tick_equity") is not None
            else None
        ),
    )

    for sym, pos_data in data.get("positions", {}).items():
        state.positions[sym] = LivePosition(
            symbol=pos_data["symbol"],
            side=pos_data["side"],
            entry_price=float(pos_data["entry_price"]),
            entry_bar=int(pos_data["entry_bar"]),
            size=Decimal(pos_data["size"]),
            trailing_stop=float(pos_data["trailing_stop"]),
            highest_pnl=float(pos_data.get("highest_pnl", 0.0)),
            bars_held=int(pos_data.get("bars_held", 0)),
        )
    return state


def save_state(state: LiveState, path: str | Path = DEFAULT_STATE_FILE) -> None:
    """Persist engine state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode_state_payload(state)

    # Write atomically: write to tmp then move.
    # shutil.move handles Docker bind-mounted files where os.rename fails
    # with ERRNO 16 (Device or resource busy).
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, cls=_DecimalEncoder))
    shutil.move(str(tmp_path), str(path))
    logger.info(
        "State saved: %d positions, bar #%d", len(state.positions), state.bar_count
    )


def load_state(path: str | Path = DEFAULT_STATE_FILE) -> LiveState | None:
    """Load engine state from disk. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        logger.info("No state file found at %s — starting fresh", path)
        return None

    try:
        data = json.loads(path.read_text())
        state = decode_state_payload(data)
        if state is None:
            return None

        saved_at = data.get("saved_at", "unknown")
        logger.info(
            "State loaded: %d positions, bar #%d (saved %s)",
            len(state.positions),
            state.bar_count,
            saved_at,
        )
        return state

    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to load state from %s: %s", path, e)
        return None
