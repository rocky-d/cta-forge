"""State persistence for the live trading engine.

Saves engine state to JSON on every tick so restarts can resume
without losing position tracking, equity history, or signal state.

File format: JSON with ISO timestamps for human readability.
Location: configurable, defaults to `{project}/engine-state.json`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .live import LivePosition, LiveState

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "engine-state.json"


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, o: object) -> object:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


def save_state(state: LiveState, path: str | Path = DEFAULT_STATE_FILE) -> None:
    """Persist engine state to disk."""
    path = Path(path)
    data = {
        "version": 1,
        "saved_at": datetime.now(tz=UTC).isoformat(),
        "bar_count": state.bar_count,
        "initial_equity": state.initial_equity,
        "peak_equity": state.peak_equity,
        "dd_breaker_active": state.dd_breaker_active,
        "last_signals": state.last_signals,
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

    # Write atomically: write to tmp then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, cls=_DecimalEncoder))
    tmp_path.rename(path)
    logger.info("State saved: %d positions, bar #%d", len(state.positions), state.bar_count)


def load_state(path: str | Path = DEFAULT_STATE_FILE) -> LiveState | None:
    """Load engine state from disk. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        logger.info("No state file found at %s — starting fresh", path)
        return None

    try:
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            logger.warning("Unknown state version %s — ignoring", data.get("version"))
            return None

        state = LiveState(
            bar_count=data["bar_count"],
            initial_equity=data["initial_equity"],
            peak_equity=data["peak_equity"],
            dd_breaker_active=data.get("dd_breaker_active", False),
            last_signals=data.get("last_signals", {}),
        )

        for sym, pos_data in data.get("positions", {}).items():
            state.positions[sym] = LivePosition(
                symbol=pos_data["symbol"],
                side=pos_data["side"],
                entry_price=pos_data["entry_price"],
                entry_bar=pos_data["entry_bar"],
                size=Decimal(pos_data["size"]),
                trailing_stop=pos_data["trailing_stop"],
                highest_pnl=pos_data.get("highest_pnl", 0.0),
                bars_held=pos_data.get("bars_held", 0),
            )

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
