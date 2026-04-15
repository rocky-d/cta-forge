"""Trade journal: append-only log for live engine events.

Records every tick's equity snapshot and every trade action to JSONL files.
These files are the data source for post-hoc performance reports.

Files written (inside the configured directory):
  - equity.jsonl: one line per tick {ts, bar, equity, peak, dd_pct, n_positions, positions}
  - trades.jsonl: one line per trade action {ts, bar, kind, symbol, qty, price, reason, pnl, ...}
  - signals.jsonl: one line per tick {ts, bar, signals: {symbol: value}}
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeJournal:
    """Append-only trade journal backed by JSONL files."""

    def __init__(self, directory: str | Path = "journal") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._equity_file = self._dir / "equity.jsonl"
        self._trades_file = self._dir / "trades.jsonl"
        self._signals_file = self._dir / "signals.jsonl"
        logger.info("TradeJournal initialized: %s", self._dir)

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
    ) -> None:
        """Record an equity snapshot for the current tick."""
        dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "equity": round(equity, 2),
            "peak": round(peak_equity, 2),
            "dd_pct": round(dd_pct, 2),
            "n_positions": len(positions),
            "positions": positions,
        }
        self._append(self._equity_file, record)

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
    ) -> None:
        """Record a trade action (open, close, partial close, flatten)."""
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "kind": kind,
            "symbol": symbol,
            "side": side,
            "qty": round(qty, 8),
            "price": round(price, 4),
            "reason": reason,
        }
        # Only include PnL fields for closes
        if kind in ("close", "partial_close", "flatten_all"):
            record["entry_price"] = round(entry_price, 4)
            record["pnl"] = round(pnl, 4)
            record["pnl_pct"] = round(pnl_pct, 4)
            record["held_bars"] = held_bars

        self._append(self._trades_file, record)

    def record_signals(
        self,
        bar: int,
        signals: dict[str, float],
    ) -> None:
        """Record signal values for the current tick."""
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "signals": {k: round(v, 4) for k, v in signals.items()},
        }
        self._append(self._signals_file, record)

    def _append(self, path: Path, record: dict) -> None:
        """Append a JSON record to a JSONL file."""
        try:
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to write to %s", path)
