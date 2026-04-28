"""Trade journal: append-only log for live engine events.

Records every tick's equity snapshot and every trade action to JSONL files.
These files are the data source for post-hoc performance reports.

Files written (inside the configured directory):
  - equity.jsonl: one line per tick {ts, bar, equity, peak, dd_pct, n_positions, positions}
  - trades.jsonl: one line per trade action {ts, bar, kind, symbol, qty, price, reason, pnl, ...}
  - signals.jsonl: one line per tick {ts, bar, signals: {symbol: value}}
  - targets.jsonl: one line per target tick {ts, bar, profile, target_ts, weights, ignored_weights, orders}
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
        self._targets_file = self._dir / "targets.jsonl"
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
    ) -> None:
        """Record target-weight diagnostics for shadow/live reconciliation."""
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "profile": profile,
            "target_ts": target_ts,
            "staleness_seconds": round(staleness_seconds, 3),
            "target_gross": round(target_gross, 6),
            "normalized_gross": round(normalized_gross, 6),
            "weights": {k: round(v, 8) for k, v in weights.items() if abs(v) > 1e-12},
            "ignored_weights": {
                k: round(v, 8)
                for k, v in (ignored_weights or {}).items()
                if abs(v) > 1e-12
            },
            "orders": orders,
        }
        self._append(self._targets_file, record)

    def load_equity(self) -> list[dict]:
        """Load all equity snapshots from JSONL."""
        return self._read(self._equity_file)

    def load_trades(self) -> list[dict]:
        """Load all trade records from JSONL."""
        return self._read(self._trades_file)

    def load_signals(self) -> list[dict]:
        """Load all signal records from JSONL."""
        return self._read(self._signals_file)

    def load_targets(self) -> list[dict]:
        """Load all target diagnostics from JSONL."""
        return self._read(self._targets_file)

    def _read(self, path: Path) -> list[dict]:
        """Read all JSON records from a JSONL file."""
        if not path.exists():
            return []
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _append(self, path: Path, record: dict) -> None:
        """Append a JSON record to a JSONL file."""
        try:
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to write to %s", path)
