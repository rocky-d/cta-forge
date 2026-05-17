"""Trade journal: append-only log for live engine events.

Records every tick's equity snapshot and every trade action to JSONL files.
These files are the data source for post-hoc performance reports.

Files written (inside the configured directory):
  - equity.jsonl: one line per tick {ts, bar, equity, peak, dd_pct, n_positions, positions}
  - trades.jsonl: one line per trade action {ts, bar, kind, symbol, qty, price, reason, pnl, ...}
  - signals.jsonl: one line per tick {ts, bar, signals: {symbol: value}}
  - targets.jsonl: one line per target tick {ts, bar, profile, target_ts, weights, ignored_weights, orders}

When configured, records also include additive runtime identity fields such as
live_instance_id, run_id, and public_instance_slug. Older readers can ignore
these unknown fields.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class LiveJournalStore(Protocol):
    """Persistence port for live journal records."""

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
    ) -> None:
        """Record an equity snapshot for the current tick."""
        ...

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
        """Record a trade action."""
        ...

    def record_signals(self, bar: int, signals: dict[str, float]) -> None:
        """Record signal values for the current tick."""
        ...

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
        """Record target-weight diagnostics."""
        ...

    def load_equity(self) -> list[dict]:
        """Load equity snapshots."""
        ...

    def load_trades(self) -> list[dict]:
        """Load trade records."""
        ...

    def load_signals(self) -> list[dict]:
        """Load signal records."""
        ...

    def load_targets(self) -> list[dict]:
        """Load target diagnostics."""
        ...


class TradeJournal:
    """Append-only trade journal backed by JSONL files."""

    def __init__(
        self,
        directory: str | Path = "journal",
        *,
        live_instance_id: str | None = None,
        run_id: str | None = None,
        public_instance_slug: str | None = None,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._equity_file = self._dir / "equity.jsonl"
        self._trades_file = self._dir / "trades.jsonl"
        self._signals_file = self._dir / "signals.jsonl"
        self._targets_file = self._dir / "targets.jsonl"
        self._identity_fields = {
            key: value
            for key, value in {
                "live_instance_id": live_instance_id,
                "run_id": run_id,
                "public_instance_slug": public_instance_slug,
            }.items()
            if value
        }
        logger.info("TradeJournal initialized: %s", self._dir)

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
    ) -> None:
        """Record an equity snapshot for the current tick.

        ``dd_pct`` is stored as a positive percentage below the running peak.
        """
        peak = max(float(peak_equity), float(equity))
        dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "equity": float(equity),
            "peak": float(peak),
            "dd_pct": float(dd_pct),
            "n_positions": len(positions),
            "positions": positions,
        }
        self._append(self._equity_file, self._with_identity(record))

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
            "qty": float(qty),
            "price": float(price),
            "reason": reason,
        }
        # Only include PnL fields for closes
        if kind in ("close", "partial_close", "flatten_all"):
            record["entry_price"] = float(entry_price)
            record["pnl"] = float(pnl)
            record["pnl_pct"] = float(pnl_pct)
            record["held_bars"] = held_bars

        self._append(self._trades_file, self._with_identity(record))

    def record_signals(
        self,
        bar: int,
        signals: dict[str, float],
    ) -> None:
        """Record signal values for the current tick."""
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "signals": {k: float(v) for k, v in signals.items()},
        }
        self._append(self._signals_file, self._with_identity(record))

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
        ignored_gross = sum(abs(v) for v in (ignored_weights or {}).values())
        execution_coverage = (
            normalized_gross / target_gross if abs(target_gross) > 1e-12 else 1.0
        )
        ignored_gross_ratio = (
            ignored_gross / target_gross if abs(target_gross) > 1e-12 else 0.0
        )
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "bar": bar,
            "profile": profile,
            "target_ts": target_ts,
            "staleness_seconds": float(staleness_seconds),
            "target_gross": float(target_gross),
            "normalized_gross": float(normalized_gross),
            "ignored_gross": float(ignored_gross),
            "ignored_gross_ratio": float(ignored_gross_ratio),
            "execution_coverage": float(execution_coverage),
            "weights": {k: float(v) for k, v in weights.items() if abs(v) > 1e-12},
            "ignored_weights": {
                k: float(v)
                for k, v in (ignored_weights or {}).items()
                if abs(v) > 1e-12
            },
            "orders": orders,
        }
        self._append(self._targets_file, self._with_identity(record))

    def load_equity(self) -> list[dict]:
        """Load all equity snapshots from JSONL."""
        return self._read(self._equity_file)

    def load_equity_decimal_safe(self) -> list[dict]:
        """Load equity snapshots with Decimal-preserved JSON floats."""
        return self._read(self._equity_file, parse_float=Decimal)

    def load_trades(self) -> list[dict]:
        """Load all trade records from JSONL."""
        return self._read(self._trades_file)

    def load_trades_decimal_safe(self) -> list[dict]:
        """Load trade records with Decimal-preserved JSON floats."""
        return self._read(self._trades_file, parse_float=Decimal)

    def load_signals(self) -> list[dict]:
        """Load all signal records from JSONL."""
        return self._read(self._signals_file)

    def load_signals_decimal_safe(self) -> list[dict]:
        """Load signal records with Decimal-preserved JSON floats."""
        return self._read(self._signals_file, parse_float=Decimal)

    def load_targets(self) -> list[dict]:
        """Load all target diagnostics from JSONL."""
        return self._read(self._targets_file)

    def load_targets_decimal_safe(self) -> list[dict]:
        """Load target diagnostics with Decimal-preserved JSON floats."""
        return self._read(self._targets_file, parse_float=Decimal)

    def _read(self, path: Path, *, parse_float=None) -> list[dict]:
        """Read all JSON records from a JSONL file."""
        if not path.exists():
            return []
        records = []
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line, parse_float=parse_float))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt journal line %s:%d", path, line_no)
        return records

    def _with_identity(self, record: dict) -> dict:
        """Attach configured runtime identity fields to a journal record."""
        if self._identity_fields:
            record.update(self._identity_fields)
        return record

    def _append(self, path: Path, record: dict) -> None:
        """Append a JSON record to a JSONL file."""
        try:
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to write to %s", path)
