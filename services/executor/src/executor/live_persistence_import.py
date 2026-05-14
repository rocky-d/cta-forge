"""Helpers for importing existing live persistence files.

This module is intentionally pure file parsing. It does not open database
connections and does not participate in live trading. The goal is to make the
future JSONL/state -> PostgreSQL importer reuse one Decimal-safe loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

JOURNAL_FILES = {
    "equity": "equity.jsonl",
    "trades": "trades.jsonl",
    "signals": "signals.jsonl",
    "targets": "targets.jsonl",
}


class LivePersistenceImportError(ValueError):
    """Raised when historical live persistence files cannot be imported safely."""


@dataclass(frozen=True)
class LivePersistenceImportBatch:
    """Decimal-safe records loaded from the current file-backed live persistence."""

    journal_dir: Path
    state_file: Path | None
    state: dict[str, Any] | None
    equity: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    targets: list[dict[str, Any]]


def load_existing_live_persistence(
    journal_dir: str | Path,
    *,
    state_file: str | Path | None = None,
) -> LivePersistenceImportBatch:
    """Load current JSONL journals and optional state file for DB import.

    Missing journal files are treated as empty streams so partially populated
    historical directories can still be inspected. Malformed JSON fails closed
    with path and line context; migration should not silently skip records.
    """

    journal_path = Path(journal_dir)
    state_path = Path(state_file) if state_file is not None else None
    return LivePersistenceImportBatch(
        journal_dir=journal_path,
        state_file=state_path,
        state=_load_json_object(state_path) if state_path is not None else None,
        equity=load_jsonl_records(journal_path / JOURNAL_FILES["equity"]),
        trades=load_jsonl_records(journal_path / JOURNAL_FILES["trades"]),
        signals=load_jsonl_records(journal_path / JOURNAL_FILES["signals"]),
        targets=load_jsonl_records(journal_path / JOURNAL_FILES["targets"]),
    )


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records using Decimal for JSON floats."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = _loads_json(line, jsonl_path, line_no=line_no)
            if not isinstance(record, dict):
                raise LivePersistenceImportError(
                    f"{jsonl_path}:{line_no}: expected JSON object record"
                )
            records.append(record)
    return records


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = _loads_json(path.read_text(), path, line_no=None)
    if not isinstance(record, dict):
        raise LivePersistenceImportError(f"{path}: expected JSON object")
    return record


def _loads_json(raw: str, path: Path, *, line_no: int | None) -> Any:
    try:
        return json.loads(
            raw,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        location = f"{path}:{line_no}" if line_no is not None else str(path)
        raise LivePersistenceImportError(
            f"{location}: invalid JSON: {exc.msg}"
        ) from exc
    except ValueError as exc:
        location = f"{path}:{line_no}" if line_no is not None else str(path)
        raise LivePersistenceImportError(f"{location}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


@dataclass(frozen=True)
class LivePersistenceImportKeys:
    """DB identity keys applied while importing historical file records."""

    live_instance_id: str
    run_id: str

    def __post_init__(self) -> None:
        if not self.live_instance_id.strip():
            raise LivePersistenceImportError("live_instance_id is required")
        if not self.run_id.strip():
            raise LivePersistenceImportError("run_id is required")


@dataclass(frozen=True)
class LivePersistenceImportRows:
    """Rows normalized to the first live persistence schema shape."""

    checkpoint: dict[str, Any] | None
    ticks: list[dict[str, Any]]
    positions: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    signals: list[dict[str, Any]]


def build_live_persistence_import_rows(
    batch: LivePersistenceImportBatch,
    keys: LivePersistenceImportKeys,
) -> LivePersistenceImportRows:
    """Normalize file-backed live records into schema-shaped import rows.

    The returned position rows carry ``tick_bar`` rather than ``tick_id`` because
    PostgreSQL will allocate tick ids during insert. The DB writer can resolve
    ``tick_bar`` through the inserted tick row map.
    """

    checkpoint = _checkpoint_row(batch, keys)
    ticks, positions = _tick_and_position_rows(batch, keys)
    return LivePersistenceImportRows(
        checkpoint=checkpoint,
        ticks=ticks,
        positions=positions,
        targets=[_target_row(record, keys) for record in batch.targets],
        trades=[_trade_row(record, keys) for record in batch.trades],
        signals=[_signal_row(record, keys) for record in batch.signals],
    )


def _checkpoint_row(
    batch: LivePersistenceImportBatch,
    keys: LivePersistenceImportKeys,
) -> dict[str, Any] | None:
    if batch.state is None:
        return None
    return {
        "live_instance_id": keys.live_instance_id,
        "run_id": keys.run_id,
        "bar_count": _required(batch.state, "bar_count", "state"),
        "payload_json": batch.state,
    }


def _tick_and_position_rows(
    batch: LivePersistenceImportBatch,
    keys: LivePersistenceImportKeys,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ticks: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    for record in batch.equity:
        bar = _required(record, "bar", "equity")
        tick = {
            "live_instance_id": keys.live_instance_id,
            "run_id": keys.run_id,
            "bar": bar,
            "ts": _required(record, "ts", "equity"),
            "account_equity": _required(record, "equity", "equity"),
            "peak_equity": _required(record, "peak", "equity"),
            "dd_pct": _required(record, "dd_pct", "equity"),
            "n_positions": _required(record, "n_positions", "equity"),
            "summary_json": _unknown_fields(
                record,
                {
                    "ts",
                    "bar",
                    "equity",
                    "peak",
                    "dd_pct",
                    "n_positions",
                    "positions",
                },
            ),
        }
        ticks.append(tick)
        raw_positions = record.get("positions", {})
        if not isinstance(raw_positions, dict):
            raise LivePersistenceImportError(
                f"equity bar {bar}: positions must be object"
            )
        for symbol, raw_position in sorted(raw_positions.items()):
            if not isinstance(raw_position, dict):
                raise LivePersistenceImportError(
                    f"equity bar {bar} position {symbol}: expected object"
                )
            positions.append(
                _position_row(
                    bar, str(symbol), cast(dict[str, Any], raw_position), keys
                )
            )
    return ticks, positions


def _position_row(
    bar: Any,
    symbol: str,
    record: dict[str, Any],
    keys: LivePersistenceImportKeys,
) -> dict[str, Any]:
    side = _required(record, "side", f"position {symbol}")
    if side not in {"long", "short"}:
        raise LivePersistenceImportError(f"position {symbol}: invalid side {side!r}")
    return {
        "tick_bar": bar,
        "live_instance_id": keys.live_instance_id,
        "symbol": symbol,
        "side": side,
        "qty": _required(record, "qty", f"position {symbol}"),
        "entry_price": record.get("entry", record.get("entry_price")),
        "best_price": record.get("best", record.get("best_price")),
        "raw_json": record,
    }


def _target_row(
    record: dict[str, Any], keys: LivePersistenceImportKeys
) -> dict[str, Any]:
    return {
        "live_instance_id": keys.live_instance_id,
        "run_id": keys.run_id,
        "bar": _required(record, "bar", "target"),
        "ts": _required(record, "ts", "target"),
        "profile": _required(record, "profile", "target"),
        "target_ts": _required(record, "target_ts", "target"),
        "staleness_seconds": _required(record, "staleness_seconds", "target"),
        "target_gross": _required(record, "target_gross", "target"),
        "normalized_gross": _required(record, "normalized_gross", "target"),
        "ignored_gross": _required(record, "ignored_gross", "target"),
        "ignored_gross_ratio": _required(record, "ignored_gross_ratio", "target"),
        "execution_coverage": _required(record, "execution_coverage", "target"),
        "weights_json": _required(record, "weights", "target"),
        "ignored_weights_json": record.get("ignored_weights", {}),
        "orders_json": record.get("orders", []),
    }


def _trade_row(
    record: dict[str, Any], keys: LivePersistenceImportKeys
) -> dict[str, Any]:
    return {
        "live_instance_id": keys.live_instance_id,
        "run_id": keys.run_id,
        "bar": _required(record, "bar", "trade"),
        "ts": _required(record, "ts", "trade"),
        "kind": _required(record, "kind", "trade"),
        "symbol": _required(record, "symbol", "trade"),
        "side": record.get("side") or None,
        "qty": _required(record, "qty", "trade"),
        "price": _required(record, "price", "trade"),
        "reason": _required(record, "reason", "trade"),
        "pnl": record.get("pnl"),
        "pnl_pct": record.get("pnl_pct"),
        "held_bars": record.get("held_bars"),
        "exchange_order_id": record.get("exchange_order_id"),
        "raw_json": record,
    }


def _signal_row(
    record: dict[str, Any], keys: LivePersistenceImportKeys
) -> dict[str, Any]:
    return {
        "live_instance_id": keys.live_instance_id,
        "run_id": keys.run_id,
        "bar": _required(record, "bar", "signals"),
        "ts": _required(record, "ts", "signals"),
        "signals_json": _required(record, "signals", "signals"),
    }


def _required(record: dict[str, Any], key: str, context: str) -> Any:
    if key not in record:
        raise LivePersistenceImportError(f"{context}: missing required field {key!r}")
    return record[key]


def _unknown_fields(record: dict[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in known}
