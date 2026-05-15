"""Parity checks for live persistence imports.

The helpers here compare schema-shaped rows before and after a database import.
They do not open database connections and do not participate in live trading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .live_persistence_import import LivePersistenceImportRows

_SECTIONS = ("checkpoint", "ticks", "positions", "targets", "trades", "signals")
_TIMESTAMP_KEYS = {"ts", "target_ts"}


@dataclass(frozen=True)
class LivePersistenceParityReport:
    """Comparison result for one live persistence import."""

    counts: dict[str, dict[str, int]]
    mismatches: list[str]

    @property
    def ok(self) -> bool:
        """Return true when no parity mismatches were found."""

        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "ok": self.ok,
            "counts": self.counts,
            "mismatch_count": len(self.mismatches),
            "mismatches": self.mismatches,
        }


def compare_live_persistence_import_rows(
    expected: LivePersistenceImportRows,
    actual: LivePersistenceImportRows,
    *,
    max_examples: int = 20,
    ignored_keys: set[str] | None = None,
) -> LivePersistenceParityReport:
    """Compare normalized import rows with rows read back from the database.

    Decimal values are compared as strings so precision is not truncated during
    parity checks. Timestamp strings and ``datetime`` objects are normalized to
    UTC ``Z`` form before comparison. ``ignored_keys`` removes matching dict
    keys recursively, which is useful when comparing a file snapshot against a
    DB history that intentionally used different run identifiers.
    """

    counts: dict[str, dict[str, int]] = {}
    mismatches: list[str] = []
    ignored_keys = ignored_keys or set()
    for section in _SECTIONS:
        expected_records = _section_records(expected, section)
        actual_records = _section_records(actual, section)
        counts[section] = {
            "expected": len(expected_records),
            "actual": len(actual_records),
        }
        if len(expected_records) != len(actual_records):
            _append_mismatch(
                mismatches,
                max_examples,
                f"{section}: expected {len(expected_records)} rows, got {len(actual_records)}",
            )
            continue
        for index, (expected_record, actual_record) in enumerate(
            zip(expected_records, actual_records, strict=True)
        ):
            expected_canonical = _canonical(expected_record, ignored_keys=ignored_keys)
            actual_canonical = _canonical(actual_record, ignored_keys=ignored_keys)
            if expected_canonical != actual_canonical:
                _append_mismatch(
                    mismatches,
                    max_examples,
                    f"{section}[{index}] expected {_short(expected_canonical)} got {_short(actual_canonical)}",
                )
    return LivePersistenceParityReport(counts=counts, mismatches=mismatches)


def _section_records(rows: LivePersistenceImportRows, section: str) -> list[Any]:
    if section == "checkpoint":
        return [] if rows.checkpoint is None else [rows.checkpoint]
    return list(getattr(rows, section))


def _append_mismatch(mismatches: list[str], max_examples: int, message: str) -> None:
    if len(mismatches) < max_examples:
        mismatches.append(message)
    elif len(mismatches) == max_examples:
        mismatches.append("additional mismatches omitted")


def _canonical(
    value: Any,
    *,
    key: str | None = None,
    ignored_keys: set[str] | None = None,
) -> Any:
    ignored_keys = ignored_keys or set()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, dict):
        return {
            str(k): _canonical(v, key=str(k), ignored_keys=ignored_keys)
            for k, v in sorted(value.items())
            if str(k) not in ignored_keys
        }
    if isinstance(value, list):
        return [_canonical(item, ignored_keys=ignored_keys) for item in value]
    if key in _TIMESTAMP_KEYS and isinstance(value, str):
        return _canonical_timestamp(value)
    return value


def _canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return _canonical_datetime(parsed)


def _canonical_datetime(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")


def _short(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return raw if len(raw) <= 500 else raw[:497] + "..."
