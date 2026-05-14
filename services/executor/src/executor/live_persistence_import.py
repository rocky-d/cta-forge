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
from typing import Any

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
