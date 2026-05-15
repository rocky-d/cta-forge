"""Inventory helpers for historical live persistence imports.

The inventory is a planning aid for migrating local JSONL/state artifacts into
PostgreSQL. It does not open database connections and does not participate in
live trading.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from .live_persistence_import import (
    JOURNAL_FILES,
    LivePersistenceImportBatch,
    LivePersistenceImportError,
    load_existing_live_persistence,
)

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class LivePersistenceInventoryItem:
    """One discovered local journal directory and its import readiness."""

    journal_dir: Path
    counts: dict[str, int]
    duplicate_bars: dict[str, list[Any]]
    bar_ranges: dict[str, dict[str, Any] | None]
    content_hashes: dict[str, str]
    combined_content_hash: str | None
    state_file_candidates: list[Path]
    first_tick: dict[str, Any] | None
    latest_tick: dict[str, Any] | None
    latest_target: dict[str, Any] | None
    ready_for_import: bool
    equity_bar_keys: frozenset[str]
    signal_bar_keys: frozenset[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest item."""

        return {
            "journal_dir": str(self.journal_dir),
            "counts": self.counts,
            "duplicate_bars": self.duplicate_bars,
            "bar_ranges": self.bar_ranges,
            "content_hashes": self.content_hashes,
            "combined_content_hash": self.combined_content_hash,
            "state_file_candidates": [str(path) for path in self.state_file_candidates],
            "first_tick": self.first_tick,
            "latest_tick": self.latest_tick,
            "latest_target": self.latest_target,
            "ready_for_import": self.ready_for_import,
            "error": self.error,
        }


@dataclass(frozen=True)
class LivePersistenceInventoryReport:
    """Import-planning manifest for local historical live persistence data."""

    roots: list[Path]
    items: list[LivePersistenceInventoryItem]

    @property
    def ready_count(self) -> int:
        return sum(1 for item in self.items if item.ready_for_import)

    @property
    def blocked_count(self) -> int:
        return sum(1 for item in self.items if not item.ready_for_import)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""

        duplicate_content_groups = _duplicate_content_groups(self.items)
        bar_overlaps = _bar_overlaps(self.items)
        return {
            "schema_version": "cta_forge.live_persistence_inventory.v1",
            "roots": [str(root) for root in self.roots],
            "summary": {
                "journal_dirs": len(self.items),
                "ready_for_import": self.ready_count,
                "blocked": self.blocked_count,
                "duplicate_content_groups": len(duplicate_content_groups),
                "bar_overlaps": len(bar_overlaps),
            },
            "duplicate_content_groups": duplicate_content_groups,
            "bar_overlaps": bar_overlaps,
            "items": [item.to_dict() for item in self.items],
        }


def scan_live_persistence_journal_dirs(
    roots: Iterable[str | Path],
) -> LivePersistenceInventoryReport:
    """Discover and classify local journal directories under roots."""

    root_paths = [Path(root) for root in roots]
    journal_dirs = _discover_journal_dirs(root_paths)
    return LivePersistenceInventoryReport(
        roots=root_paths,
        items=[_inventory_item(path, root_paths) for path in journal_dirs],
    )


def _discover_journal_dirs(roots: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for current, dirnames, _filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in _EXCLUDED_DIR_NAMES]
            path = Path(current)
            if _is_journal_dir(path):
                discovered.add(path)
    return sorted(discovered)


def _is_journal_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / filename).exists() for filename in JOURNAL_FILES.values()
    )


def _inventory_item(
    journal_dir: Path,
    roots: list[Path],
) -> LivePersistenceInventoryItem:
    content_hashes = _content_hashes(journal_dir)
    try:
        batch = load_existing_live_persistence(journal_dir)
    except LivePersistenceImportError as exc:
        return LivePersistenceInventoryItem(
            journal_dir=journal_dir,
            counts={key: 0 for key in JOURNAL_FILES},
            duplicate_bars={},
            bar_ranges={},
            content_hashes=content_hashes,
            combined_content_hash=_combined_content_hash(content_hashes),
            state_file_candidates=_state_file_candidates(journal_dir, roots),
            first_tick=None,
            latest_tick=None,
            latest_target=None,
            ready_for_import=False,
            equity_bar_keys=frozenset(),
            signal_bar_keys=frozenset(),
            error=str(exc),
        )

    duplicate_bars = _duplicate_bar_report(batch)
    equity_bar_keys = _bar_keys(batch.equity)
    signal_bar_keys = _bar_keys(batch.signals)
    ready = bool(batch.equity) and not duplicate_bars
    return LivePersistenceInventoryItem(
        journal_dir=journal_dir,
        counts={
            "equity": len(batch.equity),
            "trades": len(batch.trades),
            "signals": len(batch.signals),
            "targets": len(batch.targets),
        },
        duplicate_bars=duplicate_bars,
        bar_ranges={
            "equity": _bar_range(batch.equity),
            "signals": _bar_range(batch.signals),
            "targets": _bar_range(batch.targets),
        },
        content_hashes=content_hashes,
        combined_content_hash=_combined_content_hash(content_hashes),
        state_file_candidates=_state_file_candidates(journal_dir, roots),
        first_tick=_tick_summary(batch.equity[0]) if batch.equity else None,
        latest_tick=_tick_summary(batch.equity[-1]) if batch.equity else None,
        latest_target=_target_summary(batch.targets[-1]) if batch.targets else None,
        ready_for_import=ready,
        equity_bar_keys=equity_bar_keys,
        signal_bar_keys=signal_bar_keys,
        error=None,
    )


def _duplicate_content_groups(
    items: list[LivePersistenceInventoryItem],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[LivePersistenceInventoryItem]] = {}
    for item in items:
        if item.combined_content_hash is not None:
            grouped.setdefault(item.combined_content_hash, []).append(item)
    return [
        {
            "combined_content_hash": content_hash,
            "journal_dirs": [str(item.journal_dir) for item in group],
        }
        for content_hash, group in sorted(grouped.items())
        if len(group) > 1
    ]


def _bar_overlaps(items: list[LivePersistenceInventoryItem]) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            shared_equity = sorted(
                left.equity_bar_keys & right.equity_bar_keys, key=_bar_sort_key
            )
            shared_signals = sorted(
                left.signal_bar_keys & right.signal_bar_keys, key=_bar_sort_key
            )
            if not shared_equity and not shared_signals:
                continue
            overlaps.append(
                {
                    "left_journal_dir": str(left.journal_dir),
                    "right_journal_dir": str(right.journal_dir),
                    "shared_equity_bars": _shared_bar_summary(shared_equity),
                    "shared_signal_bars": _shared_bar_summary(shared_signals),
                    "identical_content": left.combined_content_hash is not None
                    and left.combined_content_hash == right.combined_content_hash,
                }
            )
    return overlaps


def _shared_bar_summary(bar_keys: list[str]) -> dict[str, Any]:
    return {
        "count": len(bar_keys),
        "sample": bar_keys[:10],
    }


def _bar_sort_key(bar_key: str) -> tuple[int, Decimal | str]:
    try:
        return (0, Decimal(bar_key))
    except InvalidOperation:
        return (1, bar_key)


def _duplicate_bar_report(batch: LivePersistenceImportBatch) -> dict[str, list[Any]]:
    duplicates: dict[str, list[Any]] = {}
    for key, records in {"equity": batch.equity, "signals": batch.signals}.items():
        duplicate_bars = _duplicate_bars([record.get("bar") for record in records])
        if duplicate_bars:
            duplicates[key] = duplicate_bars
    return duplicates


def _duplicate_bars(bars: list[Any]) -> list[Any]:
    counts = Counter(bars)
    return [bar for bar, count in counts.items() if count > 1]


def _bar_keys(records: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(_bar_key(record.get("bar")) for record in records)


def _bar_key(bar: Any) -> str:
    return str(_json_safe_number(bar))


def _bar_range(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    bars = [record.get("bar") for record in records]
    return {
        "first": _json_safe_number(bars[0]),
        "latest": _json_safe_number(bars[-1]),
        "records": len(records),
        "unique_bars": len(frozenset(_bar_key(bar) for bar in bars)),
    }


def _content_hashes(journal_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, filename in JOURNAL_FILES.items():
        path = journal_dir / filename
        if path.exists():
            hashes[key] = _file_sha256(path)
    return hashes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_content_hash(content_hashes: dict[str, str]) -> str | None:
    if not content_hashes:
        return None
    digest = hashlib.sha256()
    for key, value in sorted(content_hashes.items()):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _state_file_candidates(journal_dir: Path, roots: list[Path]) -> list[Path]:
    candidates = [
        journal_dir / "engine-state.json",
        journal_dir.parent / "engine-state.json",
        *[root / "engine-state.json" for root in roots],
    ]
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            existing.append(path)
    return existing


def _tick_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "bar": record.get("bar"),
        "ts": record.get("ts"),
        "equity": _json_safe_number(record.get("equity")),
        "n_positions": record.get("n_positions"),
    }


def _target_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "bar": record.get("bar"),
        "ts": record.get("ts"),
        "profile": record.get("profile"),
        "target_ts": record.get("target_ts"),
        "target_gross": _json_safe_number(record.get("target_gross")),
        "normalized_gross": _json_safe_number(record.get("normalized_gross")),
    }


def _json_safe_number(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value
