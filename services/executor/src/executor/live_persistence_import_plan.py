"""Canonical import-plan helpers for historical live persistence data.

The plan is intentionally review-first. It classifies local JSONL/state artifacts
before PostgreSQL import, but it does not merge records, open database
connections, or participate in live trading.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .live_persistence_inventory import (
    LivePersistenceInventoryItem,
    LivePersistenceInventoryReport,
    scan_live_persistence_journal_dirs,
)

ACTION_IMPORT_CANDIDATE = "import_candidate"
ACTION_EXCLUDE_DUPLICATE = "exclude_exact_duplicate"
ACTION_EXCLUDE_COVERED = "exclude_covered_snapshot"
ACTION_REVIEW_BLOCKED = "review_blocked"
ACTION_REVIEW_OVERLAP = "review_overlap_identity"


@dataclass(frozen=True)
class LivePersistenceImportPlanItem:
    """One journal directory classification for future DB import."""

    journal_dir: Path
    action: str
    reason: str
    representative_journal_dir: Path | None
    counts: dict[str, int]
    duplicate_bar_details: dict[str, list[dict[str, Any]]]
    bar_ranges: dict[str, dict[str, Any] | None]
    first_tick: dict[str, Any] | None
    latest_tick: dict[str, Any] | None
    latest_target: dict[str, Any] | None
    state_file_candidates: list[Path]
    combined_content_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "journal_dir": str(self.journal_dir),
            "action": self.action,
            "reason": self.reason,
            "representative_journal_dir": str(self.representative_journal_dir)
            if self.representative_journal_dir is not None
            else None,
            "counts": self.counts,
            "duplicate_bar_details": self.duplicate_bar_details,
            "bar_ranges": self.bar_ranges,
            "first_tick": self.first_tick,
            "latest_tick": self.latest_tick,
            "latest_target": self.latest_target,
            "state_file_candidates": [str(path) for path in self.state_file_candidates],
            "combined_content_hash": self.combined_content_hash,
        }


@dataclass(frozen=True)
class LivePersistenceImportPlan:
    """Review-first canonical import plan for historical local artifacts."""

    inventory: LivePersistenceInventoryReport
    items: list[LivePersistenceImportPlanItem]

    @property
    def action_counts(self) -> dict[str, int]:
        return dict(Counter(item.action for item in self.items))

    @property
    def review_count(self) -> int:
        return sum(
            1
            for item in self.items
            if item.action in {ACTION_REVIEW_BLOCKED, ACTION_REVIEW_OVERLAP}
        )

    @property
    def import_count(self) -> int:
        return sum(1 for item in self.items if item.action == ACTION_IMPORT_CANDIDATE)

    def to_dict(self) -> dict[str, Any]:
        action_counts = self.action_counts
        return {
            "schema_version": "cta_forge.live_persistence_import_plan.v1",
            "roots": [str(root) for root in self.inventory.roots],
            "summary": {
                "journal_dirs": len(self.items),
                "import_candidates": self.import_count,
                "requires_review": self.review_count,
                "action_counts": action_counts,
            },
            "items": [item.to_dict() for item in self.items],
        }


def build_live_persistence_import_plan(
    inventory: LivePersistenceInventoryReport,
) -> LivePersistenceImportPlan:
    """Build a conservative canonical plan from an inventory report.

    The planner only auto-excludes exact duplicate copies and snapshots that are
    fully covered by a newer/larger representative. Partial overlaps stay in
    review state because they may represent separate live instances or require
    an explicit reindexing/import policy.
    """

    item_by_path = {item.journal_dir: item for item in inventory.items}
    decisions: dict[Path, tuple[str, str, Path | None]] = {}

    for item in inventory.items:
        if not item.ready_for_import:
            decisions[item.journal_dir] = (
                ACTION_REVIEW_BLOCKED,
                _blocked_reason(item),
                None,
            )

    duplicate_representatives = _duplicate_representatives(inventory.items)
    for duplicate, representative in duplicate_representatives.items():
        if duplicate not in decisions:
            decisions[duplicate] = (
                ACTION_EXCLUDE_DUPLICATE,
                "same combined content hash as representative",
                representative,
            )

    active_items = [
        item
        for item in inventory.items
        if item.ready_for_import and item.journal_dir not in decisions
    ]
    for component in _overlap_components(active_items):
        if len(component) == 1:
            item = component[0]
            decisions[item.journal_dir] = (
                ACTION_IMPORT_CANDIDATE,
                "ready and no overlap with other non-duplicate ready artifacts",
                None,
            )
            continue

        representative = _select_representative(component)
        ambiguous = False
        for item in component:
            if item.journal_dir == representative.journal_dir:
                continue
            if _covered_by(item, representative):
                decisions[item.journal_dir] = (
                    ACTION_EXCLUDE_COVERED,
                    "bar and tick-time range are covered by representative snapshot",
                    representative.journal_dir,
                )
            else:
                ambiguous = True
                decisions[item.journal_dir] = (
                    ACTION_REVIEW_OVERLAP,
                    "overlaps by bar but is not fully covered by representative snapshot",
                    representative.journal_dir,
                )

        if ambiguous:
            decisions[representative.journal_dir] = (
                ACTION_REVIEW_OVERLAP,
                "representative proposal has unresolved partial-overlap peers",
                None,
            )
        else:
            decisions[representative.journal_dir] = (
                ACTION_IMPORT_CANDIDATE,
                "largest/latest representative for covered snapshot group",
                None,
            )

    plan_items = [
        _plan_item(item_by_path[path], action, reason, representative)
        for path, (action, reason, representative) in sorted(
            decisions.items(), key=lambda row: str(row[0])
        )
    ]
    return LivePersistenceImportPlan(inventory=inventory, items=plan_items)


def build_live_persistence_import_plan_from_roots(
    roots: list[str | Path],
) -> LivePersistenceImportPlan:
    return build_live_persistence_import_plan(scan_live_persistence_journal_dirs(roots))


def _plan_item(
    item: LivePersistenceInventoryItem,
    action: str,
    reason: str,
    representative: Path | None,
) -> LivePersistenceImportPlanItem:
    return LivePersistenceImportPlanItem(
        journal_dir=item.journal_dir,
        action=action,
        reason=reason,
        representative_journal_dir=representative,
        counts=item.counts,
        duplicate_bar_details=item.duplicate_bar_details,
        bar_ranges=item.bar_ranges,
        first_tick=item.first_tick,
        latest_tick=item.latest_tick,
        latest_target=item.latest_target,
        state_file_candidates=item.state_file_candidates,
        combined_content_hash=item.combined_content_hash,
    )


def _blocked_reason(item: LivePersistenceInventoryItem) -> str:
    if item.error:
        return item.error
    if item.duplicate_bars:
        return f"duplicate bars within artifact: {item.duplicate_bars}"
    if not item.counts.get("equity"):
        return "missing equity journal records"
    return "not ready for import"


def _duplicate_representatives(
    items: list[LivePersistenceInventoryItem],
) -> dict[Path, Path]:
    grouped: dict[str, list[LivePersistenceInventoryItem]] = {}
    for item in items:
        if item.ready_for_import and item.combined_content_hash is not None:
            grouped.setdefault(item.combined_content_hash, []).append(item)

    representatives: dict[Path, Path] = {}
    for group in grouped.values():
        if len(group) < 2:
            continue
        representative = _select_representative(group)
        for item in group:
            if item.journal_dir != representative.journal_dir:
                representatives[item.journal_dir] = representative.journal_dir
    return representatives


def _overlap_components(
    items: list[LivePersistenceInventoryItem],
) -> list[list[LivePersistenceInventoryItem]]:
    pending = set(range(len(items)))
    components: list[list[LivePersistenceInventoryItem]] = []
    while pending:
        start = pending.pop()
        component_indexes = {start}
        queue = [start]
        while queue:
            left_index = queue.pop()
            for right_index in list(pending):
                if _overlaps(items[left_index], items[right_index]):
                    pending.remove(right_index)
                    component_indexes.add(right_index)
                    queue.append(right_index)
        components.append([items[index] for index in sorted(component_indexes)])
    return components


def _overlaps(
    left: LivePersistenceInventoryItem,
    right: LivePersistenceInventoryItem,
) -> bool:
    return bool(
        (left.equity_bar_keys & right.equity_bar_keys)
        or (left.signal_bar_keys & right.signal_bar_keys)
    )


def _covered_by(
    item: LivePersistenceInventoryItem,
    representative: LivePersistenceInventoryItem,
) -> bool:
    if not item.equity_bar_keys <= representative.equity_bar_keys:
        return False
    if not item.signal_bar_keys <= representative.signal_bar_keys:
        return False
    item_range = _tick_time_range(item)
    representative_range = _tick_time_range(representative)
    if item_range is None or representative_range is None:
        return False
    item_first, item_latest = item_range
    rep_first, rep_latest = representative_range
    return rep_first <= item_first and item_latest <= rep_latest


def _select_representative(
    items: list[LivePersistenceInventoryItem],
) -> LivePersistenceInventoryItem:
    return max(items, key=_representative_score)


def _representative_score(
    item: LivePersistenceInventoryItem,
) -> tuple[datetime, int, str]:
    latest = _parse_tick_ts(item.latest_tick) or datetime.min.replace(tzinfo=UTC)
    return (latest, item.counts.get("equity", 0), str(item.journal_dir))


def _tick_time_range(
    item: LivePersistenceInventoryItem,
) -> tuple[datetime, datetime] | None:
    first = _parse_tick_ts(item.first_tick)
    latest = _parse_tick_ts(item.latest_tick)
    if first is None or latest is None:
        return None
    return first, latest


def _parse_tick_ts(tick: dict[str, Any] | None) -> datetime | None:
    if not tick:
        return None
    value = tick.get("ts")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
