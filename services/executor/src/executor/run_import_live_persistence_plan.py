"""Import approved historical live-persistence candidates from a plan.

This is an offline migration helper. It is not imported by live runtime and does
not enable DB source-of-truth or dual-write behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from .live_persistence_import import (
    LivePersistenceImportError,
    LivePersistenceImportKeys,
    build_live_persistence_import_rows,
    load_existing_live_persistence,
)
from .live_persistence_import_plan import (
    ACTION_IMPORT_CANDIDATE,
    LivePersistenceImportPlanItem,
    build_live_persistence_import_plan_from_roots,
)
from .run_import_live_persistence import _summary, _write


@dataclass(frozen=True)
class PlannedImportCandidate:
    """One approved plan item with normalized import rows."""

    plan_item: LivePersistenceImportPlanItem
    row_summary: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_live_persistence_import_plan_from_roots(args.root)
    candidates = [
        _candidate(item, args)
        for item in plan.items
        if item.action == ACTION_IMPORT_CANDIDATE
    ]
    summary: dict[str, Any] = {
        "schema_version": "cta_forge.live_persistence_planned_import.v1",
        "write_requested": bool(args.write),
        "wrote": False,
        "plan": plan.to_dict()["summary"],
        "candidates": [
            {
                "journal_dir": str(candidate.plan_item.journal_dir),
                "reason": candidate.plan_item.reason,
                "summary": candidate.row_summary,
            }
            for candidate in candidates
        ],
    }

    if args.write:
        _validate_write_gate(
            args, plan_review_count=plan.review_count, candidates=candidates
        )
        candidate = candidates[0]
        rows = _rows(candidate.plan_item.journal_dir, args)
        parity = _write(
            _single_import_args(args, candidate.plan_item.journal_dir), rows
        )
        summary["wrote"] = True
        if parity is not None:
            summary["parity"] = parity

    json.dump(summary, sys.stdout, indent=2, sort_keys=True, default=_json_default)
    sys.stdout.write("\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or write import candidates from a historical live-persistence plan."
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan. Can be supplied multiple times. Defaults to cwd.",
    )
    parser.add_argument("--live-instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Allow --write even when non-candidate artifacts still require review.",
    )
    parser.add_argument("--strategy-slug", default="cta-forge")
    parser.add_argument("--strategy-name", default="CTA Forge")
    parser.add_argument("--profile-id")
    parser.add_argument("--profile-slug")
    parser.add_argument("--profile-version", default="")
    parser.add_argument("--account-id")
    parser.add_argument("--exchange", default="hyperliquid")
    parser.add_argument("--network", choices=["testnet", "mainnet"])
    parser.add_argument("--account-label")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "testnet_live", "mainnet_pilot", "mainnet_live"],
    )
    parser.add_argument("--public-instance-slug")
    parser.add_argument("--public-display-name")
    parser.add_argument("--public-enabled", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.root is None:
        parsed.root = [Path.cwd()]
    return parsed


def _candidate(
    item: LivePersistenceImportPlanItem, args: argparse.Namespace
) -> PlannedImportCandidate:
    rows = _rows(item.journal_dir, args)
    return PlannedImportCandidate(
        plan_item=item,
        row_summary=_summary(rows, journal_dir=item.journal_dir, state_file=None),
    )


def _rows(journal_dir: Path, args: argparse.Namespace):
    batch = load_existing_live_persistence(journal_dir)
    return build_live_persistence_import_rows(
        batch,
        LivePersistenceImportKeys(
            live_instance_id=args.live_instance_id,
            run_id=args.run_id,
        ),
    )


def _validate_write_gate(
    args: argparse.Namespace,
    *,
    plan_review_count: int,
    candidates: list[PlannedImportCandidate],
) -> None:
    if not args.database_url:
        raise LivePersistenceImportError("--database-url is required with --write")
    if plan_review_count and not args.allow_review:
        raise LivePersistenceImportError(
            "plan still contains review items; rerun with --allow-review only after explicit approval"
        )
    if len(candidates) != 1:
        raise LivePersistenceImportError(
            f"--write currently requires exactly one import candidate, found {len(candidates)}"
        )


def _single_import_args(
    args: argparse.Namespace, journal_dir: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        journal_dir=journal_dir,
        state_file=None,
        live_instance_id=args.live_instance_id,
        run_id=args.run_id,
        write=args.write,
        parity_check=True,
        database_url=args.database_url,
        strategy_slug=args.strategy_slug,
        strategy_name=args.strategy_name,
        profile_id=args.profile_id,
        profile_slug=args.profile_slug,
        profile_version=args.profile_version,
        account_id=args.account_id,
        exchange=args.exchange,
        network=args.network,
        account_label=args.account_label,
        mode=args.mode,
        public_instance_slug=args.public_instance_slug,
        public_display_name=args.public_display_name,
        public_enabled=args.public_enabled,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
