"""Import existing live JSONL/state persistence into PostgreSQL.

Default mode is a dry run that prints row counts and inferred metadata. Actual
DB writes require both ``--write`` and ``--database-url``.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence, cast

from .live_persistence_import import (
    LivePersistenceImportError,
    LivePersistenceImportKeys,
    LivePersistenceImportRows,
    build_live_persistence_import_rows,
    load_existing_live_persistence,
)
from .live_persistence_parity import compare_live_persistence_import_rows
from .live_persistence_postgres import (
    DbConnection,
    LivePersistenceReferenceData,
    load_live_import_rows,
    write_live_import_rows,
    write_live_reference_rows,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    batch = load_existing_live_persistence(
        args.journal_dir,
        state_file=args.state_file,
    )
    rows = build_live_persistence_import_rows(
        batch,
        LivePersistenceImportKeys(
            live_instance_id=args.live_instance_id,
            run_id=args.run_id,
        ),
    )
    summary = _summary(rows, journal_dir=args.journal_dir, state_file=args.state_file)
    if args.parity_check and not args.write:
        raise LivePersistenceImportError("--parity-check requires --write")
    if args.write:
        parity = _write(args, rows)
        summary["wrote"] = True
        if parity is not None:
            summary["parity"] = parity
    else:
        summary["wrote"] = False
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or write a live JSONL/state PostgreSQL import."
    )
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--live-instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="After --write, read rows back from the same DB transaction and compare.",
    )
    parser.add_argument("--database-url")
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
    return parser.parse_args(argv)


def _summary(
    rows: LivePersistenceImportRows,
    *,
    journal_dir: Path,
    state_file: Path | None,
) -> dict[str, Any]:
    latest_tick = rows.ticks[-1] if rows.ticks else None
    latest_target = rows.targets[-1] if rows.targets else None
    return {
        "journal_dir": str(journal_dir),
        "state_file": str(state_file) if state_file is not None else None,
        "counts": {
            "checkpoint": 1 if rows.checkpoint is not None else 0,
            "ticks": len(rows.ticks),
            "positions": len(rows.positions),
            "targets": len(rows.targets),
            "trades": len(rows.trades),
            "signals": len(rows.signals),
        },
        "latest_tick": {
            "bar": latest_tick["bar"],
            "ts": latest_tick["ts"],
        }
        if latest_tick is not None
        else None,
        "latest_target": {
            "bar": latest_target["bar"],
            "target_ts": latest_target["target_ts"],
            "profile": latest_target["profile"],
        }
        if latest_target is not None
        else None,
    }


def _write(
    args: argparse.Namespace, rows: LivePersistenceImportRows
) -> dict[str, Any] | None:
    if not args.database_url:
        raise LivePersistenceImportError("--database-url is required with --write")
    reference = _reference_from_args(args, rows)
    try:
        psycopg = import_module("psycopg")
    except ImportError as exc:
        raise LivePersistenceImportError(
            "psycopg is required for --write; install the PostgreSQL driver first"
        ) from exc

    with psycopg.connect(args.database_url) as raw_conn:
        conn = cast(DbConnection, raw_conn)
        with raw_conn.transaction():
            write_live_reference_rows(conn, reference)
            write_live_import_rows(conn, rows)
            if not args.parity_check:
                return None
            report = compare_live_persistence_import_rows(
                rows,
                load_live_import_rows(conn, live_instance_id=args.live_instance_id),
            )
            if not report.ok:
                details = "; ".join(report.mismatches[:3])
                raise LivePersistenceImportError(
                    f"post-import parity check failed: {details}"
                )
            return report.to_dict()
    return None


def _reference_from_args(
    args: argparse.Namespace,
    rows: LivePersistenceImportRows,
) -> LivePersistenceReferenceData:
    inferred_profile = _infer_profile_slug(rows)
    profile_slug = args.profile_slug or inferred_profile
    profile_id = args.profile_id or profile_slug
    missing = [
        name
        for name, value in {
            "profile_slug": profile_slug,
            "profile_id": profile_id,
            "account_id": args.account_id,
            "network": args.network,
            "account_label": args.account_label,
            "mode": args.mode,
        }.items()
        if not value
    ]
    if missing:
        raise LivePersistenceImportError(
            "missing required --write args: " + ", ".join(sorted(missing))
        )
    return LivePersistenceReferenceData(
        strategy_slug=str(args.strategy_slug),
        strategy_name=str(args.strategy_name),
        profile_id=_required_text(profile_id, "profile_id"),
        profile_slug=_required_text(profile_slug, "profile_slug"),
        profile_version=str(args.profile_version),
        account_id=_required_text(args.account_id, "account_id"),
        exchange=str(args.exchange),
        network=_required_text(args.network, "network"),
        account_label=_required_text(args.account_label, "account_label"),
        live_instance_id=str(args.live_instance_id),
        run_id=str(args.run_id),
        mode=_required_text(args.mode, "mode"),
        public_instance_slug=args.public_instance_slug,
        public_display_name=args.public_display_name,
        public_enabled=bool(args.public_enabled),
    )


def _required_text(value: Any, name: str) -> str:
    if not value:
        raise LivePersistenceImportError(f"{name} is required")
    return str(value)


def _infer_profile_slug(rows: LivePersistenceImportRows) -> str | None:
    if not rows.targets:
        return None
    profile = rows.targets[-1].get("profile")
    return str(profile) if profile else None


if __name__ == "__main__":
    raise SystemExit(main())
