"""Compare file-backed live persistence with PostgreSQL rows.

This is an observation/rehearsal tool. It does not write to the database and is
not wired into live runtime.
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
from .live_persistence_postgres import DbConnection, load_live_import_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected = _load_file_rows(args)
    actual = _load_db_rows(args)
    ignored_keys = {"run_id"} if args.ignore_run_id else set()
    report = compare_live_persistence_import_rows(
        expected,
        actual,
        max_examples=args.max_examples,
        ignored_keys=ignored_keys,
    )
    summary = {
        "ok": report.ok,
        "journal_dir": str(args.journal_dir),
        "state_file": str(args.state_file) if args.state_file is not None else None,
        "live_instance_id": args.live_instance_id,
        "run_id": args.run_id,
        "ignored_keys": sorted(ignored_keys),
        "file": _summary(expected),
        "database": _summary(actual),
        "parity": report.to_dict(),
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.ok else 2


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare file-backed live JSONL/state rows with PostgreSQL rows."
    )
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--live-instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--ignore-run-id",
        action="store_true",
        help="Ignore run_id keys recursively when comparing mixed historical/runtime rows.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args(argv)


def _load_file_rows(args: argparse.Namespace) -> LivePersistenceImportRows:
    batch = load_existing_live_persistence(
        args.journal_dir,
        state_file=args.state_file,
    )
    return build_live_persistence_import_rows(
        batch,
        LivePersistenceImportKeys(
            live_instance_id=args.live_instance_id,
            run_id=args.run_id,
        ),
    )


def _load_db_rows(args: argparse.Namespace) -> LivePersistenceImportRows:
    try:
        psycopg = import_module("psycopg")
    except ImportError as exc:
        raise LivePersistenceImportError(
            "psycopg is required for DB parity checks; install the PostgreSQL driver first"
        ) from exc

    with psycopg.connect(args.database_url) as raw_conn:
        conn = cast(DbConnection, raw_conn)
        return load_live_import_rows(conn, live_instance_id=args.live_instance_id)


def _summary(rows: LivePersistenceImportRows) -> dict[str, Any]:
    latest_tick = rows.ticks[-1] if rows.ticks else None
    latest_target = rows.targets[-1] if rows.targets else None
    latest_trade = rows.trades[-1] if rows.trades else None
    latest_signal = rows.signals[-1] if rows.signals else None
    return {
        "counts": {
            "checkpoint": 1 if rows.checkpoint is not None else 0,
            "ticks": len(rows.ticks),
            "positions": len(rows.positions),
            "targets": len(rows.targets),
            "trades": len(rows.trades),
            "signals": len(rows.signals),
        },
        "latest_tick": _latest_summary(latest_tick, ["bar", "ts"]),
        "latest_target": _latest_summary(
            latest_target, ["bar", "target_ts", "profile"]
        ),
        "latest_trade": _latest_summary(latest_trade, ["bar", "ts", "kind", "symbol"]),
        "latest_signal": _latest_summary(latest_signal, ["bar", "ts"]),
    }


def _latest_summary(
    record: dict[str, Any] | None, keys: list[str]
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {key: record.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
