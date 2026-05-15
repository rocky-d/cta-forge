"""Inventory local live JSONL/state artifacts before PostgreSQL import."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from .live_persistence_inventory import scan_live_persistence_journal_dirs


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = scan_live_persistence_journal_dirs(args.root)
    json.dump(
        report.to_dict(), sys.stdout, indent=2, sort_keys=True, default=_json_default
    )
    sys.stdout.write("\n")
    if args.fail_on_blocked and report.blocked_count:
        return 1
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory local live persistence directories for DB import planning."
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan. Can be supplied multiple times. Defaults to cwd.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when any discovered journal directory is not import-ready.",
    )
    parsed = parser.parse_args(argv)
    if parsed.root is None:
        parsed.root = [Path.cwd()]
    return parsed


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
