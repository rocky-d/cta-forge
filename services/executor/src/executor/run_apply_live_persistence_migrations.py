"""Apply live persistence PostgreSQL migrations.

This is intentionally small and deployment-oriented: it applies checked-in SQL
migration files in lexical order and does not print connection strings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence, cast

from .live_persistence_postgres import DbConnection

DEFAULT_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[4] / "infra" / "db" / "migrations"
)


class MigrationError(RuntimeError):
    """Raised when migration input or execution is unsafe."""


def main(
    argv: Sequence[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    stdout: Any = sys.stdout,
    stderr: Any = sys.stderr,
) -> int:
    args = _parse_args(argv)
    effective_env = os.environ if env is None else env
    database_url = args.database_url or effective_env.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is required", file=stderr)
        return 2
    migrations_dir = args.migrations_dir
    try:
        migration_files = _migration_files(migrations_dir)
        _apply(database_url, migration_files, dry_run=args.dry_run)
    except MigrationError as exc:
        print(str(exc), file=stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"migration failed: {type(exc).__name__}", file=stderr)
        return 1

    json.dump(
        {
            "database_url_configured": True,
            "dry_run": bool(args.dry_run),
            "migrations": [path.name for path in migration_files],
            "migrations_dir": str(migrations_dir),
        },
        stdout,
        indent=2,
        sort_keys=True,
    )
    stdout.write("\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply live persistence DB migrations."
    )
    parser.add_argument("--database-url")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=DEFAULT_MIGRATIONS_DIR,
        help="Directory containing checked-in .sql migrations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate migration file discovery without opening a DB connection.",
    )
    return parser.parse_args(argv)


def _migration_files(migrations_dir: Path) -> list[Path]:
    if not migrations_dir.exists():
        raise MigrationError(f"migrations directory not found: {migrations_dir}")
    files = sorted(path for path in migrations_dir.iterdir() if path.suffix == ".sql")
    if not files:
        raise MigrationError(f"no .sql migrations found in {migrations_dir}")
    return files


def _apply(database_url: str, migration_files: list[Path], *, dry_run: bool) -> None:
    if dry_run:
        return
    psycopg = import_module("psycopg")
    with psycopg.connect(database_url) as raw_conn:
        conn = cast(DbConnection, raw_conn)
        with raw_conn.transaction():
            for path in migration_files:
                conn.execute(path.read_text())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
