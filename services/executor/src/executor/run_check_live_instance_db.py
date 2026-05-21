"""Read-only DB readiness check for one live instance.

This is the DB-only companion to exchange preflight. It confirms the instance
reference row exists, reports safe status metadata, and can require an active
instance/account plus an available runtime advisory lock before a dry-run start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, TextIO, cast

from .live_runtime_lock import DbConnection, probe_live_runtime_lock


class ClosableConnection(DbConnection, Protocol):
    def close(self) -> None:
        """Close DB connection."""
        ...


ConnectDatabase = Callable[[str], ClosableConnection]


@dataclass(frozen=True)
class DbReadiness:
    status: str
    live_instance_id: str
    instance_status: str | None = None
    account_status: str | None = None
    mode: str | None = None
    public_instance_slug: str | None = None
    public_enabled: bool | None = None
    public_status: str | None = None
    latest_checkpoint_bar: int | None = None
    latest_tick_bar: int | None = None
    lock_available: bool | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "live_instance_id": self.live_instance_id,
            "instance_status": self.instance_status,
            "account_status": self.account_status,
            "mode": self.mode,
            "public_instance_slug": self.public_instance_slug,
            "public_enabled": self.public_enabled,
            "public_status": self.public_status,
            "latest_checkpoint_bar": self.latest_checkpoint_bar,
            "latest_tick_bar": self.latest_tick_bar,
            "lock_available": self.lock_available,
            "errors": list(self.errors),
        }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    connect_database: ConnectDatabase | None = None,
) -> int:
    args = _parse_args(argv)
    source_env = env if env is not None else os.environ
    database_url = _optional_text(args.database_url or source_env.get("DATABASE_URL"))
    live_instance_id = _optional_text(
        args.live_instance_id or source_env.get("LIVE_INSTANCE_ID")
    )
    if database_url is None or live_instance_id is None:
        missing = []
        if database_url is None:
            missing.append("DATABASE_URL")
        if live_instance_id is None:
            missing.append("LIVE_INSTANCE_ID")
        print(
            f"missing required DB readiness settings: {', '.join(missing)}", file=stderr
        )
        return 2

    connector = connect_database or _connect_psycopg
    conn = connector(database_url)
    try:
        readiness = check_live_instance_db(
            conn,
            live_instance_id=live_instance_id,
            require_active=args.require_active,
            require_lock_available=args.require_lock_available,
        )
    finally:
        conn.close()
    json.dump(readiness.to_dict(), stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    if readiness.status != "ok":
        print("live instance DB readiness invalid", file=stderr)
        return 2
    return 0


def check_live_instance_db(
    conn: DbConnection,
    *,
    live_instance_id: str,
    require_active: bool = False,
    require_lock_available: bool = False,
) -> DbReadiness:
    """Return safe DB readiness metadata for one live instance."""

    instance_id = _required_text("LIVE_INSTANCE_ID", live_instance_id)
    row = _fetch_instance_row(conn, live_instance_id=instance_id)
    if row is None:
        return DbReadiness(
            status="error",
            live_instance_id=instance_id,
            errors=("live instance identity row is missing",),
        )

    instance_status = _as_optional_str(_row_value(row, "instance_status", 0))
    account_status = _as_optional_str(_row_value(row, "account_status", 1))
    mode = _as_optional_str(_row_value(row, "mode", 2))
    public_slug = _as_optional_str(_row_value(row, "public_instance_slug", 3))
    public_enabled = bool(_row_value(row, "public_enabled", 4))
    public_status = _as_optional_str(_row_value(row, "public_status", 5))
    checkpoint_bar = _as_optional_int(_row_value(row, "latest_checkpoint_bar", 6))
    latest_tick_bar = _as_optional_int(_row_value(row, "latest_tick_bar", 7))

    errors: list[str] = []
    if require_active:
        if instance_status != "active":
            errors.append(
                f"live instance status is {instance_status or 'missing'}, expected active"
            )
        if account_status != "active":
            errors.append(
                f"exchange account status is {account_status or 'missing'}, expected active"
            )

    lock_available: bool | None = None
    if require_lock_available:
        lock_status = probe_live_runtime_lock(conn, live_instance_id=instance_id)
        lock_available = lock_status.acquired
        if not lock_status.acquired:
            errors.append("runtime advisory lock is already held")

    return DbReadiness(
        status="ok" if not errors else "error",
        live_instance_id=instance_id,
        instance_status=instance_status,
        account_status=account_status,
        mode=mode,
        public_instance_slug=public_slug,
        public_enabled=public_enabled,
        public_status=public_status,
        latest_checkpoint_bar=checkpoint_bar,
        latest_tick_bar=latest_tick_bar,
        lock_available=lock_available,
        errors=tuple(errors),
    )


def _fetch_instance_row(conn: DbConnection, *, live_instance_id: str):
    cursor = conn.execute(
        """
        select
            li.status as instance_status,
            ea.status as account_status,
            li.mode,
            li.public_instance_slug,
            li.public_enabled,
            pdi.status as public_status,
            ec.bar_count as latest_checkpoint_bar,
            max(lt.bar) as latest_tick_bar
        from live_instances li
        join exchange_accounts ea on ea.account_id = li.account_id
        left join public_dashboard_instances pdi
            on pdi.live_instance_id = li.live_instance_id
            and pdi.strategy_slug = li.strategy_slug
        left join engine_checkpoints ec
            on ec.live_instance_id = li.live_instance_id
        left join live_ticks lt
            on lt.live_instance_id = li.live_instance_id
        where li.live_instance_id = %(live_instance_id)s
        group by
            li.status,
            ea.status,
            li.mode,
            li.public_instance_slug,
            li.public_enabled,
            pdi.status,
            ec.bar_count
        """,
        {"live_instance_id": live_instance_id},
    )
    return cursor.fetchone()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only DB readiness check for a cta-forge live instance."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--live-instance-id", default=None)
    parser.add_argument("--require-active", action="store_true")
    parser.add_argument("--require-lock-available", action="store_true")
    return parser.parse_args(argv)


def _connect_psycopg(database_url: str) -> ClosableConnection:
    psycopg = import_module("psycopg")
    return cast(ClosableConnection, psycopg.connect(database_url, autocommit=True))


def _optional_text(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _required_text(name: str, value: str) -> str:
    stripped = _optional_text(value)
    if stripped is None:
        raise ValueError(f"{name} is required")
    return stripped


def _row_value(row: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        mapping = cast(Mapping[str, Any], row)
        return mapping[key]
    sequence = cast(Sequence[Any], row)
    return sequence[index]


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
