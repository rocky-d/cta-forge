"""Read-only DB status summary for all live instances.

This is an ops-friendly companion to ``run_check_live_instance_db``. It avoids
ad-hoc SQL during multi-live operation by reporting only safe, non-secret live
instance metadata plus latest checkpoint/tick progress.
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

from .live_runtime_lock import DbConnection


class ClosableConnection(DbConnection, Protocol):
    def close(self) -> None:
        """Close DB connection."""
        ...


ConnectDatabase = Callable[[str], ClosableConnection]


class FetchAllCursor(Protocol):
    def fetchall(self) -> list[tuple[Any, ...] | Mapping[str, Any]]:
        """Return all result rows."""
        ...


@dataclass(frozen=True)
class LiveInstanceStatus:
    live_instance_id: str
    instance_status: str | None
    account_id: str | None
    account_status: str | None
    mode: str | None
    public_instance_slug: str | None
    public_enabled: bool | None
    public_status: str | None
    latest_checkpoint_bar: int | None
    latest_checkpoint_saved_at: str | None
    latest_tick_bar: int | None
    latest_tick_ts: str | None
    tick_count: int
    latest_run_id: str | None
    latest_run_status: str | None
    latest_run_started_at: str | None
    latest_target_ts: str | None
    latest_target_gross: float | None
    latest_normalized_gross: float | None
    errors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "ok" if not self.errors else "error"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "live_instance_id": self.live_instance_id,
            "instance_status": self.instance_status,
            "account_id": self.account_id,
            "account_status": self.account_status,
            "mode": self.mode,
            "public_instance_slug": self.public_instance_slug,
            "public_enabled": self.public_enabled,
            "public_status": self.public_status,
            "latest_checkpoint_bar": self.latest_checkpoint_bar,
            "latest_checkpoint_saved_at": self.latest_checkpoint_saved_at,
            "latest_tick_bar": self.latest_tick_bar,
            "latest_tick_ts": self.latest_tick_ts,
            "tick_count": self.tick_count,
            "latest_run_id": self.latest_run_id,
            "latest_run_status": self.latest_run_status,
            "latest_run_started_at": self.latest_run_started_at,
            "latest_target_ts": self.latest_target_ts,
            "latest_target_gross": self.latest_target_gross,
            "latest_normalized_gross": self.latest_normalized_gross,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class LiveInstancesStatusReport:
    status: str
    instances: tuple[LiveInstanceStatus, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": {
                "instances": len(self.instances),
                "active_instances": sum(
                    1 for item in self.instances if item.instance_status == "active"
                ),
                "error_instances": sum(1 for item in self.instances if item.errors),
            },
            "instances": [item.to_dict() for item in self.instances],
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
    if database_url is None:
        print("missing required setting: DATABASE_URL", file=stderr)
        return 2

    connector = connect_database or _connect_psycopg
    conn = connector(database_url)
    try:
        report = check_live_instances_status(
            conn,
            instance_ids=tuple(args.instance_id or ()),
            expect_instances=tuple(args.expect_instance or ()),
            require_active=args.require_active,
            require_progress=args.require_progress,
        )
    finally:
        conn.close()

    json.dump(report.to_dict(), stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    if report.status != "ok":
        print("live instances DB status invalid", file=stderr)
        return 2
    return 0


def check_live_instances_status(
    conn: DbConnection,
    *,
    instance_ids: tuple[str, ...] = (),
    expect_instances: tuple[str, ...] = (),
    require_active: bool = False,
    require_progress: bool = False,
) -> LiveInstancesStatusReport:
    """Return safe DB status metadata for multiple live instances."""

    normalized_ids = tuple(
        _required_text("instance id", value) for value in instance_ids
    )
    expected_ids = tuple(
        _required_text("expected instance id", value) for value in expect_instances
    )
    rows = _fetch_instance_rows(conn, instance_ids=normalized_ids)
    instances = tuple(
        _status_from_row(
            row,
            require_active=require_active,
            require_progress=require_progress,
        )
        for row in rows
    )

    seen = {item.live_instance_id for item in instances}
    errors = [
        f"expected live instance is missing: {instance_id}"
        for instance_id in expected_ids
        if instance_id not in seen
    ]
    if normalized_ids and not instances:
        errors.append("no live instances matched requested filters")

    status = "ok"
    if errors or any(item.errors for item in instances):
        status = "error"
    return LiveInstancesStatusReport(
        status=status,
        instances=instances,
        errors=tuple(errors),
    )


def _fetch_instance_rows(conn: DbConnection, *, instance_ids: tuple[str, ...]):
    params: dict[str, object] = {"instance_ids": list(instance_ids)}
    filter_sql = ""
    if instance_ids:
        filter_sql = "where li.live_instance_id = any(%(instance_ids)s)"
    cursor = cast(
        FetchAllCursor,
        # nosemgrep: sqlalchemy-execute-raw-query — filter_sql hardcoded, values parameterized
        conn.execute(
        f"""
        select
            li.live_instance_id,
            li.status as instance_status,
            ea.account_id,
            ea.status as account_status,
            li.mode,
            li.public_instance_slug,
            li.public_enabled,
            pdi.status as public_status,
            ec.bar_count as latest_checkpoint_bar,
            ec.saved_at as latest_checkpoint_saved_at,
            tick_summary.latest_tick_bar,
            tick_summary.latest_tick_ts,
            coalesce(tick_summary.tick_count, 0) as tick_count,
            latest_run.run_id as latest_run_id,
            latest_run.status as latest_run_status,
            latest_run.started_at as latest_run_started_at,
            latest_target.target_ts as latest_target_ts,
            latest_target.target_gross as latest_target_gross,
            latest_target.normalized_gross as latest_normalized_gross
        from live_instances li
        join exchange_accounts ea on ea.account_id = li.account_id
        left join public_dashboard_instances pdi
            on pdi.live_instance_id = li.live_instance_id
            and pdi.strategy_slug = li.strategy_slug
        left join engine_checkpoints ec
            on ec.live_instance_id = li.live_instance_id
        left join lateral (
            select
                max(bar) as latest_tick_bar,
                max(ts) as latest_tick_ts,
                count(*) as tick_count
            from live_ticks
            where live_instance_id = li.live_instance_id
        ) tick_summary on true
        left join lateral (
            select run_id, status, started_at
            from live_runs
            where live_instance_id = li.live_instance_id
            order by started_at desc
            limit 1
        ) latest_run on true
        left join lateral (
            select target_ts, target_gross, normalized_gross
            from live_targets
            where live_instance_id = li.live_instance_id
            order by bar desc, created_at desc
            limit 1
        ) latest_target on true
        {filter_sql}
        order by li.live_instance_id
        """,
            params,
        ),
    )
    return cursor.fetchall()


def _status_from_row(
    row: Sequence[Any] | Mapping[str, Any],
    *,
    require_active: bool,
    require_progress: bool,
) -> LiveInstanceStatus:
    live_instance_id = str(_row_value(row, "live_instance_id", 0))
    instance_status = _as_optional_str(_row_value(row, "instance_status", 1))
    account_id = _as_optional_str(_row_value(row, "account_id", 2))
    account_status = _as_optional_str(_row_value(row, "account_status", 3))
    mode = _as_optional_str(_row_value(row, "mode", 4))
    public_slug = _as_optional_str(_row_value(row, "public_instance_slug", 5))
    public_enabled = bool(_row_value(row, "public_enabled", 6))
    public_status = _as_optional_str(_row_value(row, "public_status", 7))
    checkpoint_bar = _as_optional_int(_row_value(row, "latest_checkpoint_bar", 8))
    checkpoint_saved_at = _as_optional_iso(
        _row_value(row, "latest_checkpoint_saved_at", 9)
    )
    tick_bar = _as_optional_int(_row_value(row, "latest_tick_bar", 10))
    tick_ts = _as_optional_iso(_row_value(row, "latest_tick_ts", 11))
    tick_count = int(_row_value(row, "tick_count", 12) or 0)
    latest_run_id = _as_optional_str(_row_value(row, "latest_run_id", 13))
    latest_run_status = _as_optional_str(_row_value(row, "latest_run_status", 14))
    latest_run_started_at = _as_optional_iso(
        _row_value(row, "latest_run_started_at", 15)
    )
    latest_target_ts = _as_optional_iso(_row_value(row, "latest_target_ts", 16))
    latest_target_gross = _as_optional_float(_row_value(row, "latest_target_gross", 17))
    latest_normalized_gross = _as_optional_float(
        _row_value(row, "latest_normalized_gross", 18)
    )

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
    if require_progress:
        if checkpoint_bar is None:
            errors.append("latest checkpoint is missing")
        if tick_bar is None:
            errors.append("latest tick is missing")
        if (
            checkpoint_bar is not None
            and tick_bar is not None
            and checkpoint_bar != tick_bar
        ):
            errors.append(
                f"checkpoint/tick mismatch: checkpoint={checkpoint_bar}, tick={tick_bar}"
            )

    return LiveInstanceStatus(
        live_instance_id=live_instance_id,
        instance_status=instance_status,
        account_id=account_id,
        account_status=account_status,
        mode=mode,
        public_instance_slug=public_slug,
        public_enabled=public_enabled,
        public_status=public_status,
        latest_checkpoint_bar=checkpoint_bar,
        latest_checkpoint_saved_at=checkpoint_saved_at,
        latest_tick_bar=tick_bar,
        latest_tick_ts=tick_ts,
        tick_count=tick_count,
        latest_run_id=latest_run_id,
        latest_run_status=latest_run_status,
        latest_run_started_at=latest_run_started_at,
        latest_target_ts=latest_target_ts,
        latest_target_gross=latest_target_gross,
        latest_normalized_gross=latest_normalized_gross,
        errors=tuple(errors),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only DB status summary for cta-forge live instances."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Limit output to one live instance. May be repeated.",
    )
    parser.add_argument(
        "--expect-instance",
        action="append",
        default=[],
        help="Fail if the live instance is missing. May be repeated.",
    )
    parser.add_argument("--require-active", action="store_true")
    parser.add_argument(
        "--require-progress",
        action="store_true",
        help="Require latest checkpoint and tick bars to exist and match.",
    )
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


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
