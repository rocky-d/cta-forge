"""PostgreSQL write helpers for live persistence imports.

The helpers in this module are intentionally connection-protocol based. They do
not import a PostgreSQL driver, open sockets, or participate in live trading.
Callers are expected to provide an active DB-API/psycopg-style connection and
own the surrounding transaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence, cast

from .live import LiveState
from .live_persistence_import import (
    LivePersistenceImportError,
    LivePersistenceImportRows,
)
from .live_public_instances import PublicDashboardInstance
from .state import decode_state_payload, encode_state_payload


class DbCursor(Protocol):
    """Minimal cursor shape used by the import writer and query helpers."""

    def fetchone(self) -> Sequence[Any] | Mapping[str, Any] | None:
        """Return one result row."""
        ...

    def fetchall(self) -> Sequence[Sequence[Any] | Mapping[str, Any]]:
        """Return all result rows."""
        ...


class DbConnection(Protocol):
    """Minimal psycopg-style connection shape used by the import writer."""

    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> DbCursor:
        """Execute one SQL statement."""
        ...


class PostgresLiveStateStore:
    """PostgreSQL-backed live checkpoint store.

    The store depends on an injected connection and is not wired into the live
    runtime by default. Callers own transaction/connection lifecycle.
    """

    def __init__(
        self,
        conn: DbConnection,
        *,
        live_instance_id: str,
        run_id: str,
    ) -> None:
        if not live_instance_id.strip():
            raise LivePersistenceImportError("live_instance_id is required")
        if not run_id.strip():
            raise LivePersistenceImportError("run_id is required")
        self._conn = conn
        self._live_instance_id = live_instance_id
        self._run_id = run_id

    def load(self) -> LiveState | None:
        """Load the latest DB checkpoint for this live instance."""

        cursor = self._conn.execute(
            """
            select payload_json
            from engine_checkpoints
            where live_instance_id = %(live_instance_id)s
            """,
            {"live_instance_id": self._live_instance_id},
        )
        row = cursor.fetchone()
        if row is None:
            return None
        payload = _row_value(row, "payload_json", 0)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise LivePersistenceImportError("checkpoint payload_json must be object")
        return decode_state_payload(payload)

    def save(self, state: LiveState) -> None:
        """Persist the latest DB checkpoint for this live instance."""

        payload = encode_state_payload(state)
        _write_checkpoint(
            self._conn,
            {
                "live_instance_id": self._live_instance_id,
                "run_id": self._run_id,
                "bar_count": state.bar_count,
                "payload_json": payload,
            },
        )


def load_public_dashboard_instances(
    conn: DbConnection,
    *,
    strategy_slug: str,
) -> list[PublicDashboardInstance]:
    """Load public-safe active dashboard instances for one strategy."""

    if not strategy_slug.strip():
        raise LivePersistenceImportError("strategy_slug is required")
    cursor = conn.execute(
        """
        select public_instance_slug, display_name, status, is_default
        from public_dashboard_instances
        where strategy_slug = %(strategy_slug)s and status = 'active'
        order by is_default desc, public_instance_slug asc
        """,
        {"strategy_slug": strategy_slug},
    )
    return [
        PublicDashboardInstance(
            public_instance_slug=str(_row_value(row, "public_instance_slug", 0)),
            display_name=str(_row_value(row, "display_name", 1)),
            status=str(_row_value(row, "status", 2)),
            is_default=bool(_row_value(row, "is_default", 3)),
        )
        for row in cursor.fetchall()
    ]


@dataclass(frozen=True)
class LivePersistenceReferenceData:
    """Reference rows required before importing operational live records."""

    strategy_slug: str
    strategy_name: str
    profile_id: str
    profile_slug: str
    account_id: str
    exchange: str
    network: str
    account_label: str
    live_instance_id: str
    run_id: str
    mode: str
    profile_version: str = ""
    public_instance_slug: str | None = None
    public_display_name: str | None = None
    public_enabled: bool = False
    git_sha: str | None = None
    image_ref: str | None = None
    profile_config_json: dict[str, Any] = field(default_factory=dict)
    risk_config_json: dict[str, Any] = field(default_factory=dict)
    runtime_config_json: dict[str, Any] = field(default_factory=dict)
    address_hash: str | None = None
    address_prefix: str | None = None


def write_live_reference_rows(
    conn: DbConnection,
    reference: LivePersistenceReferenceData,
) -> None:
    """Upsert identity/reference rows for one live import.

    The caller should run this inside the same transaction as
    ``write_live_import_rows``.
    """

    _validate_reference(reference)
    conn.execute(
        """
        insert into strategies (slug, name)
        values (%(strategy_slug)s, %(strategy_name)s)
        on conflict (slug) do update set name = excluded.name
        """,
        {
            "strategy_slug": reference.strategy_slug,
            "strategy_name": reference.strategy_name,
        },
    )
    conn.execute(
        """
        insert into strategy_profiles (
            profile_id, strategy_slug, slug, version, config_json
        )
        values (
            %(profile_id)s, %(strategy_slug)s, %(profile_slug)s,
            %(profile_version)s, %(config_json)s::jsonb
        )
        on conflict (profile_id) do update set
            strategy_slug = excluded.strategy_slug,
            slug = excluded.slug,
            version = excluded.version,
            config_json = excluded.config_json
        """,
        {
            "profile_id": reference.profile_id,
            "strategy_slug": reference.strategy_slug,
            "profile_slug": reference.profile_slug,
            "profile_version": reference.profile_version,
            "config_json": _jsonb(reference.profile_config_json),
        },
    )
    conn.execute(
        """
        insert into exchange_accounts (
            account_id, exchange, network, account_label, address_hash,
            address_prefix
        )
        values (
            %(account_id)s, %(exchange)s, %(network)s, %(account_label)s,
            %(address_hash)s, %(address_prefix)s
        )
        on conflict (account_id) do update set
            exchange = excluded.exchange,
            network = excluded.network,
            account_label = excluded.account_label,
            address_hash = excluded.address_hash,
            address_prefix = excluded.address_prefix
        """,
        {
            "account_id": reference.account_id,
            "exchange": reference.exchange,
            "network": reference.network,
            "account_label": reference.account_label,
            "address_hash": reference.address_hash,
            "address_prefix": reference.address_prefix,
        },
    )
    conn.execute(
        """
        insert into live_instances (
            live_instance_id, strategy_slug, profile_id, account_id,
            public_instance_slug, mode, risk_config_json, public_enabled
        )
        values (
            %(live_instance_id)s, %(strategy_slug)s, %(profile_id)s,
            %(account_id)s, %(public_instance_slug)s, %(mode)s,
            %(risk_config_json)s::jsonb, %(public_enabled)s
        )
        on conflict (live_instance_id) do update set
            strategy_slug = excluded.strategy_slug,
            profile_id = excluded.profile_id,
            account_id = excluded.account_id,
            public_instance_slug = excluded.public_instance_slug,
            mode = excluded.mode,
            risk_config_json = excluded.risk_config_json,
            public_enabled = excluded.public_enabled
        """,
        {
            "live_instance_id": reference.live_instance_id,
            "strategy_slug": reference.strategy_slug,
            "profile_id": reference.profile_id,
            "account_id": reference.account_id,
            "public_instance_slug": reference.public_instance_slug,
            "mode": reference.mode,
            "risk_config_json": _jsonb(reference.risk_config_json),
            "public_enabled": reference.public_enabled,
        },
    )
    conn.execute(
        """
        insert into live_runs (
            run_id, live_instance_id, git_sha, image_ref, runtime_config_json
        )
        values (
            %(run_id)s, %(live_instance_id)s, %(git_sha)s, %(image_ref)s,
            %(runtime_config_json)s::jsonb
        )
        on conflict (run_id) do update set
            live_instance_id = excluded.live_instance_id,
            git_sha = excluded.git_sha,
            image_ref = excluded.image_ref,
            runtime_config_json = excluded.runtime_config_json
        """,
        {
            "run_id": reference.run_id,
            "live_instance_id": reference.live_instance_id,
            "git_sha": reference.git_sha,
            "image_ref": reference.image_ref,
            "runtime_config_json": _jsonb(reference.runtime_config_json),
        },
    )
    if reference.public_enabled:
        _write_public_dashboard_instance(conn, reference)


def write_live_import_rows(conn: DbConnection, rows: LivePersistenceImportRows) -> None:
    """Upsert schema-shaped operational rows.

    The caller owns transaction boundaries. This function writes checkpoint,
    ticks, positions, targets, trades, and signals in FK-safe order.
    """

    if rows.checkpoint is not None:
        _write_checkpoint(conn, rows.checkpoint)
    tick_ids = _write_ticks(conn, rows.ticks)
    _write_positions(conn, rows.positions, tick_ids)
    for row in rows.targets:
        _write_target(conn, row)
    for row in rows.trades:
        _write_trade(conn, row)
    for row in rows.signals:
        _write_signal(conn, row)


def _write_public_dashboard_instance(
    conn: DbConnection,
    reference: LivePersistenceReferenceData,
) -> None:
    if not reference.public_instance_slug:
        raise LivePersistenceImportError(
            "public_instance_slug is required when public_enabled is true"
        )
    conn.execute(
        """
        insert into public_dashboard_instances (
            strategy_slug, public_instance_slug, live_instance_id, display_name,
            status, is_default
        )
        values (
            %(strategy_slug)s, %(public_instance_slug)s, %(live_instance_id)s,
            %(display_name)s, 'hidden', true
        )
        on conflict (strategy_slug, public_instance_slug) do update set
            live_instance_id = excluded.live_instance_id,
            display_name = excluded.display_name,
            updated_at = now()
        """,
        {
            "strategy_slug": reference.strategy_slug,
            "public_instance_slug": reference.public_instance_slug,
            "live_instance_id": reference.live_instance_id,
            "display_name": reference.public_display_name
            or reference.public_instance_slug,
        },
    )


def _write_checkpoint(conn: DbConnection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into engine_checkpoints (
            live_instance_id, run_id, bar_count, payload_json
        )
        values (
            %(live_instance_id)s, %(run_id)s, %(bar_count)s,
            %(payload_json)s::jsonb
        )
        on conflict (live_instance_id) do update set
            run_id = excluded.run_id,
            bar_count = excluded.bar_count,
            payload_json = excluded.payload_json,
            saved_at = now()
        """,
        {**row, "payload_json": _jsonb(row["payload_json"])},
    )


def _write_ticks(conn: DbConnection, rows: list[dict[str, Any]]) -> dict[Any, Any]:
    tick_ids: dict[Any, Any] = {}
    for row in rows:
        cursor = conn.execute(
            """
            insert into live_ticks (
                live_instance_id, run_id, bar, ts, account_equity, peak_equity,
                dd_pct, n_positions, summary_json
            )
            values (
                %(live_instance_id)s, %(run_id)s, %(bar)s, %(ts)s,
                %(account_equity)s, %(peak_equity)s, %(dd_pct)s,
                %(n_positions)s, %(summary_json)s::jsonb
            )
            on conflict (live_instance_id, bar) do update set
                run_id = excluded.run_id,
                ts = excluded.ts,
                account_equity = excluded.account_equity,
                peak_equity = excluded.peak_equity,
                dd_pct = excluded.dd_pct,
                n_positions = excluded.n_positions,
                summary_json = excluded.summary_json
            returning id, bar
            """,
            {**row, "summary_json": _jsonb(row["summary_json"])},
        )
        result = cursor.fetchone()
        if result is None:
            raise LivePersistenceImportError(
                f"live tick upsert returned no id for bar {row['bar']}"
            )
        tick_id, bar = _row_values(result, "id", "bar")
        tick_ids[bar] = tick_id
    return tick_ids


def _write_positions(
    conn: DbConnection,
    rows: list[dict[str, Any]],
    tick_ids: Mapping[Any, Any],
) -> None:
    for row in rows:
        tick_bar = row["tick_bar"]
        if tick_bar not in tick_ids:
            raise LivePersistenceImportError(
                f"position row references missing tick bar {tick_bar!r}"
            )
        params = {key: value for key, value in row.items() if key != "tick_bar"} | {
            "tick_id": tick_ids[tick_bar],
            "raw_json": _jsonb(row["raw_json"]),
        }
        conn.execute(
            """
            insert into live_positions (
                tick_id, live_instance_id, symbol, side, qty, entry_price,
                best_price, raw_json
            )
            values (
                %(tick_id)s, %(live_instance_id)s, %(symbol)s, %(side)s,
                %(qty)s, %(entry_price)s, %(best_price)s, %(raw_json)s::jsonb
            )
            on conflict (tick_id, symbol) do update set
                live_instance_id = excluded.live_instance_id,
                side = excluded.side,
                qty = excluded.qty,
                entry_price = excluded.entry_price,
                best_price = excluded.best_price,
                raw_json = excluded.raw_json
            """,
            params,
        )


def _write_target(conn: DbConnection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into live_targets (
            live_instance_id, run_id, bar, ts, profile, target_ts,
            staleness_seconds, target_gross, normalized_gross, ignored_gross,
            ignored_gross_ratio, execution_coverage, weights_json,
            ignored_weights_json, orders_json
        )
        values (
            %(live_instance_id)s, %(run_id)s, %(bar)s, %(ts)s, %(profile)s,
            %(target_ts)s, %(staleness_seconds)s, %(target_gross)s,
            %(normalized_gross)s, %(ignored_gross)s, %(ignored_gross_ratio)s,
            %(execution_coverage)s, %(weights_json)s::jsonb,
            %(ignored_weights_json)s::jsonb, %(orders_json)s::jsonb
        )
        on conflict (live_instance_id, bar, profile, target_ts) do update set
            run_id = excluded.run_id,
            ts = excluded.ts,
            staleness_seconds = excluded.staleness_seconds,
            target_gross = excluded.target_gross,
            normalized_gross = excluded.normalized_gross,
            ignored_gross = excluded.ignored_gross,
            ignored_gross_ratio = excluded.ignored_gross_ratio,
            execution_coverage = excluded.execution_coverage,
            weights_json = excluded.weights_json,
            ignored_weights_json = excluded.ignored_weights_json,
            orders_json = excluded.orders_json
        """,
        {
            **row,
            "weights_json": _jsonb(row["weights_json"]),
            "ignored_weights_json": _jsonb(row["ignored_weights_json"]),
            "orders_json": _jsonb(row["orders_json"]),
        },
    )


def _write_trade(conn: DbConnection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into live_trades (
            live_instance_id, run_id, bar, ts, kind, symbol, side, qty, price,
            reason, pnl, pnl_pct, held_bars, exchange_order_id, raw_json
        )
        values (
            %(live_instance_id)s, %(run_id)s, %(bar)s, %(ts)s, %(kind)s,
            %(symbol)s, %(side)s, %(qty)s, %(price)s, %(reason)s, %(pnl)s,
            %(pnl_pct)s, %(held_bars)s, %(exchange_order_id)s,
            %(raw_json)s::jsonb
        )
        on conflict (
            live_instance_id, run_id, bar, ts, kind, symbol, qty, price, reason
        ) do update set
            side = excluded.side,
            pnl = excluded.pnl,
            pnl_pct = excluded.pnl_pct,
            held_bars = excluded.held_bars,
            exchange_order_id = excluded.exchange_order_id,
            raw_json = excluded.raw_json
        """,
        {**row, "raw_json": _jsonb(row["raw_json"])},
    )


def _write_signal(conn: DbConnection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into live_signals (
            live_instance_id, run_id, bar, ts, signals_json
        )
        values (
            %(live_instance_id)s, %(run_id)s, %(bar)s, %(ts)s,
            %(signals_json)s::jsonb
        )
        on conflict (live_instance_id, bar) do update set
            run_id = excluded.run_id,
            ts = excluded.ts,
            signals_json = excluded.signals_json
        """,
        {**row, "signals_json": _jsonb(row["signals_json"])},
    )


def _validate_reference(reference: LivePersistenceReferenceData) -> None:
    required = {
        "strategy_slug": reference.strategy_slug,
        "strategy_name": reference.strategy_name,
        "profile_id": reference.profile_id,
        "profile_slug": reference.profile_slug,
        "account_id": reference.account_id,
        "exchange": reference.exchange,
        "network": reference.network,
        "account_label": reference.account_label,
        "live_instance_id": reference.live_instance_id,
        "run_id": reference.run_id,
        "mode": reference.mode,
    }
    missing = [key for key, value in required.items() if not value.strip()]
    if missing:
        raise LivePersistenceImportError(
            "missing required reference fields: " + ", ".join(sorted(missing))
        )
    if reference.public_enabled and not reference.public_instance_slug:
        raise LivePersistenceImportError(
            "public_instance_slug is required when public_enabled is true"
        )


def _jsonb(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _row_value(
    row: Sequence[Any] | Mapping[str, Any],
    key: str,
    index: int,
) -> Any:
    if isinstance(row, Mapping):
        return cast(Mapping[str, Any], row)[key]
    return cast(Sequence[Any], row)[index]


def _row_values(row: Sequence[Any] | Mapping[str, Any], *keys: str) -> tuple[Any, ...]:
    return tuple(_row_value(row, key, index) for index, key in enumerate(keys))
