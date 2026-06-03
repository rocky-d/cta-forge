"""Runtime store factory for live persistence modes.

This module is the narrow wiring boundary between ``run_live`` and the
file/PostgreSQL persistence implementations.  It intentionally keeps the safe
production default as file-backed persistence, and PostgreSQL source-of-truth
mode is guarded by an explicit runtime allow flag.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from .journal import LiveJournalStore, TradeJournal
from .live_persistence_dual import ShadowWriteFailurePolicy
from .live_persistence_postgres import (
    DbConnection,
    PostgresLiveJournalStore,
    PostgresLiveStateStore,
)
from .live_persistence_runtime import LivePersistenceRuntimeConfig
from .live_runtime_lock import acquire_live_runtime_lock
from .state import JsonFileLiveStateStore, LiveStateStore

logger = logging.getLogger(__name__)


class ClosableConnection(DbConnection, Protocol):
    """DB connection shape needed by the runtime store bundle."""

    def close(self) -> None:
        """Close the DB connection."""
        ...


ConnectDatabase = Callable[[str], ClosableConnection]


@dataclass(frozen=True)
class LivePersistenceStoreBundle:
    """Persistence stores and owned resources for one live runtime."""

    journal: LiveJournalStore
    state_store: LiveStateStore
    close: Callable[[], None]


def build_live_persistence_stores(
    config: LivePersistenceRuntimeConfig,
    *,
    journal_dir: str,
    state_file: str,
    public_instance_slug: str | None,
    connect_database: ConnectDatabase | None = None,
) -> LivePersistenceStoreBundle:
    """Build journal/state stores for the approved live persistence mode.

    ``file`` is the safe default and does not import or connect to psycopg.
    ``dual`` keeps file reads/writes primary and mirrors writes to PostgreSQL.
    ``postgres`` uses PostgreSQL for live journal and checkpoint reads/writes,
    but only after ``load_live_persistence_runtime_config`` has accepted the
    explicit source-of-truth allow flag.
    """

    if config.backend == "file":
        file_journal = TradeJournal(
            journal_dir,
            live_instance_id=config.live_instance_id,
            run_id=config.run_id,
            public_instance_slug=public_instance_slug,
        )
        file_state = JsonFileLiveStateStore(state_file)
        return LivePersistenceStoreBundle(
            journal=file_journal,
            state_store=file_state,
            close=_noop,
        )

    _validate_database_config(config)
    assert config.database_url is not None
    assert config.live_instance_id is not None
    assert config.run_id is not None

    connector = connect_database or _connect_psycopg
    conn = connector(config.database_url)
    try:
        _ensure_live_instance_active(conn, config)
        lock_status = acquire_live_runtime_lock(
            conn,
            live_instance_id=config.live_instance_id,
        )
        if not lock_status.acquired:
            raise ValueError(
                "live runtime lock is already held for "
                f"LIVE_INSTANCE_ID={config.live_instance_id}"
            )
        _ensure_live_run(conn, config)
    except Exception:
        conn.close()
        raise
    db_journal = PostgresLiveJournalStore(
        conn,
        live_instance_id=config.live_instance_id,
        run_id=config.run_id,
    )
    db_state = PostgresLiveStateStore(
        conn,
        live_instance_id=config.live_instance_id,
        run_id=config.run_id,
    )
    if config.backend == "postgres":
        return LivePersistenceStoreBundle(
            journal=db_journal,
            state_store=db_state,
            close=conn.close,
        )

    file_journal = TradeJournal(
        journal_dir,
        live_instance_id=config.live_instance_id,
        run_id=config.run_id,
        public_instance_slug=public_instance_slug,
    )
    file_state = JsonFileLiveStateStore(state_file)
    return LivePersistenceStoreBundle(
        journal=FileFirstPostgresMirrorJournalStore(
            file_journal,
            db_journal,
            shadow_failure_policy=config.shadow_failure_policy,
        ),
        state_store=FileFirstPostgresMirrorStateStore(
            file_state,
            db_state,
            state_file=state_file,
            shadow_failure_policy=config.shadow_failure_policy,
        ),
        close=conn.close,
    )


class FileFirstPostgresMirrorJournalStore:
    """Mirror exact file journal rows to PostgreSQL after file writes succeed.

    The generic dual store can call both stores through the semantic journal
    interface, but then file and DB rows get independently generated timestamps.
    This runtime-specific wrapper reads the just-appended file row and writes
    that exact record to PostgreSQL, keeping later parity checks strict.
    """

    def __init__(
        self,
        primary: TradeJournal,
        shadow: PostgresLiveJournalStore,
        *,
        shadow_failure_policy: ShadowWriteFailurePolicy = "warn",
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._shadow_failure_policy = _validate_shadow_failure_policy(
            shadow_failure_policy
        )

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
        *,
        dry_run: bool = False,
    ) -> None:
        self._primary.record_tick(bar, equity, peak_equity, positions, dry_run=dry_run)
        record = self._latest_matching(
            self._load_exact_file("equity"), "equity", bar=bar
        )
        self._write_shadow("record_file_equity", record)

    def record_trade(
        self,
        bar: int,
        kind: str,
        symbol: str,
        qty: float,
        price: float,
        reason: str,
        *,
        side: str = "",
        entry_price: float = 0.0,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        held_bars: int = 0,
        exchange_order_id: str | None = None,
        fee: float | None = None,
    ) -> None:
        self._primary.record_trade(
            bar,
            kind,
            symbol,
            qty,
            price,
            reason,
            side=side,
            entry_price=entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held_bars,
            exchange_order_id=exchange_order_id,
            fee=fee,
        )
        record = self._latest_matching(
            self._load_exact_file("trades"),
            "trade",
            bar=bar,
            kind=kind,
            symbol=symbol,
        )
        self._write_shadow("record_file_trade", record)

    def record_signals(
        self, bar: int, signals: dict[str, float], *, dry_run: bool = False
    ) -> None:
        self._primary.record_signals(bar, signals, dry_run=dry_run)
        record = self._latest_matching(
            self._load_exact_file("signals"), "signals", bar=bar
        )
        self._write_shadow("record_file_signals", record)

    def record_target(
        self,
        *,
        bar: int,
        profile: str,
        target_ts: str,
        staleness_seconds: float,
        target_gross: float,
        normalized_gross: float,
        weights: dict[str, float],
        orders: list[dict],
        ignored_weights: dict[str, float] | None = None,
        submitted_orders: list[dict] | None = None,
        filled_trades: list[dict] | None = None,
        failed_orders: list[dict] | None = None,
    ) -> None:
        self._primary.record_target(
            bar=bar,
            profile=profile,
            target_ts=target_ts,
            staleness_seconds=staleness_seconds,
            target_gross=target_gross,
            normalized_gross=normalized_gross,
            weights=weights,
            orders=orders,
            ignored_weights=ignored_weights,
            submitted_orders=submitted_orders,
            filled_trades=filled_trades,
            failed_orders=failed_orders,
        )
        record = self._latest_matching(
            self._load_exact_file("targets"),
            "target",
            bar=bar,
            profile=profile,
            target_ts=target_ts,
        )
        self._write_shadow("record_file_target", record)

    def load_equity(self) -> list[dict]:
        return self._primary.load_equity()

    def load_trades(self) -> list[dict]:
        return self._primary.load_trades()

    def load_signals(self) -> list[dict]:
        return self._primary.load_signals()

    def load_targets(self) -> list[dict]:
        return self._primary.load_targets()

    def _load_exact_file(self, stream: str) -> list[dict]:
        loader_by_stream = {
            "equity": self._primary.load_equity_decimal_safe,
            "trades": self._primary.load_trades_decimal_safe,
            "signals": self._primary.load_signals_decimal_safe,
            "targets": self._primary.load_targets_decimal_safe,
        }
        return loader_by_stream[stream]()

    def _latest_matching(
        self,
        records: list[dict],
        stream: str,
        **expected: object,
    ) -> dict:
        if not records:
            msg = f"primary file journal did not append {stream} record"
            raise RuntimeError(msg)
        record = records[-1]
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            msg = (
                f"primary file journal latest {stream} record does not match "
                f"expected write: {mismatches!r}"
            )
            raise RuntimeError(msg)
        return record

    def _write_shadow(self, operation: str, record: dict) -> None:
        try:
            getattr(self._shadow, operation)(record)
        except Exception:
            logger.exception("shadow live journal write failed: %s", operation)
            if self._shadow_failure_policy == "raise":
                raise


class FileFirstPostgresMirrorStateStore:
    """Mirror the exact file checkpoint payload to PostgreSQL after save."""

    def __init__(
        self,
        primary: JsonFileLiveStateStore,
        shadow: PostgresLiveStateStore,
        *,
        state_file: str | Path,
        shadow_failure_policy: ShadowWriteFailurePolicy = "warn",
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._state_file = Path(state_file)
        self._shadow_failure_policy = _validate_shadow_failure_policy(
            shadow_failure_policy
        )

    def load(self):
        return self._primary.load()

    def save(self, state) -> None:
        self._primary.save(state)
        try:
            self._shadow.save_file_payload_path(self._state_file)
        except Exception:
            logger.exception("shadow live checkpoint write failed: save")
            if self._shadow_failure_policy == "raise":
                raise


def _connect_psycopg(database_url: str) -> ClosableConnection:
    psycopg = import_module("psycopg")
    return cast(ClosableConnection, psycopg.connect(database_url, autocommit=True))


def _ensure_live_instance_active(
    conn: DbConnection,
    config: LivePersistenceRuntimeConfig,
) -> None:
    """Fail closed unless the DB identity row is ready to run.

    This is intentionally a narrow status gate, not a dynamic config system.
    Secrets and risk caps remain in private env/hard-code guards.
    """

    cursor = conn.execute(
        """
        select li.status, ea.status
        from live_instances li
        join exchange_accounts ea on ea.account_id = li.account_id
        where li.live_instance_id = %(live_instance_id)s
        """,
        {"live_instance_id": config.live_instance_id},
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(
            "DB live instance identity is missing for "
            f"LIVE_INSTANCE_ID={config.live_instance_id}"
        )
    instance_status = _row_value(row, "status", 0)
    account_status = _row_value(row, "account_status", 1)
    if instance_status != "active":
        raise ValueError(
            "DB live instance must be active before runtime start: "
            f"LIVE_INSTANCE_ID={config.live_instance_id}, status={instance_status}"
        )
    if account_status != "active":
        raise ValueError(
            "DB exchange account must be active before runtime start: "
            f"LIVE_INSTANCE_ID={config.live_instance_id}, account_status={account_status}"
        )


def _ensure_live_run(
    conn: DbConnection,
    config: LivePersistenceRuntimeConfig,
) -> None:
    """Ensure the current runtime run id exists before shadow writes.

    When DRY_RUN is True, skip the live_runs row — the dry-run run
    event does not need DB traceability.
    """
    if config.dry_run:
        return

    Journal and checkpoint tables reference ``live_runs``. Runtime-generated
    ``RUN_ID`` values are intentionally unique per process, so dual mode must
    register the run before the first mirrored write or PostgreSQL will reject
    rows by foreign key. The live instance itself must already exist from the
    approved historical import/cutover setup; otherwise this fails closed.
    """

    conn.execute(
        """
        insert into live_runs (run_id, live_instance_id, runtime_config_json)
        values (%(run_id)s, %(live_instance_id)s, %(runtime_config_json)s::jsonb)
        on conflict (run_id) do update set
            live_instance_id = excluded.live_instance_id,
            runtime_config_json = excluded.runtime_config_json,
            status = 'running'
        """,
        {
            "run_id": config.run_id,
            "live_instance_id": config.live_instance_id,
            "runtime_config_json": json.dumps(config.to_safe_dict()),
        },
    )


def _row_value(
    row: Sequence[Any] | Mapping[str, Any],
    key: str,
    index: int,
) -> Any:
    if isinstance(row, Mapping):
        mapping = cast(Mapping[str, Any], row)
        return mapping[key]
    sequence = cast(Sequence[Any], row)
    return sequence[index]


def _validate_database_config(config: LivePersistenceRuntimeConfig) -> None:
    if config.database_url is None:
        raise ValueError(f"PERSISTENCE_BACKEND={config.backend} requires DATABASE_URL")
    if config.live_instance_id is None:
        raise ValueError(
            f"PERSISTENCE_BACKEND={config.backend} requires LIVE_INSTANCE_ID"
        )
    if config.run_id is None:
        raise ValueError(f"PERSISTENCE_BACKEND={config.backend} requires RUN_ID")
    if config.backend == "postgres" and not config.allow_postgres_source_of_truth:
        raise ValueError(
            "PERSISTENCE_BACKEND=postgres requires ALLOW_POSTGRES_SOURCE_OF_TRUTH=true"
        )


def _validate_shadow_failure_policy(policy: str) -> ShadowWriteFailurePolicy:
    if policy in {"warn", "raise"}:
        return cast(ShadowWriteFailurePolicy, policy)
    msg = f"invalid shadow write failure policy {policy!r}; expected warn or raise"
    raise ValueError(msg)


def _noop() -> None:
    return None
