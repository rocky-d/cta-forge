"""Runtime store factory for live persistence modes.

This module is the narrow wiring boundary between ``run_live`` and the
file/PostgreSQL persistence implementations.  It intentionally keeps the safe
production default as file-backed persistence, and PostgreSQL source-of-truth
mode remains disabled for live runtime until a later approval gate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from .journal import LiveJournalStore, TradeJournal
from .live_persistence_dual import ShadowWriteFailurePolicy
from .live_persistence_postgres import (
    DbConnection,
    PostgresLiveJournalStore,
    PostgresLiveStateStore,
)
from .live_persistence_runtime import LivePersistenceRuntimeConfig
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
    ``postgres`` intentionally fails closed here; making DB the live source of
    truth is a later cutover phase, not part of shadow-mode wiring.
    """

    file_journal = TradeJournal(
        journal_dir,
        live_instance_id=config.live_instance_id,
        run_id=config.run_id,
        public_instance_slug=public_instance_slug,
    )
    file_state = JsonFileLiveStateStore(state_file)

    if config.backend == "file":
        return LivePersistenceStoreBundle(
            journal=file_journal,
            state_store=file_state,
            close=_noop,
        )

    if config.backend == "postgres":
        msg = "PERSISTENCE_BACKEND=postgres is not wired into live runtime yet"
        raise ValueError(msg)

    if config.database_url is None:
        raise ValueError("PERSISTENCE_BACKEND=dual requires DATABASE_URL")
    if config.live_instance_id is None:
        raise ValueError("PERSISTENCE_BACKEND=dual requires LIVE_INSTANCE_ID")
    if config.run_id is None:
        raise ValueError("PERSISTENCE_BACKEND=dual requires RUN_ID")

    connector = connect_database or _connect_psycopg
    conn = connector(config.database_url)
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
    ) -> None:
        self._primary.record_tick(bar, equity, peak_equity, positions)
        record = self._latest_matching(self._primary.load_equity(), "equity", bar=bar)
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
        )
        record = self._latest_matching(
            self._primary.load_trades(),
            "trade",
            bar=bar,
            kind=kind,
            symbol=symbol,
        )
        self._write_shadow("record_file_trade", record)

    def record_signals(self, bar: int, signals: dict[str, float]) -> None:
        self._primary.record_signals(bar, signals)
        record = self._latest_matching(self._primary.load_signals(), "signals", bar=bar)
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
        )
        record = self._latest_matching(
            self._primary.load_targets(),
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


def _validate_shadow_failure_policy(policy: str) -> ShadowWriteFailurePolicy:
    if policy in {"warn", "raise"}:
        return cast(ShadowWriteFailurePolicy, policy)
    msg = f"invalid shadow write failure policy {policy!r}; expected warn or raise"
    raise ValueError(msg)


def _noop() -> None:
    return None
