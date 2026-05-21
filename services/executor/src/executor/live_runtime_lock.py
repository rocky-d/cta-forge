"""PostgreSQL advisory lock helpers for one live runtime owner.

The lock is intentionally narrow: it only prevents duplicate runtimes for the
same ``LIVE_INSTANCE_ID``. It does not become a config store or scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class DbCursor(Protocol):
    """Minimal cursor shape returned by psycopg-style execute."""

    def fetchone(self) -> Sequence[Any] | Mapping[str, Any] | None:
        """Return one result row."""
        ...


class DbConnection(Protocol):
    """Minimal connection shape used for runtime advisory locks."""

    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> DbCursor:
        """Execute one SQL statement."""
        ...


@dataclass(frozen=True)
class RuntimeLockStatus:
    """Safe runtime lock probe result."""

    live_instance_id: str
    acquired: bool

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "live_instance_id": self.live_instance_id,
            "acquired": self.acquired,
        }


def acquire_live_runtime_lock(
    conn: DbConnection,
    *,
    live_instance_id: str,
) -> RuntimeLockStatus:
    """Acquire a session-level PostgreSQL advisory lock for one live instance.

    The lock is held by the DB session and released automatically when the
    connection closes. Runtime code should keep the same connection open until
    shutdown.
    """

    instance_id = _normalize_live_instance_id(live_instance_id)
    cursor = conn.execute(
        """
        select pg_try_advisory_lock(hashtextextended(%(live_instance_id)s, 0))
        """,
        {"live_instance_id": instance_id},
    )
    acquired = _row_bool(cursor.fetchone())
    return RuntimeLockStatus(live_instance_id=instance_id, acquired=acquired)


def release_live_runtime_lock(
    conn: DbConnection,
    *,
    live_instance_id: str,
) -> bool:
    """Release a session-level advisory lock previously acquired by this session."""

    instance_id = _normalize_live_instance_id(live_instance_id)
    cursor = conn.execute(
        """
        select pg_advisory_unlock(hashtextextended(%(live_instance_id)s, 0))
        """,
        {"live_instance_id": instance_id},
    )
    return _row_bool(cursor.fetchone())


def probe_live_runtime_lock(
    conn: DbConnection,
    *,
    live_instance_id: str,
) -> RuntimeLockStatus:
    """Check whether the runtime lock is currently available, then release it.

    This is for preflight diagnostics before starting a runtime. It does not
    prove future ownership; startup must still call ``acquire_live_runtime_lock``.
    """

    status = acquire_live_runtime_lock(conn, live_instance_id=live_instance_id)
    if status.acquired:
        release_live_runtime_lock(conn, live_instance_id=live_instance_id)
    return status


def _normalize_live_instance_id(value: str) -> str:
    instance_id = value.strip()
    if not instance_id:
        raise ValueError("LIVE_INSTANCE_ID is required for runtime lock")
    return instance_id


def _row_bool(row: Sequence[Any] | Mapping[str, Any] | None) -> bool:
    if row is None:
        raise ValueError("runtime lock query returned no rows")
    if isinstance(row, Mapping):
        value = next(iter(row.values()))
    else:
        value = row[0]
    return bool(value)
