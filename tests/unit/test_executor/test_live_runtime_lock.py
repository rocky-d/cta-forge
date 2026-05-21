from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest
from executor.live_runtime_lock import (
    acquire_live_runtime_lock,
    probe_live_runtime_lock,
    release_live_runtime_lock,
)


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...]:
        return self._row


class FakeConnection:
    def __init__(self, acquired: bool = True) -> None:
        self.acquired = acquired
        self.calls: list[RecordedExecute] = []

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        self.calls.append(RecordedExecute(query, params or {}))
        if "pg_try_advisory_lock" in query:
            return FakeCursor((self.acquired,))
        if "pg_advisory_unlock" in query:
            return FakeCursor((True,))
        raise AssertionError(query)


def test_acquire_live_runtime_lock_uses_live_instance_id_key() -> None:
    conn = FakeConnection(acquired=True)

    status = acquire_live_runtime_lock(conn, live_instance_id=" mainnet-400-01 ")

    assert status.acquired is True
    assert status.live_instance_id == "mainnet-400-01"
    assert conn.calls[0].params == {"live_instance_id": "mainnet-400-01"}


def test_probe_releases_lock_only_when_acquired() -> None:
    conn = FakeConnection(acquired=True)

    status = probe_live_runtime_lock(conn, live_instance_id="mainnet-400-01")

    assert status.acquired is True
    assert len(conn.calls) == 2
    assert "pg_try_advisory_lock" in conn.calls[0].sql
    assert "pg_advisory_unlock" in conn.calls[1].sql


def test_probe_does_not_release_lock_when_not_acquired() -> None:
    conn = FakeConnection(acquired=False)

    status = probe_live_runtime_lock(conn, live_instance_id="mainnet-400-01")

    assert status.acquired is False
    assert len(conn.calls) == 1


def test_release_live_runtime_lock_returns_db_result() -> None:
    conn = FakeConnection()

    assert release_live_runtime_lock(conn, live_instance_id="mainnet-400-01") is True


def test_live_instance_id_is_required() -> None:
    with pytest.raises(ValueError, match="LIVE_INSTANCE_ID"):
        acquire_live_runtime_lock(FakeConnection(), live_instance_id=" ")
