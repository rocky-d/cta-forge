from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Mapping

from executor import run_check_live_instance_db as runner


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(self, row: tuple[Any, ...] | Mapping[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | Mapping[str, Any] | None:
        return self._row


class FakeConnection:
    def __init__(
        self,
        *,
        row: tuple[Any, ...] | Mapping[str, Any] | None = None,
        lock_acquired: bool = True,
    ) -> None:
        self.row = row
        self.lock_acquired = lock_acquired
        self.calls: list[RecordedExecute] = []
        self.closed = False

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        self.calls.append(RecordedExecute(query, params or {}))
        lowered = query.lower()
        if "pg_try_advisory_lock" in lowered:
            return FakeCursor((self.lock_acquired,))
        if "pg_advisory_unlock" in lowered:
            return FakeCursor((True,))
        return FakeCursor(self.row)

    def close(self) -> None:
        self.closed = True


def _active_row() -> tuple[Any, ...]:
    return (
        "active",
        "active",
        "mainnet_pilot",
        "mainnet-400-01",
        False,
        "hidden",
        12,
        13,
    )


def _run(conn: FakeConnection, *args: str) -> tuple[int, dict, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = runner.main(
        list(args),
        env={
            "DATABASE_URL": "postgresql://user:secret@postgres/db",
            "LIVE_INSTANCE_ID": "mainnet-400-01",
        },
        stdout=stdout,
        stderr=stderr,
        connect_database=lambda _: conn,
    )
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


def test_check_live_instance_db_reports_safe_active_metadata() -> None:
    conn = FakeConnection(row=_active_row())

    code, payload, stderr = _run(conn, "--require-active")

    assert code == 0
    assert stderr == ""
    assert conn.closed is True
    assert payload == {
        "status": "ok",
        "live_instance_id": "mainnet-400-01",
        "instance_status": "active",
        "account_status": "active",
        "mode": "mainnet_pilot",
        "public_instance_slug": "mainnet-400-01",
        "public_enabled": False,
        "public_status": "hidden",
        "latest_checkpoint_bar": 12,
        "latest_tick_bar": 13,
        "lock_available": None,
        "errors": [],
    }
    assert "secret" not in json.dumps(payload)


def test_check_live_instance_db_fails_when_identity_missing() -> None:
    conn = FakeConnection(row=None)

    code, payload, stderr = _run(conn)

    assert code == 2
    assert payload["status"] == "error"
    assert "live instance identity row is missing" in payload["errors"]
    assert "invalid" in stderr


def test_check_live_instance_db_can_require_active_status() -> None:
    conn = FakeConnection(
        row=(
            "paused",
            "active",
            "mainnet_pilot",
            "mainnet-400-01",
            False,
            "hidden",
            None,
            None,
        )
    )

    code, payload, _ = _run(conn, "--require-active")

    assert code == 2
    assert payload["status"] == "error"
    assert any("expected active" in error for error in payload["errors"])


def test_check_live_instance_db_can_require_lock_available() -> None:
    conn = FakeConnection(row=_active_row(), lock_acquired=False)

    code, payload, _ = _run(conn, "--require-active", "--require-lock-available")

    assert code == 2
    assert payload["lock_available"] is False
    assert "runtime advisory lock is already held" in payload["errors"]


def test_check_live_instance_db_returns_two_for_missing_required_env() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = runner.main([], env={}, stdout=stdout, stderr=stderr)

    assert code == 2
    assert stdout.getvalue() == ""
    assert "DATABASE_URL" in stderr.getvalue()
    assert "LIVE_INSTANCE_ID" in stderr.getvalue()
