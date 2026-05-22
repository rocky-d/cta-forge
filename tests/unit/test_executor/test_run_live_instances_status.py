from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping

from executor import run_live_instances_status as runner


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...] | Mapping[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...] | Mapping[str, Any]]:
        return self._rows


class FakeConnection:
    def __init__(self, rows: list[tuple[Any, ...] | Mapping[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[RecordedExecute] = []
        self.closed = False

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        self.calls.append(RecordedExecute(query, params or {}))
        return FakeCursor(self.rows)

    def close(self) -> None:
        self.closed = True


def _row(instance_id: str, checkpoint: int, tick: int) -> tuple[Any, ...]:
    return (
        instance_id,
        "active",
        f"acct-{instance_id}",
        "active",
        "mainnet_pilot",
        instance_id,
        False,
        "hidden",
        checkpoint,
        datetime(2026, 5, 22, 2, 5, tzinfo=UTC),
        tick,
        datetime(2026, 5, 22, 2, 0, tzinfo=UTC),
        tick,
        f"run-{instance_id}",
        "running",
        datetime(2026, 5, 22, 1, 0, tzinfo=UTC),
        datetime(2026, 5, 22, 0, 0, tzinfo=UTC),
        Decimal("0.81"),
        Decimal("0.80"),
    )


def _run(conn: FakeConnection, *args: str) -> tuple[int, dict, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = runner.main(
        list(args),
        env={"DATABASE_URL": "postgresql://user:secret@postgres/db"},
        stdout=stdout,
        stderr=stderr,
        connect_database=lambda _: conn,
    )
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


def test_live_instances_status_reports_safe_multi_instance_progress() -> None:
    conn = FakeConnection(
        rows=[_row("mainnet-400-01", 13, 13), _row("mainnet-pilot", 425, 425)]
    )

    code, payload, stderr = _run(
        conn,
        "--require-active",
        "--require-progress",
        "--expect-instance",
        "mainnet-pilot",
        "--expect-instance",
        "mainnet-400-01",
    )

    assert code == 0
    assert stderr == ""
    assert conn.closed is True
    assert payload["status"] == "ok"
    assert payload["summary"] == {
        "instances": 2,
        "active_instances": 2,
        "error_instances": 0,
    }
    assert [item["live_instance_id"] for item in payload["instances"]] == [
        "mainnet-400-01",
        "mainnet-pilot",
    ]
    assert payload["instances"][0]["latest_checkpoint_bar"] == 13
    assert payload["instances"][0]["latest_tick_bar"] == 13
    assert payload["instances"][0]["latest_target_gross"] == 0.81
    assert "secret" not in json.dumps(payload)


def test_live_instances_status_fails_when_expected_instance_missing() -> None:
    conn = FakeConnection(rows=[_row("mainnet-pilot", 425, 425)])

    code, payload, stderr = _run(conn, "--expect-instance", "mainnet-400-01")

    assert code == 2
    assert payload["status"] == "error"
    assert payload["errors"] == ["expected live instance is missing: mainnet-400-01"]
    assert "invalid" in stderr


def test_live_instances_status_can_filter_instance_ids() -> None:
    conn = FakeConnection(rows=[_row("mainnet-400-01", 13, 13)])

    code, payload, _ = _run(conn, "--instance-id", "mainnet-400-01")

    assert code == 0
    assert payload["instances"][0]["live_instance_id"] == "mainnet-400-01"
    assert conn.calls[0].params == {"instance_ids": ["mainnet-400-01"]}
    assert "any(%(instance_ids)s)" in conn.calls[0].sql


def test_live_instances_status_can_require_progress() -> None:
    bad = list(_row("mainnet-400-01", 12, 13))
    conn = FakeConnection(rows=[tuple(bad)])

    code, payload, _ = _run(conn, "--require-progress")

    assert code == 2
    assert payload["instances"][0]["status"] == "error"
    assert payload["instances"][0]["errors"] == [
        "checkpoint/tick mismatch: checkpoint=12, tick=13"
    ]


def test_live_instances_status_returns_two_for_missing_database_url() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = runner.main([], env={}, stdout=stdout, stderr=stderr)

    assert code == 2
    assert stdout.getvalue() == ""
    assert "DATABASE_URL" in stderr.getvalue()
