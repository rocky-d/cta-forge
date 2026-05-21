from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

import pytest
from executor.live import LiveState
from executor.live_persistence_runtime import LivePersistenceRuntimeConfig
from executor.live_persistence_store_factory import build_live_persistence_stores


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(self, row: tuple[Any, ...] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


class FakeConnection:
    def __init__(
        self,
        *,
        lock_acquired: bool = True,
        instance_status: str | None = "active",
        account_status: str = "active",
    ) -> None:
        self.calls: list[RecordedExecute] = []
        self.closed = False
        self.lock_acquired = lock_acquired
        self.instance_status = instance_status
        self.account_status = account_status

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        bound = params or {}
        self.calls.append(RecordedExecute(query, bound))
        lowered = query.lower()
        if "from live_instances" in lowered:
            if self.instance_status is None:
                return FakeCursor()
            return FakeCursor((self.instance_status, self.account_status))
        if "pg_try_advisory_lock" in lowered:
            return FakeCursor((self.lock_acquired,))
        if "pg_advisory_unlock" in lowered:
            return FakeCursor((True,))
        if "returning id, bar" in lowered:
            bar = int(bound["bar"])
            return FakeCursor((10_000 + bar, bar))
        return FakeCursor()

    def close(self) -> None:
        self.closed = True


def test_file_mode_builds_file_stores_without_db_connection(tmp_path) -> None:
    def fail_connect(_: str) -> FakeConnection:
        pytest.fail("file mode must not connect to the database")

    config = LivePersistenceRuntimeConfig(
        backend="file",
        live_instance_id="instance-1",
        run_id="run-1",
    )

    bundle = build_live_persistence_stores(
        config,
        journal_dir=str(tmp_path / "journal"),
        state_file=str(tmp_path / "state.json"),
        public_instance_slug="mainnet-pilot",
        connect_database=fail_connect,
    )

    bundle.journal.record_signals(1, {"BTC": 0.5})
    bundle.state_store.save(LiveState(bar_count=1, initial_equity=100, peak_equity=100))
    bundle.close()

    signal_record = json.loads((tmp_path / "journal" / "signals.jsonl").read_text())
    assert signal_record["live_instance_id"] == "instance-1"
    assert signal_record["run_id"] == "run-1"
    assert signal_record["public_instance_slug"] == "mainnet-pilot"
    assert (tmp_path / "state.json").exists()


def test_dual_mode_mirrors_exact_file_rows_to_postgres(tmp_path) -> None:
    conn = FakeConnection()
    config = LivePersistenceRuntimeConfig(
        backend="dual",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
        shadow_failure_policy="raise",
    )

    bundle = build_live_persistence_stores(
        config,
        journal_dir=str(tmp_path / "journal"),
        state_file=str(tmp_path / "state.json"),
        public_instance_slug="mainnet-pilot",
        connect_database=lambda _: conn,
    )

    bundle.journal.record_tick(
        7,
        100.5,
        101.25,
        {"BTC": {"side": "long", "qty": 0.00009, "entry": 81465.9, "best": 81182.0}},
    )
    bundle.journal.record_signals(7, {"BTC": 0.25})
    bundle.state_store.save(
        LiveState(bar_count=7, initial_equity=100.5, peak_equity=101.25)
    )
    bundle.close()

    run_call = next(call for call in conn.calls if "insert into live_runs" in call.sql)
    assert run_call.params["live_instance_id"] == "instance-1"
    assert run_call.params["run_id"] == "run-1"
    runtime_config = json.loads(run_call.params["runtime_config_json"])
    assert runtime_config["backend"] == "dual"
    assert runtime_config["database_url_configured"] is True

    tick_call = next(
        call for call in conn.calls if "insert into live_ticks" in call.sql
    )
    position_call = next(
        call for call in conn.calls if "insert into live_positions" in call.sql
    )
    assert tick_call.params["account_equity"] == Decimal("100.5")
    assert '"qty":"0.00009"' in position_call.params["raw_json"]
    assert '"entry":"81465.9"' in position_call.params["raw_json"]

    file_signal = json.loads((tmp_path / "journal" / "signals.jsonl").read_text())
    signal_call = next(
        call for call in conn.calls if "insert into live_signals" in call.sql
    )
    assert signal_call.params["live_instance_id"] == "instance-1"
    assert signal_call.params["run_id"] == "run-1"
    assert signal_call.params["bar"] == file_signal["bar"]
    assert signal_call.params["ts"] == file_signal["ts"]
    assert json.loads(signal_call.params["signals_json"]) == {"BTC": "0.25"}
    assert file_signal["signals"] == {"BTC": 0.25}

    file_state = json.loads((tmp_path / "state.json").read_text())
    checkpoint_call = next(
        call for call in conn.calls if "insert into engine_checkpoints" in call.sql
    )
    assert checkpoint_call.params["bar_count"] == 7
    checkpoint_payload = json.loads(checkpoint_call.params["payload_json"])
    assert file_state["initial_equity"] == 100.5
    assert checkpoint_payload["initial_equity"] == "100.5"
    assert checkpoint_payload["peak_equity"] == "101.25"
    assert conn.closed is True


def test_db_mode_fails_closed_when_live_instance_is_missing(tmp_path) -> None:
    conn = FakeConnection(instance_status=None)
    config = LivePersistenceRuntimeConfig(
        backend="dual",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
    )

    with pytest.raises(ValueError, match="identity is missing"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
            connect_database=lambda _: conn,
        )

    assert conn.closed is True


def test_db_mode_fails_closed_when_live_instance_is_not_active(tmp_path) -> None:
    conn = FakeConnection(instance_status="paused")
    config = LivePersistenceRuntimeConfig(
        backend="dual",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
    )

    with pytest.raises(ValueError, match="must be active"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
            connect_database=lambda _: conn,
        )

    assert conn.closed is True


def test_db_mode_fails_closed_when_exchange_account_is_not_active(tmp_path) -> None:
    conn = FakeConnection(account_status="paused")
    config = LivePersistenceRuntimeConfig(
        backend="dual",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
    )

    with pytest.raises(ValueError, match="account must be active"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
            connect_database=lambda _: conn,
        )

    assert conn.closed is True


def test_db_mode_fails_closed_when_runtime_lock_is_held(tmp_path) -> None:
    conn = FakeConnection(lock_acquired=False)
    config = LivePersistenceRuntimeConfig(
        backend="dual",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
    )

    with pytest.raises(ValueError, match="runtime lock is already held"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
            connect_database=lambda _: conn,
        )

    assert conn.closed is True


def test_postgres_mode_requires_source_of_truth_allow_flag(tmp_path) -> None:
    config = LivePersistenceRuntimeConfig(
        backend="postgres",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
        allow_postgres_source_of_truth=False,
    )

    with pytest.raises(ValueError, match="ALLOW_POSTGRES_SOURCE_OF_TRUTH=true"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
        )


def test_postgres_mode_builds_postgres_stores_without_file_writes(tmp_path) -> None:
    conn = FakeConnection()
    config = LivePersistenceRuntimeConfig(
        backend="postgres",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
        allow_postgres_source_of_truth=True,
    )

    bundle = build_live_persistence_stores(
        config,
        journal_dir=str(tmp_path / "journal"),
        state_file=str(tmp_path / "state.json"),
        public_instance_slug="mainnet-pilot",
        connect_database=lambda _: conn,
    )

    bundle.journal.record_signals(8, {"BTC": 0.5})
    bundle.state_store.save(
        LiveState(bar_count=8, initial_equity=101.0, peak_equity=102.0)
    )
    bundle.close()

    run_call = next(call for call in conn.calls if "insert into live_runs" in call.sql)
    runtime_config = json.loads(run_call.params["runtime_config_json"])
    assert runtime_config["backend"] == "postgres"
    assert runtime_config["allow_postgres_source_of_truth"] is True

    signal_call = next(
        call for call in conn.calls if "insert into live_signals" in call.sql
    )
    assert signal_call.params["live_instance_id"] == "instance-1"
    assert signal_call.params["run_id"] == "run-1"
    assert signal_call.params["bar"] == 8

    checkpoint_call = next(
        call for call in conn.calls if "insert into engine_checkpoints" in call.sql
    )
    assert checkpoint_call.params["bar_count"] == 8
    checkpoint_payload = json.loads(checkpoint_call.params["payload_json"])
    assert checkpoint_payload["initial_equity"] == 101.0
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "journal" / "signals.jsonl").exists()
    assert conn.closed is True
