from __future__ import annotations

import json
from dataclasses import dataclass
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
    def __init__(self) -> None:
        self.calls: list[RecordedExecute] = []
        self.closed = False

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        bound = params or {}
        self.calls.append(RecordedExecute(query, bound))
        if "returning id, bar" in query.lower():
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

    bundle.journal.record_signals(7, {"BTC": 0.25})
    bundle.state_store.save(LiveState(bar_count=7, initial_equity=100, peak_equity=101))
    bundle.close()

    file_signal = json.loads((tmp_path / "journal" / "signals.jsonl").read_text())
    signal_call = next(
        call for call in conn.calls if "insert into live_signals" in call.sql
    )
    assert signal_call.params["live_instance_id"] == "instance-1"
    assert signal_call.params["run_id"] == "run-1"
    assert signal_call.params["bar"] == file_signal["bar"]
    assert signal_call.params["ts"] == file_signal["ts"]
    assert json.loads(signal_call.params["signals_json"]) == file_signal["signals"]

    file_state = json.loads((tmp_path / "state.json").read_text())
    checkpoint_call = next(
        call for call in conn.calls if "insert into engine_checkpoints" in call.sql
    )
    assert checkpoint_call.params["bar_count"] == 7
    assert json.loads(checkpoint_call.params["payload_json"]) == file_state
    assert conn.closed is True


def test_postgres_mode_fails_closed_for_live_runtime(tmp_path) -> None:
    config = LivePersistenceRuntimeConfig(
        backend="postgres",
        database_url="postgresql://example.invalid/cta",
        live_instance_id="instance-1",
        run_id="run-1",
        allow_postgres_source_of_truth=True,
    )

    with pytest.raises(ValueError, match="not wired into live runtime"):
        build_live_persistence_stores(
            config,
            journal_dir=str(tmp_path / "journal"),
            state_file=str(tmp_path / "state.json"),
            public_instance_slug=None,
        )
