from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Mapping

import pytest
from executor.live import LiveState
from executor.live_persistence_import import (
    LivePersistenceImportError,
    LivePersistenceImportKeys,
    LivePersistenceImportRows,
    build_live_persistence_import_rows,
    load_existing_live_persistence,
)
from executor.live_persistence_postgres import (
    LivePersistenceReferenceData,
    PostgresLiveStateStore,
    write_live_import_rows,
    write_live_reference_rows,
)


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(
        self,
        row: tuple[Any, ...] | Mapping[str, Any] | None = None,
    ) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | Mapping[str, Any] | None:
        return self._row


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[RecordedExecute] = []
        self.next_row: tuple[Any, ...] | Mapping[str, Any] | None = None

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        bound = params or {}
        self.calls.append(RecordedExecute(query, bound))
        lowered = query.lower()
        if lowered.lstrip().startswith("select"):
            return FakeCursor(self.next_row)
        if "returning id, bar" in lowered:
            bar = bound["bar"]
            return FakeCursor((10_000 + bar, bar))
        return FakeCursor()


def _reference() -> LivePersistenceReferenceData:
    return LivePersistenceReferenceData(
        strategy_slug="cta-forge",
        strategy_name="CTA Forge",
        profile_id="v16a-mainnet-pilot",
        profile_slug="v16a-mainnet-pilot",
        profile_version="2026-05-14",
        profile_config_json={"target_scale": Decimal("5.0")},
        account_id="hl-mainnet-pilot-01",
        exchange="hyperliquid",
        network="mainnet",
        account_label="Mainnet pilot",
        live_instance_id="cta-forge-mainnet-pilot-01",
        run_id="20260514T060000Z-test",
        mode="mainnet_pilot",
        public_instance_slug="mainnet-pilot",
        public_display_name="Mainnet Pilot",
        public_enabled=True,
        risk_config_json={"target_gross_cap": Decimal("4.00")},
        runtime_config_json={"dry_run": False},
    )


def _write_fixture(tmp_path) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"equity":106.294634123456789,"peak":112.438665987654321,'
        '"dd_pct":5.464000000000001,"n_positions":1,'
        '"positions":{"LINK":{"side":"long",'
        '"qty":2.000000000000001,"entry":9.91873,"best":9.7069}}}\n'
    )
    (journal_dir / "targets.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"profile":"v16a-mainnet-pilot",'
        '"target_ts":"2026-05-14T06:00:00Z",'
        '"staleness_seconds":180.125,'
        '"target_gross":0.5461864655472283,'
        '"normalized_gross":0.5461864655472283,'
        '"ignored_gross":0.0,"ignored_gross_ratio":0.0,'
        '"execution_coverage":1.0,'
        '"weights":{"LINK":0.18428112026966947},'
        '"ignored_weights":{},"orders":[]}\n'
    )
    (journal_dir / "trades.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:01Z","bar":91,'
        '"kind":"target_buy","symbol":"LINK","side":"long",'
        '"qty":2.000000000000001,"price":9.7069,'
        '"reason":"target:v16a-mainnet-pilot"}\n'
    )
    (journal_dir / "signals.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"signals":{"LINK":0.12345678901234567}}\n'
    )
    (tmp_path / "engine-state.json").write_text(
        '{"bar_count":91,"equity_peak":112.438665987654321}'
    )


def _rows(tmp_path):
    _write_fixture(tmp_path)
    batch = load_existing_live_persistence(
        tmp_path / "journal",
        state_file=tmp_path / "engine-state.json",
    )
    return build_live_persistence_import_rows(
        batch,
        LivePersistenceImportKeys(
            live_instance_id="cta-forge-mainnet-pilot-01",
            run_id="20260514T060000Z-test",
        ),
    )


def test_postgres_live_state_store_saves_checkpoint_payload() -> None:
    conn = FakeConnection()
    store = PostgresLiveStateStore(
        conn,
        live_instance_id="cta-forge-mainnet-pilot-01",
        run_id="20260514T060000Z-test",
    )

    store.save(
        LiveState(
            bar_count=91,
            initial_equity=99.7,
            peak_equity=112.438665987654321,
            last_tick_equity=106.294634123456789,
        )
    )

    call = conn.calls[-1]
    assert "insert into engine_checkpoints" in call.sql.lower()
    assert call.params["live_instance_id"] == "cta-forge-mainnet-pilot-01"
    assert call.params["run_id"] == "20260514T060000Z-test"
    assert call.params["bar_count"] == 91
    assert "106.29463412345679" in call.params["payload_json"]


def test_postgres_live_state_store_loads_checkpoint_payload_from_mapping() -> None:
    conn = FakeConnection()
    conn.next_row = {
        "payload_json": {
            "version": 1,
            "saved_at": "2026-05-14T06:03:00+00:00",
            "bar_count": 91,
            "initial_equity": 99.7,
            "peak_equity": 112.438665987654321,
            "dd_breaker_active": False,
            "last_signals": {},
            "recent_returns": [],
            "last_tick_equity": 106.294634123456789,
            "positions": {},
        }
    }
    store = PostgresLiveStateStore(conn, live_instance_id="instance", run_id="run")

    state = store.load()

    assert state is not None
    assert state.bar_count == 91
    assert state.last_tick_equity == 106.294634123456789
    assert "from engine_checkpoints" in conn.calls[-1].sql.lower()


def test_postgres_live_state_store_loads_checkpoint_payload_from_json_text() -> None:
    conn = FakeConnection()
    conn.next_row = (
        '{"version":1,"saved_at":"2026-05-14T06:03:00+00:00",'
        '"bar_count":91,"initial_equity":99.7,"peak_equity":112.4,'
        '"positions":{}}',
    )
    store = PostgresLiveStateStore(conn, live_instance_id="instance", run_id="run")

    state = store.load()

    assert state is not None
    assert state.bar_count == 91


def test_postgres_live_state_store_returns_none_when_checkpoint_missing() -> None:
    conn = FakeConnection()
    store = PostgresLiveStateStore(conn, live_instance_id="instance", run_id="run")

    assert store.load() is None


def test_write_live_reference_rows_upserts_identity_and_public_instance() -> None:
    conn = FakeConnection()

    write_live_reference_rows(conn, _reference())

    joined_sql = "\n".join(call.sql.lower() for call in conn.calls)
    assert "insert into strategies" in joined_sql
    assert "insert into strategy_profiles" in joined_sql
    assert "insert into exchange_accounts" in joined_sql
    assert "insert into live_instances" in joined_sql
    assert "insert into live_runs" in joined_sql
    assert "insert into public_dashboard_instances" in joined_sql
    assert conn.calls[1].params["config_json"] == '{"target_scale":"5.0"}'
    assert conn.calls[3].params["risk_config_json"] == '{"target_gross_cap":"4.00"}'


def test_write_live_import_rows_upserts_operational_rows(tmp_path) -> None:
    conn = FakeConnection()

    write_live_import_rows(conn, _rows(tmp_path))

    joined_sql = "\n".join(call.sql.lower() for call in conn.calls)
    assert "insert into engine_checkpoints" in joined_sql
    assert "insert into live_ticks" in joined_sql
    assert "insert into live_positions" in joined_sql
    assert "insert into live_targets" in joined_sql
    assert "insert into live_trades" in joined_sql
    assert "insert into live_signals" in joined_sql
    assert "on conflict (\n            live_instance_id, run_id, bar" in joined_sql

    tick_call = next(
        call for call in conn.calls if "insert into live_ticks" in call.sql
    )
    assert tick_call.params["account_equity"] == Decimal("106.294634123456789")
    position_call = next(
        call for call in conn.calls if "insert into live_positions" in call.sql
    )
    assert position_call.params["tick_id"] == 10_091
    assert position_call.params["qty"] == Decimal("2.000000000000001")
    target_call = next(
        call for call in conn.calls if "insert into live_targets" in call.sql
    )
    assert target_call.params["target_gross"] == Decimal("0.5461864655472283")
    assert target_call.params["weights_json"] == ('{"LINK":"0.18428112026966947"}')


def test_write_live_import_rows_fails_when_position_tick_is_missing() -> None:
    conn = FakeConnection()
    with pytest.raises(LivePersistenceImportError, match="missing tick bar"):
        write_live_import_rows(
            conn,
            LivePersistenceImportRows(
                checkpoint=None,
                ticks=[],
                positions=[
                    {
                        "tick_bar": 1,
                        "live_instance_id": "instance",
                        "symbol": "BTC",
                        "side": "long",
                        "qty": Decimal("0.1"),
                        "entry_price": Decimal("100"),
                        "best_price": Decimal("101"),
                        "raw_json": {},
                    }
                ],
                targets=[],
                trades=[],
                signals=[],
            ),
        )


def test_write_live_reference_rows_requires_public_slug_when_public() -> None:
    conn = FakeConnection()
    reference = _reference()
    bad_reference = replace(reference, public_instance_slug=None)

    with pytest.raises(LivePersistenceImportError, match="public_instance_slug"):
        write_live_reference_rows(conn, bad_reference)
