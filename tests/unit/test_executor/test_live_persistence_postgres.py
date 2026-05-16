from __future__ import annotations

import json
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
from executor.live_persistence_parity import compare_live_persistence_import_rows
from executor.live_persistence_postgres import (
    LivePersistenceReferenceData,
    PostgresLiveJournalStore,
    PostgresLiveStateStore,
    load_live_import_rows,
    load_public_dashboard_instances,
    write_live_import_rows,
    write_live_reference_rows,
)
from executor.routes import _journal_to_report_format


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeCursor:
    def __init__(
        self,
        row: tuple[Any, ...] | Mapping[str, Any] | None = None,
        rows: list[tuple[Any, ...] | Mapping[str, Any]] | None = None,
    ) -> None:
        self._row = row
        self._rows = rows or []

    def fetchone(self) -> tuple[Any, ...] | Mapping[str, Any] | None:
        return self._row

    def fetchall(self) -> list[tuple[Any, ...] | Mapping[str, Any]]:
        return self._rows


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[RecordedExecute] = []
        self.next_row: tuple[Any, ...] | Mapping[str, Any] | None = None
        self.next_rows: list[tuple[Any, ...] | Mapping[str, Any]] | None = None
        self.queued_rows: list[list[tuple[Any, ...] | Mapping[str, Any]]] = []

    def execute(
        self, query: str, params: Mapping[str, Any] | None = None
    ) -> FakeCursor:
        bound = params or {}
        self.calls.append(RecordedExecute(query, bound))
        lowered = query.lower()
        if lowered.lstrip().startswith("select"):
            rows = self.queued_rows.pop(0) if self.queued_rows else self.next_rows
            return FakeCursor(self.next_row, rows)
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


def test_postgres_live_journal_store_records_tick_with_positions() -> None:
    conn = FakeConnection()
    store = PostgresLiveJournalStore(
        conn,
        live_instance_id="cta-forge-mainnet-pilot-01",
        run_id="20260514T060000Z-test",
    )

    store.record_tick(
        91,
        106.294634123456789,
        112.438665987654321,
        {
            "LINK": {
                "side": "long",
                "qty": 2.000000000000001,
                "entry": 9.91873,
                "best": 9.7069,
            }
        },
    )

    tick_call = next(
        call for call in conn.calls if "insert into live_ticks" in call.sql
    )
    assert tick_call.params["live_instance_id"] == "cta-forge-mainnet-pilot-01"
    assert tick_call.params["run_id"] == "20260514T060000Z-test"
    assert tick_call.params["bar"] == 91
    assert tick_call.params["account_equity"] == Decimal("106.29463412345679")
    assert tick_call.params["peak_equity"] == Decimal("112.43866598765432")
    assert tick_call.params["n_positions"] == 1
    assert tick_call.params["summary_json"] == "{}"

    delete_call = next(
        call for call in conn.calls if "delete from live_positions" in call.sql
    )
    assert delete_call.params == {"tick_id": 10_091}

    position_call = next(
        call for call in conn.calls if "insert into live_positions" in call.sql
    )
    assert position_call.params["tick_id"] == 10_091
    assert position_call.params["symbol"] == "LINK"
    assert position_call.params["side"] == "long"
    assert position_call.params["qty"] == Decimal("2.000000000000001")
    assert '"best":9.7069' in position_call.params["raw_json"]


def test_postgres_live_journal_store_records_trade_signals_and_target() -> None:
    conn = FakeConnection()
    store = PostgresLiveJournalStore(conn, live_instance_id="instance", run_id="run")

    store.record_trade(
        92,
        "close",
        "LINK",
        2.5,
        9.8,
        "risk:stop",
        side="long",
        entry_price=10.0,
        pnl=-0.5,
        pnl_pct=-2.0,
        held_bars=7,
    )
    store.record_signals(92, {"LINK": 0.12345678901234567})
    store.record_target(
        bar=92,
        profile="v16a-mainnet-pilot",
        target_ts="2026-05-14T06:00:00Z",
        staleness_seconds=180.125,
        target_gross=0.6,
        normalized_gross=0.3,
        weights={"LINK": 0.3, "BTC": 0.0},
        ignored_weights={"ETH": -0.2},
        orders=[{"symbol": "LINK", "side": "sell", "qty": 1.0}],
    )

    trade_call = next(
        call for call in conn.calls if "insert into live_trades" in call.sql
    )
    assert trade_call.params["pnl"] == Decimal("-0.5")
    assert trade_call.params["pnl_pct"] == Decimal("-2.0")
    assert trade_call.params["held_bars"] == 7
    assert '"entry_price":10.0' in trade_call.params["raw_json"]

    signal_call = next(
        call for call in conn.calls if "insert into live_signals" in call.sql
    )
    assert signal_call.params["signals_json"] == '{"LINK":0.12345678901234566}'

    target_call = next(
        call for call in conn.calls if "insert into live_targets" in call.sql
    )
    assert target_call.params["target_gross"] == Decimal("0.6")
    assert target_call.params["ignored_gross"] == Decimal("0.2")
    assert target_call.params["execution_coverage"] == Decimal("0.5")
    assert target_call.params["weights_json"] == '{"LINK":0.3}'
    assert target_call.params["ignored_weights_json"] == '{"ETH":-0.2}'


def test_postgres_live_journal_store_records_exact_file_rows() -> None:
    conn = FakeConnection()
    store = PostgresLiveJournalStore(conn, live_instance_id="instance", run_id="run")

    store.record_file_equity(
        {
            "ts": "2026-05-16T08:03:31.123456+00:00",
            "bar": 287,
            "equity": 103.7,
            "peak": 112.4,
            "dd_pct": 7.7,
            "n_positions": 1,
            "positions": {
                "BTC": {"side": "long", "qty": 0.1, "entry": 100.0, "best": 101.0}
            },
        }
    )
    store.record_file_trade(
        {
            "ts": "2026-05-16T08:03:32.123456+00:00",
            "bar": 287,
            "kind": "target_buy",
            "symbol": "BTC",
            "side": "long",
            "qty": 0.1,
            "price": 100.0,
            "reason": "target:v16a-mainnet-pilot",
        }
    )
    store.record_file_signals(
        {
            "ts": "2026-05-16T08:03:33.123456+00:00",
            "bar": 287,
            "signals": {"BTC": 0.25},
        }
    )
    store.record_file_target(
        {
            "ts": "2026-05-16T08:03:34.123456+00:00",
            "bar": 287,
            "profile": "v16a-mainnet-pilot",
            "target_ts": "2026-05-16T07:00:00+00:00",
            "staleness_seconds": 200.0,
            "target_gross": 0.5,
            "normalized_gross": 0.5,
            "ignored_gross": 0.0,
            "ignored_gross_ratio": 0.0,
            "execution_coverage": 1.0,
            "weights": {"BTC": 0.5},
            "ignored_weights": {},
            "orders": [],
        }
    )

    tick_call = next(
        call for call in conn.calls if "insert into live_ticks" in call.sql
    )
    trade_call = next(
        call for call in conn.calls if "insert into live_trades" in call.sql
    )
    signal_call = next(
        call for call in conn.calls if "insert into live_signals" in call.sql
    )
    target_call = next(
        call for call in conn.calls if "insert into live_targets" in call.sql
    )

    assert tick_call.params["ts"] == "2026-05-16T08:03:31.123456+00:00"
    assert trade_call.params["ts"] == "2026-05-16T08:03:32.123456+00:00"
    assert signal_call.params["ts"] == "2026-05-16T08:03:33.123456+00:00"
    assert target_call.params["ts"] == "2026-05-16T08:03:34.123456+00:00"
    assert (
        json.loads(trade_call.params["raw_json"])["reason"]
        == "target:v16a-mainnet-pilot"
    )


def test_postgres_live_journal_store_loads_journal_shapes() -> None:
    conn = FakeConnection()
    conn.queued_rows = [
        [
            {
                "id": 10_091,
                "ts": "2026-05-14T06:03:00+00:00",
                "bar": 91,
                "account_equity": Decimal("106.294634123456789"),
                "peak_equity": Decimal("112.438665987654321"),
                "dd_pct": Decimal("5.464"),
                "n_positions": 1,
                "summary_json": {"status": "ok"},
            }
        ],
        [
            {
                "tick_id": 10_091,
                "symbol": "LINK",
                "side": "long",
                "qty": Decimal("2.000000000000001"),
                "entry_price": Decimal("9.91873"),
                "best_price": Decimal("9.7069"),
                "raw_json": {
                    "side": "long",
                    "source": "core",
                    "qty": "2.000000000000001",
                },
            }
        ],
        [
            {
                "ts": "2026-05-14T06:03:01+00:00",
                "bar": 91,
                "kind": "target_buy",
                "symbol": "LINK",
                "side": "long",
                "qty": Decimal("2.000000000000001"),
                "price": Decimal("9.7069"),
                "reason": "target:v16a-mainnet-pilot",
                "pnl": None,
                "pnl_pct": None,
                "held_bars": None,
                "exchange_order_id": None,
                "raw_json": {
                    "ts": "2026-05-14T06:03:01+00:00",
                    "bar": 91,
                    "kind": "target_buy",
                    "symbol": "LINK",
                    "side": "long",
                    "qty": "2.000000000000001",
                    "price": "9.7069",
                    "reason": "target:v16a-mainnet-pilot",
                },
            }
        ],
        [
            {
                "ts": "2026-05-14T06:03:00+00:00",
                "bar": 91,
                "signals_json": {"LINK": "0.12345678901234567"},
            }
        ],
        [
            {
                "ts": "2026-05-14T06:03:00+00:00",
                "bar": 91,
                "profile": "v16a-mainnet-pilot",
                "target_ts": "2026-05-14T06:00:00+00:00",
                "staleness_seconds": Decimal("180.125"),
                "target_gross": Decimal("0.5461864655472283"),
                "normalized_gross": Decimal("0.5461864655472283"),
                "ignored_gross": Decimal("0"),
                "ignored_gross_ratio": Decimal("0"),
                "execution_coverage": Decimal("1"),
                "weights_json": {"LINK": "0.18428112026966947"},
                "ignored_weights_json": {},
                "orders_json": [],
            }
        ],
    ]
    store = PostgresLiveJournalStore(conn, live_instance_id="instance", run_id="run")

    equity = store.load_equity()
    trades = store.load_trades()
    signals = store.load_signals()
    targets = store.load_targets()

    assert equity == [
        {
            "status": "ok",
            "ts": "2026-05-14T06:03:00+00:00",
            "bar": 91,
            "equity": 106.29463412345679,
            "peak": 112.43866598765432,
            "dd_pct": 5.464,
            "n_positions": 1,
            "positions": {
                "LINK": {
                    "side": "long",
                    "source": "core",
                    "qty": 2.000000000000001,
                    "entry": 9.91873,
                    "best": 9.7069,
                }
            },
        }
    ]
    assert trades[0]["kind"] == "target_buy"
    assert trades[0]["qty"] == 2.000000000000001
    assert trades[0]["price"] == 9.7069
    assert signals == [
        {
            "ts": "2026-05-14T06:03:00+00:00",
            "bar": 91,
            "signals": {"LINK": 0.12345678901234567},
        }
    ]
    assert targets[0]["profile"] == "v16a-mainnet-pilot"
    assert targets[0]["target_gross"] == 0.5461864655472283
    assert targets[0]["weights"] == {"LINK": 0.18428112026966947}


def test_postgres_live_journal_store_matches_report_conversion_shape() -> None:
    conn = FakeConnection()
    conn.queued_rows = [
        [
            {
                "id": 10_001,
                "ts": "2026-05-14T06:03:00+00:00",
                "bar": 1,
                "account_equity": Decimal("100.0"),
                "peak_equity": Decimal("100.0"),
                "dd_pct": Decimal("0"),
                "n_positions": 1,
                "summary_json": {},
            },
            {
                "id": 10_002,
                "ts": "2026-05-14T07:03:00+00:00",
                "bar": 2,
                "account_equity": Decimal("102.0"),
                "peak_equity": Decimal("102.0"),
                "dd_pct": Decimal("0"),
                "n_positions": 0,
                "summary_json": {},
            },
        ],
        [
            {
                "tick_id": 10_001,
                "symbol": "BTC",
                "side": "long",
                "qty": Decimal("0.1"),
                "entry_price": Decimal("50000"),
                "best_price": Decimal("52000"),
                "raw_json": {"side": "long", "qty": "0.1"},
            }
        ],
        [
            {
                "ts": "2026-05-14T07:03:01+00:00",
                "bar": 2,
                "kind": "close",
                "symbol": "BTC",
                "side": "long",
                "qty": Decimal("0.1"),
                "price": Decimal("52000"),
                "reason": "tp",
                "pnl": Decimal("200"),
                "pnl_pct": Decimal("4.0"),
                "held_bars": 1,
                "exchange_order_id": None,
                "raw_json": {
                    "ts": "2026-05-14T07:03:01+00:00",
                    "bar": 2,
                    "kind": "close",
                    "symbol": "BTC",
                    "side": "long",
                    "qty": "0.1",
                    "price": "52000",
                    "reason": "tp",
                    "entry_price": "50000",
                    "pnl": "200",
                    "pnl_pct": "4.0",
                    "held_bars": 1,
                },
            }
        ],
    ]
    store = PostgresLiveJournalStore(conn, live_instance_id="instance", run_id="run")

    report = _journal_to_report_format(store)

    assert report["bars"] == 2
    assert report["equity_curve"] == [
        ("2026-05-14T06:03:00+00:00", 100.0),
        ("2026-05-14T07:03:00+00:00", 102.0),
    ]
    assert report["positions"] == {}
    assert len(report["trades"]) == 1
    assert report["trades"][0]["pnl"] == 200


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


def test_load_public_dashboard_instances_queries_only_non_hidden_public_rows() -> None:
    conn = FakeConnection()
    conn.next_rows = [
        {
            "public_instance_slug": "mainnet-pilot",
            "display_name": "Mainnet Pilot",
            "status": "live",
            "is_default": True,
        },
        ("testnet-shadow", "Testnet Shadow", "stale", False),
    ]

    instances = load_public_dashboard_instances(conn, strategy_slug="cta-forge")

    assert [instance.public_instance_slug for instance in instances] == [
        "mainnet-pilot",
        "testnet-shadow",
    ]
    assert instances[0].is_default is True
    assert "status <> 'hidden'" in conn.calls[-1].sql.lower()
    assert conn.calls[-1].params == {"strategy_slug": "cta-forge"}


def test_load_public_dashboard_instances_requires_strategy_slug() -> None:
    conn = FakeConnection()

    with pytest.raises(LivePersistenceImportError, match="strategy_slug"):
        load_public_dashboard_instances(conn, strategy_slug=" ")


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


def test_write_live_import_rows_rejects_duplicate_tick_bars(tmp_path) -> None:
    conn = FakeConnection()
    rows = _rows(tmp_path)

    with pytest.raises(LivePersistenceImportError, match="duplicate tick bars"):
        write_live_import_rows(conn, replace(rows, ticks=[*rows.ticks, rows.ticks[0]]))


def test_write_live_reference_rows_requires_public_slug_when_public() -> None:
    conn = FakeConnection()
    reference = _reference()
    bad_reference = replace(reference, public_instance_slug=None)

    with pytest.raises(LivePersistenceImportError, match="public_instance_slug"):
        write_live_reference_rows(conn, bad_reference)


def test_load_live_import_rows_supports_post_import_parity(tmp_path) -> None:
    expected = _rows(tmp_path)
    assert expected.checkpoint is not None
    conn = FakeConnection()
    conn.next_row = {
        "live_instance_id": expected.checkpoint["live_instance_id"],
        "run_id": expected.checkpoint["run_id"],
        "bar_count": expected.checkpoint["bar_count"],
        "payload_json": _db_json(expected.checkpoint["payload_json"]),
    }
    conn.queued_rows = [
        [],
        [
            {
                **expected.ticks[0],
                "ts": "2026-05-14T06:03:00+00:00",
                "summary_json": _db_json(expected.ticks[0]["summary_json"]),
            }
        ],
        [
            {
                **expected.positions[0],
                "raw_json": _db_json(expected.positions[0]["raw_json"]),
            }
        ],
        [
            {
                **expected.targets[0],
                "ts": "2026-05-14T06:03:00+00:00",
                "target_ts": "2026-05-14T06:00:00+00:00",
                "weights_json": _db_json(expected.targets[0]["weights_json"]),
                "ignored_weights_json": _db_json(
                    expected.targets[0]["ignored_weights_json"]
                ),
                "orders_json": _db_json(expected.targets[0]["orders_json"]),
            }
        ],
        [
            {
                **expected.trades[0],
                "ts": "2026-05-14T06:03:01+00:00",
                "raw_json": _db_json(expected.trades[0]["raw_json"]),
            }
        ],
        [
            {
                **expected.signals[0],
                "ts": "2026-05-14T06:03:00+00:00",
                "signals_json": _db_json(expected.signals[0]["signals_json"]),
            }
        ],
    ]

    actual = load_live_import_rows(
        conn,
        live_instance_id="cta-forge-mainnet-pilot-01",
    )
    report = compare_live_persistence_import_rows(expected, actual)

    assert report.ok is True
    assert report.to_dict()["counts"]["ticks"] == {"expected": 1, "actual": 1}
    assert all(
        call.params == {"live_instance_id": "cta-forge-mainnet-pilot-01"}
        for call in conn.calls
    )


def test_compare_live_persistence_import_rows_reports_mismatch(tmp_path) -> None:
    expected = _rows(tmp_path)
    actual = replace(expected, trades=[])

    report = compare_live_persistence_import_rows(expected, actual)

    assert report.ok is False
    assert report.counts["trades"] == {"expected": 1, "actual": 0}
    assert report.mismatches == ["trades: expected 1 rows, got 0"]


def test_compare_live_persistence_import_rows_can_ignore_run_id(tmp_path) -> None:
    expected = _rows(tmp_path)
    actual = replace(
        expected,
        checkpoint={**expected.checkpoint, "run_id": "db-run"}
        if expected.checkpoint is not None
        else None,
        ticks=[{**row, "run_id": "db-run"} for row in expected.ticks],
        targets=[{**row, "run_id": "db-run"} for row in expected.targets],
        trades=[
            {
                **row,
                "run_id": "db-run",
                "raw_json": {**row["raw_json"], "run_id": "db-run"},
            }
            for row in expected.trades
        ],
        signals=[{**row, "run_id": "db-run"} for row in expected.signals],
    )

    strict_report = compare_live_persistence_import_rows(expected, actual)
    ignored_report = compare_live_persistence_import_rows(
        expected,
        actual,
        ignored_keys={"run_id"},
    )

    assert strict_report.ok is False
    assert ignored_report.ok is True


def _db_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _db_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_db_json(item) for item in value]
    return value
