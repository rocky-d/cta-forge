from decimal import Decimal

import pytest
from executor.live_persistence_import import (
    LivePersistenceImportError,
    LivePersistenceImportKeys,
    build_live_persistence_import_rows,
    load_existing_live_persistence,
    load_jsonl_records,
)


def test_load_existing_live_persistence_preserves_decimal_precision_and_identity(
    tmp_path,
) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"equity":106.294634123456789,"peak":112.438665987654321,'
        '"dd_pct":5.464000000000001,"n_positions":1,'
        '"positions":{"BTC":{"qty":0.000090000000000001}},'
        '"live_instance_id":"cta-forge-mainnet-pilot-01",'
        '"run_id":"20260514T060000Z-test",'
        '"public_instance_slug":"mainnet-pilot"}\n'
    )
    (journal_dir / "targets.jsonl").write_text(
        '{"bar":91,"profile":"v16a-mainnet-pilot",'
        '"target_ts":"2026-05-14T06:00:00Z",'
        '"target_gross":0.5461864655472283,'
        '"normalized_gross":0.5461864655472283,'
        '"weights":{"LINK":0.18428112026966947},"orders":[]}\n'
    )
    state_file = tmp_path / "engine-state.json"
    state_file.write_text(
        '{"bar_count":91,"equity_peak":112.438665987654321,'
        '"positions":{"BTC":{"qty":0.000090000000000001}}}'
    )

    batch = load_existing_live_persistence(journal_dir, state_file=state_file)

    equity = batch.equity[0]
    assert equity["bar"] == 91
    assert equity["equity"] == Decimal("106.294634123456789")
    assert equity["peak"] == Decimal("112.438665987654321")
    assert equity["positions"]["BTC"]["qty"] == Decimal("0.000090000000000001")
    assert equity["live_instance_id"] == "cta-forge-mainnet-pilot-01"
    assert equity["run_id"] == "20260514T060000Z-test"
    assert equity["public_instance_slug"] == "mainnet-pilot"
    assert batch.targets[0]["target_gross"] == Decimal("0.5461864655472283")
    assert batch.state is not None
    assert batch.state["equity_peak"] == Decimal("112.438665987654321")


def test_build_live_persistence_import_rows_maps_schema_shape(tmp_path) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"equity":106.294634123456789,"peak":112.438665987654321,'
        '"dd_pct":5.464000000000001,"n_positions":1,'
        '"positions":{"LINK":{"side":"long",'
        '"qty":2.000000000000001,"entry":9.91873,"best":9.7069}},'
        '"public_instance_slug":"mainnet-pilot"}\n'
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
    state_file = tmp_path / "engine-state.json"
    state_file.write_text('{"bar_count":91,"equity_peak":112.438665987654321}')
    batch = load_existing_live_persistence(journal_dir, state_file=state_file)

    rows = build_live_persistence_import_rows(
        batch,
        LivePersistenceImportKeys(
            live_instance_id="cta-forge-mainnet-pilot-01",
            run_id="20260514T060000Z-test",
        ),
    )

    assert rows.checkpoint is not None
    assert rows.checkpoint["bar_count"] == 91
    assert rows.ticks[0]["account_equity"] == Decimal("106.294634123456789")
    assert rows.ticks[0]["summary_json"] == {"public_instance_slug": "mainnet-pilot"}
    assert rows.positions[0]["tick_bar"] == 91
    assert rows.positions[0]["qty"] == Decimal("2.000000000000001")
    assert rows.positions[0]["entry_price"] == Decimal("9.91873")
    assert rows.targets[0]["weights_json"]["LINK"] == Decimal("0.18428112026966947")
    assert rows.trades[0]["qty"] == Decimal("2.000000000000001")
    assert rows.signals[0]["signals_json"]["LINK"] == Decimal("0.12345678901234567")


def test_build_live_persistence_import_rows_fails_on_missing_required_field(
    tmp_path,
) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text('{"bar":1}\n')
    batch = load_existing_live_persistence(journal_dir)

    with pytest.raises(LivePersistenceImportError, match="equity.*missing.*ts"):
        build_live_persistence_import_rows(
            batch,
            LivePersistenceImportKeys(live_instance_id="instance", run_id="run"),
        )


def test_build_live_persistence_import_rows_requires_identity_keys() -> None:
    with pytest.raises(LivePersistenceImportError, match="live_instance_id"):
        LivePersistenceImportKeys(live_instance_id=" ", run_id="run")


def test_load_existing_live_persistence_treats_missing_files_as_empty(tmp_path) -> None:
    batch = load_existing_live_persistence(tmp_path / "missing-journal")

    assert batch.state is None
    assert batch.equity == []
    assert batch.trades == []
    assert batch.signals == []
    assert batch.targets == []


def test_load_jsonl_records_fails_closed_on_malformed_json(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    path.write_text('{"bar": 1}\nnot-json\n')

    with pytest.raises(LivePersistenceImportError, match="equity.jsonl:2"):
        load_jsonl_records(path)


def test_load_jsonl_records_rejects_non_object_records(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    path.write_text("[1, 2, 3]\n")

    with pytest.raises(LivePersistenceImportError, match="expected JSON object"):
        load_jsonl_records(path)


def test_load_jsonl_records_rejects_non_standard_json_constants(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    path.write_text('{"equity": NaN}\n')

    with pytest.raises(LivePersistenceImportError, match="non-standard JSON"):
        load_jsonl_records(path)
