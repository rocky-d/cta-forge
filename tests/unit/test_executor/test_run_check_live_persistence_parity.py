from __future__ import annotations

import json
from dataclasses import replace

from executor import run_check_live_persistence_parity as runner
from executor.live_persistence_import import (
    LivePersistenceImportKeys,
    build_live_persistence_import_rows,
    load_existing_live_persistence,
)


def _write_journal(tmp_path) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-15T02:00:00Z","bar":7,'
        '"equity":101.25,"peak":102.00,"dd_pct":0.7352941176470589,'
        '"n_positions":0,"positions":{}}\n'
    )
    (journal_dir / "targets.jsonl").write_text(
        '{"ts":"2026-05-15T02:00:01Z","bar":7,'
        '"profile":"v16a-mainnet-pilot",'
        '"target_ts":"2026-05-15T01:00:00Z",'
        '"staleness_seconds":3601.0,'
        '"target_gross":0.1,"normalized_gross":0.1,'
        '"ignored_gross":0.0,"ignored_gross_ratio":0.0,'
        '"execution_coverage":1.0,'
        '"weights":{"BTC":0.1},"ignored_weights":{},"orders":[]}\n'
    )
    (journal_dir / "trades.jsonl").write_text(
        '{"ts":"2026-05-15T02:00:02Z","bar":7,'
        '"kind":"target_buy","symbol":"BTC","side":"long",'
        '"qty":0.01,"price":100000,"reason":"target:v16a"}\n'
    )
    (journal_dir / "signals.jsonl").write_text(
        '{"ts":"2026-05-15T02:00:00Z","bar":7,"signals":{"BTC":0.2}}\n'
    )


def _rows(tmp_path, *, run_id: str = "run-1"):
    return build_live_persistence_import_rows(
        load_existing_live_persistence(tmp_path / "journal"),
        LivePersistenceImportKeys(live_instance_id="instance-1", run_id=run_id),
    )


def test_check_live_persistence_parity_cli_reports_ok(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _write_journal(tmp_path)
    expected = _rows(tmp_path)
    monkeypatch.setattr(runner, "_load_db_rows", lambda _args: expected)

    code = runner.main(
        [
            "--journal-dir",
            str(tmp_path / "journal"),
            "--database-url",
            "postgresql:///unused",
            "--live-instance-id",
            "instance-1",
            "--run-id",
            "run-1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["parity"]["mismatch_count"] == 0
    assert payload["file"]["latest_tick"] == {
        "bar": 7,
        "ts": "2026-05-15T02:00:00Z",
    }


def test_check_live_persistence_parity_cli_exits_nonzero_on_mismatch(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _write_journal(tmp_path)
    expected = _rows(tmp_path)
    actual = replace(expected, trades=[])
    monkeypatch.setattr(runner, "_load_db_rows", lambda _args: actual)

    code = runner.main(
        [
            "--journal-dir",
            str(tmp_path / "journal"),
            "--database-url",
            "postgresql:///unused",
            "--live-instance-id",
            "instance-1",
            "--run-id",
            "run-1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False
    assert payload["parity"]["mismatches"] == ["trades: expected 1 rows, got 0"]


def test_check_live_persistence_parity_cli_can_ignore_run_id(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _write_journal(tmp_path)
    actual = _rows(tmp_path, run_id="db-run")
    monkeypatch.setattr(runner, "_load_db_rows", lambda _args: actual)

    code = runner.main(
        [
            "--journal-dir",
            str(tmp_path / "journal"),
            "--database-url",
            "postgresql:///unused",
            "--live-instance-id",
            "instance-1",
            "--run-id",
            "file-run",
            "--ignore-run-id",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["ignored_keys"] == ["run_id"]
