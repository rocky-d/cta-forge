import json

import pytest
from executor.live_persistence_import import LivePersistenceImportError
from executor.run_import_live_persistence import main


def _write_fixture(tmp_path) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,'
        '"equity":106.294634,"peak":112.438665,'
        '"dd_pct":5.464,"n_positions":1,'
        '"positions":{"LINK":{"side":"long",'
        '"qty":2.0,"entry":9.91873,"best":9.7069}}}\n'
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
    (journal_dir / "trades.jsonl").write_text("")
    (journal_dir / "signals.jsonl").write_text(
        '{"ts":"2026-05-14T06:03:00Z","bar":91,"signals":{"LINK":0.1}}\n'
    )
    (tmp_path / "engine-state.json").write_text('{"bar_count":91}')


def test_import_live_persistence_cli_dry_run_prints_summary(tmp_path, capsys) -> None:
    _write_fixture(tmp_path)

    exit_code = main(
        [
            "--journal-dir",
            str(tmp_path / "journal"),
            "--state-file",
            str(tmp_path / "engine-state.json"),
            "--live-instance-id",
            "cta-forge-mainnet-pilot-01",
            "--run-id",
            "20260514T060000Z-test",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wrote"] is False
    assert payload["counts"] == {
        "checkpoint": 1,
        "ticks": 1,
        "positions": 1,
        "targets": 1,
        "trades": 0,
        "signals": 1,
    }
    assert payload["latest_target"] == {
        "bar": 91,
        "profile": "v16a-mainnet-pilot",
        "target_ts": "2026-05-14T06:00:00Z",
    }


def test_import_live_persistence_cli_write_requires_database_url(tmp_path) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(LivePersistenceImportError, match="database-url"):
        main(
            [
                "--journal-dir",
                str(tmp_path / "journal"),
                "--live-instance-id",
                "instance",
                "--run-id",
                "run",
                "--write",
            ]
        )


def test_import_live_persistence_cli_parity_requires_write(tmp_path) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(LivePersistenceImportError, match="parity-check.*write"):
        main(
            [
                "--journal-dir",
                str(tmp_path / "journal"),
                "--live-instance-id",
                "instance",
                "--run-id",
                "run",
                "--parity-check",
            ]
        )
