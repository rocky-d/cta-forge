from __future__ import annotations

import json

from executor.live_persistence_inventory import scan_live_persistence_journal_dirs
from executor.run_inventory_live_persistence import main


def _write_journal(journal_dir, *, duplicate: bool = False) -> None:
    journal_dir.mkdir(parents=True)
    second_bar = 1 if duplicate else 2
    (journal_dir / "equity.jsonl").write_text(
        '{"ts":"2026-05-14T06:00:00Z","bar":1,"equity":100.1,'
        '"peak":101.0,"dd_pct":0.9,"n_positions":0,"positions":{}}\n'
        f'{{"ts":"2026-05-14T07:00:00Z","bar":{second_bar},"equity":102.2,'
        '"peak":102.2,"dd_pct":0,"n_positions":0,"positions":{}}\n'
    )
    (journal_dir / "signals.jsonl").write_text(
        '{"ts":"2026-05-14T06:00:00Z","bar":1,"signals":{"BTC":0.1}}\n'
        f'{{"ts":"2026-05-14T07:00:00Z","bar":{second_bar},"signals":{{"BTC":0.2}}}}\n'
    )
    (journal_dir / "targets.jsonl").write_text(
        '{"ts":"2026-05-14T07:00:00Z","bar":2,'
        '"profile":"v16a-mainnet-pilot",'
        '"target_ts":"2026-05-14T06:00:00Z",'
        '"target_gross":0.3,"normalized_gross":0.2}\n'
    )
    (journal_dir / "trades.jsonl").write_text("")


def test_scan_live_persistence_journal_dirs_reports_ready_and_blocked(tmp_path) -> None:
    ready_dir = tmp_path / "ready-journal"
    blocked_dir = tmp_path / "blocked-journal"
    _write_journal(ready_dir)
    _write_journal(blocked_dir, duplicate=True)
    (tmp_path / "engine-state.json").write_text('{"bar_count":2}')

    report = scan_live_persistence_journal_dirs([tmp_path])
    items = {item.journal_dir.name: item for item in report.items}

    assert report.ready_count == 1
    assert report.blocked_count == 1
    assert items["ready-journal"].ready_for_import is True
    assert items["ready-journal"].counts == {
        "equity": 2,
        "trades": 0,
        "signals": 2,
        "targets": 1,
    }
    assert items["ready-journal"].latest_tick == {
        "bar": 2,
        "ts": "2026-05-14T07:00:00Z",
        "equity": "102.2",
        "n_positions": 0,
    }
    assert items["ready-journal"].bar_ranges["equity"] == {
        "first": 1,
        "latest": 2,
        "records": 2,
        "unique_bars": 2,
    }
    assert set(items["ready-journal"].content_hashes) == {
        "equity",
        "signals",
        "targets",
        "trades",
    }
    assert items["ready-journal"].combined_content_hash is not None
    assert items["ready-journal"].state_file_candidates == [
        tmp_path / "engine-state.json"
    ]
    assert items["blocked-journal"].ready_for_import is False
    assert items["blocked-journal"].duplicate_bars == {
        "equity": [1],
        "signals": [1],
    }
    assert items["blocked-journal"].duplicate_bar_details["equity"] == [
        {
            "bar": 1,
            "ts": "2026-05-14T06:00:00Z",
            "equity": "100.1",
            "peak": "101.0",
            "dd_pct": "0.9",
            "n_positions": 0,
        },
        {
            "bar": 1,
            "ts": "2026-05-14T07:00:00Z",
            "equity": "102.2",
            "peak": "102.2",
            "dd_pct": 0,
            "n_positions": 0,
        },
    ]
    manifest = report.to_dict()
    assert manifest["summary"]["bar_overlaps"] == 1
    assert manifest["bar_overlaps"][0]["shared_equity_bars"] == {
        "count": 1,
        "sample": ["1"],
    }


def test_scan_live_persistence_journal_dirs_reports_duplicate_content(tmp_path) -> None:
    _write_journal(tmp_path / "journal-a")
    _write_journal(tmp_path / "journal-b")

    manifest = scan_live_persistence_journal_dirs([tmp_path]).to_dict()

    assert manifest["summary"]["duplicate_content_groups"] == 1
    assert manifest["duplicate_content_groups"][0]["journal_dirs"] == [
        str(tmp_path / "journal-a"),
        str(tmp_path / "journal-b"),
    ]
    assert manifest["bar_overlaps"][0]["identical_content"] is True


def test_scan_live_persistence_journal_dirs_reports_parse_errors(tmp_path) -> None:
    journal_dir = tmp_path / "bad-journal"
    journal_dir.mkdir()
    (journal_dir / "equity.jsonl").write_text("not-json\n")

    report = scan_live_persistence_journal_dirs([tmp_path])

    assert report.ready_count == 0
    assert report.blocked_count == 1
    assert report.items[0].error is not None
    assert "invalid JSON" in report.items[0].error


def test_inventory_cli_prints_manifest(tmp_path, capsys) -> None:
    _write_journal(tmp_path / "journal")

    exit_code = main(["--root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "cta_forge.live_persistence_inventory.v1"
    assert payload["summary"] == {
        "bar_overlaps": 0,
        "blocked": 0,
        "duplicate_content_groups": 0,
        "journal_dirs": 1,
        "ready_for_import": 1,
    }


def test_inventory_cli_can_fail_on_blocked_items(tmp_path, capsys) -> None:
    _write_journal(tmp_path / "journal", duplicate=True)

    exit_code = main(["--root", str(tmp_path), "--fail-on-blocked"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["summary"]["blocked"] == 1
