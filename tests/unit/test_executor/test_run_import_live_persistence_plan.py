from __future__ import annotations

import json

import pytest
from executor.live_persistence_import import LivePersistenceImportError
from executor import run_import_live_persistence_plan as runner


def _write_journal(
    root, name: str, *, bars: list[int], duplicate: bool = False
) -> None:
    journal_dir = root / name
    journal_dir.mkdir(parents=True)
    equity_lines: list[str] = []
    signal_lines: list[str] = []
    target_lines: list[str] = []
    for index, bar in enumerate(bars):
        output_bar = bars[0] if duplicate and index == len(bars) - 1 else bar
        ts = f"2026-05-{index + 1:02d}T00:00:00Z"
        equity_lines.append(
            json.dumps(
                {
                    "ts": ts,
                    "bar": output_bar,
                    "equity": 100 + index,
                    "peak": 100 + index,
                    "dd_pct": 0,
                    "n_positions": 0,
                    "positions": {},
                },
                separators=(",", ":"),
            )
        )
        signal_lines.append(
            json.dumps(
                {"ts": ts, "bar": output_bar, "signals": {"BTC": index / 10}},
                separators=(",", ":"),
            )
        )
        target_lines.append(
            json.dumps(
                {
                    "ts": ts,
                    "bar": output_bar,
                    "profile": "v16a-mainnet-pilot",
                    "target_ts": ts,
                    "staleness_seconds": 0,
                    "target_gross": 0.1,
                    "normalized_gross": 0.1,
                    "ignored_gross": 0,
                    "ignored_gross_ratio": 0,
                    "execution_coverage": 1,
                    "weights": {"BTC": 0.1},
                    "ignored_weights": {},
                    "orders": [],
                },
                separators=(",", ":"),
            )
        )
    (journal_dir / "equity.jsonl").write_text("\n".join(equity_lines) + "\n")
    (journal_dir / "signals.jsonl").write_text("\n".join(signal_lines) + "\n")
    (journal_dir / "targets.jsonl").write_text("\n".join(target_lines) + "\n")
    (journal_dir / "trades.jsonl").write_text("")


def test_planned_import_cli_dry_run_summarizes_import_candidates(
    tmp_path, capsys
) -> None:
    _write_journal(tmp_path, "covered", bars=[1])
    _write_journal(tmp_path, "candidate", bars=[1, 2])

    code = runner.main(
        [
            "--root",
            str(tmp_path),
            "--live-instance-id",
            "mainnet-pilot",
            "--run-id",
            "historical-import",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["wrote"] is False
    assert payload["plan"]["import_candidates"] == 1
    assert payload["plan"]["requires_review"] == 0
    assert payload["candidates"][0]["journal_dir"].endswith("candidate")
    assert payload["candidates"][0]["summary"]["counts"]["ticks"] == 2


def test_planned_import_write_blocks_unreviewed_items(tmp_path) -> None:
    _write_journal(tmp_path, "candidate", bars=[1, 2])
    _write_journal(tmp_path, "blocked", bars=[1, 2], duplicate=True)

    with pytest.raises(LivePersistenceImportError, match="review items"):
        runner.main(
            [
                "--root",
                str(tmp_path),
                "--live-instance-id",
                "mainnet-pilot",
                "--run-id",
                "historical-import",
                "--write",
                "--database-url",
                "postgresql:///unused",
            ]
        )


def test_planned_import_write_requires_single_candidate(tmp_path) -> None:
    _write_journal(tmp_path, "candidate-a", bars=[1])
    _write_journal(tmp_path, "candidate-b", bars=[3])

    with pytest.raises(
        LivePersistenceImportError, match="exactly one import candidate"
    ):
        runner.main(
            [
                "--root",
                str(tmp_path),
                "--live-instance-id",
                "mainnet-pilot",
                "--run-id",
                "historical-import",
                "--write",
                "--database-url",
                "postgresql:///unused",
            ]
        )


def test_planned_import_write_uses_single_import_writer(
    tmp_path, monkeypatch, capsys
) -> None:
    _write_journal(tmp_path, "candidate", bars=[1])
    calls = []

    def fake_write(args, rows):
        calls.append((args, rows))
        return {"ok": True, "mismatch_count": 0}

    monkeypatch.setattr(runner, "_write", fake_write)

    code = runner.main(
        [
            "--root",
            str(tmp_path),
            "--live-instance-id",
            "mainnet-pilot",
            "--run-id",
            "historical-import",
            "--write",
            "--database-url",
            "postgresql:///unused",
            "--profile-id",
            "v16a-mainnet-pilot",
            "--profile-slug",
            "v16a-mainnet-pilot",
            "--account-id",
            "historical-mainnet-pilot",
            "--network",
            "mainnet",
            "--account-label",
            "historical-mainnet-pilot",
            "--mode",
            "mainnet_pilot",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["wrote"] is True
    assert payload["parity"] == {"ok": True, "mismatch_count": 0}
    assert len(calls) == 1
    assert calls[0][0].parity_check is True
    assert len(calls[0][1].ticks) == 1
