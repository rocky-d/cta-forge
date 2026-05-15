from __future__ import annotations

import json
from pathlib import Path

from executor.live_persistence_import_plan import (
    ACTION_EXCLUDE_COVERED,
    ACTION_EXCLUDE_DUPLICATE,
    ACTION_IMPORT_CANDIDATE,
    ACTION_REVIEW_BLOCKED,
    ACTION_REVIEW_OVERLAP,
    build_live_persistence_import_plan_from_roots,
)
from executor.run_plan_live_persistence_import import main


def _write_journal(
    journal_dir: Path,
    *,
    bars: list[int],
    start_day: int = 1,
    duplicate: bool = False,
) -> None:
    journal_dir.mkdir(parents=True)
    equity_lines: list[str] = []
    signal_lines: list[str] = []
    for index, bar in enumerate(bars):
        output_bar = bars[0] if duplicate and index == len(bars) - 1 else bar
        ts = f"2026-05-{start_day + index:02d}T00:00:00Z"
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
                {
                    "ts": ts,
                    "bar": output_bar,
                    "signals": {"BTC": index / 10},
                },
                separators=(",", ":"),
            )
        )
    (journal_dir / "equity.jsonl").write_text("\n".join(equity_lines) + "\n")
    (journal_dir / "signals.jsonl").write_text("\n".join(signal_lines) + "\n")
    (journal_dir / "targets.jsonl").write_text("")
    (journal_dir / "trades.jsonl").write_text("")


def _actions_by_name(tmp_path: Path) -> dict[str, str]:
    plan = build_live_persistence_import_plan_from_roots([tmp_path])
    return {item.journal_dir.name: item.action for item in plan.items}


def test_import_plan_selects_largest_covering_snapshot(tmp_path) -> None:
    _write_journal(tmp_path / "snapshot-early", bars=[1, 2], start_day=1)
    _write_journal(tmp_path / "snapshot-latest", bars=[1, 2, 3], start_day=1)

    plan = build_live_persistence_import_plan_from_roots([tmp_path])
    actions = {item.journal_dir.name: item for item in plan.items}

    assert actions["snapshot-latest"].action == ACTION_IMPORT_CANDIDATE
    assert actions["snapshot-early"].action == ACTION_EXCLUDE_COVERED
    assert (
        actions["snapshot-early"].representative_journal_dir
        == tmp_path / "snapshot-latest"
    )
    assert plan.to_dict()["summary"]["requires_review"] == 0


def test_import_plan_excludes_exact_duplicate_copies(tmp_path) -> None:
    _write_journal(tmp_path / "copy-a", bars=[1, 2], start_day=1)
    _write_journal(tmp_path / "copy-b", bars=[1, 2], start_day=1)

    actions = _actions_by_name(tmp_path)

    assert sorted(actions.values()) == [
        ACTION_EXCLUDE_DUPLICATE,
        ACTION_IMPORT_CANDIDATE,
    ]


def test_import_plan_blocks_duplicate_bars_inside_artifact(tmp_path) -> None:
    _write_journal(tmp_path / "bad", bars=[1, 2], duplicate=True)

    plan = build_live_persistence_import_plan_from_roots([tmp_path])

    assert plan.items[0].action == ACTION_REVIEW_BLOCKED
    assert "duplicate bars" in plan.items[0].reason


def test_import_plan_reviews_partial_overlaps(tmp_path) -> None:
    _write_journal(tmp_path / "left", bars=[1, 2], start_day=1)
    _write_journal(tmp_path / "right", bars=[2, 3], start_day=2)

    actions = _actions_by_name(tmp_path)

    assert actions == {
        "left": ACTION_REVIEW_OVERLAP,
        "right": ACTION_REVIEW_OVERLAP,
    }


def test_import_plan_cli_prints_plan(tmp_path, capsys) -> None:
    _write_journal(tmp_path / "journal", bars=[1, 2], start_day=1)

    exit_code = main(["--root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "cta_forge.live_persistence_import_plan.v1"
    assert payload["summary"]["action_counts"] == {ACTION_IMPORT_CANDIDATE: 1}


def test_import_plan_cli_can_fail_on_review(tmp_path, capsys) -> None:
    _write_journal(tmp_path / "bad", bars=[1, 2], duplicate=True)

    exit_code = main(["--root", str(tmp_path), "--fail-on-review"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["summary"]["requires_review"] == 1
