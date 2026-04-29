"""Tests for one-shot target shadow CLI helpers."""

from __future__ import annotations

import tempfile
from datetime import timedelta

import pytest
from executor.journal import TradeJournal
from executor.live import V16A_PROFILE_SLUG
from executor.run_shadow_tick import load_shadow_tick_config, summarize_latest_target


def _base_env() -> dict[str, str]:
    return {
        "HL_PRIVATE_KEY": "redacted-key",
        "HL_ACCOUNT_ADDRESS": "redacted-address",
        "DRY_RUN": "true",
        "STRATEGY_PROFILE": V16A_PROFILE_SLUG,
    }


def test_load_shadow_tick_config_defaults_to_safe_shadow_paths() -> None:
    config = load_shadow_tick_config(_base_env())

    assert config.testnet is True
    assert config.data_dir == "data"
    assert config.journal_dir == "journal/shadow-v16a"
    assert config.state_file == "engine-state-shadow.json"
    assert config.min_order_notional == 10.0
    assert config.max_staleness == timedelta(hours=8)


def test_load_shadow_tick_config_rejects_non_dry_run() -> None:
    env = _base_env() | {"DRY_RUN": "false"}

    with pytest.raises(ValueError, match="DRY_RUN=true"):
        load_shadow_tick_config(env)


def test_load_shadow_tick_config_accepts_shadow_overrides() -> None:
    env = _base_env() | {
        "DATA_DIR": "custom-data",
        "JOURNAL_DIR": "custom-journal",
        "STATE_FILE": "custom-state.json",
        "MIN_ORDER_NOTIONAL": "25",
        "V16A_MAX_STALENESS_HOURS": "2.5",
    }

    config = load_shadow_tick_config(env)

    assert config.data_dir == "custom-data"
    assert config.journal_dir == "custom-journal"
    assert config.state_file == "custom-state.json"
    assert config.min_order_notional == 25.0
    assert config.max_staleness == timedelta(hours=2.5)


def test_load_shadow_tick_config_rejects_other_profiles() -> None:
    env = _base_env() | {"STRATEGY_PROFILE": "v10g-engine-6h"}

    with pytest.raises(ValueError, match="only supports"):
        load_shadow_tick_config(env)


def test_load_shadow_tick_config_requires_exchange_identity() -> None:
    env = _base_env() | {"HL_PRIVATE_KEY": ""}

    with pytest.raises(ValueError, match="HL_PRIVATE_KEY"):
        load_shadow_tick_config(env)


def test_summarize_latest_target_empty() -> None:
    with tempfile.TemporaryDirectory() as directory:
        summary = summarize_latest_target(directory)

    assert summary["status"] == "error"


def test_summarize_latest_target_has_no_warning_when_coverage_is_good() -> None:
    with tempfile.TemporaryDirectory() as directory:
        journal = TradeJournal(directory)
        journal.record_target(
            bar=1,
            profile=V16A_PROFILE_SLUG,
            target_ts="2026-04-28T12:00:00+00:00",
            staleness_seconds=12.0,
            target_gross=0.2,
            normalized_gross=0.18,
            weights={"BTC": 0.18},
            ignored_weights={"XRPUSDT": 0.02},
            orders=[],
        )

        summary = summarize_latest_target(directory)

    assert summary["ignored_gross_ratio"] == 0.1
    assert summary["execution_coverage"] == 0.9
    assert summary["warnings"] == []


def test_summarize_latest_target_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        journal = TradeJournal(directory)
        journal.record_target(
            bar=1,
            profile=V16A_PROFILE_SLUG,
            target_ts="2026-04-28T12:00:00+00:00",
            staleness_seconds=12.0,
            target_gross=0.2,
            normalized_gross=0.1,
            weights={"BTC": 0.06},
            ignored_weights={"XRPUSDT": 0.05},
            orders=[{"symbol": "BTC", "side": "buy"}],
        )

        summary = summarize_latest_target(directory)

    assert summary["status"] == "ok"
    assert summary["profile"] == V16A_PROFILE_SLUG
    assert summary["n_weights"] == 1
    assert summary["n_ignored_weights"] == 1
    assert summary["n_orders"] == 1
    assert summary["ignored_gross"] == 0.05
    assert summary["ignored_gross_ratio"] == 0.25
    assert summary["execution_coverage"] == 0.5
    assert summary["warnings"] == [
        "execution coverage degraded: ignored_gross_ratio 25.0% exceeds 20%"
    ]
