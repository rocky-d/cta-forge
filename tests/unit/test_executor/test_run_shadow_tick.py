"""Tests for one-shot target shadow CLI helpers."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from executor.journal import TradeJournal
from executor.live import V16A_PROFILE_SLUG
from exchange.adapter import AccountState, Position

from executor.run_shadow_tick import (
    ShadowTickConfig,
    load_phase_comparisons,
    load_shadow_tick_config,
    phase_diff_metrics,
    record_phase_comparison,
    summarize_latest_target,
)


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
    assert config.max_order_notional is None
    assert config.max_staleness == timedelta(hours=8)
    assert config.target_scale == 1.0
    assert config.gross_cap == 1.0
    assert config.core_phase_hours == 0
    assert config.compare_core_phase_hours is None
    assert config.phase_comparison_journal_dir == "journal/phase-shadow"
    assert config.symbols is None


def test_load_shadow_tick_config_rejects_non_dry_run() -> None:
    env = _base_env() | {"DRY_RUN": "false"}

    with pytest.raises(ValueError, match="DRY_RUN=true"):
        load_shadow_tick_config(env)


def test_load_shadow_tick_config_rejects_unknown_network() -> None:
    env = _base_env() | {"HL_NETWORK": "mainet"}

    with pytest.raises(ValueError, match="HL_NETWORK"):
        load_shadow_tick_config(env)


def test_load_shadow_tick_config_accepts_shadow_overrides() -> None:
    env = _base_env() | {
        "DATA_DIR": "custom-data",
        "JOURNAL_DIR": "custom-journal",
        "STATE_FILE": "custom-state.json",
        "MIN_ORDER_NOTIONAL": "25",
        "MAX_ORDER_NOTIONAL": "50",
        "TARGET_SCALE": "5",
        "TARGET_GROSS_CAP": "4",
        "V16A_MAX_STALENESS_HOURS": "2.5",
        "V16A_CORE_PHASE_HOURS": "2",
        "V16A_COMPARE_CORE_PHASE_HOURS": "0",
        "PHASE_COMPARISON_JOURNAL_DIR": "phase-journal",
        "LIVE_SYMBOLS": "BTC,ETH",
    }

    config = load_shadow_tick_config(env)

    assert config.data_dir == "custom-data"
    assert config.journal_dir == "custom-journal"
    assert config.state_file == "custom-state.json"
    assert config.min_order_notional == 25.0
    assert config.max_order_notional == 50.0
    assert config.target_scale == 5.0
    assert config.gross_cap == 4.0
    assert config.max_staleness == timedelta(hours=2.5)
    assert config.core_phase_hours == 2
    assert config.compare_core_phase_hours == 0
    assert config.phase_comparison_journal_dir == "phase-journal"
    assert config.symbols == ["BTC", "ETH"]


def test_load_shadow_tick_config_rejects_other_profiles() -> None:
    env = _base_env() | {"STRATEGY_PROFILE": "v10g-engine-6h"}

    with pytest.raises(ValueError, match="only supports"):
        load_shadow_tick_config(env)


def test_load_shadow_tick_config_requires_exchange_identity() -> None:
    env = _base_env() | {"HL_PRIVATE_KEY": ""}

    with pytest.raises(ValueError, match="HL_PRIVATE_KEY"):
        load_shadow_tick_config(env)


def test_phase_diff_metrics_reports_l1_cosine_and_flips() -> None:
    metrics = phase_diff_metrics({"BTC": 0.1, "ETH": -0.2}, {"BTC": -0.1, "SOL": 0.3})

    assert metrics["l1"] == 0.7
    assert metrics["max_abs"] == 0.3
    assert metrics["sign_flips"] == 1
    assert float(metrics["cosine"]) < 0


def test_load_phase_comparisons_empty() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assert load_phase_comparisons(directory) == []


def test_load_phase_comparisons_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "phase_comparisons.jsonl"
    path.write_text('{"metrics":{"l1":0.1}}\n')

    assert load_phase_comparisons(tmp_path) == [{"metrics": {"l1": 0.1}}]


class _FakeExchange:
    async def get_account_state(self) -> AccountState:
        return AccountState(
            equity=Decimal("1000"),
            available_balance=Decimal("1000"),
            total_margin_used=Decimal("0"),
            positions=[
                Position(
                    symbol="BTC",
                    size=Decimal("0.001"),
                    entry_price=Decimal("100000"),
                    unrealized_pnl=Decimal("0"),
                    leverage=1,
                )
            ],
        )


def test_record_phase_comparison_writes_read_only_diagnostics(
    tmp_path, monkeypatch
) -> None:
    fixed_ts = datetime(2026, 5, 6, 12, tzinfo=UTC)

    class FakeStrategy:
        def __init__(self, *_args, core_phase_hours: int, **_kwargs) -> None:
            self.core_phase_hours = core_phase_hours

        def target(self, _now):
            weights = (
                {"BTC": 0.1, "ETH": -0.05}
                if self.core_phase_hours == 0
                else {"BTC": -0.05, "ETH": -0.05}
            )
            return SimpleNamespace(
                timestamp=fixed_ts,
                gross=sum(abs(weight) for weight in weights.values()),
                weights=weights,
            )

    async def fake_prices(_exchange, _symbols):
        return {"BTC": 100000.0, "ETH": 4000.0}

    import executor.run_shadow_tick as module

    monkeypatch.setattr(module, "V16aOnlineTargetStrategy", FakeStrategy)
    monkeypatch.setattr(module, "fetch_target_prices", fake_prices)
    config = ShadowTickConfig(
        private_key="redacted-key",
        account_address="redacted-address",
        testnet=True,
        data_dir="unused-data",
        journal_dir="unused-journal",
        state_file="unused-state.json",
        min_order_notional=10.0,
        max_order_notional=None,
        max_staleness=timedelta(hours=8),
        target_scale=1.0,
        gross_cap=1.0,
        core_phase_hours=0,
        compare_core_phase_hours=2,
        phase_comparison_journal_dir=str(tmp_path),
        symbols=None,
    )

    record = asyncio.run(
        record_phase_comparison(
            config=config,
            exchange=_FakeExchange(),
            allowed_symbols={"BTC", "ETH"},
        )
    )

    assert record is not None
    assert record["metrics"] == {
        "l1": pytest.approx(0.15),
        "max_abs": pytest.approx(0.15),
        "cosine": pytest.approx(-0.3162277660168379),
        "sign_flips": 1,
    }
    assert record["phases"]["0"]["n_orders"] >= 1
    assert record["phases"]["2"]["n_orders"] >= 1
    assert load_phase_comparisons(tmp_path) == [record]


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

    assert summary["ignored_gross_ratio"] == pytest.approx(0.1)
    assert summary["execution_coverage"] == pytest.approx(0.9)
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
