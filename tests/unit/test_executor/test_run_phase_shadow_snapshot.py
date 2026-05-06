"""Tests for cache-only v16a phase shadow snapshots."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from executor.run_phase_shadow_snapshot import (
    phase_shadow_symbols,
    run_phase_shadow_snapshot,
)
from executor.run_shadow_tick import ShadowTickConfig


def _config(**overrides) -> ShadowTickConfig:
    config = ShadowTickConfig(
        private_key="redacted-key",
        account_address="redacted-address",
        testnet=False,
        data_dir="data",
        journal_dir="journal/shadow-v16a",
        state_file="engine-state-shadow.json",
        min_order_notional=10.0,
        max_order_notional=50.0,
        max_staleness=timedelta(hours=8),
        target_scale=5.0,
        gross_cap=4.0,
        core_phase_hours=0,
        compare_core_phase_hours=2,
        phase_comparison_journal_dir="journal/phase-shadow",
        symbols=["BTC", "ETH", "XRP", "LINK"],
    )
    return ShadowTickConfig(**(config.__dict__ | overrides))


def test_phase_shadow_symbols_uses_live_symbols() -> None:
    assert phase_shadow_symbols(_config()) == {"BTC", "ETH", "XRP", "LINK"}


def test_phase_shadow_symbols_filters_testnet_exclusions() -> None:
    assert phase_shadow_symbols(_config(testnet=True)) == {"BTC", "ETH"}


def test_run_phase_shadow_snapshot_requires_compare_phase() -> None:
    with pytest.raises(ValueError, match="V16A_COMPARE_CORE_PHASE_HOURS"):
        asyncio.run(run_phase_shadow_snapshot(_config(compare_core_phase_hours=None)))


def test_run_phase_shadow_snapshot_records_compact_summary(monkeypatch) -> None:
    class FakeAdapter:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def close(self) -> None:
            return None

    async def fake_record_phase_comparison(*, config, exchange, allowed_symbols):
        assert config.compare_core_phase_hours == 2
        assert isinstance(exchange, FakeAdapter)
        assert allowed_symbols == {"BTC", "ETH", "XRP", "LINK"}
        return {
            "base_core_phase_hours": 0,
            "compare_core_phase_hours": 2,
            "metrics": {"l1": 0.1, "cosine": 0.9, "max_abs": 0.05, "sign_flips": 1},
            "phases": {
                "0": {"target_ts": "2026-05-06T12:00:00+00:00", "n_orders": 1},
                "2": {"target_ts": "2026-05-06T14:00:00+00:00", "n_orders": 2},
            },
        }

    import executor.run_phase_shadow_snapshot as module

    monkeypatch.setattr(module, "HyperliquidAdapter", FakeAdapter)
    monkeypatch.setattr(module, "record_phase_comparison", fake_record_phase_comparison)

    summary = asyncio.run(run_phase_shadow_snapshot(_config()))

    assert summary == {
        "status": "ok",
        "journal_dir": "journal/phase-shadow",
        "base_core_phase_hours": 0,
        "compare_core_phase_hours": 2,
        "target_timestamps": {
            "0": "2026-05-06T12:00:00+00:00",
            "2": "2026-05-06T14:00:00+00:00",
        },
        "metrics": {"l1": 0.1, "cosine": 0.9, "max_abs": 0.05, "sign_flips": 1},
        "phase_order_counts": {"0": 1, "2": 2},
    }
