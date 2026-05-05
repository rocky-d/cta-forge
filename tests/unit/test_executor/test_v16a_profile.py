"""Tests for v16a target-profile helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import executor.profiles.v16a_badscore_overlay as v16a_module
from executor.profiles.v16a_badscore_overlay import (
    V16A_MAINNET_PILOT_PROFILE,
    V16A_PROFILE,
    V16aHistoricalStrategy,
    V16aOnlineTargetStrategy,
    V16aTargetSet,
    latest_forward_filled_hour,
    latest_target_index,
    top_n_signals,
)


def test_v16a_profile_metadata_is_stable() -> None:
    assert V16A_PROFILE.slug == "v16a-badscore-overlay"
    assert "Badscore" in V16A_PROFILE.name
    assert V16A_PROFILE.timeframe_hours == 1


def test_v16a_online_strategy_declares_live_cache_warmup_needs() -> None:
    assert V16aOnlineTargetStrategy.required_timeframes == (
        ("1h", 1, 5000),
        ("6h", 6, 500),
    )


def test_v16a_mainnet_pilot_profile_metadata_is_distinct() -> None:
    assert V16A_MAINNET_PILOT_PROFILE.slug == "v16a-mainnet-pilot"
    assert V16A_MAINNET_PILOT_PROFILE.timeframe_hours == 1


def test_top_n_signals_rejects_empty_signal_set() -> None:
    with pytest.raises(ValueError, match="No overlay signals built"):
        top_n_signals({}, top_n=2)


def test_load_bars_backfills_cache_with_executor_fetch_path(
    monkeypatch, tmp_path
) -> None:
    calls = []
    expected = {
        "BTCUSDT": pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)]})
    }

    async def fake_fetch_bars(data_dir, *, symbols, timeframe, min_bars):
        calls.append((data_dir, symbols, timeframe, min_bars))
        return expected

    monkeypatch.setattr(v16a_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(v16a_module, "fetch_cached_bars", fake_fetch_bars)

    assert v16a_module.load_bars("1h", min_bars=5_000) is expected
    assert calls == [(str(tmp_path), v16a_module.DEFAULT_SYMBOLS, "1h", 5_000)]


def test_load_bars_can_read_local_cache_without_backfill(monkeypatch, tmp_path) -> None:
    async def fail_fetch_bars(*args, **kwargs):
        raise AssertionError("live cache-only load must not fetch/backfill")

    monkeypatch.setattr(v16a_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(v16a_module, "DEFAULT_SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(v16a_module, "fetch_cached_bars", fail_fetch_bars)

    store = v16a_module.ParquetStore(tmp_path)
    store.write(
        "BTCUSDT",
        "1h",
        pl.DataFrame(
            {
                "open_time": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, hour=1, tzinfo=UTC),
                ]
            }
        ),
    )
    store.write(
        "ETHUSDT",
        "1h",
        pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)]}),
    )

    bars = v16a_module.load_bars("1h", min_bars=2, backfill=False)

    assert list(bars) == ["BTCUSDT"]
    assert len(bars["BTCUSDT"]) == 2


def test_v16a_historical_strategy_returns_capped_portfolio_target() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    target_set = V16aTargetSet(
        timeline=[ts],
        symbols=["BTCUSDT", "ETHUSDT"],
        returns=np.zeros((1, 2)),
        target_weights=np.array([[0.8, -0.7]]),
        v10g_weights=np.zeros((1, 2)),
        overlay_weights=np.zeros((1, 2)),
        gate=np.ones(1),
    )

    target = V16aHistoricalStrategy(target_set).target(ts)

    assert target.timestamp == ts
    assert target.gross == pytest.approx(1.0)
    assert target.weights["BTCUSDT"] == pytest.approx(0.8 / 1.5)
    assert target.weights["ETHUSDT"] == pytest.approx(-0.7 / 1.5)


def test_v16a_historical_strategy_rejects_unknown_timestamp() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    target_set = V16aTargetSet(
        timeline=[ts],
        symbols=["BTCUSDT"],
        returns=np.zeros((1, 1)),
        target_weights=np.zeros((1, 1)),
        v10g_weights=np.zeros((1, 1)),
        overlay_weights=np.zeros((1, 1)),
        gate=np.ones(1),
    )

    with pytest.raises(KeyError):
        V16aHistoricalStrategy(target_set).target(datetime(2024, 1, 2, tzinfo=UTC))


def _target_set(timeline: list[datetime], weights: np.ndarray) -> V16aTargetSet:
    symbols = ["BTCUSDT", "ETHUSDT"]
    return V16aTargetSet(
        timeline=timeline,
        symbols=symbols,
        returns=np.zeros((len(timeline), len(symbols))),
        target_weights=weights,
        v10g_weights=np.zeros((len(timeline), len(symbols))),
        overlay_weights=np.zeros((len(timeline), len(symbols))),
        gate=np.ones(len(timeline)),
    )


def test_latest_forward_filled_hour_extends_core_until_next_bar_closes() -> None:
    core_ts = datetime(2024, 1, 1, hour=12, tzinfo=UTC)

    assert latest_forward_filled_hour(
        core_ts,
        datetime(2024, 1, 1, hour=16, tzinfo=UTC),
        core_timeframe_hours=6,
    ) == datetime(2024, 1, 1, hour=16, tzinfo=UTC)
    assert latest_forward_filled_hour(
        core_ts,
        datetime(2024, 1, 1, hour=23, tzinfo=UTC),
        core_timeframe_hours=6,
    ) == datetime(2024, 1, 1, hour=23, tzinfo=UTC)
    assert latest_forward_filled_hour(
        core_ts,
        datetime(2024, 1, 2, hour=0, tzinfo=UTC),
        core_timeframe_hours=6,
    ) == datetime(2024, 1, 1, hour=23, tzinfo=UTC)


def test_latest_target_index_uses_latest_target_at_or_before_timestamp() -> None:
    timeline = [
        datetime(2024, 1, 1, hour=0, tzinfo=UTC),
        datetime(2024, 1, 1, hour=1, tzinfo=UTC),
        datetime(2024, 1, 1, hour=2, tzinfo=UTC),
    ]

    assert (
        latest_target_index(
            timeline, datetime(2024, 1, 1, hour=1, minute=30, tzinfo=UTC)
        )
        == 1
    )
    with pytest.raises(KeyError):
        latest_target_index(timeline, datetime(2023, 12, 31, hour=23, tzinfo=UTC))


def test_v16a_online_strategy_returns_latest_non_stale_target(
    monkeypatch, tmp_path
) -> None:
    timeline = [
        datetime(2024, 1, 1, hour=0, tzinfo=UTC),
        datetime(2024, 1, 1, hour=1, tzinfo=UTC),
    ]
    target_set = _target_set(timeline, np.array([[0.1, 0.0], [0.2, -0.1]]))
    calls: list[tuple[str, bool]] = []

    def fake_build(data_dir, *, backfill=True):
        calls.append((str(data_dir), backfill))
        return target_set

    monkeypatch.setattr(v16a_module, "build_v16a_target_set", fake_build)

    strategy = V16aOnlineTargetStrategy(
        tmp_path,
        refresh_seconds=3600.0,
        gross_cap=0.2,
        profile=V16A_MAINNET_PILOT_PROFILE,
    )
    target = strategy.target(datetime(2024, 1, 1, hour=1, minute=30, tzinfo=UTC))

    assert calls == [(str(tmp_path), False)]
    assert strategy.profile == V16A_MAINNET_PILOT_PROFILE
    assert target.timestamp == timeline[1]
    assert target.gross == pytest.approx(0.2)
    assert target.weights["BTCUSDT"] == pytest.approx(0.2 / 0.3 * 0.2)
    assert target.weights["ETHUSDT"] == pytest.approx(-0.1 / 0.3 * 0.2)

    # Cached target set should be reused inside refresh_seconds.
    strategy.target(datetime(2024, 1, 1, hour=1, minute=45, tzinfo=UTC))
    assert calls == [(str(tmp_path), False)]


def test_v16a_online_strategy_applies_target_scale_before_gross_cap(
    monkeypatch, tmp_path
) -> None:
    timeline = [datetime(2024, 1, 1, hour=0, tzinfo=UTC)]
    target_set = _target_set(timeline, np.array([[0.06, -0.02]]))
    monkeypatch.setattr(
        v16a_module,
        "build_v16a_target_set",
        lambda data_dir, *, backfill=True: target_set,
    )

    strategy = V16aOnlineTargetStrategy(
        tmp_path,
        target_scale=5.0,
        gross_cap=1.0,
    )
    target = strategy.target(datetime(2024, 1, 1, hour=0, minute=30, tzinfo=UTC))

    assert target.gross == pytest.approx(0.4)
    assert target.weights["BTCUSDT"] == pytest.approx(0.3)
    assert target.weights["ETHUSDT"] == pytest.approx(-0.1)


def test_v16a_online_strategy_caps_scaled_target(monkeypatch, tmp_path) -> None:
    timeline = [datetime(2024, 1, 1, hour=0, tzinfo=UTC)]
    target_set = _target_set(timeline, np.array([[0.2, -0.1]]))
    monkeypatch.setattr(
        v16a_module,
        "build_v16a_target_set",
        lambda data_dir, *, backfill=True: target_set,
    )

    strategy = V16aOnlineTargetStrategy(
        tmp_path,
        target_scale=5.0,
        gross_cap=1.0,
    )
    target = strategy.target(datetime(2024, 1, 1, hour=0, minute=30, tzinfo=UTC))

    assert target.gross == pytest.approx(1.0)
    assert target.weights["BTCUSDT"] == pytest.approx(2 / 3)
    assert target.weights["ETHUSDT"] == pytest.approx(-1 / 3)


def test_v16a_online_strategy_rejects_stale_target(monkeypatch, tmp_path) -> None:
    timeline = [datetime(2024, 1, 1, hour=0, tzinfo=UTC)]
    target_set = _target_set(timeline, np.array([[0.1, 0.0]]))
    monkeypatch.setattr(
        v16a_module,
        "build_v16a_target_set",
        lambda data_dir, *, backfill=True: target_set,
    )

    strategy = V16aOnlineTargetStrategy(tmp_path, max_staleness=timedelta(hours=1))

    with pytest.raises(ValueError, match="stale"):
        strategy.target(datetime(2024, 1, 1, hour=2, minute=1, tzinfo=UTC))
