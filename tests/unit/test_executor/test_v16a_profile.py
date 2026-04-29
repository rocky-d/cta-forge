"""Tests for v16a target-profile helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import executor.profiles.v16a_badscore_overlay as v16a_module
from executor.profiles.v16a_badscore_overlay import (
    V16A_PROFILE,
    V16aHistoricalStrategy,
    V16aOnlineTargetStrategy,
    V16aTargetSet,
    latest_forward_filled_hour,
    latest_target_index,
)


def test_v16a_profile_metadata_is_stable() -> None:
    assert V16A_PROFILE.slug == "v16a-badscore-overlay"
    assert "Badscore" in V16A_PROFILE.name
    assert V16A_PROFILE.timeframe_hours == 1


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


def test_latest_forward_filled_hour_extends_core_within_timeframe() -> None:
    core_ts = datetime(2024, 1, 1, hour=12, tzinfo=UTC)

    assert latest_forward_filled_hour(
        core_ts,
        datetime(2024, 1, 1, hour=16, tzinfo=UTC),
        core_timeframe_hours=6,
    ) == datetime(2024, 1, 1, hour=16, tzinfo=UTC)
    assert latest_forward_filled_hour(
        core_ts,
        datetime(2024, 1, 1, hour=18, tzinfo=UTC),
        core_timeframe_hours=6,
    ) == datetime(2024, 1, 1, hour=17, tzinfo=UTC)


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
    calls: list[str] = []

    def fake_build(data_dir):
        calls.append(str(data_dir))
        return target_set

    monkeypatch.setattr(v16a_module, "build_v16a_target_set", fake_build)

    strategy = V16aOnlineTargetStrategy(tmp_path, refresh_seconds=3600.0)
    target = strategy.target(datetime(2024, 1, 1, hour=1, minute=30, tzinfo=UTC))

    assert calls == [str(tmp_path)]
    assert target.timestamp == timeline[1]
    assert target.weights["BTCUSDT"] == pytest.approx(0.2)
    assert target.weights["ETHUSDT"] == pytest.approx(-0.1)

    # Cached target set should be reused inside refresh_seconds.
    strategy.target(datetime(2024, 1, 1, hour=1, minute=45, tzinfo=UTC))
    assert calls == [str(tmp_path)]


def test_v16a_online_strategy_rejects_stale_target(monkeypatch, tmp_path) -> None:
    timeline = [datetime(2024, 1, 1, hour=0, tzinfo=UTC)]
    target_set = _target_set(timeline, np.array([[0.1, 0.0]]))
    monkeypatch.setattr(
        v16a_module, "build_v16a_target_set", lambda data_dir: target_set
    )

    strategy = V16aOnlineTargetStrategy(tmp_path, max_staleness=timedelta(hours=1))

    with pytest.raises(ValueError, match="stale"):
        strategy.target(datetime(2024, 1, 1, hour=2, minute=1, tzinfo=UTC))
