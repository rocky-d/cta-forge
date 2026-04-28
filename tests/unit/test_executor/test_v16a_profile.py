"""Tests for v16a target-profile helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from executor.profiles.v16a_badscore_overlay import (
    V16A_PROFILE,
    V16aHistoricalStrategy,
    V16aTargetSet,
)


def test_v16a_profile_metadata_is_stable() -> None:
    assert V16A_PROFILE.slug == "v16a-badscore-overlay"
    assert "Badscore" in V16A_PROFILE.name


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
