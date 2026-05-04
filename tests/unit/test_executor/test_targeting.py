"""Tests for target-weight portfolio utilities."""

from __future__ import annotations

import pytest
from datetime import UTC, datetime

from executor.targeting import (
    PortfolioTarget,
    SleeveTarget,
    combine_sleeves,
    current_weights,
    normalize_gross,
    weights_to_orders,
)


def test_portfolio_target_caps_requested_gross() -> None:
    target = PortfolioTarget(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        weights={"BTC": 0.8, "ETH": -0.7},
        gross_cap=1.0,
    ).capped()

    assert target.gross == pytest.approx(1.0)
    assert target.weights["BTC"] == pytest.approx(0.8 / 1.5)
    assert target.weights["ETH"] == pytest.approx(-0.7 / 1.5)


def test_normalize_gross_scales_when_over_cap() -> None:
    weights = normalize_gross({"BTC": 0.8, "ETH": -0.7}, gross_cap=1.0)

    assert sum(abs(v) for v in weights.values()) == pytest.approx(1.0)
    assert weights["BTC"] == pytest.approx(0.8 / 1.5)
    assert weights["ETH"] == pytest.approx(-0.7 / 1.5)


def test_combine_sleeves_applies_allocation_gate_and_cap() -> None:
    target = combine_sleeves(
        [
            SleeveTarget("core", {"BTC": 0.4, "ETH": -0.2}, allocation=0.5),
            SleeveTarget("overlay", {"BTC": 0.2, "SOL": 0.6}, allocation=0.5),
        ],
        gate_scale=0.5,
        gross_cap=1.0,
    )

    assert target["BTC"] == pytest.approx(0.15)
    assert target["ETH"] == pytest.approx(-0.05)
    assert target["SOL"] == pytest.approx(0.15)


def test_current_weights_from_signed_positions() -> None:
    weights = current_weights(
        {"BTC": 0.1, "ETH": -2.0},
        {"BTC": 50_000.0, "ETH": 2_500.0},
        equity=10_000.0,
    )

    assert weights["BTC"] == pytest.approx(0.5)
    assert weights["ETH"] == pytest.approx(-0.5)


def test_weights_to_orders_splits_sign_flip_into_reduce_then_open() -> None:
    orders = weights_to_orders(
        positions={"BTC": 0.1},
        prices={"BTC": 50_000.0},
        equity=10_000.0,
        target_weights={"BTC": -0.2},
        min_notional=20.0,
    )

    assert len(orders) == 2
    assert orders[0].reduce_only is True
    assert orders[0].side == "sell"
    assert orders[0].delta_weight == pytest.approx(-0.5)
    assert orders[1].reduce_only is False
    assert orders[1].side == "sell"
    assert orders[1].delta_weight == pytest.approx(-0.2)


def test_weights_to_orders_reduce_first_and_skip_small_orders() -> None:
    orders = weights_to_orders(
        positions={"BTC": 0.1, "ETH": -1.0},
        prices={"BTC": 50_000.0, "ETH": 2_000.0, "SOL": 100.0},
        equity=10_000.0,
        target_weights={"BTC": 0.7, "ETH": 0.0, "SOL": 0.001},
        min_notional=20.0,
    )

    assert [order.symbol for order in orders] == ["ETH", "BTC"]
    assert orders[0].reduce_only is True
    assert orders[0].side == "buy"
    assert orders[1].reduce_only is False
    assert orders[1].side == "buy"


def test_weights_to_orders_splits_short_to_long_sign_flip() -> None:
    orders = weights_to_orders(
        positions={"ETH": -2.0},
        prices={"ETH": 2_500.0},
        equity=10_000.0,
        target_weights={"ETH": 0.3},
        min_notional=20.0,
    )

    assert len(orders) == 2
    assert orders[0].reduce_only is True
    assert orders[0].side == "buy"
    assert orders[0].delta_weight == pytest.approx(0.5)
    assert orders[1].reduce_only is False
    assert orders[1].side == "buy"
    assert orders[1].delta_weight == pytest.approx(0.3)


def test_weights_to_orders_skips_symbols_without_valid_price() -> None:
    orders = weights_to_orders(
        positions={"BTC": 0.1, "ETH": 1.0},
        prices={"BTC": 0.0},
        equity=10_000.0,
        target_weights={"BTC": 0.0, "ETH": 0.5, "SOL": 0.1},
        min_notional=20.0,
    )

    assert orders == []


def test_weights_to_orders_caps_single_order_notional() -> None:
    orders = weights_to_orders(
        positions={},
        prices={"BTC": 50_000.0},
        equity=10_000.0,
        target_weights={"BTC": 0.5},
        min_notional=20.0,
        max_notional=100.0,
    )

    assert len(orders) == 1
    assert orders[0].delta_notional == pytest.approx(100.0)
    assert orders[0].delta_weight == pytest.approx(0.01)
    assert orders[0].qty == pytest.approx(0.002)


def test_weights_to_orders_does_not_open_flip_before_capped_reduce_completes() -> None:
    orders = weights_to_orders(
        positions={"BTC": 0.1},
        prices={"BTC": 50_000.0},
        equity=10_000.0,
        target_weights={"BTC": -0.2},
        min_notional=20.0,
        max_notional=100.0,
    )

    assert len(orders) == 1
    assert orders[0].reduce_only is True
    assert orders[0].side == "sell"
    assert orders[0].delta_notional == pytest.approx(-100.0)
