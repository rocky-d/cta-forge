"""Tests for live CLI guard helpers."""

from __future__ import annotations

import logging

import pytest

from executor.profiles.v16a_badscore_overlay import V16A_MAINNET_PILOT_PROFILE
from executor.run_live import (
    _is_truthy,
    _parse_symbols,
    _suppress_secret_bearing_http_logs,
    _validate_v16a_live_mode,
)


def test_is_truthy_accepts_common_env_values() -> None:
    assert _is_truthy("true") is True
    assert _is_truthy("1") is True
    assert _is_truthy("yes") is True
    assert _is_truthy("y") is True
    assert _is_truthy("false") is False
    assert _is_truthy(None) is False


def test_validate_v16a_live_mode_allows_shadow_without_flag() -> None:
    _validate_v16a_live_mode(
        dry_run=True,
        testnet=False,
        allow_testnet_live=False,
    )


def test_validate_v16a_live_mode_requires_testnet_for_non_dry_run() -> None:
    with pytest.raises(ValueError, match="HL_NETWORK=testnet"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            allow_testnet_live=True,
        )


def test_validate_v16a_live_mode_requires_explicit_testnet_live_flag() -> None:
    with pytest.raises(ValueError, match="ALLOW_V16A_TESTNET_LIVE=true"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=True,
            allow_testnet_live=False,
        )


def test_validate_v16a_live_mode_allows_explicit_testnet_live() -> None:
    _validate_v16a_live_mode(
        dry_run=False,
        testnet=True,
        allow_testnet_live=True,
    )


def test_validate_mainnet_pilot_requires_mainnet_network() -> None:
    with pytest.raises(ValueError, match="HL_NETWORK=mainnet"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=True,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
        )


def test_validate_mainnet_pilot_requires_explicit_live_flag() -> None:
    with pytest.raises(ValueError, match="ALLOW_MAINNET_PILOT_LIVE=true"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=False,
        )


def test_validate_mainnet_pilot_allows_explicit_live_flag() -> None:
    _validate_v16a_live_mode(
        dry_run=False,
        testnet=False,
        strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
        allow_mainnet_pilot_live=True,
    )


def test_parse_symbols_normalizes_comma_separated_list() -> None:
    assert _parse_symbols("btc, ETH,, sol ") == ["BTC", "ETH", "SOL"]
    assert _parse_symbols("") is None


def test_suppress_secret_bearing_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    _suppress_secret_bearing_http_logs()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
