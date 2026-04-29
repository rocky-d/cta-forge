"""Tests for live CLI guard helpers."""

from __future__ import annotations

import logging

import pytest

from executor.run_live import (
    _is_truthy,
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


def test_suppress_secret_bearing_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    _suppress_secret_bearing_http_logs()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
