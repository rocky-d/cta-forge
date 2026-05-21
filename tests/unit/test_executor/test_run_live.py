"""Tests for live CLI guard helpers."""

from __future__ import annotations

import logging
import re

import pytest

from executor.profiles.v16a_badscore_overlay import V16A_MAINNET_PILOT_PROFILE
from executor.run_mainnet_preflight import _check_runtime_paths, _report_has_errors
from executor.run_live import (
    MAINNET_400_LIVE_INSTANCE_ID,
    _is_truthy,
    _load_runtime_identity,
    _parse_hl_network,
    _parse_symbols,
    _suppress_secret_bearing_http_logs,
    _validate_mainnet_non_dry_run_profile,
    _validate_v16a_live_mode,
)


def test_is_truthy_accepts_common_env_values() -> None:
    assert _is_truthy("true") is True
    assert _is_truthy("1") is True
    assert _is_truthy("yes") is True
    assert _is_truthy("y") is True
    assert _is_truthy("false") is False
    assert _is_truthy(None) is False


def test_load_runtime_identity_uses_explicit_values() -> None:
    identity = _load_runtime_identity(
        {
            "LIVE_INSTANCE_ID": " cta-forge-mainnet-pilot-01 ",
            "PUBLIC_INSTANCE_SLUG": " mainnet-pilot ",
            "RUN_ID": " run-001 ",
        }
    )

    assert identity.live_instance_id == "cta-forge-mainnet-pilot-01"
    assert identity.public_instance_slug == "mainnet-pilot"
    assert identity.run_id == "run-001"


def test_load_runtime_identity_generates_run_id_and_ignores_blank_values() -> None:
    identity = _load_runtime_identity(
        {
            "LIVE_INSTANCE_ID": " ",
            "PUBLIC_INSTANCE_SLUG": "",
        }
    )

    assert identity.live_instance_id is None
    assert identity.public_instance_slug is None
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", identity.run_id)


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


def test_validate_mainnet_400_requires_dedicated_live_flag_and_caps() -> None:
    with pytest.raises(ValueError, match="ALLOW_MAINNET_400_LIVE=true"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            allow_mainnet_400_live=False,
            live_instance_id=MAINNET_400_LIVE_INSTANCE_ID,
            enforce_pilot_caps=True,
            max_equity=450,
            max_order_notional=50,
            target_gross_cap=4.0,
            leverage=5,
        )

    _validate_v16a_live_mode(
        dry_run=False,
        testnet=False,
        strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
        allow_mainnet_pilot_live=False,
        allow_mainnet_400_live=True,
        live_instance_id=MAINNET_400_LIVE_INSTANCE_ID,
        enforce_pilot_caps=True,
        max_equity=450,
        max_order_notional=50,
        target_gross_cap=4.0,
        leverage=5,
    )

    with pytest.raises(ValueError, match="MAX_EQUITY"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_400_live=True,
            live_instance_id=MAINNET_400_LIVE_INSTANCE_ID,
            enforce_pilot_caps=True,
            max_equity=501,
            max_order_notional=50,
            target_gross_cap=4.0,
            leverage=5,
        )


def test_validate_mainnet_pilot_enforced_caps() -> None:
    _validate_v16a_live_mode(
        dry_run=False,
        testnet=False,
        strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
        allow_mainnet_pilot_live=True,
        enforce_pilot_caps=True,
        max_equity=200,
        max_order_notional=50,
        target_gross_cap=4.0,
        leverage=5,
    )

    with pytest.raises(ValueError, match="MAX_EQUITY"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=None,
            max_order_notional=50,
            target_gross_cap=4.0,
            leverage=5,
        )
    with pytest.raises(ValueError, match="MAX_ORDER_NOTIONAL"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=200,
            max_order_notional=None,
            target_gross_cap=4.0,
            leverage=5,
        )
    with pytest.raises(ValueError, match="MAX_ORDER_NOTIONAL"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=200,
            max_order_notional=51,
            target_gross_cap=4.0,
            leverage=5,
        )
    _validate_v16a_live_mode(
        dry_run=False,
        testnet=False,
        strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
        allow_mainnet_pilot_live=True,
        enforce_pilot_caps=True,
        max_equity=200,
        max_order_notional=None,
        target_gross_cap=4.0,
        leverage=5,
        allow_uncapped_orders=True,
    )

    with pytest.raises(ValueError, match="MAX_ORDER_NOTIONAL"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=200,
            max_order_notional=51,
            target_gross_cap=4.0,
            leverage=5,
            allow_uncapped_orders=True,
        )

    with pytest.raises(ValueError, match="TARGET_GROSS_CAP"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=200,
            max_order_notional=50,
            target_gross_cap=4.01,
            leverage=5,
        )
    with pytest.raises(ValueError, match="HL_LEVERAGE"):
        _validate_v16a_live_mode(
            dry_run=False,
            testnet=False,
            strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
            allow_mainnet_pilot_live=True,
            enforce_pilot_caps=True,
            max_equity=200,
            max_order_notional=50,
            target_gross_cap=4.0,
            leverage=6,
        )


def test_validate_mainnet_non_dry_run_rejects_non_pilot_profiles() -> None:
    _validate_mainnet_non_dry_run_profile(
        dry_run=False,
        testnet=False,
        strategy_profile=V16A_MAINNET_PILOT_PROFILE.slug,
    )
    _validate_mainnet_non_dry_run_profile(
        dry_run=True,
        testnet=False,
        strategy_profile="v10g-engine-6h",
    )
    with pytest.raises(ValueError, match="STRATEGY_PROFILE=v16a-mainnet-pilot"):
        _validate_mainnet_non_dry_run_profile(
            dry_run=False,
            testnet=False,
            strategy_profile="v10g-engine-6h",
        )


def test_parse_symbols_normalizes_comma_separated_list() -> None:
    assert _parse_symbols("btc, ETH,, sol ") == ["BTC", "ETH", "SOL"]
    assert _parse_symbols("") is None


def test_parse_hl_network_rejects_unknown_values() -> None:
    assert _parse_hl_network("testnet") is True
    assert _parse_hl_network(" mainnet ") is False
    with pytest.raises(ValueError, match="HL_NETWORK"):
        _parse_hl_network("prod")


def test_suppress_secret_bearing_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    _suppress_secret_bearing_http_logs()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_mainnet_preflight_report_fails_on_target_or_path_errors(tmp_path) -> None:
    good = {
        "paths": {"status": "ok"},
        "target": {"status": "ok"},
        "symbols": [{"symbol": "BTC", "exists": True}],
    }
    assert _report_has_errors(good) is False

    target_error = good | {"target": {"status": "error", "error": "stale cache"}}
    assert _report_has_errors(target_error) is True

    path_error = good | {"paths": {"status": "error", "error": "readonly"}}
    assert _report_has_errors(path_error) is True

    missing_symbol = good | {"symbols": [{"symbol": "BTC", "exists": False}]}
    assert _report_has_errors(missing_symbol) is True

    path_report = _check_runtime_paths(
        tmp_path / "state" / "engine-state.json",
        tmp_path / "journal" / "mainnet-pilot",
    )
    assert path_report["status"] == "ok"
    assert (tmp_path / "state").exists()
    assert (tmp_path / "journal" / "mainnet-pilot").exists()
