from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping

import pytest
from executor.run_bootstrap_live_instance import _bootstrap, _build_params


@dataclass
class RecordedExecute:
    sql: str
    params: Mapping[str, Any]


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[RecordedExecute] = []

    def execute(self, query: str, params: Mapping[str, Any] | None = None):
        self.calls.append(RecordedExecute(query, params or {}))
        return None


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "strategy_slug": "cta-forge",
        "strategy_name": "CTA Forge",
        "profile_id": None,
        "profile_slug": "v16a-mainnet-pilot",
        "profile_version": "",
        "account_id": "hl-mainnet-400-01",
        "exchange": "hyperliquid",
        "network": "mainnet",
        "account_label": None,
        "account_address": "0xABCDEF1234567890",
        "live_instance_id": "mainnet-400-01",
        "public_instance_slug": "mainnet-400-01",
        "display_name": None,
        "mode": "mainnet_pilot",
        "status": "paused",
        "public_status": "hidden",
        "public_enabled": False,
        "default_public_instance": False,
        "risk_config_json": '{"max_equity": "500"}',
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_params_keeps_only_non_secret_identity_fields() -> None:
    params = _build_params(_args())

    assert params["live_instance_id"] == "mainnet-400-01"
    assert params["profile_id"] == "v16a-mainnet-pilot"
    assert params["account_label"] == "hl-mainnet-400-01"
    assert params["address_prefix"] == "0xABCDEF12"
    assert params["address_hash"]
    assert "0xABCDEF1234567890" not in json.dumps(params)
    assert json.loads(params["risk_config_json"]) == {"max_equity": "500"}


def test_build_params_requires_core_identity() -> None:
    with pytest.raises(ValueError, match="LIVE_INSTANCE_ID"):
        _build_params(_args(live_instance_id=" "))

    with pytest.raises(ValueError, match="STRATEGY_PROFILE"):
        _build_params(_args(profile_slug=None))

    with pytest.raises(ValueError, match="EXCHANGE_ACCOUNT_ID"):
        _build_params(_args(account_id=None))


def test_build_params_rejects_non_object_risk_config() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _build_params(_args(risk_config_json="[]"))


def test_bootstrap_writes_minimal_reference_rows() -> None:
    conn = FakeConnection()
    params = _build_params(_args(status="active", public_enabled=True))

    _bootstrap(conn, params)

    joined_sql = "\n".join(call.sql.lower() for call in conn.calls)
    assert "insert into strategies" in joined_sql
    assert "insert into strategy_profiles" in joined_sql
    assert "insert into exchange_accounts" in joined_sql
    assert "insert into live_instances" in joined_sql
    assert "insert into public_dashboard_instances" in joined_sql
    live_call = next(
        call for call in conn.calls if "insert into live_instances" in call.sql.lower()
    )
    assert live_call.params["live_instance_id"] == "mainnet-400-01"
    assert live_call.params["status"] == "active"
    assert live_call.params["public_enabled"] is True
