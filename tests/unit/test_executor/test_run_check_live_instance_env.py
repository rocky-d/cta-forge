from __future__ import annotations

import io
import json

from executor import run_check_live_instance_env as runner


def _mainnet_400_env(**overrides: str) -> dict[str, str]:
    env = {
        "LIVE_INSTANCE_ID": "mainnet-400-01",
        "PERSISTENCE_BACKEND": "postgres",
        "ALLOW_POSTGRES_SOURCE_OF_TRUTH": "true",
        "HL_NETWORK": "mainnet",
        "STRATEGY_PROFILE": "v16a-mainnet-pilot",
        "LIVE_INSTANCE_STATUS": "paused",
        "PUBLIC_INSTANCE_STATUS": "hidden",
        "DRY_RUN": "true",
        "ALLOW_LIVE": "false",
        "DATA_DIR": "/app/data",
        "STATE_FILE": "/app/state/engine-state-mainnet-400-01.json",
        "JOURNAL_DIR": "/app/journal/mainnet-400-01",
        "MAX_EQUITY": "450",
        "MAX_ORDER_NOTIONAL": "50",
        "TARGET_GROSS_CAP": "1.00",
        "HL_LEVERAGE": "5",
        "HL_PRIVATE_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
    }
    env.update(overrides)
    return env


def _run(env: dict[str, str], *args: str) -> tuple[int, dict, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = runner.main(list(args), env=env, stdout=stdout, stderr=stderr)
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


def test_check_live_instance_env_accepts_safe_mainnet_400_prep_env() -> None:
    code, payload, stderr = _run(_mainnet_400_env())

    assert code == 0
    assert stderr == ""
    assert payload["status"] == "ok"
    assert payload["live_instance_id"] == "mainnet-400-01"
    assert all(check["ok"] for check in payload["checks"])


def test_check_live_instance_env_rejects_non_dry_run_without_approval_flag() -> None:
    code, payload, stderr = _run(_mainnet_400_env(DRY_RUN="false"))

    assert code == 2
    assert payload["status"] == "error"
    assert "invalid" in stderr
    dry_run_check = next(
        check for check in payload["checks"] if check["name"] == "dry_run"
    )
    assert dry_run_check["ok"] is False

    approved_code, approved_payload, _ = _run(
        _mainnet_400_env(DRY_RUN="false", ALLOW_LIVE="true"),
        "--allow-non-dry-run",
    )
    assert approved_code == 0
    assert approved_payload["status"] == "ok"

    still_blocked_code, still_blocked_payload, _ = _run(
        _mainnet_400_env(DRY_RUN="false", ALLOW_LIVE="false"),
        "--allow-non-dry-run",
    )
    assert still_blocked_code == 2
    live_flag_check = next(
        check
        for check in still_blocked_payload["checks"]
        if check["name"] == "allow_live"
    )
    assert live_flag_check["ok"] is False


def test_check_live_instance_env_rejects_enabled_live_flag_during_prep() -> None:
    code, payload, _ = _run(_mainnet_400_env(ALLOW_LIVE="true"))

    assert code == 2
    live_flag_check = next(
        check for check in payload["checks"] if check["name"] == "allow_live"
    )
    assert live_flag_check["ok"] is False


def test_check_live_instance_env_rejects_bad_values_and_paths() -> None:
    code, payload, _ = _run(
        _mainnet_400_env(
            MAX_EQUITY="abc",
            MAX_ORDER_NOTIONAL="abc",
            STATE_FILE="/app/state/engine-state.json",
        )
    )

    assert code == 2
    failed = {check["name"] for check in payload["checks"] if not check["ok"]}
    assert {"max_equity", "max_order_notional", "state_file"}.issubset(failed)


def test_check_live_instance_env_can_require_private_secrets_without_printing_them() -> (
    None
):
    code, payload, _ = _run(_mainnet_400_env(), "--require-secrets")

    assert code == 2
    secrets_check = next(
        check for check in payload["checks"] if check["name"] == "secrets"
    )
    assert secrets_check["ok"] is False

    ok_code, ok_payload, _ = _run(
        _mainnet_400_env(
            HL_PRIVATE_KEY="super-secret-private-key",
            HL_ACCOUNT_ADDRESS="0xabc123",
        ),
        "--require-secrets",
    )
    assert ok_code == 0
    rendered = json.dumps(ok_payload)
    assert "super-secret-private-key" not in rendered
    assert "0xabc123" not in rendered


def test_check_live_instance_env_accepts_generic_instance_with_basic_fields() -> None:
    code, payload, stderr = _run(
        {
            "LIVE_INSTANCE_ID": "mainnet-pilot",
            "PERSISTENCE_BACKEND": "postgres",
            "HL_NETWORK": "mainnet",
            "STRATEGY_PROFILE": "v16a-mainnet-pilot",
            "STATE_FILE": "/app/state/engine-state-mainnet-pilot.json",
            "JOURNAL_DIR": "/app/journal/mainnet-pilot",
            "DATA_DIR": "/app/data",
            "DRY_RUN": "true",
        }
    )

    assert code == 0
    assert stderr == ""
    assert payload["status"] == "ok"
