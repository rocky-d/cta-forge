from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "docker-compose.prod.yml"
ENV_EXAMPLE = ROOT / "infra/env.mainnet-400-01.example"


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_second_live_executor_is_profile_gated_and_isolated() -> None:
    compose = COMPOSE.read_text()

    assert "executor-mainnet-400-01:" in compose
    assert "container_name: cta-forge-executor-mainnet-400-01" in compose
    assert 'profiles: ["mainnet-400-01"]' in compose
    assert "env_file: .env.mainnet-400-01" in compose
    assert "./data-mainnet-400-01:/app/data" in compose
    assert "./state-mainnet-400-01:/app/state" in compose
    assert "./journal-mainnet-400-01:/app/journal" in compose


def test_second_live_env_example_starts_paused_hidden_and_dry_run() -> None:
    env = _env_example()

    assert env["LIVE_INSTANCE_ID"] == "mainnet-400-01"
    assert env["LIVE_INSTANCE_STATUS"] == "paused"
    assert env["PUBLIC_INSTANCE_STATUS"] == "hidden"
    assert env["DRY_RUN"] == "true"
    assert env["ALLOW_LIVE"] == "false"
    assert env["REQUIRE_LIVE_INSTANCE_LOCK_AVAILABLE"] == "true"


def test_second_live_env_example_uses_db_primary_and_bounded_caps() -> None:
    env = _env_example()

    assert env["PERSISTENCE_BACKEND"] == "postgres"
    assert env["ALLOW_POSTGRES_SOURCE_OF_TRUTH"] == "true"
    assert env["HL_NETWORK"] == "mainnet"
    assert env["STRATEGY_PROFILE"] == "v16a-mainnet-pilot"
    assert float(env["MAX_EQUITY"]) <= 500
    assert float(env["TARGET_GROSS_CAP"]) <= 4
    assert int(env["HL_LEVERAGE"]) <= 5


def test_second_live_env_example_keeps_secret_values_blank() -> None:
    env = _env_example()

    for key in [
        "POSTGRES_PASSWORD",
        "HL_PRIVATE_KEY",
        "HL_ACCOUNT_ADDRESS",
        "TG_BOT_TOKEN",
        "TG_CHAT_ID",
        "LARK_WEBHOOK_URL",
        "LARK_WEBHOOK_SECRET",
    ]:
        assert env[key] == ""

    assert "REPLACE_ME" in env["DATABASE_URL"]


def test_second_live_env_example_is_plain_shell_compatible() -> None:
    for raw_line in ENV_EXAMPLE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assert key.isidentifier()
        assert " " not in value
        assert "<" not in value
        assert ">" not in value
