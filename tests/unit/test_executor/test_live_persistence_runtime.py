from __future__ import annotations

import pytest
from executor.live_persistence_runtime import (
    LivePersistenceRuntimeConfig,
    load_live_persistence_runtime_config,
    validate_live_persistence_runtime_config,
)


def test_persistence_runtime_config_defaults_to_file() -> None:
    config = load_live_persistence_runtime_config({})

    assert config.backend == "file"
    assert config.uses_database is False
    assert config.to_safe_dict() == {
        "backend": "file",
        "database_url_configured": False,
        "live_instance_id_configured": False,
        "run_id_configured": False,
        "shadow_failure_policy": "warn",
        "allow_postgres_source_of_truth": False,
    }


def test_persistence_runtime_config_loads_dual_mode() -> None:
    config = load_live_persistence_runtime_config(
        {
            "PERSISTENCE_BACKEND": " dual ",
            "DATABASE_URL": "postgresql://user:secret@example/db",
            "LIVE_INSTANCE_ID": " cta-forge-mainnet-pilot-01 ",
            "RUN_ID": " run-001 ",
            "SHADOW_WRITE_FAILURE_POLICY": " raise ",
            "ALLOW_POSTGRES_SOURCE_OF_TRUTH": "false",
        }
    )

    assert config.backend == "dual"
    assert config.uses_database is True
    assert config.database_url == "postgresql://user:secret@example/db"
    assert config.live_instance_id == "cta-forge-mainnet-pilot-01"
    assert config.run_id == "run-001"
    assert config.shadow_failure_policy == "raise"
    assert config.allow_postgres_source_of_truth is False
    assert config.to_safe_dict()["database_url_configured"] is True
    assert "postgresql://" not in str(config.to_safe_dict())
    assert "secret" not in str(config.to_safe_dict())


def test_persistence_runtime_config_can_use_explicit_runtime_run_id() -> None:
    config = load_live_persistence_runtime_config(
        {
            "PERSISTENCE_BACKEND": "dual",
            "DATABASE_URL": "postgresql:///cta",
            "LIVE_INSTANCE_ID": "instance-1",
            "RUN_ID": "env-run",
        },
        run_id="generated-run",
    )

    assert config.run_id == "generated-run"


@pytest.mark.parametrize("missing", ["DATABASE_URL", "LIVE_INSTANCE_ID", "RUN_ID"])
def test_db_modes_require_connection_and_identity(missing: str) -> None:
    env = {
        "PERSISTENCE_BACKEND": "dual",
        "DATABASE_URL": "postgresql:///cta",
        "LIVE_INSTANCE_ID": "instance-1",
        "RUN_ID": "run-1",
    }
    env.pop(missing)

    with pytest.raises(ValueError, match=f"requires {missing}"):
        load_live_persistence_runtime_config(env)


def test_postgres_mode_requires_source_of_truth_allow_flag() -> None:
    env = {
        "PERSISTENCE_BACKEND": "postgres",
        "DATABASE_URL": "postgresql:///cta",
        "LIVE_INSTANCE_ID": "instance-1",
        "RUN_ID": "run-1",
    }

    with pytest.raises(ValueError, match="ALLOW_POSTGRES_SOURCE_OF_TRUTH=true"):
        load_live_persistence_runtime_config(env)

    config = load_live_persistence_runtime_config(
        {**env, "ALLOW_POSTGRES_SOURCE_OF_TRUTH": "true"}
    )
    assert config.backend == "postgres"
    assert config.allow_postgres_source_of_truth is True


def test_invalid_shadow_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="SHADOW_WRITE_FAILURE_POLICY"):
        load_live_persistence_runtime_config(
            {
                "PERSISTENCE_BACKEND": "dual",
                "DATABASE_URL": "postgresql:///cta",
                "LIVE_INSTANCE_ID": "instance-1",
                "RUN_ID": "run-1",
                "SHADOW_WRITE_FAILURE_POLICY": "ignore",
            }
        )


def test_validate_file_mode_allows_absent_db_settings() -> None:
    validate_live_persistence_runtime_config(LivePersistenceRuntimeConfig())
