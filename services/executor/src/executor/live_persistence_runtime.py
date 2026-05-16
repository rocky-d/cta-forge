"""Runtime persistence configuration helpers.

This module prepares the future DB-backed runtime boundary without wiring the
live executor to PostgreSQL yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .live_persistence_dual import (
    PersistenceBackend,
    ShadowWriteFailurePolicy,
    parse_persistence_backend,
)

POSTGRES_SOURCE_OF_TRUTH_ALLOW_ENV = "ALLOW_POSTGRES_SOURCE_OF_TRUTH"
SHADOW_WRITE_FAILURE_POLICY_ENV = "LIVE_PERSISTENCE_SHADOW_FAILURE_POLICY"
LEGACY_SHADOW_WRITE_FAILURE_POLICY_ENV = "SHADOW_WRITE_FAILURE_POLICY"


def _optional_env_text(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class LivePersistenceRuntimeConfig:
    """Non-secret persistence runtime config.

    ``database_url`` is stored for connection construction, but ``to_safe_dict``
    intentionally avoids returning it.
    """

    backend: PersistenceBackend = "file"
    database_url: str | None = None
    live_instance_id: str | None = None
    run_id: str | None = None
    shadow_failure_policy: ShadowWriteFailurePolicy = "warn"
    allow_postgres_source_of_truth: bool = False

    @property
    def uses_database(self) -> bool:
        return self.backend in {"dual", "postgres"}

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "database_url_configured": bool(self.database_url),
            "live_instance_id_configured": bool(self.live_instance_id),
            "run_id_configured": bool(self.run_id),
            "shadow_failure_policy": self.shadow_failure_policy,
            "allow_postgres_source_of_truth": self.allow_postgres_source_of_truth,
        }


def load_live_persistence_runtime_config(
    env: Mapping[str, str],
    *,
    run_id: str | None = None,
) -> LivePersistenceRuntimeConfig:
    """Load future live persistence config from environment-like mapping.

    Safe default is ``backend=file``. DB modes fail closed unless the required
    non-secret identity and connection settings are present. ``postgres`` mode
    also requires an explicit source-of-truth allow flag.
    """

    backend = parse_persistence_backend(env.get("PERSISTENCE_BACKEND"))
    database_url = _optional_env_text(env.get("DATABASE_URL"))
    live_instance_id = _optional_env_text(env.get("LIVE_INSTANCE_ID"))
    resolved_run_id = _optional_env_text(run_id) or _optional_env_text(
        env.get("RUN_ID")
    )
    shadow_failure_policy = _parse_shadow_failure_policy(
        env.get(SHADOW_WRITE_FAILURE_POLICY_ENV)
        or env.get(LEGACY_SHADOW_WRITE_FAILURE_POLICY_ENV)
    )
    allow_postgres_source_of_truth = _is_truthy(
        env.get(POSTGRES_SOURCE_OF_TRUTH_ALLOW_ENV)
    )

    config = LivePersistenceRuntimeConfig(
        backend=backend,
        database_url=database_url,
        live_instance_id=live_instance_id,
        run_id=resolved_run_id,
        shadow_failure_policy=shadow_failure_policy,
        allow_postgres_source_of_truth=allow_postgres_source_of_truth,
    )
    validate_live_persistence_runtime_config(config)
    return config


def validate_live_persistence_runtime_config(
    config: LivePersistenceRuntimeConfig,
) -> None:
    """Fail closed on unsafe or incomplete DB runtime settings."""

    if config.backend == "file":
        return
    missing = [
        name
        for name, value in {
            "DATABASE_URL": config.database_url,
            "LIVE_INSTANCE_ID": config.live_instance_id,
            "RUN_ID": config.run_id,
        }.items()
        if value is None
    ]
    if missing:
        msg = f"PERSISTENCE_BACKEND={config.backend} requires " + ", ".join(missing)
        raise ValueError(msg)
    if config.backend == "postgres" and not config.allow_postgres_source_of_truth:
        msg = (
            "PERSISTENCE_BACKEND=postgres requires "
            f"{POSTGRES_SOURCE_OF_TRUTH_ALLOW_ENV}=true"
        )
        raise ValueError(msg)


def _parse_shadow_failure_policy(value: str | None) -> ShadowWriteFailurePolicy:
    normalized = (value or "warn").strip().lower()
    if normalized == "warn":
        return "warn"
    if normalized == "raise":
        return "raise"
    msg = f"{SHADOW_WRITE_FAILURE_POLICY_ENV} must be one of: warn, raise"
    raise ValueError(msg)
