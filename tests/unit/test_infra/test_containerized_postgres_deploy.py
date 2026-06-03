from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_executor_image_includes_db_migrations() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "COPY infra/db/ infra/db/" in dockerfile


def test_production_compose_has_private_postgres_without_published_ports() -> None:
    compose = (ROOT / "docker-compose.prod.yml").read_text()

    assert "postgres:17-alpine" in compose
    assert "container_name: cta-forge-postgres" in compose
    assert "cta_forge_postgres_data:/var/lib/postgresql/data" in compose
    assert "db-migrate:" in compose
    assert "executor.run_apply_live_persistence_migrations" in compose
    assert "ports:" not in compose


def test_deploy_migrates_db_before_recreating_executor_without_downing_db() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()

    assert "docker compose $COMPOSE_FILES up -d postgres" in workflow
    assert "--profile db-maintenance run --rm db-migrate" in workflow
    assert "up -d --no-deps executor-live" in workflow
    assert "docker compose $COMPOSE_FILES down --remove-orphans" not in workflow
