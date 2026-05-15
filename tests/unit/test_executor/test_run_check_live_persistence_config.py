from __future__ import annotations

import io
import json

from executor import run_check_live_persistence_config as runner


def test_check_live_persistence_config_cli_outputs_safe_file_default() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = runner.main([], env={}, stdout=stdout, stderr=stderr)

    assert code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "backend": "file",
        "database_url_configured": False,
        "live_instance_id_configured": False,
        "run_id_configured": False,
        "shadow_failure_policy": "warn",
        "allow_postgres_source_of_truth": False,
    }


def test_check_live_persistence_config_cli_outputs_safe_dual_config() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {
        "PERSISTENCE_BACKEND": "dual",
        "DATABASE_URL": "postgresql://user:secret@example/db",
        "LIVE_INSTANCE_ID": "instance-1",
        "RUN_ID": "run-1",
    }

    code = runner.main([], env=env, stdout=stdout, stderr=stderr)

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload == {
        "backend": "dual",
        "database_url_configured": True,
        "live_instance_id_configured": True,
        "run_id_configured": True,
        "shadow_failure_policy": "warn",
        "allow_postgres_source_of_truth": False,
    }
    assert "secret" not in stdout.getvalue()
    assert "postgresql://" not in stdout.getvalue()


def test_check_live_persistence_config_cli_uses_explicit_run_id() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {
        "PERSISTENCE_BACKEND": "dual",
        "DATABASE_URL": "postgresql:///cta",
        "LIVE_INSTANCE_ID": "instance-1",
    }

    code = runner.main(
        ["--run-id", "runtime-run"],
        env=env,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert json.loads(stdout.getvalue())["run_id_configured"] is True


def test_check_live_persistence_config_cli_returns_two_for_invalid_config() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {
        "PERSISTENCE_BACKEND": "postgres",
        "DATABASE_URL": "postgresql:///cta",
        "LIVE_INSTANCE_ID": "instance-1",
        "RUN_ID": "run-1",
    }

    code = runner.main([], env=env, stdout=stdout, stderr=stderr)

    assert code == 2
    assert stdout.getvalue() == ""
    assert "ALLOW_POSTGRES_SOURCE_OF_TRUTH=true" in stderr.getvalue()
