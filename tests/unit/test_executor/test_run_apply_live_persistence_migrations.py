from __future__ import annotations

import io
import json

from executor import run_apply_live_persistence_migrations as runner


def test_apply_migrations_dry_run_outputs_safe_summary() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = runner.main(
        ["--dry-run", "--database-url", "postgresql://user:secret@example/db"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["database_url_configured"] is True
    assert payload["dry_run"] is True
    assert payload["migrations"] == [
        "001_live_persistence.sql",
        "002_live_target_execution_buckets.sql",
    ]
    assert "secret" not in stdout.getvalue()
    assert "postgresql://" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_apply_migrations_requires_database_url() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = runner.main(["--dry-run"], env={}, stdout=stdout, stderr=stderr)

    assert code == 2
    assert stdout.getvalue() == ""
    assert "DATABASE_URL is required" in stderr.getvalue()
