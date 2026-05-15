from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_live_persistence_migrations_are_in_docker_build_context() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "infra/db/migrations/001_live_persistence.sql",
        ],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 1
