"""Print a safe live persistence runtime-config preflight.

This CLI is read-only and does not construct stores or connect to PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from typing import Sequence, TextIO

from .live_persistence_runtime import load_live_persistence_runtime_config


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = _parse_args(argv)
    source_env = env if env is not None else os.environ
    try:
        config = load_live_persistence_runtime_config(
            source_env,
            run_id=args.run_id,
        )
    except ValueError as exc:
        print(f"live persistence config invalid: {exc}", file=stderr)
        return 2
    json.dump(config.to_safe_dict(), stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate live persistence runtime config without connecting to DB."
    )
    parser.add_argument(
        "--run-id",
        help="Runtime-generated run id to validate instead of RUN_ID env.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
