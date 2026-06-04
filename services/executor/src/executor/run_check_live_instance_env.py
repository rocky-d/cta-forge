"""Validate non-secret live-instance env guardrails.

This read-only CLI is intentionally narrower than exchange preflight: it does
not connect to Hyperliquid or PostgreSQL. Use it before deployment to catch
unsafe per-instance env defaults and path/cap mistakes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TextIO

from .live import V16A_PROFILE_SLUG
from .profiles.v16a_badscore_overlay import V16A_MAINNET_PILOT_PROFILE
from .run_live import (
    MAINNET_400_LIVE_INSTANCE_ID,
    _is_truthy,
    _read_mainnet_caps_from_env,
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "message": self.message}


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = _parse_args(argv)
    source_env = env if env is not None else os.environ
    checks = validate_live_instance_env(
        source_env,
        allow_non_dry_run=args.allow_non_dry_run,
        require_secrets=args.require_secrets,
    )
    ok = all(check.ok for check in checks)
    payload = {
        "status": "ok" if ok else "error",
        "live_instance_id": _optional_text(source_env.get("LIVE_INSTANCE_ID")),
        "checks": [check.to_dict() for check in checks],
    }
    json.dump(payload, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    if not ok:
        print("live instance env invalid", file=stderr)
        return 2
    return 0


def validate_live_instance_env(
    env: Mapping[str, str],
    *,
    allow_non_dry_run: bool = False,
    require_secrets: bool = False,
) -> list[CheckResult]:
    """Return non-secret env safety checks for one live instance."""

    instance_id = _optional_text(env.get("LIVE_INSTANCE_ID"))
    checks = [
        _check_required(env, "LIVE_INSTANCE_ID"),
        _check_required(env, "PERSISTENCE_BACKEND"),
        _check_required(env, "HL_NETWORK"),
        _check_required(env, "STRATEGY_PROFILE"),
        _check_required(env, "STATE_FILE"),
        _check_required(env, "JOURNAL_DIR"),
        _check_required(env, "DATA_DIR"),
        _check_bool_false_unless_allowed(
            env,
            "DRY_RUN",
            allow_non_dry_run=allow_non_dry_run,
        ),
    ]
    if instance_id == MAINNET_400_LIVE_INSTANCE_ID:
        checks.extend(
            _validate_mainnet_400_env(
                env,
                allow_non_dry_run=allow_non_dry_run,
                require_secrets=require_secrets,
            )
        )
    else:
        checks.append(
            CheckResult(
                "known_instance_profile",
                True,
                "no instance-specific checks beyond generic guardrails",
            )
        )
    return checks


def _validate_mainnet_400_env(
    env: Mapping[str, str],
    *,
    allow_non_dry_run: bool,
    require_secrets: bool,
) -> list[CheckResult]:
    caps = _read_mainnet_caps_from_env(env)
    return [
        _check_equals(
            "persistence_backend", env.get("PERSISTENCE_BACKEND"), "postgres"
        ),
        _check_truthy(
            "allow_postgres_source_of_truth",
            env.get("ALLOW_POSTGRES_SOURCE_OF_TRUTH"),
        ),
        _check_equals("hl_network", env.get("HL_NETWORK"), "mainnet"),
        _check_in(
            "strategy_profile",
            env.get("STRATEGY_PROFILE"),
            {V16A_MAINNET_PILOT_PROFILE.slug, V16A_PROFILE_SLUG},
        ),
        _check_live_allow_flag(
            "allow_live",
            env.get("ALLOW_LIVE"),
            allow_non_dry_run=allow_non_dry_run,
        ),
        _check_status_default("live_instance_status", env.get("LIVE_INSTANCE_STATUS")),
        _check_equals(
            "public_instance_status", env.get("PUBLIC_INSTANCE_STATUS"), "hidden"
        ),
        _check_path_mentions_instance("state_file", env.get("STATE_FILE")),
        _check_path_mentions_instance("journal_dir", env.get("JOURNAL_DIR")),
        _check_cap(
            "max_equity",
            env.get("MAX_EQUITY"),
            caps["equity"],
        ),
        _check_cap(
            "max_order_notional",
            env.get("MAX_ORDER_NOTIONAL"),
            caps["order_notional"],
        ),
        _check_cap(
            "target_gross_cap",
            env.get("TARGET_GROSS_CAP"),
            caps["gross_cap"],
        ),
        _check_int_cap(
            "hl_leverage",
            env.get("HL_LEVERAGE"),
            caps["leverage"],
        ),
        _check_secrets(env, require_secrets=require_secrets),
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate non-secret per-live-instance env guardrails."
    )
    parser.add_argument(
        "--allow-non-dry-run",
        action="store_true",
        help="Allow DRY_RUN=false for an explicitly approved live promotion check.",
    )
    parser.add_argument(
        "--require-secrets",
        action="store_true",
        help="Require private exchange secrets to be present without printing them.",
    )
    return parser.parse_args(argv)


def _optional_text(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _check_required(env: Mapping[str, str], key: str) -> CheckResult:
    ok = _optional_text(env.get(key)) is not None
    return CheckResult(
        key.lower(), ok, f"{key} is {'configured' if ok else 'required'}"
    )


def _check_bool_false_unless_allowed(
    env: Mapping[str, str],
    key: str,
    *,
    allow_non_dry_run: bool,
) -> CheckResult:
    dry_run = _is_truthy(env.get(key))
    ok = dry_run or allow_non_dry_run
    return CheckResult(
        key.lower(),
        ok,
        f"{key}=true for prep/dry-run" if ok else f"{key}=false requires approval flag",
    )


def _check_equals(name: str, value: str | None, expected: str) -> CheckResult:
    actual = _optional_text(value)
    ok = actual == expected
    return CheckResult(name, ok, f"expected {expected}, got {actual or '-'}")


def _check_in(name: str, value: str | None, allowed: set[str]) -> CheckResult:
    actual = _optional_text(value)
    ok = actual in allowed
    allowed_text = ", ".join(sorted(allowed))
    return CheckResult(name, ok, f"expected one of {allowed_text}, got {actual or '-'}")


def _check_truthy(name: str, value: str | None) -> CheckResult:
    ok = _is_truthy(value)
    return CheckResult(name, ok, "enabled" if ok else "must be enabled")


def _check_live_allow_flag(
    name: str,
    value: str | None,
    *,
    allow_non_dry_run: bool,
) -> CheckResult:
    enabled = _is_truthy(value)
    ok = enabled if allow_non_dry_run else not enabled
    if allow_non_dry_run:
        message = (
            "enabled for approved live promotion"
            if ok
            else "must be enabled for live promotion"
        )
    else:
        message = (
            "disabled for prep/dry-run"
            if ok
            else "must remain disabled for prep/dry-run"
        )
    return CheckResult(name, ok, message)


def _check_status_default(name: str, value: str | None) -> CheckResult:
    actual = _optional_text(value)
    ok = actual in {"paused", "active"}
    return CheckResult(
        name,
        ok,
        f"expected paused during prep or active during dry-run, got {actual or '-'}",
    )


def _check_path_mentions_instance(name: str, value: str | None) -> CheckResult:
    actual = _optional_text(value)
    if actual is None:
        return CheckResult(name, False, "path is required")
    path_name = PurePosixPath(actual).as_posix()
    ok = MAINNET_400_LIVE_INSTANCE_ID in path_name
    return CheckResult(
        name, ok, "path is instance-specific" if ok else "path must include instance id"
    )


def _check_cap(name: str, value: str | None, maximum: float) -> CheckResult:
    actual = _optional_text(value)
    try:
        parsed = float(actual or "")
    except ValueError:
        return CheckResult(name, False, f"must be a number <= {maximum:g}")
    ok = parsed <= maximum
    return CheckResult(
        name, ok, f"{parsed:g} <= {maximum:g}" if ok else f"{parsed:g} > {maximum:g}"
    )


def _check_int_cap(name: str, value: str | None, maximum: int) -> CheckResult:
    actual = _optional_text(value)
    try:
        parsed = int(actual or "")
    except ValueError:
        return CheckResult(name, False, f"must be an integer <= {maximum}")
    ok = parsed <= maximum
    return CheckResult(
        name, ok, f"{parsed} <= {maximum}" if ok else f"{parsed} > {maximum}"
    )


def _check_secrets(env: Mapping[str, str], *, require_secrets: bool) -> CheckResult:
    if not require_secrets:
        return CheckResult("secrets", True, "not required for non-secret env check")
    missing = [
        key
        for key in ["HL_PRIVATE_KEY", "HL_ACCOUNT_ADDRESS"]
        if _optional_text(env.get(key)) is None
    ]
    ok = not missing
    return CheckResult(
        "secrets", ok, "present" if ok else f"missing: {', '.join(missing)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
