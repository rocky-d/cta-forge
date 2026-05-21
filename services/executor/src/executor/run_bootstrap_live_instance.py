"""Bootstrap minimal PostgreSQL reference rows for one live instance.

This command writes only non-secret identity metadata. It does not store private
keys, full wallet addresses, or live order permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any


def _optional_text(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _address_hash(value: str | None) -> str | None:
    stripped = _optional_text(value)
    if stripped is None:
        return None
    return hashlib.sha256(stripped.lower().encode()).hexdigest()


def _address_prefix(value: str | None) -> str | None:
    stripped = _optional_text(value)
    if stripped is None:
        return None
    return stripped[:10]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap non-secret DB rows for one cta-forge live instance."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--strategy-slug", default="cta-forge")
    parser.add_argument("--strategy-name", default="CTA Forge")
    parser.add_argument("--profile-id", default=os.environ.get("STRATEGY_PROFILE_ID"))
    parser.add_argument("--profile-slug", default=os.environ.get("STRATEGY_PROFILE"))
    parser.add_argument("--profile-version", default="")
    parser.add_argument("--account-id", default=os.environ.get("EXCHANGE_ACCOUNT_ID"))
    parser.add_argument("--exchange", default="hyperliquid")
    parser.add_argument("--network", default=os.environ.get("HL_NETWORK", "mainnet"))
    parser.add_argument(
        "--account-label", default=os.environ.get("EXCHANGE_ACCOUNT_LABEL")
    )
    parser.add_argument(
        "--account-address", default=os.environ.get("HL_ACCOUNT_ADDRESS")
    )
    parser.add_argument(
        "--live-instance-id", default=os.environ.get("LIVE_INSTANCE_ID")
    )
    parser.add_argument(
        "--public-instance-slug", default=os.environ.get("PUBLIC_INSTANCE_SLUG")
    )
    parser.add_argument(
        "--display-name", default=os.environ.get("PUBLIC_INSTANCE_DISPLAY_NAME")
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "testnet_live", "mainnet_pilot", "mainnet_live"],
        default=os.environ.get("LIVE_INSTANCE_MODE", "mainnet_pilot"),
    )
    parser.add_argument(
        "--status",
        choices=["active", "paused", "retired"],
        default=os.environ.get("LIVE_INSTANCE_STATUS", "paused"),
    )
    parser.add_argument(
        "--public-status",
        choices=["live", "stale", "paused", "retired", "hidden"],
        default=os.environ.get("PUBLIC_INSTANCE_STATUS", "hidden"),
    )
    parser.add_argument("--public-enabled", action="store_true")
    parser.add_argument("--default-public-instance", action="store_true")
    parser.add_argument(
        "--risk-config-json", default=os.environ.get("RISK_CONFIG_JSON", "{}")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _required(name: str, value: str | None) -> str:
    stripped = _optional_text(value)
    if stripped is None:
        raise ValueError(f"{name} is required")
    return stripped


def _load_risk_config(value: str) -> dict[str, Any]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("--risk-config-json must be a JSON object")
    return parsed


def _build_params(args: argparse.Namespace) -> dict[str, Any]:
    profile_slug = _required("--profile-slug/STRATEGY_PROFILE", args.profile_slug)
    profile_id = _optional_text(args.profile_id) or profile_slug
    account_id = _required("--account-id/EXCHANGE_ACCOUNT_ID", args.account_id)
    live_instance_id = _required(
        "--live-instance-id/LIVE_INSTANCE_ID", args.live_instance_id
    )
    public_slug = _optional_text(args.public_instance_slug)
    display_name = _optional_text(args.display_name) or public_slug or live_instance_id
    return {
        "strategy_slug": _required("--strategy-slug", args.strategy_slug),
        "strategy_name": _required("--strategy-name", args.strategy_name),
        "profile_id": profile_id,
        "profile_slug": profile_slug,
        "profile_version": args.profile_version or "",
        "account_id": account_id,
        "exchange": _required("--exchange", args.exchange),
        "network": _required("--network", args.network),
        "account_label": _optional_text(args.account_label) or account_id,
        "address_hash": _address_hash(args.account_address),
        "address_prefix": _address_prefix(args.account_address),
        "live_instance_id": live_instance_id,
        "public_instance_slug": public_slug,
        "display_name": display_name,
        "mode": args.mode,
        "status": args.status,
        "public_enabled": bool(args.public_enabled),
        "public_status": args.public_status,
        "is_default": bool(args.default_public_instance),
        "risk_config_json": json.dumps(_load_risk_config(args.risk_config_json)),
    }


def _bootstrap(conn: Any, params: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into strategies (slug, name)
        values (%(strategy_slug)s, %(strategy_name)s)
        on conflict (slug) do update set name = excluded.name
        """,
        params,
    )
    conn.execute(
        """
        insert into strategy_profiles (profile_id, strategy_slug, slug, version)
        values (%(profile_id)s, %(strategy_slug)s, %(profile_slug)s, %(profile_version)s)
        on conflict (profile_id) do update set
            strategy_slug = excluded.strategy_slug,
            slug = excluded.slug,
            version = excluded.version
        """,
        params,
    )
    conn.execute(
        """
        insert into exchange_accounts (
            account_id, exchange, network, account_label, address_hash, address_prefix
        )
        values (
            %(account_id)s, %(exchange)s, %(network)s, %(account_label)s,
            %(address_hash)s, %(address_prefix)s
        )
        on conflict (account_id) do update set
            exchange = excluded.exchange,
            network = excluded.network,
            account_label = excluded.account_label,
            address_hash = coalesce(excluded.address_hash, exchange_accounts.address_hash),
            address_prefix = coalesce(excluded.address_prefix, exchange_accounts.address_prefix)
        """,
        params,
    )
    conn.execute(
        """
        insert into live_instances (
            live_instance_id, strategy_slug, profile_id, account_id,
            public_instance_slug, mode, status, risk_config_json, public_enabled
        )
        values (
            %(live_instance_id)s, %(strategy_slug)s, %(profile_id)s, %(account_id)s,
            %(public_instance_slug)s, %(mode)s, %(status)s,
            %(risk_config_json)s::jsonb, %(public_enabled)s
        )
        on conflict (live_instance_id) do update set
            strategy_slug = excluded.strategy_slug,
            profile_id = excluded.profile_id,
            account_id = excluded.account_id,
            public_instance_slug = excluded.public_instance_slug,
            mode = excluded.mode,
            status = excluded.status,
            risk_config_json = excluded.risk_config_json,
            public_enabled = excluded.public_enabled
        """,
        params,
    )
    if params["public_instance_slug"]:
        conn.execute(
            """
            insert into public_dashboard_instances (
                strategy_slug, public_instance_slug, live_instance_id,
                display_name, status, is_default
            )
            values (
                %(strategy_slug)s, %(public_instance_slug)s, %(live_instance_id)s,
                %(display_name)s, %(public_status)s, %(is_default)s
            )
            on conflict (strategy_slug, public_instance_slug) do update set
                live_instance_id = excluded.live_instance_id,
                display_name = excluded.display_name,
                status = excluded.status,
                is_default = excluded.is_default,
                updated_at = now()
            """,
            params,
        )


def main() -> None:
    args = _parse_args()
    try:
        database_url = _optional_text(args.database_url)
        params = _build_params(args)
        if not args.dry_run and database_url is None:
            raise ValueError("--database-url/DATABASE_URL is required")
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        sys.exit(2)

    safe_summary = {
        key: value
        for key, value in params.items()
        if key not in {"address_hash", "risk_config_json"}
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "params": safe_summary}, indent=2))
        return

    try:
        import psycopg

        assert database_url is not None
        with psycopg.connect(database_url, autocommit=True) as conn:
            _bootstrap(conn, params)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        sys.exit(1)

    print(json.dumps({"status": "ok", "instance": safe_summary}, indent=2))


if __name__ == "__main__":
    main()
