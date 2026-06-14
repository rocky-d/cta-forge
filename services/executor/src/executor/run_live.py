"""CLI entry point for running the live engine."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from exchange.hyperliquid import HyperliquidAdapter

from .live import LiveEngine, V10G_PROFILE_SLUG, V16A_PROFILE_SLUG
from .live_persistence_runtime import load_live_persistence_runtime_config
from .live_persistence_store_factory import build_live_persistence_stores
from .profiles.v16a_badscore_overlay import (
    V16A_MAINNET_PILOT_PROFILE,
    V16aOnlineTargetStrategy,
    validate_core_phase_hours,
)
from .notify import (
    LarkNotifier,
    MultiNotifier,
    NullNotifier,
    TelegramNotifier,
    _Notifier,
)


def _is_truthy(value: str | None) -> bool:
    """Return whether an environment-style flag is enabled."""
    return (value or "").lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class RuntimeIdentity:
    """Non-secret runtime identity attached to live journal records."""

    live_instance_id: str | None
    public_instance_slug: str | None
    run_id: str


def _optional_env_text(value: str | None) -> str | None:
    """Return a stripped env value, or None when absent/blank."""
    stripped = (value or "").strip()
    return stripped or None


def _new_run_id() -> str:
    """Generate a compact human-readable process run identifier."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _load_runtime_identity(env: Mapping[str, str] = os.environ) -> RuntimeIdentity:
    """Load optional live instance identity from environment variables."""
    return RuntimeIdentity(
        live_instance_id=_optional_env_text(env.get("LIVE_INSTANCE_ID")),
        public_instance_slug=_optional_env_text(env.get("PUBLIC_INSTANCE_SLUG")),
        run_id=_optional_env_text(env.get("RUN_ID")) or _new_run_id(),
    )


MAINNET_400_LIVE_INSTANCE_ID = "mainnet-400-01"


def _validate_v16a_live_mode(
    *,
    dry_run: bool,
    testnet: bool,
    strategy_profile: str = V16A_PROFILE_SLUG,
    allow_live: bool = False,
    allow_testnet_live: bool = False,
) -> None:
    """Validate v16a live-mode guardrails before constructing the engine."""
    if dry_run:
        return
    if strategy_profile == V16A_MAINNET_PILOT_PROFILE.slug:
        if testnet:
            msg = f"{V16A_MAINNET_PILOT_PROFILE.slug} requires HL_NETWORK=mainnet"
            raise ValueError(msg)
        if not allow_live:
            msg = "Non-dry-run requires ALLOW_LIVE=true"
            raise ValueError(msg)
        return
    if not testnet:
        msg = f"{V16A_PROFILE_SLUG} non-dry-run is only allowed on HL_NETWORK=testnet"
        raise ValueError(msg)
    if not allow_testnet_live:
        msg = f"{V16A_PROFILE_SLUG} testnet live requires ALLOW_V16A_TESTNET_LIVE=true"
        raise ValueError(msg)


def _validate_mainnet_non_dry_run_profile(
    *, dry_run: bool, testnet: bool, strategy_profile: str
) -> None:
    """Allow real mainnet orders only through the explicit pilot profile."""
    if (
        not dry_run
        and not testnet
        and strategy_profile != V16A_MAINNET_PILOT_PROFILE.slug
    ):
        msg = (
            "mainnet non-dry-run requires "
            f"STRATEGY_PROFILE={V16A_MAINNET_PILOT_PROFILE.slug}"
        )
        raise ValueError(msg)


def _parse_symbols(value: str | None) -> list[str] | None:
    """Parse comma-separated live symbols from env."""
    if not value:
        return None
    symbols = [symbol.strip().upper() for symbol in value.split(",")]
    return [symbol for symbol in symbols if symbol]


def _parse_optional_float(value: str | None) -> float | None:
    """Parse optional positive float env values."""
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def _parse_hl_network(value: str | None) -> bool:
    """Parse HL_NETWORK and return whether to use testnet."""
    network = (value or "testnet").strip().lower()
    if network == "testnet":
        return True
    if network == "mainnet":
        return False
    msg = "HL_NETWORK must be one of: testnet, mainnet"
    raise ValueError(msg)


def _suppress_secret_bearing_http_logs() -> None:
    """Avoid logging notification URLs that can contain webhook or bot secrets."""
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _build_notifier() -> _Notifier:
    """Build notifier from env vars. Multiple backends stack via MultiNotifier."""
    notifiers: list[_Notifier] = []

    tg_token = os.environ.get("TG_BOT_TOKEN", "")
    tg_chat = os.environ.get("TG_CHAT_ID", "")
    if tg_token and tg_chat:
        notifiers.append(TelegramNotifier(tg_token, tg_chat))

    lark_url = os.environ.get("LARK_WEBHOOK_URL", "")
    lark_secret = os.environ.get("LARK_WEBHOOK_SECRET") or None
    if lark_url:
        notifiers.append(LarkNotifier(lark_url, secret=lark_secret))

    if not notifiers:
        return NullNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return MultiNotifier(notifiers)


def main() -> None:
    """Start the live trading engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _suppress_secret_bearing_http_logs()

    pk = os.environ.get("HL_PRIVATE_KEY", "")
    addr = os.environ.get("HL_ACCOUNT_ADDRESS", "")
    if not pk or not addr:
        logging.error("Set HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS env vars")
        sys.exit(1)

    try:
        testnet = _parse_hl_network(os.environ.get("HL_NETWORK", "testnet"))
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    dry_run = _is_truthy(os.environ.get("DRY_RUN", "false"))
    identity = _load_runtime_identity(os.environ)
    try:
        persistence_config = load_live_persistence_runtime_config(
            os.environ,
            run_id=identity.run_id,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    logging.info("Live persistence config: %s", persistence_config.to_safe_dict())
    state_file = os.environ.get("STATE_FILE", "engine-state.json")
    journal_dir = os.environ.get("JOURNAL_DIR", "journal")
    data_dir = os.environ.get("DATA_DIR", "data")
    clean_start = _is_truthy(os.environ.get("CLEAN_START", "false"))
    strategy_profile = os.environ.get("STRATEGY_PROFILE", V10G_PROFILE_SLUG)
    min_order_notional = float(os.environ.get("MIN_ORDER_NOTIONAL", "10"))
    max_order_notional = _parse_optional_float(os.environ.get("MAX_ORDER_NOTIONAL"))
    min_equity = _parse_optional_float(os.environ.get("MIN_EQUITY"))
    min_available_balance = _parse_optional_float(
        os.environ.get("MIN_AVAILABLE_BALANCE")
    )
    max_equity = _parse_optional_float(os.environ.get("MAX_EQUITY"))
    target_gross_cap = float(os.environ.get("TARGET_GROSS_CAP", "1"))
    target_scale = float(os.environ.get("TARGET_SCALE", "1"))
    gate_rolling_years = float(os.environ.get("GATE_ROLLING_WINDOW", "3"))
    soft_dd_limit = float(os.environ.get("SOFT_DD_LIMIT", "0.20"))
    hard_dd_limit = float(os.environ.get("HARD_DD_LIMIT", "0.30"))
    leverage = int(os.environ.get("HL_LEVERAGE", str(LiveEngine.DEFAULT_LEVERAGE)))
    symbols = _parse_symbols(os.environ.get("LIVE_SYMBOLS"))
    v16a_max_staleness_hours = float(os.environ.get("V16A_MAX_STALENESS_HOURS", "8"))
    v16a_core_phase_hours = validate_core_phase_hours(
        int(os.environ.get("V16A_CORE_PHASE_HOURS", "2"))
    )
    allow_v16a_testnet_live = _is_truthy(os.environ.get("ALLOW_V16A_TESTNET_LIVE"))
    allow_live = _is_truthy(os.environ.get("ALLOW_LIVE"))
    try:
        _validate_mainnet_non_dry_run_profile(
            dry_run=dry_run, testnet=testnet, strategy_profile=strategy_profile
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    target_strategy = None
    if strategy_profile in {V16A_PROFILE_SLUG, V16A_MAINNET_PILOT_PROFILE.slug}:
        try:
            _validate_v16a_live_mode(
                dry_run=dry_run,
                testnet=testnet,
                strategy_profile=strategy_profile,
                allow_live=allow_live,
                allow_testnet_live=allow_v16a_testnet_live,
            )
        except ValueError as exc:
            logging.error("%s", exc)
            sys.exit(1)
        target_strategy = V16aOnlineTargetStrategy(
            data_dir,
            max_staleness=timedelta(hours=v16a_max_staleness_hours),
            target_scale=target_scale,
            gross_cap=target_gross_cap,
            core_phase_hours=v16a_core_phase_hours,
            gate_rolling_years=gate_rolling_years,
            profile=V16A_MAINNET_PILOT_PROFILE
            if strategy_profile == V16A_MAINNET_PILOT_PROFILE.slug
            else V16aOnlineTargetStrategy.profile,
        )
    elif strategy_profile == V10G_PROFILE_SLUG:
        pass
    else:
        logging.error("Unknown STRATEGY_PROFILE=%s", strategy_profile)
        sys.exit(1)

    notifier = _build_notifier()

    logging.info(
        "Runtime identity: live_instance_id=%s, public_instance_slug=%s, run_id=%s",
        identity.live_instance_id or "-",
        identity.public_instance_slug or "-",
        identity.run_id,
    )

    try:
        persistence_stores = build_live_persistence_stores(
            persistence_config,
            journal_dir=journal_dir,
            state_file=state_file,
            public_instance_slug=identity.public_instance_slug,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.error(
            "Failed to initialize live persistence stores: %s",
            exc.__class__.__name__,
        )
        sys.exit(1)

    adapter = HyperliquidAdapter(pk, addr, testnet=testnet)
    engine = LiveEngine(
        adapter,
        symbols=symbols,
        dry_run=dry_run,
        state_file=state_file,
        journal_dir=journal_dir,
        data_dir=data_dir,
        notify=notifier,
        clean_start=clean_start,
        strategy_profile=strategy_profile,
        target_strategy=target_strategy,
        min_order_notional=min_order_notional,
        max_order_notional=max_order_notional,
        min_equity=min_equity,
        min_available_balance=min_available_balance,
        max_equity=max_equity,
        leverage=leverage,
        live_instance_id=identity.live_instance_id,
        run_id=identity.run_id,
        public_instance_slug=identity.public_instance_slug,
        journal=persistence_stores.journal,
        state_store=persistence_stores.state_store,
        soft_dd_limit=soft_dd_limit,
        hard_dd_limit=hard_dd_limit,
    )

    async def run() -> None:
        try:
            await engine.start()
        except KeyboardInterrupt:
            await engine.stop()
        finally:
            await adapter.close()
            persistence_stores.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
