"""CLI entry point for running the live engine."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from exchange.hyperliquid import HyperliquidAdapter

from .live import LiveEngine, V10G_PROFILE_SLUG, V16A_PROFILE_SLUG
from .notify import (
    LarkNotifier,
    MultiNotifier,
    NullNotifier,
    TelegramNotifier,
    _Notifier,
)


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

    pk = os.environ.get("HL_PRIVATE_KEY", "")
    addr = os.environ.get("HL_ACCOUNT_ADDRESS", "")
    if not pk or not addr:
        logging.error("Set HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS env vars")
        sys.exit(1)

    testnet = os.environ.get("HL_NETWORK", "testnet") == "testnet"
    dry_run = os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")
    state_file = os.environ.get("STATE_FILE", "engine-state.json")
    journal_dir = os.environ.get("JOURNAL_DIR", "journal")
    data_dir = os.environ.get("DATA_DIR", "data")
    clean_start = os.environ.get("CLEAN_START", "false").lower() in ("true", "1", "yes")
    strategy_profile = os.environ.get("STRATEGY_PROFILE", V10G_PROFILE_SLUG)
    min_order_notional = float(os.environ.get("MIN_ORDER_NOTIONAL", "10"))

    if strategy_profile == V16A_PROFILE_SLUG:
        logging.error(
            "%s is not wired as a live target provider yet; use the research "
            "backtest path until shadow/testnet integration is complete",
            V16A_PROFILE_SLUG,
        )
        sys.exit(1)
    if strategy_profile != V10G_PROFILE_SLUG:
        logging.error("Unknown STRATEGY_PROFILE=%s", strategy_profile)
        sys.exit(1)

    notifier = _build_notifier()

    adapter = HyperliquidAdapter(pk, addr, testnet=testnet)
    engine = LiveEngine(
        adapter,
        dry_run=dry_run,
        state_file=state_file,
        journal_dir=journal_dir,
        data_dir=data_dir,
        notify=notifier,
        clean_start=clean_start,
        strategy_profile=strategy_profile,
        min_order_notional=min_order_notional,
    )

    async def run() -> None:
        try:
            await engine.start()
        except KeyboardInterrupt:
            await engine.stop()
        finally:
            await adapter.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
