"""CLI entry point for running the live engine."""

from __future__ import annotations

import asyncio
import logging
import os
import sys


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

    from exchange.hyperliquid import HyperliquidAdapter

    from .live import LiveEngine, TelegramNotifier, _NullNotifier

    # Setup Telegram notifications if configured
    tg_token = os.environ.get("TG_BOT_TOKEN", "")
    tg_chat = os.environ.get("TG_CHAT_ID", "")
    notifier = (
        TelegramNotifier(tg_token, tg_chat) if tg_token and tg_chat else _NullNotifier()
    )

    adapter = HyperliquidAdapter(pk, addr, testnet=testnet)
    engine = LiveEngine(
        adapter,
        dry_run=dry_run,
        state_file=state_file,
        notify=notifier,
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
