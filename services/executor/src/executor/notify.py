"""Notification backends for the live trading engine.

All notifiers follow the _Notifier protocol: async def send(message: str) -> None.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from lark_bots import ABot

logger = logging.getLogger(__name__)


class _Notifier:
    """Protocol-ish base for trade notifications."""

    async def send(self, message: str) -> None:
        """Send a notification message."""


class NullNotifier(_Notifier):
    """No-op notifier for when notifications aren't configured."""

    async def send(self, message: str) -> None:
        pass


class TelegramNotifier(_Notifier):
    """Send notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": message,
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Telegram API %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("Telegram notification failed: %s", e)


class LarkNotifier(_Notifier):
    """Send notifications via Lark/Feishu webhook (lark-bots)."""

    def __init__(self, webhook_url: str, secret: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._secret = secret

    async def send(self, message: str) -> None:
        try:
            async with ABot(self._webhook_url, secret=self._secret) as bot:
                await bot.asend_text(message)
        except Exception as e:
            logger.warning("Lark notification failed: %s", e)


class MultiNotifier(_Notifier):
    """Fan-out notifier: sends to all backends concurrently."""

    def __init__(self, notifiers: list[_Notifier]) -> None:
        self._notifiers = notifiers

    async def send(self, message: str) -> None:
        await asyncio.gather(
            *(n.send(message) for n in self._notifiers),
            return_exceptions=True,
        )
