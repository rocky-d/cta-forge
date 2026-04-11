"""Live execution engine (Hyperliquid integration)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LiveEngine:
    """Live trading engine for Hyperliquid.

    Placeholder — will be implemented in Phase 4.
    """

    def __init__(self, private_key: str, wallet_address: str) -> None:
        self._private_key = private_key
        self._wallet_address = wallet_address
        self._running = False

    async def start(self) -> None:
        logger.info("LiveEngine starting (not implemented)")
        self._running = True

    async def stop(self) -> None:
        logger.info("LiveEngine stopping")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
