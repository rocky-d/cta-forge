"""Cache-only v16a phase comparison snapshot CLI.

This module records phase side-by-side diagnostics without running a live engine
tick and without refreshing Binance data. It reads the existing parquet cache and
Hyperliquid account/prices only, so it is suitable for forward evidence
collection when the production/live target remains phase 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from core.constants import V10G_SYMBOLS, V10G_TESTNET_EXCLUDED
from exchange.hyperliquid import HyperliquidAdapter

from .run_shadow_tick import (
    ShadowTickConfig,
    load_shadow_tick_config,
    record_phase_comparison,
)

logger = logging.getLogger(__name__)


def phase_shadow_symbols(config: ShadowTickConfig) -> set[str]:
    """Return the account universe used for phase shadow comparison."""
    symbols = set(config.symbols or list(V10G_SYMBOLS))
    if config.testnet:
        symbols -= set(V10G_TESTNET_EXCLUDED)
    return symbols


async def run_phase_shadow_snapshot(config: ShadowTickConfig) -> dict[str, object]:
    """Record one cache-only phase comparison snapshot."""
    if config.compare_core_phase_hours is None:
        msg = "Set V16A_COMPARE_CORE_PHASE_HOURS for phase shadow snapshots"
        raise ValueError(msg)

    allowed_symbols = phase_shadow_symbols(config)
    if not allowed_symbols:
        msg = "No symbols available for phase shadow snapshot"
        raise ValueError(msg)

    adapter = HyperliquidAdapter(
        config.private_key,
        config.account_address,
        testnet=config.testnet,
    )
    try:
        record = await record_phase_comparison(
            config=config,
            exchange=adapter,
            allowed_symbols=allowed_symbols,
        )
    finally:
        await adapter.close()

    if record is None:  # Defensive; checked above.
        msg = "Phase comparison was not recorded"
        raise RuntimeError(msg)
    return {
        "status": "ok",
        "journal_dir": config.phase_comparison_journal_dir,
        "base_core_phase_hours": record["base_core_phase_hours"],
        "compare_core_phase_hours": record["compare_core_phase_hours"],
        "target_timestamps": {
            phase: summary["target_ts"] for phase, summary in record["phases"].items()
        },
        "metrics": record["metrics"],
        "phase_order_counts": {
            phase: summary["n_orders"] for phase, summary in record["phases"].items()
        },
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        config = load_shadow_tick_config()
        summary = asyncio.run(run_phase_shadow_snapshot(config))
    except Exception:
        logger.exception("Phase shadow snapshot failed")
        sys.exit(1)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
