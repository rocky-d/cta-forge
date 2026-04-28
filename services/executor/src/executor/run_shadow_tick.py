"""One-shot shadow target tick CLI.

This module runs the live target reconciliation path exactly once in dry-run mode.
It is intended for v16a shadow validation from the same executor image used by
production deploys, without enabling real order submission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Any

from exchange.hyperliquid import HyperliquidAdapter

from .journal import TradeJournal
from .live import LiveEngine, V16A_PROFILE_SLUG
from .notify import NullNotifier
from .profiles.v16a_badscore_overlay import V16aOnlineTargetStrategy

logger = logging.getLogger(__name__)

COVERAGE_WARNING_THRESHOLD = 0.20


@dataclass(frozen=True)
class ShadowTickConfig:
    """Environment-derived config for one-shot target shadow validation."""

    private_key: str
    account_address: str
    testnet: bool
    data_dir: str
    journal_dir: str
    state_file: str
    min_order_notional: float
    max_staleness: timedelta


def _is_truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "y"}


def load_shadow_tick_config(env: Mapping[str, str] = os.environ) -> ShadowTickConfig:
    """Load and validate shadow-tick config from environment variables."""
    profile = env.get("STRATEGY_PROFILE", V16A_PROFILE_SLUG)
    if profile != V16A_PROFILE_SLUG:
        msg = f"run_shadow_tick only supports STRATEGY_PROFILE={V16A_PROFILE_SLUG}"
        raise ValueError(msg)

    if not _is_truthy(env.get("DRY_RUN", "true")):
        msg = "run_shadow_tick requires DRY_RUN=true"
        raise ValueError(msg)

    private_key = env.get("HL_PRIVATE_KEY", "")
    account_address = env.get("HL_ACCOUNT_ADDRESS", "")
    if not private_key or not account_address:
        msg = "Set HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS env vars"
        raise ValueError(msg)

    return ShadowTickConfig(
        private_key=private_key,
        account_address=account_address,
        testnet=env.get("HL_NETWORK", "testnet") == "testnet",
        data_dir=env.get("DATA_DIR", "data"),
        journal_dir=env.get("JOURNAL_DIR", "journal/shadow-v16a"),
        state_file=env.get("STATE_FILE", "engine-state-shadow.json"),
        min_order_notional=float(env.get("MIN_ORDER_NOTIONAL", "10")),
        max_staleness=timedelta(hours=float(env.get("V16A_MAX_STALENESS_HOURS", "8"))),
    )


def summarize_latest_target(journal_dir: str | Path) -> dict[str, Any]:
    """Return a compact JSON-serializable summary of the latest target record."""
    journal = TradeJournal(journal_dir)
    targets = journal.load_targets()
    if not targets:
        return {
            "status": "error",
            "journal_dir": str(journal_dir),
            "message": "No target diagnostics were written",
        }

    latest = targets[-1]
    orders = latest.get("orders", [])
    ignored_weights = latest.get("ignored_weights", {})
    weights = latest.get("weights", {})
    ignored_gross_ratio = float(latest.get("ignored_gross_ratio", 0.0) or 0.0)
    warnings = []
    if ignored_gross_ratio > COVERAGE_WARNING_THRESHOLD:
        warnings.append(
            "execution coverage degraded: ignored_gross_ratio "
            f"{ignored_gross_ratio:.1%} exceeds {COVERAGE_WARNING_THRESHOLD:.0%}"
        )

    return {
        "status": "ok",
        "journal_dir": str(journal_dir),
        "profile": latest.get("profile"),
        "target_ts": latest.get("target_ts"),
        "staleness_seconds": latest.get("staleness_seconds"),
        "target_gross": latest.get("target_gross"),
        "normalized_gross": latest.get("normalized_gross"),
        "ignored_gross": latest.get("ignored_gross", 0.0),
        "ignored_gross_ratio": ignored_gross_ratio,
        "execution_coverage": latest.get("execution_coverage", 1.0),
        "n_weights": len(weights),
        "n_ignored_weights": len(ignored_weights),
        "n_orders": len(orders),
        "weights": weights,
        "ignored_weights": ignored_weights,
        "orders": orders,
        "warnings": warnings,
    }


async def run_shadow_tick(config: ShadowTickConfig) -> dict[str, Any]:
    """Run one v16a target-mode tick and return the latest target summary."""
    strategy = V16aOnlineTargetStrategy(
        config.data_dir,
        max_staleness=config.max_staleness,
    )
    adapter = HyperliquidAdapter(
        config.private_key,
        config.account_address,
        testnet=config.testnet,
    )
    try:
        engine = LiveEngine(
            adapter,
            dry_run=True,
            state_file=config.state_file,
            journal_dir=config.journal_dir,
            data_dir=config.data_dir,
            notify=NullNotifier(),
            target_strategy=strategy,
            min_order_notional=config.min_order_notional,
        )
        await engine._tick()
    finally:
        await adapter.close()

    return summarize_latest_target(config.journal_dir)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        config = load_shadow_tick_config()
        summary = asyncio.run(run_shadow_tick(config))
    except Exception:
        logger.exception("Shadow tick failed")
        sys.exit(1)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
