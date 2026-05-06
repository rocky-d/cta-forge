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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping, Any

from exchange.hyperliquid import HyperliquidAdapter

from .journal import TradeJournal
from .live import LiveEngine, V16A_PROFILE_SLUG
from .live_target import fetch_target_prices, normalize_target_weights
from .notify import NullNotifier
from .profiles.v16a_badscore_overlay import (
    V16aOnlineTargetStrategy,
    validate_core_phase_hours,
)
from .run_live import _parse_hl_network, _parse_optional_float, _parse_symbols
from .targeting import TargetOrder, weights_to_orders

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
    max_order_notional: float | None
    max_staleness: timedelta
    target_scale: float
    gross_cap: float
    core_phase_hours: int
    compare_core_phase_hours: int | None
    phase_comparison_journal_dir: str
    symbols: list[str] | None


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

    compare_phase = env.get("V16A_COMPARE_CORE_PHASE_HOURS")
    return ShadowTickConfig(
        private_key=private_key,
        account_address=account_address,
        testnet=_parse_hl_network(env.get("HL_NETWORK", "testnet")),
        data_dir=env.get("DATA_DIR", "data"),
        journal_dir=env.get("JOURNAL_DIR", "journal/shadow-v16a"),
        state_file=env.get("STATE_FILE", "engine-state-shadow.json"),
        min_order_notional=float(env.get("MIN_ORDER_NOTIONAL", "10")),
        max_order_notional=_parse_optional_float(env.get("MAX_ORDER_NOTIONAL")),
        max_staleness=timedelta(hours=float(env.get("V16A_MAX_STALENESS_HOURS", "8"))),
        target_scale=float(env.get("TARGET_SCALE", "1")),
        gross_cap=float(env.get("TARGET_GROSS_CAP", "1")),
        core_phase_hours=validate_core_phase_hours(
            int(env.get("V16A_CORE_PHASE_HOURS", "0"))
        ),
        compare_core_phase_hours=(
            validate_core_phase_hours(int(compare_phase))
            if compare_phase is not None
            else None
        ),
        phase_comparison_journal_dir=env.get(
            "PHASE_COMPARISON_JOURNAL_DIR", "journal/phase-shadow"
        ),
        symbols=_parse_symbols(env.get("LIVE_SYMBOLS")),
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_phase_comparisons(journal_dir: str | Path) -> list[dict[str, Any]]:
    """Load recorded side-by-side phase shadow diagnostics."""
    path = Path(journal_dir) / "phase_comparisons.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _order_to_dict(order: TargetOrder) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "qty": round(order.qty, 8),
        "current_weight": round(order.current_weight, 8),
        "target_weight": round(order.target_weight, 8),
        "delta_weight": round(order.delta_weight, 8),
        "delta_notional": round(order.delta_notional, 4),
        "reduce_only": order.reduce_only,
    }


def phase_diff_metrics(
    base_weights: Mapping[str, float], compare_weights: Mapping[str, float]
) -> dict[str, float | int]:
    """Return compact target-difference metrics for two normalized portfolios."""
    symbols = sorted(set(base_weights) | set(compare_weights))
    base = [float(base_weights.get(symbol, 0.0)) for symbol in symbols]
    compare = [float(compare_weights.get(symbol, 0.0)) for symbol in symbols]
    l1 = sum(abs(a - b) for a, b in zip(base, compare, strict=True))
    max_abs = max((abs(a - b) for a, b in zip(base, compare, strict=True)), default=0.0)
    base_norm = sum(a * a for a in base) ** 0.5
    compare_norm = sum(b * b for b in compare) ** 0.5
    dot = sum(a * b for a, b in zip(base, compare, strict=True))
    cosine = (
        dot / (base_norm * compare_norm) if base_norm > 0 and compare_norm > 0 else 1.0
    )
    flips = sum(
        1
        for a, b in zip(base, compare, strict=True)
        if abs(a) > 1e-12 and abs(b) > 1e-12 and a * b < 0
    )
    return {
        "l1": round(l1, 8),
        "max_abs": round(max_abs, 8),
        "cosine": round(cosine, 8),
        "sign_flips": flips,
    }


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


async def record_phase_comparison(
    *,
    config: ShadowTickConfig,
    exchange: HyperliquidAdapter,
    allowed_symbols: set[str],
) -> dict[str, Any] | None:
    """Record read-only phase-0-vs-candidate diagnostics from live cache/account."""
    if config.compare_core_phase_hours is None:
        return None

    now = datetime.now(tz=UTC)
    account = await exchange.get_account_state()
    prices = await fetch_target_prices(exchange, allowed_symbols)
    positions = {
        pos.symbol: float(pos.size)
        for pos in account.positions
        if pos.symbol in allowed_symbols
    }
    phase_summaries: dict[str, dict[str, Any]] = {}
    normalized_by_phase: dict[int, dict[str, float]] = {}

    for phase in (config.core_phase_hours, config.compare_core_phase_hours):
        strategy = V16aOnlineTargetStrategy(
            config.data_dir,
            max_staleness=config.max_staleness,
            target_scale=config.target_scale,
            gross_cap=config.gross_cap,
            core_phase_hours=phase,
        )
        target = strategy.target(now)
        normalized, ignored = normalize_target_weights(
            dict(target.weights), allowed_symbols
        )
        orders = weights_to_orders(
            positions,
            prices,
            float(account.equity),
            normalized,
            min_notional=config.min_order_notional,
            max_notional=config.max_order_notional,
        )
        normalized_by_phase[phase] = normalized
        phase_summaries[str(phase)] = {
            "target_ts": target.timestamp.isoformat(),
            "staleness_seconds": round((now - target.timestamp).total_seconds(), 3),
            "target_gross": round(target.gross, 6),
            "normalized_gross": round(sum(abs(w) for w in normalized.values()), 6),
            "ignored_gross": round(sum(abs(w) for w in ignored.values()), 6),
            "n_orders": len(orders),
            "weights": {
                k: round(v, 8) for k, v in normalized.items() if abs(v) > 1e-12
            },
            "ignored_weights": {
                k: round(v, 8) for k, v in ignored.items() if abs(v) > 1e-12
            },
            "orders": [_order_to_dict(order) for order in orders],
        }

    record = {
        "ts": now.isoformat(),
        "base_core_phase_hours": config.core_phase_hours,
        "compare_core_phase_hours": config.compare_core_phase_hours,
        "equity": round(float(account.equity), 6),
        "allowed_symbols": sorted(allowed_symbols),
        "metrics": phase_diff_metrics(
            normalized_by_phase[config.core_phase_hours],
            normalized_by_phase[config.compare_core_phase_hours],
        ),
        "phases": phase_summaries,
    }
    _append_jsonl(
        Path(config.phase_comparison_journal_dir) / "phase_comparisons.jsonl", record
    )
    return record


async def run_shadow_tick(config: ShadowTickConfig) -> dict[str, Any]:
    """Run one v16a target-mode tick and return the latest target summary."""
    strategy = V16aOnlineTargetStrategy(
        config.data_dir,
        max_staleness=config.max_staleness,
        target_scale=config.target_scale,
        gross_cap=config.gross_cap,
        core_phase_hours=config.core_phase_hours,
    )
    adapter = HyperliquidAdapter(
        config.private_key,
        config.account_address,
        testnet=config.testnet,
    )
    try:
        engine = LiveEngine(
            adapter,
            symbols=config.symbols,
            dry_run=True,
            state_file=config.state_file,
            journal_dir=config.journal_dir,
            data_dir=config.data_dir,
            notify=NullNotifier(),
            target_strategy=strategy,
            min_order_notional=config.min_order_notional,
            max_order_notional=config.max_order_notional,
        )
        await engine._tick()
        comparison = await record_phase_comparison(
            config=config,
            exchange=adapter,
            allowed_symbols=set(engine._symbols),  # noqa: SLF001
        )
    finally:
        await adapter.close()

    summary = summarize_latest_target(config.journal_dir)
    summary["core_phase_hours"] = config.core_phase_hours
    if comparison is not None:
        summary["phase_comparison"] = {
            "journal_dir": config.phase_comparison_journal_dir,
            "base_core_phase_hours": comparison["base_core_phase_hours"],
            "compare_core_phase_hours": comparison["compare_core_phase_hours"],
            "metrics": comparison["metrics"],
        }
    return summary


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
