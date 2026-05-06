"""Read-only Hyperliquid mainnet preflight report.

This command intentionally performs no exchange writes. It initializes the
mainnet adapter, reads account/order/market metadata, and optionally computes
current v16a pilot target diagnostics from the local parquet cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from core.constants import V10G_SYMBOLS
from exchange.hyperliquid import HyperliquidAdapter

from .live_target import normalize_target_weights
from .profiles.v16a_badscore_overlay import (
    V16A_MAINNET_PILOT_PROFILE,
    V16aOnlineTargetStrategy,
    validate_core_phase_hours,
)
from .run_live import (
    _parse_optional_float,
    _parse_symbols,
    _validate_mainnet_pilot_caps,
)
from .targeting import weights_to_orders


def _decimal_to_float(value: Decimal) -> float:
    return float(value)


def _round_up_size(raw_qty: float, sz_decimals: int) -> float:
    factor = 10**sz_decimals
    return math.ceil(raw_qty * factor) / factor


async def _build_report() -> dict[str, Any]:
    pk = os.environ.get("HL_PRIVATE_KEY", "")
    addr = os.environ.get("HL_ACCOUNT_ADDRESS", "")
    if not pk or not addr:
        raise ValueError("Set HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS env vars")

    if os.environ.get("HL_NETWORK", "mainnet") != "mainnet":
        raise ValueError("run_mainnet_preflight requires HL_NETWORK=mainnet")

    data_dir = os.environ.get("DATA_DIR", "data")
    state_file = Path(os.environ.get("STATE_FILE", "engine-state.json"))
    journal_dir = Path(os.environ.get("JOURNAL_DIR", "journal"))
    symbols = _parse_symbols(os.environ.get("LIVE_SYMBOLS")) or list(V10G_SYMBOLS)
    min_order_notional = float(os.environ.get("MIN_ORDER_NOTIONAL", "10"))
    target_gross_cap = float(os.environ.get("TARGET_GROSS_CAP", "0.2"))
    target_scale = float(os.environ.get("TARGET_SCALE", "1"))
    v16a_max_staleness_hours = float(os.environ.get("V16A_MAX_STALENESS_HOURS", "8"))
    v16a_core_phase_hours = validate_core_phase_hours(
        int(os.environ.get("V16A_CORE_PHASE_HOURS", "0"))
    )
    max_equity = _parse_optional_float(os.environ.get("MAX_EQUITY"))
    max_order_notional = _parse_optional_float(os.environ.get("MAX_ORDER_NOTIONAL"))
    leverage = int(os.environ.get("HL_LEVERAGE", "5"))
    _validate_mainnet_pilot_caps(
        max_equity=max_equity,
        max_order_notional=max_order_notional,
        target_gross_cap=target_gross_cap,
        leverage=leverage,
    )

    adapter = HyperliquidAdapter(pk, addr, testnet=False)
    try:
        account = await adapter.get_account_state()
        open_orders = await adapter.get_open_orders()
        meta = await adapter._run_sync(adapter._info.meta)  # noqa: SLF001
        meta_by_symbol = {asset.get("name", ""): asset for asset in meta["universe"]}

        symbol_rows = []
        prices: dict[str, float] = {}
        for symbol in symbols:
            row: dict[str, Any] = {"symbol": symbol, "exists": symbol in meta_by_symbol}
            asset = meta_by_symbol.get(symbol, {})
            row["sz_decimals"] = asset.get("szDecimals")
            row["max_leverage"] = asset.get("maxLeverage")
            if not row["exists"]:
                symbol_rows.append(row)
                continue
            try:
                snap = await adapter.get_market_snapshot(symbol)
                mark = float(snap.mark_price or snap.mid_price)
                spread_bps = float(
                    (snap.best_ask - snap.best_bid) / snap.mid_price * 10_000
                )
                sz_decimals = int(row["sz_decimals"] or 0)
                min_qty = _round_up_size(min_order_notional / mark, sz_decimals)
                row.update(
                    {
                        "best_bid": _decimal_to_float(snap.best_bid),
                        "best_ask": _decimal_to_float(snap.best_ask),
                        "mark_price": mark,
                        "spread_bps": spread_bps,
                        "min_order_qty_estimate": min_qty,
                        "min_order_notional_estimate": min_qty * mark,
                    }
                )
                prices[symbol] = mark
            except Exception as exc:  # pragma: no cover - live diagnostic path
                row["error"] = str(exc)
            symbol_rows.append(row)

        path_report = _check_runtime_paths(state_file, journal_dir)

        target_report: dict[str, Any] = {"status": "not_computed"}
        try:
            strategy = V16aOnlineTargetStrategy(
                data_dir,
                max_staleness=timedelta(hours=v16a_max_staleness_hours),
                target_scale=target_scale,
                gross_cap=target_gross_cap,
                core_phase_hours=v16a_core_phase_hours,
                profile=V16A_MAINNET_PILOT_PROFILE,
            )
            target = strategy.target(datetime.now(tz=UTC))
            normalized, ignored = normalize_target_weights(
                dict(target.weights), set(symbols)
            )
            positions = {
                pos.symbol: float(pos.size)
                for pos in account.positions
                if pos.symbol in set(symbols)
            }
            orders = weights_to_orders(
                positions,
                prices,
                float(account.equity),
                normalized,
                min_notional=min_order_notional,
                max_notional=max_order_notional,
            )
            target_report = {
                "status": "ok",
                "target_ts": target.timestamp.isoformat(),
                "target_gross": target.gross,
                "normalized_gross": sum(abs(w) for w in normalized.values()),
                "weights": normalized,
                "ignored_weights": ignored,
                "orders": [order.__dict__ for order in orders],
            }
        except Exception as exc:  # pragma: no cover - live diagnostic path
            target_report = {"status": "error", "error": str(exc)}

        return {
            "ts": datetime.now(tz=UTC).isoformat(),
            "network": "mainnet",
            "address_prefix": addr[:10],
            "caps": {
                "max_equity": max_equity,
                "max_order_notional": max_order_notional,
                "target_gross_cap": target_gross_cap,
                "v16a_core_phase_hours": v16a_core_phase_hours,
                "leverage": leverage,
            },
            "account": {
                "equity": _decimal_to_float(account.equity),
                "available_balance": _decimal_to_float(account.available_balance),
                "total_margin_used": _decimal_to_float(account.total_margin_used),
                "positions": [
                    {
                        "symbol": pos.symbol,
                        "size": _decimal_to_float(pos.size),
                        "entry_price": _decimal_to_float(pos.entry_price),
                        "unrealized_pnl": _decimal_to_float(pos.unrealized_pnl),
                        "leverage": pos.leverage,
                    }
                    for pos in account.positions
                ],
                "open_orders_count": len(open_orders),
            },
            "open_orders": open_orders,
            "paths": path_report,
            "symbols": symbol_rows,
            "target": target_report,
        }
    finally:
        await adapter.close()


def _check_runtime_paths(state_file: Path, journal_dir: Path) -> dict[str, Any]:
    """Verify runtime persistence paths are writable without changing state."""
    checks: dict[str, Any] = {
        "state_file": str(state_file),
        "journal_dir": str(journal_dir),
        "status": "ok",
    }
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        journal_dir.mkdir(parents=True, exist_ok=True)
        state_probe = state_file.parent / ".preflight-write-test"
        journal_probe = journal_dir / ".preflight-write-test"
        state_probe.write_text("ok")
        journal_probe.write_text("ok")
        state_probe.unlink(missing_ok=True)
        journal_probe.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - deployment environment dependent
        checks["status"] = "error"
        checks["error"] = str(exc)
    return checks


def _report_has_errors(report: dict[str, Any]) -> bool:
    """Return whether a read-only mainnet preflight report should fail deploy."""
    if report.get("status") == "error":
        return True
    if report.get("paths", {}).get("status") != "ok":
        return True
    if report.get("target", {}).get("status") != "ok":
        return True
    for symbol in report.get("symbols", []):
        if not symbol.get("exists", False) or symbol.get("error"):
            return True
    return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    try:
        report = asyncio.run(_build_report())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(report, indent=2, sort_keys=True))
    if _report_has_errors(report):
        sys.exit(1)


if __name__ == "__main__":
    main()
