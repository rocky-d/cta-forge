"""Hyperliquid exchange adapter implementation.

Wraps hyperliquid-python-sdk (sync) with async executor pattern.
Supports testnet and mainnet via config.
"""

from __future__ import annotations

import asyncio
import logging
import math
from decimal import Decimal
from typing import Any

import eth_account
from hyperliquid.api import API
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from .adapter import AccountState, MarketSnapshot, OrderResult, Position

logger = logging.getLogger(__name__)

# Timeout for SDK calls to prevent hangs
SDK_TIMEOUT: float = 15.0

# Transient error patterns (retry-safe)
TRANSIENT_PATTERNS = (
    "post-only orders allowed",
    "rate limit",
    "too many requests",
    "timeout",
    "maintenance",
    "service unavailable",
)


def _is_transient(msg: str) -> bool:
    return any(p in msg.lower() for p in TRANSIENT_PATTERNS)


def _safe_init_info(api_url: str, *, skip_ws: bool = True) -> Info:
    """Initialize Info with workaround for testnet spot_meta IndexError.

    The testnet spot_meta sometimes references token indices that don't exist.
    We pre-fetch and filter the data before passing to Info constructor.
    """
    api = API(api_url)
    spot_meta = api.post("/info", {"type": "spotMeta"})
    max_idx = len(spot_meta.get("tokens", [])) - 1
    if max_idx >= 0:
        spot_meta["universe"] = [
            u
            for u in spot_meta.get("universe", [])
            if u["tokens"][0] <= max_idx and u["tokens"][1] <= max_idx
        ]
    meta = api.post("/info", {"type": "meta"})
    return Info(api_url, skip_ws=skip_ws, meta=meta, spot_meta=spot_meta)


class HyperliquidAdapter:
    """Hyperliquid DEX adapter for CTA trading.

    Implements ExchangeAdapter protocol via structural subtyping.
    """

    def __init__(
        self,
        private_key: str,
        account_address: str,
        *,
        testnet: bool = True,
    ) -> None:
        self._testnet = testnet
        self._api_url = (
            constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        )
        self._address = account_address

        wallet = eth_account.Account.from_key(private_key)

        # Initialize SDK instances (safe init for testnet spot_meta bug)
        self._info = _safe_init_info(self._api_url, skip_ws=True)

        # Monkey-patch Info to avoid spot_meta IndexError in Exchange.__init__
        _orig_init = Info.__init__

        def _patched_init(self_info, *args, **kwargs):
            self_info.__dict__.update(self._info.__dict__)

        Info.__init__ = _patched_init
        try:
            self._exchange = Exchange(
                wallet,
                self._api_url,
                account_address=account_address,
            )
        finally:
            Info.__init__ = _orig_init

        # Serialize exchange writes to avoid nonce collisions
        self._order_lock = asyncio.Lock()

        # Cache asset metadata
        self._sz_decimals: dict[str, int] = {}
        self._asset_indices: dict[str, int] = {}

        logger.info(
            "HyperliquidAdapter initialized: testnet=%s, address=%s…",
            testnet,
            account_address[:10],
        )

    # ── helpers ──────────────────────────────────────────────────────

    async def _run_sync(
        self, func: Any, *args: Any, timeout: float = SDK_TIMEOUT
    ) -> Any:
        """Run sync SDK call in executor with timeout."""
        loop = asyncio.get_event_loop()
        async with asyncio.timeout(timeout):
            return await loop.run_in_executor(None, lambda: func(*args))

    async def _ensure_metadata(self, symbol: str) -> None:
        """Load and cache asset metadata."""
        if symbol in self._sz_decimals:
            return
        meta = await self._run_sync(self._info.meta)
        for i, asset in enumerate(meta.get("universe", [])):
            name = asset.get("name", "")
            self._sz_decimals[name] = asset.get("szDecimals", 0)
            self._asset_indices[name] = i
        if symbol not in self._sz_decimals:
            logger.warning("Symbol %s not in metadata, defaulting szDecimals=2", symbol)
            self._sz_decimals[symbol] = 2

    def _format_price(self, price: Decimal) -> str:
        """Format price with max 5 significant figures (HL requirement)."""
        p = float(price)
        if p == 0:
            return "0"
        integer_digits = max(1, int(math.log10(abs(p))) + 1)
        allowed_dec = max(0, 5 - integer_digits)
        rounded = round(p, allowed_dec)
        if allowed_dec <= 0:
            return str(int(rounded))
        s = f"{rounded:.{allowed_dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def _format_size(self, size: Decimal, symbol: str) -> float:
        """Round size to asset's szDecimals."""
        sz_dec = self._sz_decimals.get(symbol, 2)
        quant = Decimal(10) ** -sz_dec
        return float(size.quantize(quant))

    # ── ExchangeAdapter implementation ───────────────────────────────

    async def get_account_state(self) -> AccountState:
        state = await self._run_sync(self._info.user_state, self._address)
        margin = state.get("marginSummary", {})

        # Unified account: perp marginSummary.accountValue only reflects
        # perp-side margin. Spot USDC also serves as collateral, so we
        # must sum both to get the true equity.
        perp_equity = Decimal(str(margin.get("accountValue", "0")))

        # Always check spot balance for unified accounts
        spot_usdc = Decimal("0")
        try:
            spot = await self._run_sync(self._info.spot_user_state, self._address)
            for bal in spot.get("balances", []):
                if bal.get("coin") == "USDC":
                    total = Decimal(str(bal.get("total", "0")))
                    hold = Decimal(str(bal.get("hold", "0")))
                    # Only add the available spot balance, as the held margin
                    # is already accounted for in perp marginSummary.accountValue.
                    spot_usdc = total - hold
                    break
        except Exception as e:
            logger.warning("Failed to fetch spot balance: %s", e)

        equity = perp_equity + spot_usdc

        positions = []
        for p in state.get("assetPositions", []):
            pos = p.get("position", {})
            szi = Decimal(str(pos.get("szi", "0")))
            if szi == 0:
                continue
            positions.append(
                Position(
                    symbol=pos.get("coin", ""),
                    size=szi,
                    entry_price=Decimal(str(pos.get("entryPx", "0"))),
                    unrealized_pnl=Decimal(str(pos.get("unrealizedPnl", "0"))),
                    leverage=int(pos.get("leverage", {}).get("value", 1)),
                )
            )

        return AccountState(
            equity=equity,
            available_balance=Decimal(str(margin.get("totalRawUsd", "0")))
            if equity > 0
            else equity,
            total_margin_used=Decimal(str(margin.get("totalMarginUsed", "0"))),
            positions=positions,
        )

    async def get_position(self, symbol: str) -> Position | None:
        state = await self.get_account_state()
        for p in state.positions:
            if p.symbol == symbol:
                return p
        return None

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        # L2 for bid/ask
        l2 = await self._run_sync(self._info.l2_snapshot, symbol)
        levels = l2.get("levels", [])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []

        if not bids or not asks:
            msg = f"No L2 data for {symbol}"
            raise ValueError(msg)

        best_bid = Decimal(str(bids[0]["px"]))
        best_ask = Decimal(str(asks[0]["px"]))
        mid = (best_bid + best_ask) / 2

        # Funding rate from meta + contexts
        contexts = await self._run_sync(
            self._info.meta_and_asset_ctxs,
        )
        funding_rate = Decimal("0")
        mark_price = mid
        if len(contexts) >= 2:
            meta_universe = contexts[0].get("universe", [])
            asset_ctxs = contexts[1]
            for meta_item, ctx in zip(meta_universe, asset_ctxs, strict=False):
                if meta_item.get("name") == symbol:
                    funding_rate = Decimal(str(ctx.get("funding", "0")))
                    mark_price = Decimal(str(ctx.get("markPx", str(mid))))
                    break

        return MarketSnapshot(
            symbol=symbol,
            mid_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            mark_price=mark_price,
            funding_rate=funding_rate,
            timestamp_ms=l2.get("time", 0),
        )

    async def place_market_order(
        self,
        symbol: str,
        is_buy: bool,
        size: Decimal,
        *,
        reduce_only: bool = False,
    ) -> OrderResult:
        await self._ensure_metadata(symbol)
        sz = self._format_size(size, symbol)

        # IOC at 0.5% slippage
        snapshot = await self.get_market_snapshot(symbol)
        px = (
            float(snapshot.best_ask * Decimal("1.005"))
            if is_buy
            else float(snapshot.best_bid * Decimal("0.995"))
        )
        px_str = self._format_price(Decimal(str(px)))

        try:
            async with self._order_lock:
                result = await self._run_sync(
                    self._exchange.order,
                    symbol,
                    is_buy,
                    sz,
                    float(px_str),
                    {"limit": {"tif": "Ioc"}},
                    reduce_only,
                )
            return self._parse_order_result(result, symbol, is_buy, sz)
        except Exception as e:
            logger.error(
                "Market order failed: %s %s %s — %s",
                symbol,
                "BUY" if is_buy else "SELL",
                sz,
                e,
            )
            return OrderResult(order_id="", success=False, message=str(e))

    async def place_limit_order(
        self,
        symbol: str,
        is_buy: bool,
        size: Decimal,
        price: Decimal,
        *,
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResult:
        await self._ensure_metadata(symbol)
        sz = self._format_size(size, symbol)
        px_str = self._format_price(price)

        tif = "Alo" if post_only else "Gtc"

        try:
            async with self._order_lock:
                result = await self._run_sync(
                    self._exchange.order,
                    symbol,
                    is_buy,
                    sz,
                    float(px_str),
                    {"limit": {"tif": tif}},
                    reduce_only,
                )
            return self._parse_order_result(result, symbol, is_buy, sz)
        except Exception as e:
            logger.error(
                "Limit order failed: %s %s %s@%s — %s",
                symbol,
                "BUY" if is_buy else "SELL",
                sz,
                px_str,
                e,
            )
            return OrderResult(order_id="", success=False, message=str(e))

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            async with self._order_lock:
                result = await self._run_sync(
                    self._exchange.cancel,
                    symbol,
                    int(order_id),
                )
            success = result.get("status") == "ok"
            if success:
                logger.info("Cancelled order %s (%s)", order_id, symbol)
            else:
                logger.warning("Cancel failed %s: %s", order_id, result)
            return success
        except Exception as e:
            logger.error("Cancel exception %s: %s", order_id, e)
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        all_orders = await self._run_sync(self._info.open_orders, self._address)
        if symbol:
            all_orders = [o for o in all_orders if o.get("coin") == symbol]
        if not all_orders:
            return 0

        cancels = [{"coin": o["coin"], "oid": int(o["oid"])} for o in all_orders]
        cancelled = 0

        for i in range(0, len(cancels), 20):
            batch = cancels[i : i + 20]
            try:
                async with self._order_lock:
                    result = await self._run_sync(self._exchange.bulk_cancel, batch)
                if result.get("status") == "ok":
                    statuses = (
                        result.get("response", {}).get("data", {}).get("statuses", [])
                    )
                    cancelled += sum(1 for s in statuses if s == "success")
            except Exception as e:
                logger.error("Bulk cancel error: %s", e)

        logger.info("Cancelled %d/%d orders", cancelled, len(cancels))
        return cancelled

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get open orders, optionally filtered by symbol."""
        all_orders = await self._run_sync(self._info.open_orders, self._address)
        if symbol:
            all_orders = [o for o in all_orders if o.get("coin") == symbol]
        return all_orders

    async def set_leverage(
        self, symbol: str, leverage: int, cross: bool = True
    ) -> bool:
        try:
            result = await self._run_sync(
                self._exchange.update_leverage,
                leverage,
                symbol,
                cross,
            )
            ok = result.get("status") == "ok"
            logger.info(
                "Set leverage %s %dx %s: %s",
                symbol,
                leverage,
                "cross" if cross else "isolated",
                ok,
            )
            return ok
        except Exception as e:
            logger.error("Set leverage failed %s: %s", symbol, e)
            return False

    async def transfer_to_perp(self, amount: Decimal) -> bool:
        """Transfer USDC from spot to perp account."""
        try:
            # HL SDK: usd_class_transfer(amount, toPerp=True)
            result = await self._run_sync(
                self._exchange.usd_class_transfer,
                float(amount),
                True,  # toPerp
            )
            ok = result.get("status") == "ok"
            logger.info("Transfer %s USDC to perp: %s", amount, ok)
            return ok
        except Exception as e:
            logger.error("Transfer to perp failed: %s", e)
            return False

    async def close(self) -> None:
        logger.info("HyperliquidAdapter closed.")

    # ── internal ─────────────────────────────────────────────────────

    def _parse_order_result(
        self,
        result: dict,
        symbol: str,
        is_buy: bool,
        sz: float,
    ) -> OrderResult:
        side_str = "BUY" if is_buy else "SELL"
        if result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses:
                st = statuses[0]
                if "resting" in st:
                    oid = str(st["resting"]["oid"])
                    logger.info(
                        "Order resting: %s %s %s → %s", side_str, sz, symbol, oid
                    )
                    return OrderResult(order_id=oid, success=True, message="resting")
                if "filled" in st:
                    oid = str(st["filled"]["oid"])
                    avg_px = float(st["filled"].get("avgPx", "0"))
                    total_sz = float(st["filled"].get("totalSz", "0"))
                    logger.info(
                        "Order filled: %s %s %s @ %s",
                        side_str,
                        total_sz,
                        symbol,
                        avg_px,
                    )
                    return OrderResult(
                        order_id=oid,
                        success=True,
                        message="filled",
                        avg_price=avg_px,
                        filled_size=total_sz,
                    )
                if "error" in st:
                    err = st["error"]
                    logger.error("Order rejected: %s", err)
                    return OrderResult(order_id="", success=False, message=err)

        error_msg = str(result)
        logger.error("Order failed: %s", error_msg)
        return OrderResult(order_id="", success=False, message=error_msg)
