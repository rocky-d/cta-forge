"""REST API routes for exchange-server."""

from __future__ import annotations

import os
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .hyperliquid import HyperliquidAdapter

router = APIRouter()

# Lazy singleton — initialized on first request
_adapter: HyperliquidAdapter | None = None


def _get_adapter() -> HyperliquidAdapter:
    global _adapter
    if _adapter is None:
        pk = os.environ.get("HL_PRIVATE_KEY", "")
        addr = os.environ.get("HL_ACCOUNT_ADDRESS", "")
        testnet = os.environ.get("HL_NETWORK", "testnet") == "testnet"
        if not pk or not addr:
            msg = "HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS required"
            raise RuntimeError(msg)
        _adapter = HyperliquidAdapter(pk, addr, testnet=testnet)
    return _adapter


# ── Models ────────────────────────────────────────────────────────


class OrderRequest(BaseModel):
    symbol: str
    is_buy: bool
    size: float
    price: float | None = None  # None = market order
    reduce_only: bool = False
    post_only: bool = False


class CancelRequest(BaseModel):
    symbol: str
    order_id: str


class TransferRequest(BaseModel):
    amount: float


class LeverageRequest(BaseModel):
    symbol: str
    leverage: int
    cross: bool = True


# ── Routes ────────────────────────────────────────────────────────


@router.get("/account")
async def get_account():
    adapter = _get_adapter()
    state = await adapter.get_account_state()
    return {
        "equity": str(state.equity),
        "available_balance": str(state.available_balance),
        "margin_used": str(state.total_margin_used),
        "positions": [
            {
                "symbol": p.symbol,
                "size": str(p.size),
                "entry_price": str(p.entry_price),
                "unrealized_pnl": str(p.unrealized_pnl),
                "leverage": p.leverage,
            }
            for p in state.positions
        ],
    }


@router.get("/market/{symbol}")
async def get_market(symbol: str):
    adapter = _get_adapter()
    snap = await adapter.get_market_snapshot(symbol)
    return {
        "symbol": snap.symbol,
        "mid_price": str(snap.mid_price),
        "best_bid": str(snap.best_bid),
        "best_ask": str(snap.best_ask),
        "mark_price": str(snap.mark_price),
        "funding_rate": str(snap.funding_rate),
        "timestamp_ms": snap.timestamp_ms,
    }


@router.post("/order")
async def place_order(req: OrderRequest):
    adapter = _get_adapter()
    if req.price is None:
        result = await adapter.place_market_order(
            req.symbol,
            req.is_buy,
            Decimal(str(req.size)),
            reduce_only=req.reduce_only,
        )
    else:
        result = await adapter.place_limit_order(
            req.symbol,
            req.is_buy,
            Decimal(str(req.size)),
            Decimal(str(req.price)),
            reduce_only=req.reduce_only,
            post_only=req.post_only,
        )
    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)
    return {
        "order_id": result.order_id,
        "message": result.message,
        "avg_price": result.avg_price,
        "filled_size": result.filled_size,
    }


@router.post("/cancel")
async def cancel_order(req: CancelRequest):
    adapter = _get_adapter()
    ok = await adapter.cancel_order(req.symbol, req.order_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cancel failed")
    return {"status": "cancelled"}


@router.post("/cancel-all")
async def cancel_all(symbol: str | None = None):
    adapter = _get_adapter()
    count = await adapter.cancel_all_orders(symbol)
    return {"cancelled": count}


@router.post("/leverage")
async def set_leverage(req: LeverageRequest):
    adapter = _get_adapter()
    ok = await adapter.set_leverage(req.symbol, req.leverage, req.cross)
    if not ok:
        raise HTTPException(status_code=400, detail="Set leverage failed")
    return {"status": "ok"}


@router.post("/transfer-to-perp")
async def transfer_to_perp(req: TransferRequest):
    adapter = _get_adapter()
    ok = await adapter.transfer_to_perp(Decimal(str(req.amount)))
    if not ok:
        raise HTTPException(status_code=400, detail="Transfer failed")
    return {"status": "ok", "amount": req.amount}
