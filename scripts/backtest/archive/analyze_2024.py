"""Analyze v10g trades in 2024-2025 to find why performance lagged."""
import asyncio, sys, time, json
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/libs/cta-core/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/services/reporter/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/scripts/backtest")
from v10g_maxrange import (
    SYMBOLS, TIMEFRAME, INITIAL_EQUITY, COMMISSION, START_TS,
    BINANCE_URL, fetch_klines_from, precompute, build_timeline,
    align_data, compute_signals
)

OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")


async def main():
    import httpx
    print("Fetching data...", flush=True)
    async with httpx.AsyncClient(timeout=30) as client:
        bars = {}
        for sym in SYMBOLS:
            df = await fetch_klines_from(client, sym, START_TS)
            if df is not None and len(df) >= 500:
                bars[sym] = df
            await asyncio.sleep(0.15)

    timeline, ts_to_idx = build_timeline(bars)
    data = precompute(bars)
    align_data(bars, data, ts_to_idx)

    params = {
        "mom_lookbacks": [20, 60, 120],
        "adx_ensemble": [22, 27, 32],
        "signal_threshold": 0.40,
        "min_hold_bars": 16,
        "atr_stop_mult": 5.0,
        "risk_per_trade": 0.015,
        "max_positions": 5,
        "rebalance_every": 4,
        "partial_take_profit": 2.5,
        "target_vol": 0.12,
        "btc_filter": True,
        "max_hold_bars": 100,
        "tighten_stop_after_atr": 2.0,
        "tightened_stop_mult": 3.0,
        "dd_circuit_breaker": 0.08,
        "risk_parity": True,
        "signal_persistence": 2,
        "adaptive_lookback": True,
    }

    sigs = compute_signals(data, timeline, params)

    # Analyze signals and vol_scale in 2024-2025
    print("\n=== 2024-2025 SIGNAL ANALYSIS ===\n", flush=True)

    # Find BTC price moves in 2024
    btc_d = data.get("BTCUSDT")
    if btc_d:
        print("BTC price trajectory (quarterly):", flush=True)
        for quarter_start in ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",
                               "2025-01-01", "2025-04-01"]:
            dt = datetime.fromisoformat(quarter_start).replace(tzinfo=UTC)
            if dt in ts_to_idx:
                idx = ts_to_idx[dt]
                li = idx - btc_d["start_idx"]
                if 0 <= li < btc_d["length"]:
                    print(f"  {quarter_start}: BTC = ${btc_d['close'][li]:,.0f}", flush=True)

    # Check vol_scale behavior
    print("\nVol scaling analysis (rolling 60-bar realized vol):", flush=True)
    start_idx = 200
    recent_returns = []
    TARGET_VOL = 0.12
    equity = INITIAL_EQUITY

    vol_scales_2024 = []
    dd_scales_2024 = []
    peak_eq = equity
    
    # We need to track equity through the full backtest to get accurate vol_scale for 2024
    # Let's just sample signal strength in 2024
    print("\nSignal strength for BTC in 2024-2025:", flush=True)
    for month in range(1, 13):
        dt = datetime(2024, month, 1, tzinfo=UTC)
        if dt in ts_to_idx:
            idx = ts_to_idx[dt]
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                if sym in sigs and idx in sigs[sym]:
                    sig = sigs[sym][idx]
                    if abs(sig) > 0.01:
                        print(f"  2024-{month:02d} {sym}: signal={sig:+.3f}", flush=True)

    print("\nSignal strength for top coins in 2025:", flush=True)
    for month in range(1, 5):
        dt = datetime(2025, month, 1, tzinfo=UTC)
        if dt in ts_to_idx:
            idx = ts_to_idx[dt]
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]:
                if sym in sigs and idx in sigs[sym]:
                    sig = sigs[sym][idx]
                    if abs(sig) > 0.01:
                        print(f"  2025-{month:02d} {sym}: signal={sig:+.3f}", flush=True)

    # Check how many bars had active positions in 2024
    print("\n=== KEY PARAMETERS CHECK ===", flush=True)
    print(f"max_hold_bars: {params['max_hold_bars']} (= {params['max_hold_bars']*6}h = {params['max_hold_bars']*6/24:.0f} days)", flush=True)
    print(f"atr_stop_mult: {params['atr_stop_mult']}", flush=True)
    print(f"tighten_stop_after_atr: {params['tighten_stop_after_atr']} → tightened_stop_mult: {params['tightened_stop_mult']}", flush=True)
    print(f"partial_take_profit: {params['partial_take_profit']} ATR", flush=True)
    print(f"target_vol: {params['target_vol']}", flush=True)
    print(f"dd_circuit_breaker: {params['dd_circuit_breaker']}", flush=True)

    # The real question: max_hold = 100 bars = 25 days. BTC's big moves in 2024 were multi-month.
    # Are we exiting too early?
    print(f"\n⚠️  max_hold_bars=100 = 25 days. This forces exit even if trend continues.", flush=True)
    print(f"⚠️  BTC went from $42K (Jan) → $73K (Mar) → $54K (Jul) → $106K (Dec)", flush=True)
    print(f"⚠️  These are multi-month moves. 25-day hold limit cuts them short.", flush=True)
    
    # Also check: tighten stop at 2 ATR profit → stop at 3 ATR
    # In a strong trend, this could stop you out on a normal pullback
    print(f"\n⚠️  tighten_stop kicks in at 2 ATR profit, reduces stop to 3 ATR.", flush=True)
    print(f"    In a volatile uptrend, a 3-ATR pullback is normal → premature exit", flush=True)

    # Also: target_vol = 0.12. If realized vol spikes in bull market, vol_scale drops
    print(f"\n⚠️  target_vol=0.12. In 2024 bull, realized vol likely >20% → vol_scale shrinks positions", flush=True)


asyncio.run(main())
