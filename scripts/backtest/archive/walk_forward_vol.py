"""Walk-forward validation for target_vol parameter.

Split data into 3 folds (time-series style):
  Fold 1: 2019-10 → 2022-03 (IS) | 2022-04 → 2023-06 (OOS)
  Fold 2: 2020-10 → 2023-06 (IS) | 2023-07 → 2024-09 (OOS)
  Fold 3: 2021-10 → 2024-09 (IS) | 2024-10 → 2026-04 (OOS)

Test target_vol in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.0(disabled)]
"""
import asyncio, sys, json, time
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/libs/cta-core/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/services/reporter/src")
sys.path.insert(0, "/home/node/.openclaw/workspace/cta-forge-dev/scripts/backtest")

# We'll import the core functions from v10g and run with different params/date ranges
from v10g_maxrange import (
    SYMBOLS, TIMEFRAME, INITIAL_EQUITY, COMMISSION, START_TS,
    fetch_klines_from, precompute, build_timeline,
    align_data, compute_signals
)

OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")

# Walk-forward folds
FOLDS = [
    {"name": "Fold1", "is_start": "2019-10-28", "is_end": "2022-03-31",
     "oos_start": "2022-04-01", "oos_end": "2023-06-30"},
    {"name": "Fold2", "is_start": "2020-10-01", "is_end": "2023-06-30",
     "oos_start": "2023-07-01", "oos_end": "2024-09-30"},
    {"name": "Fold3", "is_start": "2021-10-01", "is_end": "2024-09-30",
     "oos_start": "2024-10-01", "oos_end": "2026-04-12"},
]

TARGET_VOLS = [0.0, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

def run_backtest_segment(data, timeline, ts_to_idx, sigs, start_date, end_date, target_vol):
    """Run backtest on a date segment with given target_vol."""

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)

    # Find start/end indices
    start_idx = None
    end_idx = None
    for i, ts in enumerate(timeline):
        if start_idx is None and ts >= start_dt:
            start_idx = i
        if ts <= end_dt:
            end_idx = i

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return None

    # Backtest params
    RISK_PER_TRADE = 0.015
    MAX_POS = 5
    ATR_STOP = 5.0
    REBALANCE = 4
    PARTIAL_TP = 2.5
    MIN_HOLD = 16
    DD_BREAKER = 0.08
    TIGHTEN_AFTER = 2.0
    TIGHTENED_STOP = 3.0
    MAX_HOLD = 100
    RISK_PARITY = True
    TARGET_VOL = target_vol

    equity = INITIAL_EQUITY
    peak_eq = equity
    positions = {}
    recent_returns = []
    curve = []

    def get_close(sym, t):
        d = data.get(sym)
        if d is None:
            return None
        li = t - d["start_idx"]
        if li < 0 or li >= d["length"]:
            return None
        return d["close"][li]

    def get_atr(sym, t):
        d = data.get(sym)
        if d is None:
            return None
        li = t - d["start_idx"]
        if li < 0 or li >= d["length"]:
            return None
        return d["atr"][li]

    for t in range(start_idx, end_idx + 1):
        pv = equity
        for s, po in positions.items():
            cp = get_close(s, t)
            if cp is not None:
                pv += po["qty"] * (cp - po["entry_px"]) * po["side_sign"]

        peak_eq = max(peak_eq, pv)  # use pv not equity for DD
        cur_dd = (peak_eq - pv) / peak_eq if peak_eq > 0 else 0

        if curve:
            prev = curve[-1][1]
            if prev > 0:
                recent_returns.append((pv - prev) / prev)
                if len(recent_returns) > 120:
                    recent_returns.pop(0)

        curve.append((t, pv))

        vol_scale = 1.0
        if TARGET_VOL > 0 and len(recent_returns) >= 20:
            rv = np.std(recent_returns[-60:]) * np.sqrt(4 * 365)
            if rv > 0:
                vol_scale = np.clip(TARGET_VOL / rv, 0.3, 2.0)

        dd_scale = 0.5 if DD_BREAKER > 0 and cur_dd > DD_BREAKER else 1.0

        # Close positions
        to_close = []
        for s, po in list(positions.items()):
            held = t - po["entry_bar"]
            a = get_atr(s, t)
            cp = get_close(s, t)
            if a is None or cp is None:
                continue

            pnl_atr = (cp - po["entry_px"]) * po["side_sign"] / a if a > 0 else 0

            current_stop = ATR_STOP
            if TIGHTEN_AFTER > 0 and pnl_atr >= TIGHTEN_AFTER:
                current_stop = TIGHTENED_STOP

            stop_px = po["entry_px"] - current_stop * a * po["side_sign"]
            if po["side_sign"] == 1 and cp <= stop_px:
                to_close.append(s)
            elif po["side_sign"] == -1 and cp >= stop_px:
                to_close.append(s)
            elif MAX_HOLD > 0 and held >= MAX_HOLD:
                to_close.append(s)
            elif PARTIAL_TP > 0 and pnl_atr >= PARTIAL_TP and not po.get("partial_taken"):
                realized = po["qty"] * 0.5 * (cp - po["entry_px"]) * po["side_sign"]
                equity += realized * (1 - COMMISSION)
                po["qty"] *= 0.5
                po["partial_taken"] = True

        for s in to_close:
            po = positions.pop(s)
            cp = get_close(s, t)
            if cp is not None:
                realized = po["qty"] * (cp - po["entry_px"]) * po["side_sign"]
                equity += realized * (1 - COMMISSION)

        # Open new positions
        if t % REBALANCE == 0 and len(positions) < MAX_POS:
            candidates = []
            for sym in data:
                if sym in positions:
                    continue
                if sym not in sigs or t not in sigs[sym]:
                    continue
                sig = sigs[sym][t]
                if abs(sig) < 0.40:
                    continue
                candidates.append((sym, sig))

            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            slots = MAX_POS - len(positions)

            for sym, sig in candidates[:slots]:
                a = get_atr(sym, t)
                cp = get_close(sym, t)
                if a is None or cp is None or a <= 0:
                    continue

                side_sign = 1 if sig > 0 else -1
                risk_amt = pv * RISK_PER_TRADE * vol_scale * dd_scale
                qty = risk_amt / (ATR_STOP * a)
                cost = qty * cp * COMMISSION
                equity -= cost

                positions[sym] = {
                    "entry_px": cp,
                    "entry_bar": t,
                    "qty": qty,
                    "side_sign": side_sign,
                    "partial_taken": False,
                }

    # Final PV
    final_pv = equity
    for s, po in positions.items():
        cp = get_close(s, end_idx)
        if cp is not None:
            final_pv += po["qty"] * (cp - po["entry_px"]) * po["side_sign"]

    if len(curve) < 2:
        return None

    returns = []
    for i in range(1, len(curve)):
        prev_v = curve[i-1][1]
        if prev_v > 0:
            returns.append((curve[i][1] - prev_v) / prev_v)

    returns = np.array(returns)
    if len(returns) < 10:
        return None

    ann_factor = np.sqrt(4 * 365)
    sharpe = float(np.mean(returns) / np.std(returns) * ann_factor) if np.std(returns) > 0 else 0

    # Max drawdown
    peak = INITIAL_EQUITY
    max_dd = 0
    for _, v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    total_ret = (final_pv - INITIAL_EQUITY) / INITIAL_EQUITY
    days = (end_idx - start_idx) * 6 / 24  # rough
    ann_ret = (1 + total_ret) ** (365 / max(days, 1)) - 1 if days > 0 else 0

    return {
        "total_return": round(total_ret * 100, 1),
        "ann_return": round(ann_ret * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(ann_ret / max_dd, 2) if max_dd > 0 else 0,
        "bars": end_idx - start_idx,
    }


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

    print(f"→ {len(bars)} symbols loaded\n", flush=True)

    timeline, ts_to_idx = build_timeline(bars)
    data = precompute(bars)
    align_data(bars, data, ts_to_idx)

    # Compute signals once (they don't depend on target_vol)
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

    results = {}

    for fold in FOLDS:
        print(f"=== {fold['name']} ===", flush=True)
        print(f"  IS:  {fold['is_start']} → {fold['is_end']}", flush=True)
        print(f"  OOS: {fold['oos_start']} → {fold['oos_end']}", flush=True)

        for tv in TARGET_VOLS:
            label = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"

            is_result = run_backtest_segment(
                data, timeline, ts_to_idx, sigs,
                fold["is_start"], fold["is_end"], tv
            )
            oos_result = run_backtest_segment(
                data, timeline, ts_to_idx, sigs,
                fold["oos_start"], fold["oos_end"], tv
            )

            key = f"{fold['name']}_{label}"
            results[key] = {"is": is_result, "oos": oos_result}

            is_s = f"Sharpe {is_result['sharpe']:5.2f}  Ret {is_result['total_return']:+7.1f}%  DD {is_result['max_dd']:5.1f}%" if is_result else "N/A"
            oos_s = f"Sharpe {oos_result['sharpe']:5.2f}  Ret {oos_result['total_return']:+7.1f}%  DD {oos_result['max_dd']:5.1f}%" if oos_result else "N/A"
            print(f"  {label:10s} IS: {is_s}  |  OOS: {oos_s}", flush=True)

        print(flush=True)

    # Summary: average OOS Sharpe per target_vol
    print("=" * 70, flush=True)
    print("SUMMARY: Average OOS Sharpe by target_vol", flush=True)
    print("=" * 70, flush=True)

    for tv in TARGET_VOLS:
        label = f"vol={tv:.2f}" if tv > 0 else "vol=OFF"
        oos_sharpes = []
        oos_rets = []
        oos_dds = []
        for fold in FOLDS:
            key = f"{fold['name']}_{label}"
            r = results[key]["oos"]
            if r:
                oos_sharpes.append(r["sharpe"])
                oos_rets.append(r["total_return"])
                oos_dds.append(r["max_dd"])

        if oos_sharpes:
            avg_s = np.mean(oos_sharpes)
            avg_r = np.mean(oos_rets)
            avg_dd = np.mean(oos_dds)
            print(f"  {label:10s}  OOS avg Sharpe: {avg_s:5.2f}  avg Ret: {avg_r:+6.1f}%  avg DD: {avg_dd:5.1f}%", flush=True)

    # Save full results
    out_path = OUT_DIR / "walk_forward_target_vol.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {out_path}", flush=True)


asyncio.run(main())
