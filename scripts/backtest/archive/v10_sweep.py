"""CTA-Forge v10 — Adaptive params + risk parity position sizing.

Key improvements over v9a (ensemble ADX, lowest OOS variance):
1. Adaptive lookback: when recent vol high, use shorter momentum windows; when low, use longer
2. Risk parity sizing: size each position inversely proportional to its own vol contribution
3. Portfolio-level vol targeting with exponential decay (EMA of recent returns)
4. Drawdown circuit breaker: if equity drops >8% from peak, halve position sizes until recovery
5. Signal persistence filter: require signal same direction for 2+ consecutive bars before entry
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from core.metrics import calculate_metrics

BINANCE_URL = "https://fapi.binance.com"
OUT_DIR = Path(__file__).resolve().parents[2] / "backtest-results"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT",
    "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "SUIUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT",
]
TIMEFRAME = "6h"
INITIAL_EQUITY = 10_000.0
COMMISSION = 0.0004


async def fetch_klines_paginated(client, symbol, target_bars=4500):
    all_rows = []; end_time = None
    while len(all_rows) < target_bars:
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 1500}
        if end_time is not None: params["endTime"] = end_time
        try:
            resp = await client.get(f"{BINANCE_URL}/fapi/v1/klines", params=params)
            if resp.status_code == 429: await asyncio.sleep(5); continue
            if resp.status_code != 200: return None
            raw = resp.json()
            if not raw: break
            rows = [{"open_time": datetime.fromtimestamp(k[0]/1000, tz=UTC),
                      "open": float(k[1]), "high": float(k[2]),
                      "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} for k in raw]
            all_rows = rows + all_rows; end_time = raw[0][0]-1
            if len(raw) < 1500: break
            await asyncio.sleep(0.1)
        except Exception: return None
    if not all_rows: return None
    return pl.DataFrame(all_rows).unique(subset=["open_time"]).sort("open_time")


async def fetch_all():
    bars = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for sym in SYMBOLS:
            df = await fetch_klines_paginated(client, sym)
            if df is not None and len(df) >= 500:
                bars[sym] = df
                print(f"  ✓ {sym}: {len(df)}", flush=True)
            await asyncio.sleep(0.15)
    return bars


def compute_adx(high, low, close, period=14):
    n = len(close); adx = np.zeros(n); dip_o = np.zeros(n); dim_o = np.zeros(n)
    if n < period*2: return adx, dip_o, dim_o
    tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up = high[i]-high[i-1]; down = low[i-1]-low[i]
        pdm[i] = up if up > down and up > 0 else 0
        mdm[i] = down if down > up and down > 0 else 0
    atr_s = np.zeros(n); pdm_s = np.zeros(n); mdm_s = np.zeros(n)
    atr_s[period] = tr[1:period+1].sum(); pdm_s[period] = pdm[1:period+1].sum(); mdm_s[period] = mdm[1:period+1].sum()
    for i in range(period+1, n):
        atr_s[i] = atr_s[i-1]-atr_s[i-1]/period+tr[i]
        pdm_s[i] = pdm_s[i-1]-pdm_s[i-1]/period+pdm[i]
        mdm_s[i] = mdm_s[i-1]-mdm_s[i-1]/period+mdm[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        dip_o = np.where(atr_s>0, 100*pdm_s/atr_s, 0)
        dim_o = np.where(atr_s>0, 100*mdm_s/atr_s, 0)
        ds = dip_o+dim_o; dx = np.where(ds>0, 100*np.abs(dip_o-dim_o)/ds, 0)
    if period*2 < n:
        adx[period*2] = np.nanmean(dx[period+1:period*2+1])
        for i in range(period*2+1, n): adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    return adx, dip_o, dim_o


def compute_atr(high, low, close, period=14):
    n = len(close); atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr[i] = (atr[i-1]*(period-1)+tr)/period if i >= period else tr
    return atr


def precompute_all(bars, max_len):
    data = {}
    for sym, df in bars.items():
        c = df["close"].to_numpy()[:max_len]; h = df["high"].to_numpy()[:max_len]
        lo = df["low"].to_numpy()[:max_len]; vol = df["volume"].to_numpy()[:max_len]
        atr = compute_atr(h, lo, c); adx, dip, dim = compute_adx(h, lo, c)
        # Precompute rolling vol (20-bar std of returns)
        rets = np.zeros(len(c))
        for i in range(1, len(c)):
            rets[i] = (c[i]-c[i-1])/c[i-1] if c[i-1] > 0 else 0
        rvol = np.zeros(len(c))
        for i in range(20, len(c)):
            rvol[i] = np.std(rets[i-20:i])
        data[sym] = {"close": c, "high": h, "low": lo, "volume": vol,
                      "atr": atr, "adx": adx, "dip": dip, "dim": dim, "rvol": rvol, "rets": rets}
    return data


def compute_signals(data, params):
    adx_ens = params.get("adx_ensemble", [params["adx_threshold"]])
    MOM_LBS = params["mom_lookbacks"]
    use_btc = params.get("btc_filter", False)
    btc_close = data.get("BTCUSDT", {}).get("close")
    adaptive = params.get("adaptive_lookback", False)
    persistence = params.get("signal_persistence", 1)

    sigs = {}
    for sym, d in data.items():
        c = d["close"]; h = d["high"]; lo = d["low"]; vol = d["volume"]
        adx = d["adx"]; dip = d["dip"]; dim = d["dim"]; rvol = d["rvol"]
        n = len(c); raw_sig = np.zeros(n)

        for i in range(max(MOM_LBS)+1, n):
            raw_signals = []
            for adx_t in adx_ens:
                if adx[i] < adx_t: raw_signals.append(0.0); continue
                di_long = dip[i] > dim[i]; di_short = dim[i] > dip[i]

                # Adaptive: if recent vol is high, weight shorter lookbacks more
                if adaptive and rvol[i] > 0:
                    median_rvol = np.median(rvol[max(0,i-120):i]) if i > 120 else rvol[i]
                    vol_ratio = rvol[i] / median_rvol if median_rvol > 0 else 1.0
                else:
                    vol_ratio = 1.0

                votes = 0; tw = 0
                for j, lb in enumerate(MOM_LBS):
                    base_w = 1.0/(j+1)
                    # In high-vol: boost short lookback weight; in low-vol: boost long
                    if adaptive:
                        if j == 0: w = base_w * min(vol_ratio, 2.0)  # shortest
                        elif j == len(MOM_LBS)-1: w = base_w * min(1.0/max(vol_ratio, 0.5), 2.0)  # longest
                        else: w = base_w
                    else:
                        w = base_w
                    ret = (c[i]-c[i-lb])/c[i-lb] if c[i-lb]>0 else 0
                    votes += np.sign(ret)*w*min(abs(ret)*20, 1.0); tw += w
                raw = votes/tw if tw>0 else 0

                dh = h[i-20:i].max(); dl = lo[i-20:i].min(); dm = (dh+dl)/2; dr = dh-dl
                if dr > 1e-10:
                    dp = (c[i]-dm)/(dr/2)
                    if raw>0 and dp<0: raw *= 0.2
                    elif raw<0 and dp>0: raw *= 0.2

                if i >= 20:
                    avg_vol = vol[i-20:i].mean()
                    if avg_vol > 0:
                        vr = vol[i]/avg_vol
                        if vr < 0.8: raw *= 0.5
                        elif vr > 1.5: raw *= 1.2

                if raw > 0 and not di_long: raw *= 0.3
                elif raw < 0 and not di_short: raw *= 0.3

                if use_btc and btc_close is not None and sym != "BTCUSDT" and i >= 60:
                    br = (btc_close[i]-btc_close[i-60])/btc_close[i-60] if btc_close[i-60]>0 else 0
                    if raw > 0 and br < -0.05: raw *= 0.5
                    elif raw < 0 and br > 0.05: raw *= 0.5

                raw_signals.append(np.clip(raw, -1, 1))
            raw_sig[i] = np.mean(raw_signals) if raw_signals else 0

        # Signal persistence filter
        if persistence > 1:
            filtered = np.zeros(n); streak = 0; last_dir = 0
            for i in range(n):
                d_now = np.sign(raw_sig[i])
                if d_now == last_dir and d_now != 0:
                    streak += 1
                elif d_now != 0:
                    streak = 1; last_dir = d_now
                else:
                    streak = 0; last_dir = 0
                filtered[i] = raw_sig[i] if streak >= persistence else 0
            sigs[sym] = filtered
        else:
            sigs[sym] = raw_sig
    return sigs


def run_backtest(data, sigs, ts_list, start_idx, end_idx, params):
    SIG_T = params["signal_threshold"]
    MIN_HOLD = params["min_hold_bars"]
    ATR_STOP = params["atr_stop_mult"]
    RISK_PT = params["risk_per_trade"]
    MAX_POS = params["max_positions"]
    REBAL = params["rebalance_every"]
    PT_TP = params.get("partial_take_profit", 0)
    TARGET_VOL = params.get("target_vol", 0)
    MAX_HOLD = params.get("max_hold_bars", 0)
    TIGHTEN_AFTER = params.get("tighten_stop_after_atr", 0)
    TIGHTENED_STOP = params.get("tightened_stop_mult", 3.0)
    DD_BREAKER = params.get("dd_circuit_breaker", 0)  # e.g. 0.08 = 8% DD halves size
    RISK_PARITY = params.get("risk_parity", False)

    equity = INITIAL_EQUITY; cash = INITIAL_EQUITY; peak_equity = INITIAL_EQUITY
    positions = {}; curve = []; trades = []
    recent_returns = []

    for t in range(start_idx, end_idx):
        pv = cash
        for s, po in positions.items():
            pv += abs(po["qty"])*po["entry_price"]+po["qty"]*(data[s]["close"][t]-po["entry_price"])
        equity = pv
        peak_equity = max(peak_equity, equity)
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        if len(curve) > 0:
            prev_eq = curve[-1][1]
            if prev_eq > 0:
                recent_returns.append((pv-prev_eq)/prev_eq)
                if len(recent_returns) > 120: recent_returns.pop(0)

        # Vol targeting
        vol_scale = 1.0
        if TARGET_VOL > 0 and len(recent_returns) >= 20:
            rv = np.std(recent_returns[-60:])*np.sqrt(4*365)
            if rv > 0: vol_scale = np.clip(TARGET_VOL/rv, 0.3, 2.0)

        # DD circuit breaker
        dd_scale = 1.0
        if DD_BREAKER > 0 and current_dd > DD_BREAKER:
            dd_scale = 0.5  # halve size when in deep drawdown

        # Exits
        to_close = []
        for s, po in positions.items():
            held = t-po["entry_bar"]; a = data[s]["atr"][t]
            if MAX_HOLD > 0 and held >= MAX_HOLD: to_close.append(s); continue

            cs = ATR_STOP
            if TIGHTEN_AFTER > 0:
                unr = ((data[s]["close"][t]-po["entry_price"])/a if po["qty"]>0 else
                       (po["entry_price"]-data[s]["close"][t])/a) if a>0 else 0
                if unr > TIGHTEN_AFTER: cs = TIGHTENED_STOP

            if po["qty"] > 0:
                po["best_price"] = max(po.get("best_price", po["entry_price"]), data[s]["close"][t])
                if data[s]["close"][t] < po["best_price"]-cs*a and held >= MIN_HOLD: to_close.append(s)
            else:
                po["best_price"] = min(po.get("best_price", po["entry_price"]), data[s]["close"][t])
                if data[s]["close"][t] > po["best_price"]+cs*a and held >= MIN_HOLD: to_close.append(s)

            if held >= MIN_HOLD:
                if po["qty"]>0 and sigs[s][t]<-0.15: to_close.append(s)
                elif po["qty"]<0 and sigs[s][t]>0.15: to_close.append(s)

        # Partial TP
        if PT_TP > 0:
            for s, po in positions.items():
                if s in to_close: continue
                a = data[s]["atr"][t]
                if po["qty"] > 0 and not po.get("pt"):
                    if data[s]["close"][t] > po["entry_price"]+PT_TP*a:
                        hq = po["qty"]/2
                        pnl = hq*(data[s]["close"][t]-po["entry_price"])-abs(hq)*data[s]["close"][t]*COMMISSION
                        cash += abs(hq)*po["entry_price"]+pnl; po["qty"]-=hq; po["pt"]=True; trades.append({"pnl": pnl})
                elif po["qty"] < 0 and not po.get("pt"):
                    if data[s]["close"][t] < po["entry_price"]-PT_TP*a:
                        hq = po["qty"]/2
                        pnl = hq*(data[s]["close"][t]-po["entry_price"])-abs(hq)*data[s]["close"][t]*COMMISSION
                        cash += abs(hq)*po["entry_price"]+pnl; po["qty"]-=hq; po["pt"]=True; trades.append({"pnl": pnl})

        for s in set(to_close):
            po = positions[s]
            pnl = po["qty"]*(data[s]["close"][t]-po["entry_price"])-abs(po["qty"])*data[s]["close"][t]*COMMISSION
            cash += abs(po["qty"])*po["entry_price"]+pnl; trades.append({"pnl": pnl}); del positions[s]

        # Entries
        if t % REBAL == 0 and len(positions) < MAX_POS:
            cands = [(s, sigs[s][t]) for s in data if s not in positions and abs(sigs[s][t])>=SIG_T]
            cands.sort(key=lambda x: abs(x[1]), reverse=True)
            for s, sg in cands:
                if len(positions)>=MAX_POS: break
                pr = data[s]["close"][t]; a = data[s]["atr"][t]
                if a<1e-10 or pr<1e-10: continue

                if RISK_PARITY:
                    # Risk parity: size inversely proportional to asset vol
                    asset_vol = data[s]["rvol"][t] if data[s]["rvol"][t] > 0 else 0.01
                    target_risk_per_pos = equity * RISK_PT / MAX_POS
                    qty_abs = min(target_risk_per_pos / (asset_vol * pr * np.sqrt(20)),
                                   equity * 0.15 / pr)
                else:
                    na = a/pr; ma = 0.02; iv = np.clip(ma/na, 0.5, 2.0)
                    qty_abs = min(equity*RISK_PT*iv*vol_scale/(ATR_STOP*a), equity*0.15/pr)

                qty_abs *= vol_scale * dd_scale
                if qty_abs*pr<10: continue
                qty = qty_abs if sg>0 else -qty_abs
                cash -= abs(qty)*pr+abs(qty)*pr*COMMISSION
                positions[s] = {"qty": qty, "entry_price": pr, "entry_bar": t, "best_price": pr}

        pv = cash
        for s, po in positions.items():
            pv += abs(po["qty"])*po["entry_price"]+po["qty"]*(data[s]["close"][t]-po["entry_price"])
        curve.append((ts_list[t], pv))

    for s, po in positions.items():
        t_e = end_idx-1
        pnl = po["qty"]*(data[s]["close"][t_e]-po["entry_price"])-abs(po["qty"])*data[s]["close"][t_e]*COMMISSION
        trades.append({"pnl": pnl})
    return curve, trades


def calc_ulcer(curve):
    eq = np.array([e[1] for e in curve])
    if len(eq) < 10: return 999
    rm = np.maximum.accumulate(eq); dd = (rm-eq)/rm
    return np.sqrt(np.mean(dd**2))


def walk_forward(data, ts_list, params, n_windows=6, train_bars=600, test_bars=300):
    sigs = compute_signals(data, params)
    total = len(ts_list); results = []
    step = max(1, (total - 150 - train_bars - test_bars) // max(n_windows-1, 1))
    for w in range(n_windows):
        start = 150 + w*step
        train_end = start + train_bars; test_end = min(train_end+test_bars, total)
        if test_end > total: break
        is_c, is_t = run_backtest(data, sigs, ts_list, start, train_end, params)
        oos_c, oos_t = run_backtest(data, sigs, ts_list, train_end, test_end, params)
        is_m = calculate_metrics(is_c, is_t); oos_m = calculate_metrics(oos_c, oos_t)
        results.append({
            "window": w+1, "is_sharpe": is_m.sharpe_ratio, "oos_sharpe": oos_m.sharpe_ratio,
            "is_return": is_m.total_return, "oos_return": oos_m.total_return,
            "oos_max_dd": oos_m.max_drawdown, "oos_trades": oos_m.num_trades,
        })
    return results


async def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("CTA-Forge v10 — Adaptive + Risk Parity + DD Breaker", flush=True)
    print("="*60, flush=True)

    print(f"\nFetching...", flush=True)
    bars = await fetch_all()
    min_len = min(len(df) for df in bars.values())
    ts_list = list(bars[next(iter(bars))]["open_time"][:min_len].to_list())
    data = precompute_all(bars, min_len)
    print(f"→ {len(bars)} sym, {min_len} bars ({min_len*6/24:.0f}d)\n", flush=True)

    # Base = v9a (ensemble ADX, lowest OOS variance) enriched with v8H's full-period strength
    base = {
        "mom_lookbacks": [20, 60, 120], "adx_threshold": 25, "adx_ensemble": [22, 27, 32],
        "signal_threshold": 0.40, "min_hold_bars": 16, "atr_stop_mult": 5.0,
        "risk_per_trade": 0.015, "max_positions": 5, "rebalance_every": 4,
        "partial_take_profit": 2.5, "target_vol": 0.12,
        "btc_filter": True, "max_hold_bars": 100,
        "tighten_stop_after_atr": 2.0, "tightened_stop_mult": 3.0,
    }

    configs = [
        {**base, "label": "v9a-ref (baseline)"},
        {**base, "label": "v10a: +adaptive_lookback",
         "adaptive_lookback": True},
        {**base, "label": "v10b: +risk_parity",
         "risk_parity": True},
        {**base, "label": "v10c: +dd_breaker_8%",
         "dd_circuit_breaker": 0.08},
        {**base, "label": "v10d: +persistence_2",
         "signal_persistence": 2},
        {**base, "label": "v10e: adaptive+dd_breaker",
         "adaptive_lookback": True, "dd_circuit_breaker": 0.08},
        {**base, "label": "v10f: adaptive+parity+dd",
         "adaptive_lookback": True, "risk_parity": True, "dd_circuit_breaker": 0.08},
        {**base, "label": "v10g: all_features",
         "adaptive_lookback": True, "risk_parity": True, "dd_circuit_breaker": 0.08,
         "signal_persistence": 2},
        {**base, "label": "v10h: adaptive+dd+persist+6pos",
         "adaptive_lookback": True, "dd_circuit_breaker": 0.08,
         "signal_persistence": 2, "max_positions": 6, "risk_per_trade": 0.012},
        {**base, "label": "v10i: full combo aggressive",
         "adaptive_lookback": True, "risk_parity": True, "dd_circuit_breaker": 0.06,
         "signal_persistence": 2, "signal_threshold": 0.35, "target_vol": 0.15,
         "max_hold_bars": 80},
    ]

    all_results = []
    for i, cfg in enumerate(configs):
        label = cfg["label"]
        print(f"[{i+1}/{len(configs)}] {label}", flush=True)

        wf = walk_forward(data, ts_list, cfg)
        oos_s = [r["oos_sharpe"] for r in wf]
        avg_oos = np.mean(oos_s); std_oos = np.std(oos_s)

        for r in wf:
            print(f"  W{r['window']}: IS={r['is_sharpe']:+.2f} OOS={r['oos_sharpe']:+.2f} "
                  f"ret={r['oos_return']*100:+.1f}% dd={r['oos_max_dd']*100:.1f}%", flush=True)

        sigs = compute_signals(data, cfg)
        fc, ft = run_backtest(data, sigs, ts_list, 150, min_len, cfg)
        fm = calculate_metrics(fc, ft); ulcer = calc_ulcer(fc)

        wins = sum(1 for s in oos_s if s > 0)
        inv_std = 1.0/max(std_oos, 0.1)
        composite = 0.4*avg_oos + 0.3*min(inv_std, 5) + 0.3*fm.sharpe_ratio

        print(f"  WF: avg_OOS={avg_oos:.2f}(±{std_oos:.2f}) wins={wins}/{len(wf)} | "
              f"FULL: S={fm.sharpe_ratio:.2f} Ret={fm.total_return*100:+.1f}% "
              f"DD={fm.max_drawdown*100:.1f}% Ulcer={ulcer:.4f} C={composite:.2f}\n", flush=True)

        all_results.append({
            "label": label, "wf": wf, "avg_oos": avg_oos, "std_oos": std_oos,
            "oos_wins": f"{wins}/{len(wf)}", "composite": composite,
            "full_sharpe": fm.sharpe_ratio, "full_sortino": fm.sortino_ratio,
            "full_return": fm.total_return, "full_ann": fm.annualized_return,
            "full_dd": fm.max_drawdown, "full_calmar": fm.calmar_ratio,
            "full_pf": fm.profit_factor, "full_trades": fm.num_trades,
            "ulcer": ulcer, "curve": fc,
        })

    print(f"{'='*75}", flush=True)
    print("RANKING BY COMPOSITE", flush=True)
    print(f"{'='*75}", flush=True)
    ranked = sorted(all_results, key=lambda x: x["composite"], reverse=True)
    for rank, r in enumerate(ranked):
        marker = " ★" if rank == 0 else ""
        print(f"  #{rank+1} [{r['label']}] C={r['composite']:.2f}{marker}", flush=True)
        print(f"       WF: avg_OOS={r['avg_oos']:.2f}(±{r['std_oos']:.2f}) wins={r['oos_wins']}", flush=True)
        print(f"       FULL: S={r['full_sharpe']:.2f} So={r['full_sortino']:.2f} "
              f"Ret={r['full_return']*100:+.1f}% DD={r['full_dd']*100:.1f}% "
              f"Calmar={r['full_calmar']:.2f} PF={r['full_pf']:.2f} T={r['full_trades']} "
              f"Ulcer={r['ulcer']:.4f}", flush=True)

    # Charts
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; import matplotlib.dates as mdates
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Best detailed
    best = ranked[0]
    ts_b = [e[0] for e in best["curve"]]; eq_b = [e[1] for e in best["curve"]]
    days = (ts_b[-1]-ts_b[0]).days

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), height_ratios=[3, 1, 1.5], gridspec_kw={"hspace": 0.2})
    axes[0].plot(ts_b, eq_b, linewidth=1.0, color="#2ecc71")
    axes[0].axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e>=INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#2ecc71")
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e<INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#e74c3c")
    axes[0].set_title(f"CTA-Forge v10 Best [{best['label']}] — {days}d WF-Validated", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equity ($)"); axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].grid(True, alpha=0.2)

    mt = (f"Return: {best['full_return']*100:+.1f}%  Ann: {best['full_ann']*100:+.1f}%\n"
          f"Sharpe: {best['full_sharpe']:.2f}  Sortino: {best['full_sortino']:.2f}\n"
          f"MaxDD: {best['full_dd']*100:.1f}%  Calmar: {best['full_calmar']:.2f}\n"
          f"PF: {best['full_pf']:.2f}  Trades: {best['full_trades']}\n"
          f"Ulcer: {best['ulcer']:.4f}\n"
          f"─────────────────────\n"
          f"WF avg OOS: {best['avg_oos']:.2f}(±{best['std_oos']:.2f})\n"
          f"Composite: {best['composite']:.2f}")
    axes[0].text(0.98, 0.02, mt, transform=axes[0].transAxes, fontsize=8.5, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.85), color="white", family="monospace")

    eq_arr = np.array(eq_b); rm = np.maximum.accumulate(eq_arr); dd_arr = (rm-eq_arr)/rm*100
    axes[1].fill_between(ts_b, 0, -dd_arr, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %"); axes[1].grid(True, alpha=0.2)

    monthly = {}
    for ii in range(1, len(eq_b)):
        key = ts_b[ii].strftime("%Y-%m")
        if key not in monthly: monthly[key] = {"start": eq_b[ii-1]}
        monthly[key]["end"] = eq_b[ii]
    mos = list(monthly.keys())
    rets = [(monthly[m_]["end"]-monthly[m_]["start"])/monthly[m_]["start"]*100 for m_ in mos]
    axes[2].bar(range(len(mos)), rets, color=["#2ecc71" if r>=0 else "#e74c3c" for r in rets], alpha=0.8, width=0.7)
    step = max(1, len(mos)//15)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels([mos[ii] for ii in range(0, len(mos), step)], rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Monthly %"); axes[2].axhline(y=0, color="black", linewidth=0.5); axes[2].grid(True, alpha=0.2, axis="y")
    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.savefig(OUT_DIR / "backtest_v10_best.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Top 3 + heatmap
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1.5], gridspec_kw={"hspace": 0.15})
    colors_t = ["#2ecc71", "#3498db", "#e67e22"]
    for j, r in enumerate(ranked[:3]):
        ts_c = [e[0] for e in r["curve"]]; eq_c = [e[1] for e in r["curve"]]
        axes2[0].plot(ts_c, eq_c, linewidth=1.0, color=colors_t[j],
                       label=f"#{j+1} {r['label'][:30]}\nS={r['full_sharpe']:.2f} OOS={r['avg_oos']:.2f}")
    axes2[0].axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.5)
    axes2[0].set_title("CTA-Forge v10 Top 3 + WF Heatmap", fontsize=13, fontweight="bold")
    axes2[0].set_ylabel("Equity ($)"); axes2[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes2[0].legend(loc="upper left", fontsize=7.5); axes2[0].grid(True, alpha=0.2)

    wf_arr = np.array([[w["oos_sharpe"] for w in r["wf"]] for r in ranked[:3]])
    im = axes2[1].imshow(wf_arr, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=4)
    axes2[1].set_yticks(range(3))
    axes2[1].set_yticklabels([r["label"][:30] for r in ranked[:3]], fontsize=8)
    axes2[1].set_xticks(range(wf_arr.shape[1]))
    axes2[1].set_xticklabels([f"W{i+1}" for i in range(wf_arr.shape[1])])
    axes2[1].set_title("OOS Sharpe per Window", fontsize=11)
    for yi in range(wf_arr.shape[0]):
        for xi in range(wf_arr.shape[1]):
            axes2[1].text(xi, yi, f"{wf_arr[yi,xi]:.1f}", ha="center", va="center", fontsize=9, fontweight="bold",
                           color="white" if abs(wf_arr[yi,xi])>1.0 else "black")
    fig2.colorbar(im, ax=axes2[1], shrink=0.6)
    fig2.savefig(OUT_DIR / "backtest_v10_top3.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    save_results = [{k:v for k,v in r.items() if k != "curve"} for r in ranked]
    (OUT_DIR / "sweep_v10.json").write_text(json.dumps(save_results, indent=2, default=str))
    print(f"\n✅ Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
