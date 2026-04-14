"""CTA-Forge v11c — Curated top-liquidity universe (26 OG coins only)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import polars as pl


from report_service.metrics import calculate_metrics

BINANCE_URL = "https://fapi.binance.com"
OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")

# Curated: Only Tier 1 OG coins (2019-2020 listed, deep liquidity, real projects)
# Removed: small caps, meme coins, newer listings
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "ADAUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", "DOTUSDT",
    "BCHUSDT", "AAVEUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT",
    "FILUSDT", "XLMUSDT", "ETCUSDT", "TRXUSDT", "DOGEUSDT",
    "XMRUSDT", "ZECUSDT", "ENJUSDT", "CRVUSDT", "ONTUSDT",
    "DASHUSDT",
]
TIMEFRAME = "6h"
INITIAL_EQUITY = 10_000.0
COMMISSION = 0.0004
START_TS = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1000)


async def fetch_klines_from(client, symbol, start_ms, target_bars=20000):
    all_rows = []
    start_time = start_ms
    while len(all_rows) < target_bars:
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 1500, "startTime": start_time}
        try:
            resp = await client.get(f"{BINANCE_URL}/fapi/v1/klines", params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                continue
            if resp.status_code != 200:
                break
            raw = resp.json()
            if not raw:
                break
            rows = [{"open_time": datetime.fromtimestamp(k[0]/1000, tz=UTC),
                     "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                     "close": float(k[4]), "volume": float(k[5])} for k in raw]
            all_rows.extend(rows)
            if len(raw) < 1500:
                break
            start_time = raw[-1][0] + 1
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"    err: {e}", flush=True)
            break
    if not all_rows:
        return None
    return pl.DataFrame(all_rows).unique(subset=["open_time"]).sort("open_time")


async def fetch_all():
    bars = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for sym in SYMBOLS:
            print(f"  Fetching {sym}...", end=" ", flush=True)
            df = await fetch_klines_from(client, sym, START_TS)
            if df is not None and len(df) >= 500:
                first = df["open_time"][0].strftime("%Y-%m-%d")
                last = df["open_time"][-1].strftime("%Y-%m-%d")
                bars[sym] = df
                print(f"✓ {len(df)} bars ({first} → {last})", flush=True)
            else:
                print(f"✗ skipped", flush=True)
            await asyncio.sleep(0.15)
    return bars


def compute_adx(high, low, close, period=14):
    n = len(close)
    adx = np.zeros(n)
    dip_o, dim_o = np.zeros(n), np.zeros(n)
    if n < period * 2:
        return adx, dip_o, dim_o
    tr, pdm, mdm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up, down = high[i]-high[i-1], low[i-1]-low[i]
        pdm[i] = up if up > down and up > 0 else 0
        mdm[i] = down if down > up and down > 0 else 0
    atr_s, pdm_s, mdm_s = np.zeros(n), np.zeros(n), np.zeros(n)
    atr_s[period] = tr[1:period+1].sum()
    pdm_s[period] = pdm[1:period+1].sum()
    mdm_s[period] = mdm[1:period+1].sum()
    for i in range(period+1, n):
        atr_s[i] = atr_s[i-1] - atr_s[i-1]/period + tr[i]
        pdm_s[i] = pdm_s[i-1] - pdm_s[i-1]/period + pdm[i]
        mdm_s[i] = mdm_s[i-1] - mdm_s[i-1]/period + mdm[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        dip_o = np.where(atr_s > 0, 100*pdm_s/atr_s, 0)
        dim_o = np.where(atr_s > 0, 100*mdm_s/atr_s, 0)
        ds = dip_o + dim_o
        dx = np.where(ds > 0, 100*np.abs(dip_o - dim_o)/ds, 0)
    if period*2 < n:
        adx[period*2] = np.nanmean(dx[period+1:period*2+1])
        for i in range(period*2+1, n):
            adx[i] = (adx[i-1]*(period-1) + dx[i])/period
    return adx, dip_o, dim_o


def compute_atr(high, low, close, period=14):
    n = len(close)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr[i] = (atr[i-1]*(period-1) + tr)/period if i >= period else tr
    return atr


def precompute(bars_dict):
    data = {}
    for sym, df in bars_dict.items():
        c, h, lo, vol = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
        atr = compute_atr(h, lo, c)
        adx, dip, dim = compute_adx(h, lo, c)
        rets = np.zeros(len(c))
        for i in range(1, len(c)):
            rets[i] = (c[i]-c[i-1])/c[i-1] if c[i-1] > 0 else 0
        rvol = np.zeros(len(c))
        for i in range(20, len(c)):
            rvol[i] = np.std(rets[i-20:i])
        data[sym] = {"close": c, "high": h, "low": lo, "volume": vol, "atr": atr,
                     "adx": adx, "dip": dip, "dim": dim, "rvol": rvol, "start_idx": 0, "length": len(c)}
    return data


def build_timeline(bars_dict):
    all_ts = set()
    for df in bars_dict.values():
        all_ts.update(df["open_time"].to_list())
    timeline = sorted(all_ts)
    return timeline, {ts: i for i, ts in enumerate(timeline)}


def align_data(bars_dict, data, ts_to_idx):
    for sym, df in bars_dict.items():
        data[sym]["start_idx"] = ts_to_idx[df["open_time"].to_list()[0]]


def compute_signals(data, timeline, params):
    adx_ens = params["adx_ensemble"]
    MOM_LBS = params["mom_lookbacks"]
    use_btc = params.get("btc_filter", False)
    persistence = params.get("signal_persistence", 1)
    n_global = len(timeline)
    sigs = {sym: np.zeros(n_global) for sym in data}
    
    for sym, d in data.items():
        c, h, lo, vol = d["close"], d["high"], d["low"], d["volume"]
        adx, dip, dim, rvol = d["adx"], d["dip"], d["dim"], d["rvol"]
        start, n = d["start_idx"], d["length"]
        raw_sig = np.zeros(n_global)
        
        for gt in range(start + max(MOM_LBS) + 1, start + n):
            i = gt - start
            raw_signals = []
            for adx_t in adx_ens:
                if adx[i] < adx_t:
                    raw_signals.append(0.0)
                    continue
                di_long, di_short = dip[i] > dim[i], dim[i] > dip[i]
                median_rvol = np.median(rvol[max(0,i-120):i]) if i > 120 else rvol[i]
                vol_ratio = rvol[i]/median_rvol if median_rvol > 0 else 1.0
                votes, tw = 0, 0
                for j, lb in enumerate(MOM_LBS):
                    base_w = 1.0/(j+1)
                    if j == 0: w = base_w * min(vol_ratio, 2.0)
                    elif j == len(MOM_LBS)-1: w = base_w * min(1.0/max(vol_ratio, 0.5), 2.0)
                    else: w = base_w
                    ret = (c[i]-c[i-lb])/c[i-lb] if c[i-lb] > 0 else 0
                    votes += np.sign(ret) * w * min(abs(ret)*20, 1.0)
                    tw += w
                raw = votes/tw if tw > 0 else 0
                dh, dl = h[i-20:i].max(), lo[i-20:i].min()
                dm, dr = (dh+dl)/2, dh-dl
                if dr > 1e-10:
                    dp = (c[i]-dm)/(dr/2)
                    if raw > 0 and dp < 0: raw *= 0.2
                    elif raw < 0 and dp > 0: raw *= 0.2
                if i >= 20:
                    avg_vol = vol[i-20:i].mean()
                    if avg_vol > 0:
                        vr = vol[i]/avg_vol
                        if vr < 0.8: raw *= 0.5
                        elif vr > 1.5: raw *= 1.2
                if raw > 0 and not di_long: raw *= 0.3
                elif raw < 0 and not di_short: raw *= 0.3
                if use_btc and "BTCUSDT" in data and sym != "BTCUSDT":
                    btc_d = data["BTCUSDT"]
                    btc_li = gt - btc_d["start_idx"]
                    if 0 <= btc_li < btc_d["length"] and btc_li >= 60:
                        bc = btc_d["close"]
                        br = (bc[btc_li]-bc[btc_li-60])/bc[btc_li-60] if bc[btc_li-60] > 0 else 0
                        if raw > 0 and br < -0.05: raw *= 0.5
                        elif raw < 0 and br > 0.05: raw *= 0.5
                raw_signals.append(np.clip(raw, -1, 1))
            raw_sig[gt] = np.mean(raw_signals) if raw_signals else 0
        
        if persistence > 1:
            filtered = np.zeros(n_global)
            streak, last_dir = 0, 0
            for gt in range(n_global):
                d_now = np.sign(raw_sig[gt])
                if d_now == last_dir and d_now != 0: streak += 1
                elif d_now != 0: streak, last_dir = 1, d_now
                else: streak, last_dir = 0, 0
                filtered[gt] = raw_sig[gt] if streak >= persistence else 0
            sigs[sym] = filtered
        else:
            sigs[sym] = raw_sig
    return sigs


def run_backtest(data, sigs, timeline, start_idx, end_idx, params):
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
    DD_BREAKER = params.get("dd_circuit_breaker", 0)
    RISK_PARITY = params.get("risk_parity", True)

    equity, cash, peak_eq = INITIAL_EQUITY, INITIAL_EQUITY, INITIAL_EQUITY
    positions, curve, trades, recent_returns = {}, [], [], []

    def get_close(sym, gt):
        li = gt - data[sym]["start_idx"]
        return data[sym]["close"][li] if 0 <= li < data[sym]["length"] else None

    def get_atr(sym, gt):
        li = gt - data[sym]["start_idx"]
        return data[sym]["atr"][li] if 0 <= li < data[sym]["length"] else None

    def get_rvol(sym, gt):
        li = gt - data[sym]["start_idx"]
        return data[sym]["rvol"][li] if 0 <= li < data[sym]["length"] else None

    for t in range(start_idx, end_idx):
        pv = cash
        for s, po in positions.items():
            cp = get_close(s, t)
            if cp: pv += abs(po["qty"])*po["entry_price"] + po["qty"]*(cp - po["entry_price"])
        equity = pv
        peak_eq = max(peak_eq, equity)
        cur_dd = (peak_eq - equity)/peak_eq if peak_eq > 0 else 0

        if curve:
            prev = curve[-1][1]
            if prev > 0:
                recent_returns.append((pv - prev)/prev)
                if len(recent_returns) > 120: recent_returns.pop(0)

        vol_scale = 1.0
        if TARGET_VOL > 0 and len(recent_returns) >= 20:
            rv = np.std(recent_returns[-60:]) * np.sqrt(4*365)
            if rv > 0: vol_scale = np.clip(TARGET_VOL/rv, 0.3, 2.0)
        dd_scale = 0.5 if DD_BREAKER > 0 and cur_dd > DD_BREAKER else 1.0

        to_close = []
        for s, po in positions.items():
            held = t - po["entry_bar"]
            a, cp = get_atr(s, t), get_close(s, t)
            if a is None or cp is None:
                to_close.append(s)
                continue
            if MAX_HOLD > 0 and held >= MAX_HOLD:
                to_close.append(s)
                continue
            cs = ATR_STOP
            if TIGHTEN_AFTER > 0:
                unr = ((cp - po["entry_price"])/a if po["qty"] > 0 else (po["entry_price"] - cp)/a) if a > 0 else 0
                if unr > TIGHTEN_AFTER: cs = TIGHTENED_STOP
            if po["qty"] > 0:
                po["best_price"] = max(po.get("best_price", po["entry_price"]), cp)
                if cp < po["best_price"] - cs*a and held >= MIN_HOLD: to_close.append(s)
            else:
                po["best_price"] = min(po.get("best_price", po["entry_price"]), cp)
                if cp > po["best_price"] + cs*a and held >= MIN_HOLD: to_close.append(s)
            if held >= MIN_HOLD:
                sig_val = sigs[s][t]
                if po["qty"] > 0 and sig_val < -0.15: to_close.append(s)
                elif po["qty"] < 0 and sig_val > 0.15: to_close.append(s)

        if PT_TP > 0:
            for s, po in positions.items():
                if s in to_close: continue
                a, cp = get_atr(s, t), get_close(s, t)
                if a is None or cp is None: continue
                if po["qty"] > 0 and not po.get("pt") and cp > po["entry_price"] + PT_TP*a:
                    hq = po["qty"]/2
                    pnl = hq*(cp - po["entry_price"]) - abs(hq)*cp*COMMISSION
                    cash += abs(hq)*po["entry_price"] + pnl
                    po["qty"] -= hq
                    po["pt"] = True
                    trades.append({"pnl": pnl})
                elif po["qty"] < 0 and not po.get("pt") and cp < po["entry_price"] - PT_TP*a:
                    hq = po["qty"]/2
                    pnl = hq*(cp - po["entry_price"]) - abs(hq)*cp*COMMISSION
                    cash += abs(hq)*po["entry_price"] + pnl
                    po["qty"] -= hq
                    po["pt"] = True
                    trades.append({"pnl": pnl})

        for s in set(to_close):
            po = positions[s]
            cp = get_close(s, t) or po["entry_price"]
            pnl = po["qty"]*(cp - po["entry_price"]) - abs(po["qty"])*cp*COMMISSION
            cash += abs(po["qty"])*po["entry_price"] + pnl
            trades.append({"pnl": pnl})
            del positions[s]

        if t % REBAL == 0 and len(positions) < MAX_POS:
            cands = []
            for s in data:
                if s in positions: continue
                li = t - data[s]["start_idx"]
                if li < 150 or li >= data[s]["length"]: continue
                if abs(sigs[s][t]) >= SIG_T:
                    cands.append((s, sigs[s][t]))
            cands.sort(key=lambda x: abs(x[1]), reverse=True)

            for s, sg in cands:
                if len(positions) >= MAX_POS: break
                cp, a, rv = get_close(s, t), get_atr(s, t), get_rvol(s, t)
                if cp is None or a is None or a < 1e-10 or cp < 1e-10: continue
                if RISK_PARITY:
                    av = rv if rv and rv > 0 else 0.01
                    trp = equity * RISK_PT / MAX_POS
                    qty_abs = min(trp/(av*cp*np.sqrt(20)), equity*0.15/cp)
                else:
                    na = a/cp
                    ma = 0.02
                    iv = np.clip(ma/na, 0.5, 2.0)
                    qty_abs = min(equity*RISK_PT*iv*vol_scale/(ATR_STOP*a), equity*0.15/cp)
                qty_abs *= vol_scale * dd_scale
                if qty_abs*cp < 10: continue
                qty = qty_abs if sg > 0 else -qty_abs
                cash -= abs(qty)*cp + abs(qty)*cp*COMMISSION
                positions[s] = {"qty": qty, "entry_price": cp, "entry_bar": t, "best_price": cp}

        pv = cash
        for s, po in positions.items():
            cp = get_close(s, t)
            if cp: pv += abs(po["qty"])*po["entry_price"] + po["qty"]*(cp - po["entry_price"])
        curve.append((timeline[t], pv))
        if t % 2000 == 0: print(f"    step {t}/{end_idx}: equity=${pv:,.0f}", flush=True)

    for s, po in positions.items():
        cp = get_close(s, end_idx-1) or po["entry_price"]
        pnl = po["qty"]*(cp - po["entry_price"]) - abs(po["qty"])*cp*COMMISSION
        trades.append({"pnl": pnl})
    return curve, trades


def calc_ulcer(curve):
    eq = np.array([e[1] for e in curve])
    if len(eq) < 10: return 999
    rm = np.maximum.accumulate(eq)
    dd = (rm - eq)/rm
    return np.sqrt(np.mean(dd**2))


async def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("CTA-Forge v11c — Curated OG Universe (26 symbols)", flush=True)
    print("=" * 60, flush=True)

    print(f"\nFetching from 2019-09-01...", flush=True)
    bars = await fetch_all()
    print(f"\n→ {len(bars)} symbols loaded\n", flush=True)

    timeline, ts_to_idx = build_timeline(bars)
    print(f"Unified timeline: {len(timeline)} bars ({timeline[0].strftime('%Y-%m-%d')} → {timeline[-1].strftime('%Y-%m-%d')})", flush=True)

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
    }

    print("\nComputing signals...", flush=True)
    sigs = compute_signals(data, timeline, params)

    start, end = 200, len(timeline)
    days = (timeline[end-1] - timeline[start]).days
    print(f"Backtesting: {timeline[start].strftime('%Y-%m-%d')} → {timeline[end-1].strftime('%Y-%m-%d')} ({days} days)\n", flush=True)

    curve, trades = run_backtest(data, sigs, timeline, start, end, params)
    m = calculate_metrics(curve, trades)
    ulcer = calc_ulcer(curve)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS ({days} days, {len(bars)} symbols)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Return: {m.total_return*100:+.1f}%  Ann: {m.annualized_return*100:+.1f}%", flush=True)
    print(f"  Sharpe: {m.sharpe_ratio:.2f}  Sortino: {m.sortino_ratio:.2f}", flush=True)
    print(f"  MaxDD:  {m.max_drawdown*100:.1f}%  Calmar: {m.calmar_ratio:.2f}", flush=True)
    print(f"  PF: {m.profit_factor:.2f}  Win: {m.win_rate*100:.1f}%  Trades: {m.num_trades}", flush=True)
    print(f"  Ulcer: {ulcer:.4f}", flush=True)

    yearly = {}
    for ts, eq in curve:
        yr = ts.year
        if yr not in yearly: yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq

    print(f"\nYearly returns:", flush=True)
    for yr in sorted(yearly):
        yr_ret = (yearly[yr]["last"] - yearly[yr]["first"])/yearly[yr]["first"]*100
        print(f"  {yr}: {yr_ret:+.1f}%", flush=True)

    # Charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_b, eq_b = [e[0] for e in curve], [e[1] for e in curve]

    fig, axes = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[3, 1, 1.5], gridspec_kw={"hspace": 0.2})
    axes[0].plot(ts_b, eq_b, linewidth=0.8, color="#2ecc71")
    axes[0].axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e >= INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#2ecc71")
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e < INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#e74c3c")
    axes[0].set_title(f"CTA-Forge v11c — Curated OG Universe · 6h · $10K · {days} days", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equity ($)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].grid(True, alpha=0.2)

    mt = f"Return: {m.total_return*100:+.1f}%  Ann: {m.annualized_return*100:+.1f}%\nSharpe: {m.sharpe_ratio:.2f}  Sortino: {m.sortino_ratio:.2f}\nMaxDD: {m.max_drawdown*100:.1f}%  Calmar: {m.calmar_ratio:.2f}\nPF: {m.profit_factor:.2f}  Win: {m.win_rate*100:.1f}%  Trades: {m.num_trades}\nUlcer: {ulcer:.4f}"
    axes[0].text(0.98, 0.02, mt, transform=axes[0].transAxes, fontsize=9, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.85), color="white", family="monospace")
    yr_str = " | ".join([f"{yr}:{(yearly[yr]['last']-yearly[yr]['first'])/yearly[yr]['first']*100:+.0f}%" for yr in sorted(yearly)])
    axes[0].text(0.02, 0.98, yr_str, transform=axes[0].transAxes, fontsize=8, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#34495e", alpha=0.7), color="white", family="monospace")

    eq_arr = np.array(eq_b)
    rm = np.maximum.accumulate(eq_arr)
    dd_pct = (rm - eq_arr)/rm*100
    axes[1].fill_between(ts_b, 0, -dd_pct, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.2)

    monthly = {}
    for ii in range(1, len(eq_b)):
        key = ts_b[ii].strftime("%Y-%m")
        if key not in monthly: monthly[key] = {"start": eq_b[ii-1]}
        monthly[key]["end"] = eq_b[ii]
    mos = list(monthly.keys())
    rets = [(monthly[m]["end"] - monthly[m]["start"])/monthly[m]["start"]*100 for m in mos]
    axes[2].bar(range(len(mos)), rets, color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rets], alpha=0.8, width=0.8)
    step = max(1, len(mos)//24)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels([mos[ii] for ii in range(0, len(mos), step)], rotation=45, ha="right", fontsize=6)
    axes[2].set_ylabel("Monthly %")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.2, axis="y")
    axes[2].text(0.02, 0.95, f"Positive: {sum(1 for r in rets if r > 0)}/{len(rets)} months ({sum(1 for r in rets if r > 0)/len(rets)*100:.0f}%)",
                 transform=axes[2].transAxes, fontsize=8, va="top")

    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    fig.savefig(OUT_DIR / "backtest_v11c_curated.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (OUT_DIR / "metrics_v11c_curated.json").write_text(json.dumps({
        "period": f"{timeline[start].strftime('%Y-%m-%d')} → {timeline[end-1].strftime('%Y-%m-%d')}",
        "days": days, "symbols": len(bars), "symbol_list": sorted(bars.keys()),
        "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio, "return": m.total_return,
        "ann_return": m.annualized_return, "max_dd": m.max_drawdown, "calmar": m.calmar_ratio,
        "pf": m.profit_factor, "win_rate": m.win_rate, "trades": m.num_trades, "ulcer": ulcer,
        "yearly": {str(yr): (yearly[yr]["last"]-yearly[yr]["first"])/yearly[yr]["first"]*100 for yr in sorted(yearly)},
    }, indent=2))

    print(f"\n✅ Done in {time.time()-t0:.0f}s", flush=True)
    print(f"Charts: {OUT_DIR / 'backtest_v11c_curated.png'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
