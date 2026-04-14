"""CTA-Forge v10g — Ultra long-term backtest (2021-2026, ~5 years)."""

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
OUT_DIR = Path("/home/node/.openclaw/workspace/cta-forge-dev/backtest-results")

# Some coins launched later, so we'll use what's available from 2021-01
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT",
    "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "SUIUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT",
]
TIMEFRAME = "6h"
INITIAL_EQUITY = 10_000.0
COMMISSION = 0.0004

# Target: from 2021-01-01 to now
START_TS = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000)


async def fetch_klines_from(client, symbol, start_ms, target_bars=8000):
    """Fetch forward from start_ms using pagination."""
    all_rows = []; start_time = start_ms
    while len(all_rows) < target_bars:
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 1500, "startTime": start_time}
        try:
            resp = await client.get(f"{BINANCE_URL}/fapi/v1/klines", params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5); continue
            if resp.status_code != 200:
                break
            raw = resp.json()
            if not raw:
                break
            rows = [{"open_time": datetime.fromtimestamp(k[0]/1000, tz=UTC),
                      "open": float(k[1]), "high": float(k[2]),
                      "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} for k in raw]
            all_rows.extend(rows)
            if len(raw) < 1500:
                break  # no more data
            start_time = raw[-1][0] + 1  # next bar after last
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"    err: {e}", flush=True)
            break
    if not all_rows:
        return None
    df = pl.DataFrame(all_rows).unique(subset=["open_time"]).sort("open_time")
    return df


async def fetch_all():
    bars = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for sym in SYMBOLS:
            df = await fetch_klines_from(client, sym, START_TS, target_bars=8000)
            if df is not None and len(df) >= 500:
                first = df["open_time"][0].strftime("%Y-%m-%d")
                last = df["open_time"][-1].strftime("%Y-%m-%d")
                bars[sym] = df
                print(f"  ✓ {sym}: {len(df)} bars ({first} → {last})", flush=True)
            else:
                print(f"  ✗ {sym}: skipped (insufficient data)", flush=True)
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


def precompute_all(bars_dict, max_len):
    data = {}
    for sym, df in bars_dict.items():
        c = df["close"].to_numpy()[:max_len]; h = df["high"].to_numpy()[:max_len]
        lo = df["low"].to_numpy()[:max_len]; vol = df["volume"].to_numpy()[:max_len]
        atr = compute_atr(h, lo, c); adx, dip, dim = compute_adx(h, lo, c)
        rets = np.zeros(len(c))
        for i in range(1, len(c)):
            rets[i] = (c[i]-c[i-1])/c[i-1] if c[i-1] > 0 else 0
        rvol = np.zeros(len(c))
        for i in range(20, len(c)):
            rvol[i] = np.std(rets[i-20:i])
        data[sym] = {"close": c, "high": h, "low": lo, "volume": vol,
                      "atr": atr, "adx": adx, "dip": dip, "dim": dim, "rvol": rvol}
    return data


def compute_signals(data, params):
    adx_ens = params["adx_ensemble"]
    MOM_LBS = params["mom_lookbacks"]
    use_btc = params.get("btc_filter", False)
    btc_close = data.get("BTCUSDT", {}).get("close")
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

                # Adaptive lookback weighting
                median_rvol = np.median(rvol[max(0,i-120):i]) if i > 120 else rvol[i]
                vol_ratio = rvol[i] / median_rvol if median_rvol > 0 else 1.0

                votes = 0; tw = 0
                for j, lb in enumerate(MOM_LBS):
                    base_w = 1.0/(j+1)
                    if j == 0: w = base_w * min(vol_ratio, 2.0)
                    elif j == len(MOM_LBS)-1: w = base_w * min(1.0/max(vol_ratio, 0.5), 2.0)
                    else: w = base_w
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

        if persistence > 1:
            filtered = np.zeros(n); streak = 0; last_dir = 0
            for i in range(n):
                d_now = np.sign(raw_sig[i])
                if d_now == last_dir and d_now != 0: streak += 1
                elif d_now != 0: streak = 1; last_dir = d_now
                else: streak = 0; last_dir = 0
                filtered[i] = raw_sig[i] if streak >= persistence else 0
            sigs[sym] = filtered
        else:
            sigs[sym] = raw_sig
    return sigs


def run_backtest(data, sigs, ts_list, start_idx, end_idx, params):
    SIG_T = params["signal_threshold"]; MIN_HOLD = params["min_hold_bars"]
    ATR_STOP = params["atr_stop_mult"]; RISK_PT = params["risk_per_trade"]
    MAX_POS = params["max_positions"]; REBAL = params["rebalance_every"]
    PT_TP = params.get("partial_take_profit", 0); TARGET_VOL = params.get("target_vol", 0)
    MAX_HOLD = params.get("max_hold_bars", 0)
    TIGHTEN_AFTER = params.get("tighten_stop_after_atr", 0)
    TIGHTENED_STOP = params.get("tightened_stop_mult", 3.0)
    DD_BREAKER = params.get("dd_circuit_breaker", 0)
    RISK_PARITY = params.get("risk_parity", True)

    equity = INITIAL_EQUITY; cash = INITIAL_EQUITY; peak_eq = INITIAL_EQUITY
    positions = {}; curve = []; trades = []; recent_returns = []

    for t in range(start_idx, end_idx):
        pv = cash
        for s, po in positions.items():
            pv += abs(po["qty"])*po["entry_price"]+po["qty"]*(data[s]["close"][t]-po["entry_price"])
        equity = pv; peak_eq = max(peak_eq, equity)
        cur_dd = (peak_eq-equity)/peak_eq if peak_eq>0 else 0

        if curve:
            prev = curve[-1][1]
            if prev > 0:
                recent_returns.append((pv-prev)/prev)
                if len(recent_returns) > 120: recent_returns.pop(0)

        vol_scale = 1.0
        if TARGET_VOL > 0 and len(recent_returns) >= 20:
            rv = np.std(recent_returns[-60:])*np.sqrt(4*365)
            if rv > 0: vol_scale = np.clip(TARGET_VOL/rv, 0.3, 2.0)
        dd_scale = 0.5 if DD_BREAKER > 0 and cur_dd > DD_BREAKER else 1.0

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

        if PT_TP > 0:
            for s, po in positions.items():
                if s in to_close: continue
                a = data[s]["atr"][t]
                if po["qty"] > 0 and not po.get("pt") and data[s]["close"][t] > po["entry_price"]+PT_TP*a:
                    hq = po["qty"]/2
                    pnl = hq*(data[s]["close"][t]-po["entry_price"])-abs(hq)*data[s]["close"][t]*COMMISSION
                    cash += abs(hq)*po["entry_price"]+pnl; po["qty"]-=hq; po["pt"]=True; trades.append({"pnl": pnl})
                elif po["qty"] < 0 and not po.get("pt") and data[s]["close"][t] < po["entry_price"]-PT_TP*a:
                    hq = po["qty"]/2
                    pnl = hq*(data[s]["close"][t]-po["entry_price"])-abs(hq)*data[s]["close"][t]*COMMISSION
                    cash += abs(hq)*po["entry_price"]+pnl; po["qty"]-=hq; po["pt"]=True; trades.append({"pnl": pnl})

        for s in set(to_close):
            po = positions[s]
            pnl = po["qty"]*(data[s]["close"][t]-po["entry_price"])-abs(po["qty"])*data[s]["close"][t]*COMMISSION
            cash += abs(po["qty"])*po["entry_price"]+pnl; trades.append({"pnl": pnl}); del positions[s]

        if t % REBAL == 0 and len(positions) < MAX_POS:
            cands = [(s, sigs[s][t]) for s in data if s not in positions and abs(sigs[s][t])>=SIG_T]
            cands.sort(key=lambda x: abs(x[1]), reverse=True)
            for s, sg in cands:
                if len(positions)>=MAX_POS: break
                pr = data[s]["close"][t]; a = data[s]["atr"][t]
                if a<1e-10 or pr<1e-10: continue
                if RISK_PARITY:
                    av = data[s]["rvol"][t] if data[s]["rvol"][t] > 0 else 0.01
                    trp = equity * RISK_PT / MAX_POS
                    qty_abs = min(trp/(av*pr*np.sqrt(20)), equity*0.15/pr)
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

        if t % 1000 == 0:
            print(f"    step {t}/{end_idx}: equity=${pv:,.0f}", flush=True)

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


async def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("CTA-Forge v10g — Long-Term Backtest (2021-01 → 2026)", flush=True)
    print("="*60, flush=True)

    print(f"\nFetching from 2021-01-01...", flush=True)
    bars = await fetch_all()
    print(f"→ {len(bars)} symbols loaded\n", flush=True)

    # Find common date range (only use symbols available from early enough)
    min_bars = {}
    for sym, df in bars.items():
        first = df["open_time"][0]
        min_bars[sym] = (first, len(df))

    # Filter: only keep symbols with data from before 2021-07-01
    cutoff = datetime(2021, 7, 1, tzinfo=UTC)
    valid = {s: df for s, df in bars.items() if df["open_time"][0] < cutoff}
    late = {s: df for s, df in bars.items() if s not in valid}

    print(f"Symbols from 2021H1: {len(valid)} — {sorted(valid.keys())}", flush=True)
    if late:
        print(f"Later listings (still included, shorter history): {sorted(late.keys())}", flush=True)

    # Use all symbols but align to common end
    # For simplicity: use the minimum length among valid symbols
    all_bars = {**valid, **late}
    min_len = min(len(df) for df in all_bars.values())
    print(f"Common bars (aligned to shortest): {min_len} ({min_len*6/24:.0f} days)\n", flush=True)

    # But that's limited by late listings. Instead: run with valid symbols only for full period
    # Then also run with all 19 for comparison (shorter period)
    print("=" * 60, flush=True)
    print(f"RUN 1: {len(valid)} OG symbols (full 2021→2026)", flush=True)
    print("=" * 60, flush=True)

    min_len_v = min(len(df) for df in valid.values())
    ts_v = list(valid[next(iter(valid))]["open_time"][:min_len_v].to_list())
    data_v = precompute_all(valid, min_len_v)
    print(f"Bars: {min_len_v} ({min_len_v*6/24:.0f}d), {ts_v[0].strftime('%Y-%m-%d')} → {ts_v[-1].strftime('%Y-%m-%d')}", flush=True)

    params = {
        "mom_lookbacks": [20, 60, 120], "adx_threshold": 25, "adx_ensemble": [22, 27, 32],
        "signal_threshold": 0.40, "min_hold_bars": 16, "atr_stop_mult": 5.0,
        "risk_per_trade": 0.015, "max_positions": 5, "rebalance_every": 4,
        "partial_take_profit": 2.5, "target_vol": 0.12,
        "btc_filter": True, "max_hold_bars": 100,
        "tighten_stop_after_atr": 2.0, "tightened_stop_mult": 3.0,
        "dd_circuit_breaker": 0.08, "risk_parity": True,
        "signal_persistence": 2, "adaptive_lookback": True,
    }

    sigs_v = compute_signals(data_v, params)
    curve_v, trades_v = run_backtest(data_v, sigs_v, ts_v, 150, min_len_v, params)
    m_v = calculate_metrics(curve_v, trades_v)
    ulcer_v = calc_ulcer(curve_v)

    print(f"\nFull-period results ({len(valid)} symbols, {(ts_v[-1]-ts_v[150]).days}d):", flush=True)
    print(f"  Return: {m_v.total_return*100:+.1f}%  Ann: {m_v.annualized_return*100:+.1f}%", flush=True)
    print(f"  Sharpe: {m_v.sharpe_ratio:.2f}  Sortino: {m_v.sortino_ratio:.2f}", flush=True)
    print(f"  MaxDD:  {m_v.max_drawdown*100:.1f}%  Calmar: {m_v.calmar_ratio:.2f}", flush=True)
    print(f"  PF: {m_v.profit_factor:.2f}  Win: {m_v.win_rate*100:.1f}%  Trades: {m_v.num_trades}", flush=True)
    print(f"  Ulcer: {ulcer_v:.4f}", flush=True)

    # Yearly breakdown
    yearly = {}
    eq_list = [(e[0], e[1]) for e in curve_v]
    for ts, eq in eq_list:
        yr = ts.year
        if yr not in yearly: yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq
    print(f"\nYearly returns:", flush=True)
    for yr in sorted(yearly):
        yr_ret = (yearly[yr]["last"]-yearly[yr]["first"])/yearly[yr]["first"]*100
        print(f"  {yr}: {yr_ret:+.1f}%", flush=True)

    # Charts
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; import matplotlib.dates as mdates
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_b = [e[0] for e in curve_v]; eq_b = [e[1] for e in curve_v]
    days = (ts_b[-1]-ts_b[0]).days

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), height_ratios=[3, 1, 1.5], gridspec_kw={"hspace": 0.2})

    # Equity
    axes[0].plot(ts_b, eq_b, linewidth=0.9, color="#2ecc71")
    axes[0].axhline(y=INITIAL_EQUITY, color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e>=INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#2ecc71")
    axes[0].fill_between(ts_b, INITIAL_EQUITY, eq_b, where=[e<INITIAL_EQUITY for e in eq_b], alpha=0.1, color="#e74c3c")

    # Market regime annotations
    axes[0].set_title(f"CTA-Forge v10g — {days}d Ultra Long-Term ({ts_b[0].strftime('%Y-%m')} → {ts_b[-1].strftime('%Y-%m')})\n"
                       f"{len(valid)} symbols · 6h · $10K start", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equity ($)"); axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].grid(True, alpha=0.2)

    mt = (f"Return: {m_v.total_return*100:+.1f}%  Ann: {m_v.annualized_return*100:+.1f}%\n"
          f"Sharpe: {m_v.sharpe_ratio:.2f}  Sortino: {m_v.sortino_ratio:.2f}\n"
          f"MaxDD: {m_v.max_drawdown*100:.1f}%  Calmar: {m_v.calmar_ratio:.2f}\n"
          f"PF: {m_v.profit_factor:.2f}  Win: {m_v.win_rate*100:.1f}%  Trades: {m_v.num_trades}\n"
          f"Ulcer: {ulcer_v:.4f}")
    axes[0].text(0.98, 0.02, mt, transform=axes[0].transAxes, fontsize=9, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.85), color="white", family="monospace")

    # Yearly return annotations
    yr_str = " | ".join([f"{yr}:{(yearly[yr]['last']-yearly[yr]['first'])/yearly[yr]['first']*100:+.0f}%" for yr in sorted(yearly)])
    axes[0].text(0.02, 0.98, yr_str, transform=axes[0].transAxes, fontsize=8.5, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#34495e", alpha=0.7), color="white", family="monospace")

    # Drawdown
    eq_arr = np.array(eq_b); rm = np.maximum.accumulate(eq_arr); dd_pct = (rm-eq_arr)/rm*100
    axes[1].fill_between(ts_b, 0, -dd_pct, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %"); axes[1].grid(True, alpha=0.2)

    # Monthly returns
    monthly = {}
    for ii in range(1, len(eq_b)):
        key = ts_b[ii].strftime("%Y-%m")
        if key not in monthly: monthly[key] = {"start": eq_b[ii-1]}
        monthly[key]["end"] = eq_b[ii]
    mos = list(monthly.keys())
    rets = [(monthly[m_]["end"]-monthly[m_]["start"])/monthly[m_]["start"]*100 for m_ in mos]
    axes[2].bar(range(len(mos)), rets, color=["#2ecc71" if r>=0 else "#e74c3c" for r in rets], alpha=0.8, width=0.8)
    step = max(1, len(mos)//20)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels([mos[ii] for ii in range(0, len(mos), step)], rotation=45, ha="right", fontsize=6.5)
    axes[2].set_ylabel("Monthly %"); axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.2, axis="y")

    pos_months = sum(1 for r in rets if r > 0)
    axes[2].text(0.02, 0.95, f"Positive: {pos_months}/{len(rets)} months ({pos_months/len(rets)*100:.0f}%)",
                  transform=axes[2].transAxes, fontsize=8, va="top")

    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    fig.savefig(OUT_DIR / "backtest_v10g_longterm.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save metrics
    (OUT_DIR / "metrics_v10g_longterm.json").write_text(json.dumps({
        "period": f"{ts_b[0].strftime('%Y-%m-%d')} → {ts_b[-1].strftime('%Y-%m-%d')}",
        "days": days, "symbols": len(valid), "symbol_list": sorted(valid.keys()),
        "sharpe": m_v.sharpe_ratio, "sortino": m_v.sortino_ratio,
        "return": m_v.total_return, "ann_return": m_v.annualized_return,
        "max_dd": m_v.max_drawdown, "calmar": m_v.calmar_ratio,
        "pf": m_v.profit_factor, "win_rate": m_v.win_rate, "trades": m_v.num_trades,
        "ulcer": ulcer_v,
        "yearly": {str(yr): (yearly[yr]["last"]-yearly[yr]["first"])/yearly[yr]["first"]*100 for yr in sorted(yearly)},
    }, indent=2))

    print(f"\n✅ Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
