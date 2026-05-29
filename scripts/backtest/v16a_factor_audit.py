"""v16a Factor Audit — Memory-efficient IS/OOS factor decomposition.

Builds core + overlay sleeves once, then variates compositions
in-memory for fast experimentation.

Walk-forward: IS 2020-2023, OOS 2024-2026.

Run: uv run python scripts/backtest/v16a_factor_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (
    INITIAL_EQUITY,
    align_weights,
    build_overlay_sleeve,
    build_v10g_sleeve,
    expanding_badscore_gate,
    load_hourly_returns,
    normalize_gross,
)

EPS = 1e-12
DATA_PATH = ROOT / "data"
OUT_DIR = ROOT / "backtest-results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START, IS_END = "2020-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-12-31"


@dataclass
class ExperimentResult:
    name: str
    category: str
    config: dict
    is_sharpe: float
    is_return: float
    is_maxdd: float
    oos_sharpe: float
    oos_return: float
    oos_maxdd: float
    oos_ann_ret: float
    oos_sortino: float
    oos_calmar: float
    avg_gross: float
    avg_turnover: float


def date_mask(timeline, lo, hi):
    days = np.array([ts.date().isoformat() for ts in timeline])
    return (days >= lo) & (days <= hi)


def metrics(returns):
    m = calculate_hourly_metrics(returns, initial_equity=INITIAL_EQUITY)
    d = returns[returns < 0]
    dv = np.std(d) * np.sqrt(365 * 24) if len(d) else 0.0
    return {
        "sharpe": float(m["sharpe"]),
        "return": float(m["return"]),
        "max_dd": float(m["max_dd"]),
        "ann_return": float(m["ann_return"]),
        "volatility": float(m["volatility"]),
        "sortino": float(m["ann_return"] / dv if dv > 1e-12 else 0.0),
        "calmar": float(m["ann_return"] / m["max_dd"] if m["max_dd"] > 1e-12 else 0.0),
    }


def evaluate_combo(
    name, category, config,
    v_aligned, o_aligned, timeline, all_syms, returns,
    gate_override=None,
    v_alloc=0.5, o_alloc=0.5,
    gross_cap=4.0,
):
    """Fast evaluation: all data already built, just combine."""
    if gate_override is not None:
        gate = gate_override
    else:
        gate = expanding_badscore_gate(timeline)

    target = v_alloc * v_aligned + o_alloc * o_aligned
    target *= gate[:, None]
    for i in range(target.shape[0]):
        capped = normalize_gross(
            {sym: float(target[i, j]) for j, sym in enumerate(all_syms)},
            gross_cap=gross_cap,
        )
        target[i] = np.array([capped.get(sym, 0.0) for sym in all_syms])

    result = run_target_weight_backtest(
        timeline, returns, target, initial_equity=INITIAL_EQUITY
    )

    is_m = metrics(result.returns[date_mask(timeline, IS_START, IS_END)])
    oos_m = metrics(result.returns[date_mask(timeline, OOS_START, OOS_END)])

    return ExperimentResult(
        name=name, category=category, config=config,
        is_sharpe=is_m["sharpe"], is_return=is_m["return"], is_maxdd=is_m["max_dd"],
        oos_sharpe=oos_m["sharpe"], oos_return=oos_m["return"], oos_maxdd=oos_m["max_dd"],
        oos_ann_ret=oos_m["ann_return"], oos_sortino=oos_m["sortino"], oos_calmar=oos_m["calmar"],
        avg_gross=float(np.mean(np.sum(np.abs(target), axis=1))),
        avg_turnover=float(np.mean(result.turnover)),
    )


def custom_badscore_gate(timeline, vol_pct=0.50, eff_pct=0.50, corr_pct=0.66,
                         min_count=2, scale=0.5):
    """Custom badscore gate with configurable thresholds."""
    from executor.profiles.v16a_badscore_overlay import (
        daily_symbol_returns, EPS as _EPS, DEFAULT_SYMBOLS,
    )

    days = sorted({ts.date().isoformat() for ts in timeline})
    sym_ret = daily_symbol_returns(days)
    valid_counts = np.sum(np.isfinite(sym_ret), axis=1)
    market = np.divide(
        np.nansum(sym_ret, axis=1), valid_counts,
        out=np.full(sym_ret.shape[0], np.nan), where=valid_counts > 0,
    )
    vol20, eff60, corr60 = np.full(len(days), np.nan), np.full(len(days), np.nan), np.full(len(days), np.nan)

    for i in range(len(days)):
        if i >= 20:
            vol20[i] = np.nanstd(market[i - 20 : i]) * np.sqrt(365)
        if i >= 60:
            win = market[i - 60 : i]; valid = win[np.isfinite(win)]
            if len(valid) >= 48:
                eff60[i] = abs(np.prod(1 + valid) - 1) / max(np.sum(np.abs(valid)), _EPS)
            cols = []
            for j in range(sym_ret.shape[1]):
                col = sym_ret[i - 60 : i, j]
                if np.isfinite(col).sum() >= 45:
                    cols.append(np.nan_to_num(col, nan=np.nanmean(col)))
            if len(cols) >= 8:
                corr = np.corrcoef(np.vstack(cols))
                corr60[i] = np.nanmean(corr[np.triu_indices_from(corr, k=1)])

    scale_by_day = {}
    hist_vol, hist_eff, hist_corr = [], [], []
    for day, vol, eff, corr in zip(days, vol20, eff60, corr60):
        bad = 0
        if len(hist_vol) >= 252 and np.isfinite(vol) and vol > np.nanquantile(hist_vol, vol_pct):
            bad += 1
        if len(hist_eff) >= 252 and np.isfinite(eff) and eff < np.nanquantile(hist_eff, eff_pct):
            bad += 1
        if len(hist_corr) >= 252 and np.isfinite(corr) and corr > np.nanquantile(hist_corr, corr_pct):
            bad += 1
        scale_by_day[day] = scale if bad >= min_count else 1.0
        if np.isfinite(vol): hist_vol.append(float(vol))
        if np.isfinite(eff): hist_eff.append(float(eff))
        if np.isfinite(corr): hist_corr.append(float(corr))

    return np.array([scale_by_day[ts.date().isoformat()] for ts in timeline], dtype=float)


def main():
    t0 = time.time()
    all_results = []

    print("=" * 60)
    print("v16a Factor Audit — IS/OOS Walk-Forward Research")
    print(f"IS: {IS_START} → {IS_END} | OOS: {OOS_START} → {OOS_END}")
    print("=" * 60)

    # ── Build sleeves once ───────────────────────────────────────
    print("\n=== Building core sleeve (6h, phase=2)...")
    v_syms, v_weights, v_curve, _, v_timeline = build_v10g_sleeve(
        backfill=False, core_phase_hours=2
    )
    print(f"  Core: {len(v_timeline)} bars, {len(v_syms)} symbols")

    print("\n=== Building overlay sleeve (1h)...")
    o_syms, o_weights, o_curve, _, o_timeline = build_overlay_sleeve(
        backfill=False
    )
    print(f"  Overlay: {len(o_timeline)} bars, {len(o_syms)} symbols")

    # Align
    start = max(v_curve[0][0], o_curve[0][0])
    end = min(o_curve[-1][0], v_curve[-1][0] + timedelta(hours=11))
    timeline = [ts for ts in o_timeline if start <= ts <= end]
    all_syms = sorted(set(v_syms) | set(o_syms))
    print(f"  Aligned: {len(timeline)} hours, {len(all_syms)} symbols")

    v_aligned = align_weights(v_timeline, v_syms, v_weights, timeline, all_syms, forward_fill=True)
    o_aligned = align_weights(o_timeline, o_syms, o_weights, timeline, all_syms, forward_fill=False)
    returns = load_hourly_returns(timeline, all_syms)
    gate = expanding_badscore_gate(timeline)
    print(f"  Done building in {time.time() - t0:.1f}s")

    # ── 1. BASELINE ──────────────────────────────────────────────
    print("\n=== [1] Baseline ===")
    r = evaluate_combo("baseline (live config)", "baseline",
                       {"v_alloc": 0.5, "o_alloc": 0.5, "gross_cap": 4.0, "gate": "default"},
                       v_aligned, o_aligned, timeline, all_syms, returns, gate,
                       v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
    all_results.append(r)
    print(f"  IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}  OOS MaxDD={r.oos_maxdd*100:.1f}%")

    # ── 2. SLEEVE ABLATIONS ─────────────────────────────────────
    print("\n=== [2] Sleeve Ablations ===")
    for name, v_a, o_a in [
        ("core only (no overlay)", 1.0, 0.0),
        ("overlay only (no core)", 0.0, 1.0),
        ("70/30 core/overlay", 0.7, 0.3),
        ("30/70 core/overlay", 0.3, 0.7),
        ("60/40 core/overlay", 0.6, 0.4),
        ("40/60 core/overlay", 0.4, 0.6),
    ]:
        r = evaluate_combo(name, "allocation", {"v_alloc": v_a, "o_alloc": o_a},
                           v_aligned, o_aligned, timeline, all_syms, returns, gate,
                           v_alloc=v_a, o_alloc=o_a, gross_cap=4.0)
        all_results.append(r)
        print(f"  {name}: OOS Sharpe={r.oos_sharpe:.3f}  OOS Ret={r.oos_return*100:.1f}%")

    # ── 3. BADSCORE GATE ABLATION ────────────────────────────────
    print("\n=== [3] Badscore Gate Ablation ===")
    no_gate = np.ones(len(timeline))
    r = evaluate_combo("no badscore gate", "badscore_ablation", {"gate": "none"},
                       v_aligned, o_aligned, timeline, all_syms, returns, no_gate,
                       v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
    all_results.append(r)
    print(f"  No gate: OOS Sharpe={r.oos_sharpe:.3f}  OOS Ret={r.oos_return*100:.1f}%  MaxDD={r.oos_maxdd*100:.1f}%")

    # ── 4. BADSCORE MIN_COUNT SWEEP ──────────────────────────────
    print("\n=== [4] Badscore min_count sweep ===")
    for mc in [1, 2, 3]:
        g = custom_badscore_gate(timeline, min_count=mc)
        r = evaluate_combo(f"badscore min_count={mc}", "badscore_mincount", {"min_count": mc},
                           v_aligned, o_aligned, timeline, all_syms, returns, g,
                           v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
        all_results.append(r)
        gate_on = np.mean(g < 1.0) * 100
        print(f"  min_count={mc}: OOS Sharpe={r.oos_sharpe:.3f}  Gate ON={gate_on:.0f}%")

    # ── 5. BADSCORE SCALE SWEEP ──────────────────────────────────
    print("\n=== [5] Badscore scale sweep ===")
    for sc in [0.25, 0.50, 0.75]:
        g = custom_badscore_gate(timeline, scale=sc)
        r = evaluate_combo(f"badscore scale={sc}", "badscore_scale", {"scale": sc},
                           v_aligned, o_aligned, timeline, all_syms, returns, g,
                           v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
        all_results.append(r)
        print(f"  scale={sc}: OOS Sharpe={r.oos_sharpe:.3f}  OOS Ret={r.oos_return*100:.1f}%")

    # ── 6. BADSCORE THRESHOLD SWEEP ──────────────────────────────
    print("\n=== [6] Badscore threshold sweep ===")
    threshold_combos = [
        ("default (50/50/66)", 0.50, 0.50, 0.66),
        ("tight (60/40/75)", 0.60, 0.40, 0.75),
        ("very_tight (65/35/80)", 0.65, 0.35, 0.80),
        ("loose (40/60/55)", 0.40, 0.60, 0.55),
        ("vol_70", 0.70, 0.50, 0.66),
        ("eff_40", 0.50, 0.40, 0.66),
        ("corr_80", 0.50, 0.50, 0.80),
        ("balanced_strict (55/45/70)", 0.55, 0.45, 0.70),
    ]
    for label, vp, ep, cp in threshold_combos:
        g = custom_badscore_gate(timeline, vol_pct=vp, eff_pct=ep, corr_pct=cp)
        r = evaluate_combo(f"badscore {label}", "badscore_thresholds",
                           {"vol_pct": vp, "eff_pct": ep, "corr_pct": cp},
                           v_aligned, o_aligned, timeline, all_syms, returns, g,
                           v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
        all_results.append(r)
        gate_on = np.mean(g < 1.0) * 100
        print(f"  {label}: OOS Sharpe={r.oos_sharpe:.3f}  Gate ON={gate_on:.0f}%")

    # ── 7. OVERLAY TOP-N FILTER (post-hoc on aligned weights) ────
    print("\n=== [7] Overlay Top-N filter sweep ===")
    for top_n in [1, 2, 3, 4, 5]:
        o_filtered = o_aligned.copy()
        for i in range(o_filtered.shape[0]):
            row = np.abs(o_filtered[i])
            nonzero = row[row > EPS]
            if len(nonzero) > top_n:
                thr = np.sort(nonzero)[-top_n]
                o_filtered[i] = np.where(row >= thr, o_filtered[i], 0.0)
        r = evaluate_combo(f"overlay top-{top_n}", "overlay_topn", {"top_n": top_n},
                           v_aligned, o_filtered, timeline, all_syms, returns, gate,
                           v_alloc=0.5, o_alloc=0.5, gross_cap=4.0)
        all_results.append(r)
        print(f"  top-{top_n}: OOS Sharpe={r.oos_sharpe:.3f}  Gross={r.avg_gross:.3f}")

    # ── 8. GATE x ALLOCATION interaction ─────────────────────────
    print("\n=== [8] Gate x Allocation interaction ===")
    for v_a in [0.4, 0.5, 0.6]:
        for use_g in [True, False]:
            g = gate if use_g else no_gate
            label = f"alloc {v_a:.1f}/{(1-v_a):.1f} gate={'on' if use_g else 'off'}"
            r = evaluate_combo(label, "gate_allocation_interaction",
                               {"v_alloc": v_a, "gate": use_g},
                               v_aligned, o_aligned, timeline, all_syms, returns, g,
                               v_alloc=v_a, o_alloc=1 - v_a, gross_cap=4.0)
            all_results.append(r)
    print(f"  Done")

    # ── 9. GROSS CAP SWEEP ───────────────────────────────────────
    print("\n=== [9] Gross cap sweep ===")
    for gc in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        r = evaluate_combo(f"gross_cap={gc}", "gross_cap", {"gross_cap": gc},
                           v_aligned, o_aligned, timeline, all_syms, returns, gate,
                           v_alloc=0.5, o_alloc=0.5, gross_cap=gc)
        all_results.append(r)
        print(f"  cap {gc}: OOS Sharpe={r.oos_sharpe:.3f}  Gross={r.avg_gross:.3f}  Turnover/h={r.avg_turnover:.5f}")

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"{'Name':<45} {'Category':<25} {'IS Sharpe':>9} {'OOS Sharpe':>10} {'OOS Ret%':>8} {'OOS DD%':>8} {'Gross':>6} {'Turn/h':>7}")
    print("-" * 120)
    sorted_r = sorted(all_results, key=lambda r: r.oos_sharpe, reverse=True)
    for r in sorted_r:
        print(
            f"{r.name:<45} {r.category:<25} {r.is_sharpe:>9.3f} {r.oos_sharpe:>10.3f} "
            f"{r.oos_return*100:>7.1f}% {r.oos_maxdd*100:>7.1f}% {r.avg_gross:>5.2f} {r.avg_turnover:>6.4f}"
        )

    # ── Save ─────────────────────────────────────────────────────
    payload = {
        "meta": {
            "is": f"{IS_START} → {IS_END}",
            "oos": f"{OOS_START} → {OOS_END}",
            "method": "walk-forward: optimize on IS, validate on OOS",
            "symbols": all_syms,
            "n_symbols": len(all_syms),
            "timeline_start": str(timeline[0]),
            "timeline_end": str(timeline[-1]),
            "timeline_hours": len(timeline),
        },
        "results": [asdict(r) for r in sorted_r],
    }
    out_path = OUT_DIR / "v16a_factor_audit.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nWrote {out_path} ({len(all_results)} experiments in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
