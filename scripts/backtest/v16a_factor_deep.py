"""Supplementary: Core factor decomposition — IS/OOS of individual filters.

Tests:
1. Core signal persistence [0,1,2,3,4]
2. Core momentum lookback variants 
3. Core ADX ensemble variants
4. Min/max hold bar sweeps

Run: uv run python scripts/backtest/v16a_factor_deep.py
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
    v10g_params,
    overlay_params,
)

EPS = 1e-12
DATA_PATH = ROOT / "data"
OUT_DIR = ROOT / "backtest-results"
IS_START, IS_END = "2020-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-12-31"

# We'll monkey-patch to test core-only signal params, then rebuild only core
import executor.profiles.v16a_badscore_overlay as profile_mod


@dataclass
class ExpResult:
    name: str
    category: str
    config: dict
    is_sharpe: float
    is_return: float
    is_maxdd: float
    oos_sharpe: float
    oos_return: float
    oos_maxdd: float
    avg_gross: float
    avg_turnover: float


def date_mask(timeline, lo, hi):
    days = np.array([ts.date().isoformat() for ts in timeline])
    return (days >= lo) & (days <= hi)


def metrics(returns):
    m = calculate_hourly_metrics(returns, initial_equity=INITIAL_EQUITY)
    return {
        "sharpe": float(m["sharpe"]),
        "return": float(m["return"]),
        "max_dd": float(m["max_dd"]),
        "ann_return": float(m["ann_return"]),
    }


def build_core_sleeve_patched(core_patch: dict):
    """Build v10g core sleeve with patched v10g_params."""
    orig = profile_mod.v10g_params
    orig_params = orig()
    profile_mod.v10g_params = lambda: orig_params.__class__(
        **{**vars(orig_params), **core_patch}
    )
    try:
        result = build_v10g_sleeve(backfill=False, core_phase_hours=2)
    finally:
        profile_mod.v10g_params = orig
    return result


def build_overlay_sleeve_patched(ov_patch: dict):
    """Build overlay sleeve with patched overlay_params."""
    orig = profile_mod.overlay_params
    orig_params = orig()
    profile_mod.overlay_params = lambda: orig_params.__class__(
        **{**vars(orig_params), **ov_patch}
    )
    try:
        result = build_overlay_sleeve(backfill=False)
    finally:
        profile_mod.overlay_params = orig
    return result


def evaluate_combo(
    name, category, config,
    v_aligned, o_aligned, timeline, all_syms, returns, gate,
    v_alloc=0.5, o_alloc=0.5, gross_cap=4.0,
):
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
    return ExpResult(
        name=name, category=category, config=config,
        is_sharpe=is_m["sharpe"], is_return=is_m["return"], is_maxdd=is_m["max_dd"],
        oos_sharpe=oos_m["sharpe"], oos_return=oos_m["return"], oos_maxdd=oos_m["max_dd"],
        avg_gross=float(np.mean(np.sum(np.abs(target), axis=1))),
        avg_turnover=float(np.mean(result.turnover)),
    )


def main():
    t0 = time.time()
    all_results = []

    print("=" * 60)
    print("v16a Deep Factor Audit — Core/Overlay Param Sweeps")
    print("=" * 60)

    # Pre-build baseline overlay (unchanging)
    print("\nBuilding baseline overlay sleeve...")
    o_syms, o_weights, o_curve, _, o_timeline = build_overlay_sleeve(backfill=False)
    gate = expanding_badscore_gate(o_timeline)

    # Find common timeline for alignment helper
    def run_core_var(name, cat, cfg, core_patch):
        nonlocal o_syms, o_weights, o_curve, o_timeline
        print(f"  Building core: {name}...")
        v_syms, v_weights, v_curve, _, v_timeline = build_core_sleeve_patched(core_patch)

        start = max(v_curve[0][0], o_curve[0][0])
        end = min(o_curve[-1][0], v_curve[-1][0] + timedelta(hours=11))
        timeline = [ts for ts in o_timeline if start <= ts <= end]
        all_syms = sorted(set(v_syms) | set(o_syms))

        v_aligned = align_weights(v_timeline, v_syms, v_weights, timeline, all_syms, forward_fill=True)
        o_aligned = align_weights(o_timeline, o_syms, o_weights, timeline, all_syms, forward_fill=False)
        returns = load_hourly_returns(timeline, all_syms)
        gate_local = expanding_badscore_gate(timeline)

        return evaluate_combo(name, cat, cfg, v_aligned, o_aligned, timeline, all_syms, returns, gate_local)

    def run_overlay_var(name, cat, cfg, ov_patch):
        nonlocal gate
        print(f"  Building overlay: {name}...")
        o_syms_v, o_weights_v, o_curve_v, _, o_timeline_v = build_overlay_sleeve_patched(ov_patch)

        # Use baseline core
        v_syms, v_weights, v_curve, _, v_timeline = build_v10g_sleeve(backfill=False, core_phase_hours=2)

        start = max(v_curve[0][0], o_curve_v[0][0])
        end = min(o_curve_v[-1][0], v_curve[-1][0] + timedelta(hours=11))
        timeline = [ts for ts in o_timeline_v if start <= ts <= end]
        all_syms = sorted(set(v_syms) | set(o_syms_v))

        v_aligned = align_weights(v_timeline, v_syms, v_weights, timeline, all_syms, forward_fill=True)
        o_aligned = align_weights(o_timeline_v, o_syms_v, o_weights_v, timeline, all_syms, forward_fill=False)
        returns = load_hourly_returns(timeline, all_syms)

        # Re-build gate for this timeline
        gate2 = expanding_badscore_gate(timeline)
        return evaluate_combo(name, cat, cfg, v_aligned, o_aligned, timeline, all_syms, returns, gate2)

    # ── 1. Core signal persistence ───────────────────────────────
    print("\n=== Core Signal Persistence Sweep ===")
    for sp in [0, 1, 2, 3, 4]:
        r = run_core_var(f"core persistence={sp}", "core_persistence",
                         {"signal_persistence": sp}, {"signal_persistence": sp})
        all_results.append(r)
        print(f"    persistence={sp}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 2. Core momentum lookback variants ───────────────────────
    print("\n=== Core Momentum Lookback Sweep ===")
    for lbs in [
        [10, 30, 60],
        [15, 45, 90],
        [20, 60, 120],
        [30, 90, 180],
        [40, 120, 240],
    ]:
        label = "_".join(str(x) for x in lbs)
        r = run_core_var(f"core mom=[{label}]", "core_mom_lookbacks",
                         {"mom_lookbacks": lbs}, {"mom_lookbacks": lbs})
        all_results.append(r)
        print(f"    [{label}]: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 3. Core ADX ensemble ─────────────────────────────────────
    print("\n=== Core ADX Ensemble Sweep ===")
    for ensemble in [
        [20, 25, 30],
        [22, 27, 32],
        [24, 29, 34],
        [18, 23, 28],
        [26, 31, 36],
    ]:
        label = "_".join(str(x) for x in ensemble)
        r = run_core_var(f"core adx=[{label}]", "core_adx",
                         {"adx_ensemble": ensemble}, {"adx_ensemble": ensemble})
        all_results.append(r)
        print(f"    [{label}]: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 4. Core min/max hold ─────────────────────────────────────
    print("\n=== Core Min Hold Bars Sweep ===")
    for mhb in [8, 12, 16, 20, 24]:
        r = run_core_var(f"core min_hold={mhb}", "core_minhold",
                         {"min_hold_bars": mhb}, {"min_hold_bars": mhb})
        all_results.append(r)
        print(f"    min_hold={mhb}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    print("\n=== Core Max Hold Bars Sweep ===")
    for mhb in [60, 80, 100, 150, 200]:
        r = run_core_var(f"core max_hold={mhb}", "core_maxhold",
                         {"max_hold_bars": mhb}, {"max_hold_bars": mhb})
        all_results.append(r)
        print(f"    max_hold={mhb}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 5. Core Donchian period ──────────────────────────────────
    print("\n=== Core Donchian Period Sweep ===")
    for dp in [10, 20, 30, 40]:
        r = run_core_var(f"core donchian={dp}", "core_donchian",
                         {"donchian_period": dp}, {"donchian_period": dp})
        all_results.append(r)
        print(f"    donchian={dp}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 6. Core ADX threshold ────────────────────────────────────
    print("\n=== Core ADX Threshold Sweep ===")
    for at in [15, 20, 25, 30, 35]:
        r = run_core_var(f"core adx_thr={at}", "core_adxthr",
                         {"adx_threshold": at}, {"adx_threshold": at})
        all_results.append(r)
        print(f"    adx_thr={at}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 7. Overlay signal persistence ────────────────────────────
    print("\n=== Overlay Signal Persistence Sweep ===")
    for sp in [0, 1, 2, 3, 4]:
        r = run_overlay_var(f"overlay persistence={sp}", "ov_persistence",
                            {"signal_persistence": sp}, {"signal_persistence": sp})
        all_results.append(r)
        print(f"    persistence={sp}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── 8. Core signal threshold ─────────────────────────────────
    print("\n=== Core Signal Threshold Sweep ===")
    for st in [0.30, 0.40, 0.50, 0.60]:
        r = run_core_var(f"core sig_thr={st}", "core_sigthr",
                         {"signal_threshold": st}, {"signal_threshold": st})
        all_results.append(r)
        print(f"    sig_thr={st}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}")

    # ── Print ────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'Name':<40} {'Category':<18} {'IS Sharpe':>9} {'OOS Sharpe':>10} {'OOS Ret%':>8} {'OOS DD%':>8} {'Gross':>6}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda r: r.oos_sharpe, reverse=True):
        print(
            f"{r.name:<40} {r.category:<18} {r.is_sharpe:>9.3f} {r.oos_sharpe:>10.3f} "
            f"{r.oos_return*100:>7.1f}% {r.oos_maxdd*100:>7.1f}% {r.avg_gross:>5.2f}"
        )

    # Save
    payload = {
        "meta": {"is": f"{IS_START} → {IS_END}", "oos": f"{OOS_START} → {OOS_END}"},
        "results": [asdict(r) for r in sorted(all_results, key=lambda r: r.oos_sharpe, reverse=True)],
    }
    out_path = OUT_DIR / "v16a_factor_deep.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path} ({len(all_results)} results in {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
