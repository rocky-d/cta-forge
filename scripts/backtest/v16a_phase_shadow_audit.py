"""Shadow-impact audit for switching v16a 6h core phase 0 -> phase 2.

This is a research-only feasibility audit. It does not change live defaults and
it does not submit orders. The goal is to compare target/exposure behavior using
the same target-weight and order-delta mechanisms used by the live path.

Run:
    uv run python scripts/backtest/v16a_phase_shadow_audit.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from executor.portfolio_backtest import calculate_hourly_metrics  # noqa: E402
from executor.profiles.v16a_badscore_overlay import load_hourly_returns  # noqa: E402
from executor.targeting import weights_to_orders  # noqa: E402
from scripts.backtest.v16a_phase_robustness_research import (  # noqa: E402
    PhaseTargetSet,
    build_joint_target_with_core_phase,
)

OUT_DIR = ROOT / "backtest-results"
PHASE_BASELINE = 0
PHASE_CANDIDATE = 2
INITIAL_EQUITY = 10_000.0
PILOT_EQUITY = 200.0
PILOT_TARGET_SCALE = 5.0
PILOT_GROSS_CAP = 4.0
PILOT_MIN_ORDER_NOTIONAL = 10.0
PILOT_MAX_ORDER_NOTIONAL = 50.0
RECENT_WINDOWS_DAYS = [30, 60, 90, 180, 365]
EPS = 1e-12


@dataclass(frozen=True)
class MetricSnapshot:
    ret: float
    ann_return: float
    volatility: float
    sharpe: float
    max_dd: float
    ulcer: float


@dataclass(frozen=True)
class RecentWindowResult:
    label: str
    start: str
    end: str
    bars: int
    phase0: MetricSnapshot
    phase2: MetricSnapshot
    phase2_minus_phase0_return: float
    phase2_minus_phase0_sharpe: float
    mean_l1_target_diff: float
    p95_l1_target_diff: float
    mean_sign_flips: float


@dataclass(frozen=True)
class SymbolDiffResult:
    symbol: str
    mean_abs_diff: float
    p95_abs_diff: float
    mean_phase0_weight: float
    mean_phase2_weight: float
    sign_flip_hours: int
    contribution_share: float


@dataclass(frozen=True)
class HourDiffResult:
    hour: int
    bars: int
    mean_l1_target_diff: float
    p95_l1_target_diff: float
    mean_sign_flips: float
    phase2_minus_phase0_return: float


@dataclass(frozen=True)
class PilotOrderImpactResult:
    label: str
    start: str
    end: str
    bars: int
    mean_orders_per_hour: float
    p95_orders_per_hour: float
    mean_abs_notional_per_hour: float
    p95_abs_notional_per_hour: float
    max_abs_notional_per_order: float
    max_abs_increase_notional_per_order: float
    max_abs_reduce_notional_per_order: float
    reduce_only_share: float
    mean_ignored_l1_due_to_caps: float
    p95_ignored_l1_due_to_caps: float


def metric_snapshot(returns: np.ndarray) -> MetricSnapshot:
    metrics = calculate_hourly_metrics(returns, initial_equity=INITIAL_EQUITY)
    return MetricSnapshot(
        ret=float(metrics["return"]),
        ann_return=float(metrics["ann_return"]),
        volatility=float(metrics["volatility"]),
        sharpe=float(metrics["sharpe"]),
        max_dd=float(metrics["max_dd"]),
        ulcer=float(metrics["ulcer"]),
    )


def assert_data_fidelity(phase_sets: dict[int, PhaseTargetSet]) -> dict[str, object]:
    """Fail fast on obvious data/harness drift before doing shadow conclusions."""
    coverage = {}
    for phase, phase_set in phase_sets.items():
        if len(phase_set.timeline) < 50_000:
            msg = f"phase {phase} has too few bars; expected full historical cache"
            raise RuntimeError(msg)
        if phase_set.target_weights.shape != phase_set.returns.shape:
            msg = f"phase {phase} target/return shape mismatch"
            raise RuntimeError(msg)
        if len(phase_set.timeline) != phase_set.target_weights.shape[0]:
            msg = f"phase {phase} timeline/target shape mismatch"
            raise RuntimeError(msg)
        if not np.all(np.isfinite(phase_set.target_weights)):
            msg = f"phase {phase} contains non-finite target weights"
            raise RuntimeError(msg)
        gross = np.sum(np.abs(phase_set.target_weights), axis=1)
        if float(np.nanmax(gross)) > 1.0 + 1e-9:
            msg = f"phase {phase} exceeds research gross cap"
            raise RuntimeError(msg)
        coverage[str(phase)] = {
            "bars": len(phase_set.timeline),
            "symbols": len(phase_set.symbols),
            "start": phase_set.timeline[0].isoformat(),
            "end": phase_set.timeline[-1].isoformat(),
            "max_gross": float(np.max(gross)),
            "mean_gross": float(np.mean(gross)),
        }
    return coverage


def align_phase_sets(
    phase0: PhaseTargetSet, phase2: PhaseTargetSet
) -> tuple[list[datetime], list[str], np.ndarray, np.ndarray]:
    common_timestamps = sorted(set(phase0.timeline) & set(phase2.timeline))
    if len(common_timestamps) < 50_000:
        raise RuntimeError("phase 0/2 common timestamp history is unexpectedly short")
    symbols = sorted(set(phase0.symbols) | set(phase2.symbols))

    def align(phase_set: PhaseTargetSet) -> np.ndarray:
        ts_idx = {ts: i for i, ts in enumerate(phase_set.timeline)}
        sym_idx = {symbol: i for i, symbol in enumerate(phase_set.symbols)}
        weights = np.zeros((len(common_timestamps), len(symbols)))
        for i, ts in enumerate(common_timestamps):
            src_i = ts_idx[ts]
            for j, symbol in enumerate(symbols):
                src_j = sym_idx.get(symbol)
                if src_j is not None:
                    weights[i, j] = phase_set.target_weights[src_i, src_j]
        return weights

    weights0 = align(phase0)
    weights2 = align(phase2)
    return common_timestamps, symbols, weights0, weights2


def portfolio_returns(
    weights: np.ndarray, returns: np.ndarray, *, fee: float = 0.0004
) -> np.ndarray:
    pnl = np.zeros(weights.shape[0], dtype=float)
    previous = np.zeros(weights.shape[1], dtype=float)
    for t in range(weights.shape[0] - 1):
        current = np.nan_to_num(weights[t], nan=0.0)
        turnover = float(np.sum(np.abs(current - previous)))
        pnl[t + 1] = float(
            np.sum(current * np.nan_to_num(returns[t + 1], nan=0.0)) - turnover * fee
        )
        previous = current
    return pnl


def diff_arrays(weights0: np.ndarray, weights2: np.ndarray) -> dict[str, np.ndarray]:
    diff = weights2 - weights0
    l1 = np.sum(np.abs(diff), axis=1)
    gross0 = np.sum(np.abs(weights0), axis=1)
    gross2 = np.sum(np.abs(weights2), axis=1)
    sign_flips = np.sum(
        (np.sign(weights0) != np.sign(weights2)) & (np.abs(diff) > EPS), axis=1
    )
    abs_diff = np.abs(diff)
    dot = np.sum(weights0 * weights2, axis=1)
    norm0 = np.sqrt(np.sum(weights0 * weights0, axis=1))
    norm2 = np.sqrt(np.sum(weights2 * weights2, axis=1))
    cosine = np.divide(
        dot,
        norm0 * norm2,
        out=np.full(len(dot), np.nan),
        where=(norm0 > EPS) & (norm2 > EPS),
    )
    return {
        "diff": diff,
        "abs_diff": abs_diff,
        "l1": l1,
        "gross0": gross0,
        "gross2": gross2,
        "sign_flips": sign_flips,
        "cosine": cosine,
    }


def summarize_recent_windows(
    timeline: list[datetime],
    returns0: np.ndarray,
    returns2: np.ndarray,
    diffs: dict[str, np.ndarray],
) -> list[RecentWindowResult]:
    end = timeline[-1]
    out: list[RecentWindowResult] = []
    for days in RECENT_WINDOWS_DAYS:
        start = end - timedelta(days=days)
        mask = np.array([(start <= ts <= end) for ts in timeline], dtype=bool)
        if int(np.sum(mask)) < 24:
            continue
        phase0 = metric_snapshot(returns0[mask])
        phase2 = metric_snapshot(returns2[mask])
        out.append(
            RecentWindowResult(
                label=f"last_{days}d",
                start=timeline[int(np.argmax(mask))].isoformat(),
                end=end.isoformat(),
                bars=int(np.sum(mask)),
                phase0=phase0,
                phase2=phase2,
                phase2_minus_phase0_return=phase2.ret - phase0.ret,
                phase2_minus_phase0_sharpe=phase2.sharpe - phase0.sharpe,
                mean_l1_target_diff=float(np.mean(diffs["l1"][mask])),
                p95_l1_target_diff=float(np.quantile(diffs["l1"][mask], 0.95)),
                mean_sign_flips=float(np.mean(diffs["sign_flips"][mask])),
            )
        )
    return out


def summarize_symbols(
    symbols: list[str], weights0: np.ndarray, weights2: np.ndarray
) -> list[SymbolDiffResult]:
    diff = np.abs(weights2 - weights0)
    total = float(np.sum(diff))
    out: list[SymbolDiffResult] = []
    for j, symbol in enumerate(symbols):
        sym_diff = diff[:, j]
        sign_flips = int(
            np.sum(
                (np.sign(weights0[:, j]) != np.sign(weights2[:, j])) & (sym_diff > EPS)
            )
        )
        out.append(
            SymbolDiffResult(
                symbol=symbol,
                mean_abs_diff=float(np.mean(sym_diff)),
                p95_abs_diff=float(np.quantile(sym_diff, 0.95)),
                mean_phase0_weight=float(np.mean(weights0[:, j])),
                mean_phase2_weight=float(np.mean(weights2[:, j])),
                sign_flip_hours=sign_flips,
                contribution_share=float(np.sum(sym_diff) / total)
                if total > EPS
                else 0.0,
            )
        )
    return sorted(out, key=lambda row: row.mean_abs_diff, reverse=True)


def summarize_hours(
    timeline: list[datetime],
    returns0: np.ndarray,
    returns2: np.ndarray,
    diffs: dict[str, np.ndarray],
) -> list[HourDiffResult]:
    hours = np.array([ts.hour for ts in timeline], dtype=int)
    out: list[HourDiffResult] = []
    for hour in range(24):
        mask = hours == hour
        if int(np.sum(mask)) == 0:
            continue
        out.append(
            HourDiffResult(
                hour=hour,
                bars=int(np.sum(mask)),
                mean_l1_target_diff=float(np.mean(diffs["l1"][mask])),
                p95_l1_target_diff=float(np.quantile(diffs["l1"][mask], 0.95)),
                mean_sign_flips=float(np.mean(diffs["sign_flips"][mask])),
                phase2_minus_phase0_return=metric_snapshot(returns2[mask]).ret
                - metric_snapshot(returns0[mask]).ret,
            )
        )
    return sorted(out, key=lambda row: row.mean_l1_target_diff, reverse=True)


def scale_to_pilot(weights: np.ndarray) -> np.ndarray:
    scaled = weights * PILOT_TARGET_SCALE
    gross = np.sum(np.abs(scaled), axis=1)
    capped = scaled.copy()
    scale = np.divide(
        PILOT_GROSS_CAP, gross, out=np.ones_like(gross), where=gross > PILOT_GROSS_CAP
    )
    capped *= scale[:, None]
    return capped


def summarize_pilot_order_impact(
    timeline: list[datetime],
    symbols: list[str],
    candidate_weights: np.ndarray,
    candidate_returns: np.ndarray,
) -> list[PilotOrderImpactResult]:
    prices = {symbol: 1.0 for symbol in symbols}
    scaled = scale_to_pilot(candidate_weights)
    out: list[PilotOrderImpactResult] = []
    end = timeline[-1]
    windows = [("full", timeline[0], end)] + [
        (f"last_{days}d", end - timedelta(days=days), end)
        for days in RECENT_WINDOWS_DAYS
    ]
    for label, start, stop in windows:
        mask = np.array([(start <= ts <= stop) for ts in timeline], dtype=bool)
        idxs = np.flatnonzero(mask)
        if len(idxs) < 24:
            continue
        positions = {symbol: 0.0 for symbol in symbols}
        orders_per_hour = []
        abs_notional_per_hour = []
        max_order_notional = []
        max_increase_order_notional = []
        max_reduce_order_notional = []
        reduce_only_count = 0
        total_order_count = 0
        ignored_l1 = []
        for i in idxs:
            target = {symbol: float(scaled[i, j]) for j, symbol in enumerate(symbols)}
            orders = weights_to_orders(
                positions,
                prices,
                PILOT_EQUITY,
                target,
                min_notional=PILOT_MIN_ORDER_NOTIONAL,
                max_notional=PILOT_MAX_ORDER_NOTIONAL,
            )
            next_positions = positions.copy()
            for order in orders:
                signed_qty = order.qty if order.side == "buy" else -order.qty
                next_positions[order.symbol] = (
                    next_positions.get(order.symbol, 0.0) + signed_qty
                )
                reduce_only_count += int(order.reduce_only)
            total_order_count += len(orders)
            realized_weights = np.array(
                [next_positions.get(symbol, 0.0) / PILOT_EQUITY for symbol in symbols]
            )
            target_vec = np.array([target[symbol] for symbol in symbols])
            ignored_l1.append(float(np.sum(np.abs(target_vec - realized_weights))))
            orders_per_hour.append(float(len(orders)))
            notionals = [abs(order.delta_notional) for order in orders]
            increase_notionals = [
                abs(order.delta_notional) for order in orders if not order.reduce_only
            ]
            reduce_notionals = [
                abs(order.delta_notional) for order in orders if order.reduce_only
            ]
            abs_notional_per_hour.append(float(sum(notionals)))
            max_order_notional.append(float(max(notionals)) if notionals else 0.0)
            max_increase_order_notional.append(
                float(max(increase_notionals)) if increase_notionals else 0.0
            )
            max_reduce_order_notional.append(
                float(max(reduce_notionals)) if reduce_notionals else 0.0
            )
            # Mark-to-market approximation using the next available return row.
            if i + 1 < len(candidate_returns):
                next_positions = {
                    symbol: next_positions.get(symbol, 0.0)
                    * (1.0 + float(np.nan_to_num(candidate_returns[i + 1, j], nan=0.0)))
                    for j, symbol in enumerate(symbols)
                }
            positions = next_positions
        out.append(
            PilotOrderImpactResult(
                label=label,
                start=timeline[idxs[0]].isoformat(),
                end=timeline[idxs[-1]].isoformat(),
                bars=len(idxs),
                mean_orders_per_hour=float(np.mean(orders_per_hour)),
                p95_orders_per_hour=float(np.quantile(orders_per_hour, 0.95)),
                mean_abs_notional_per_hour=float(np.mean(abs_notional_per_hour)),
                p95_abs_notional_per_hour=float(
                    np.quantile(abs_notional_per_hour, 0.95)
                ),
                max_abs_notional_per_order=float(np.max(max_order_notional)),
                max_abs_increase_notional_per_order=float(
                    np.max(max_increase_order_notional)
                ),
                max_abs_reduce_notional_per_order=float(
                    np.max(max_reduce_order_notional)
                ),
                reduce_only_share=float(reduce_only_count / total_order_count)
                if total_order_count
                else 0.0,
                mean_ignored_l1_due_to_caps=float(np.mean(ignored_l1)),
                p95_ignored_l1_due_to_caps=float(np.quantile(ignored_l1, 0.95)),
            )
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building phase 0 and phase 2 target sets...")
    phase_sets = {
        PHASE_BASELINE: build_joint_target_with_core_phase(phase_hours=PHASE_BASELINE),
        PHASE_CANDIDATE: build_joint_target_with_core_phase(
            phase_hours=PHASE_CANDIDATE
        ),
    }
    coverage = assert_data_fidelity(phase_sets)
    timeline, symbols, weights0, weights2 = align_phase_sets(
        phase_sets[PHASE_BASELINE], phase_sets[PHASE_CANDIDATE]
    )
    returns = load_hourly_returns(timeline, symbols)
    pnl0 = portfolio_returns(weights0, returns)
    pnl2 = portfolio_returns(weights2, returns)
    diffs = diff_arrays(weights0, weights2)

    full_phase0 = metric_snapshot(pnl0)
    full_phase2 = metric_snapshot(pnl2)
    recent = summarize_recent_windows(timeline, pnl0, pnl2, diffs)
    symbols_ranked = summarize_symbols(symbols, weights0, weights2)
    hours_ranked = summarize_hours(timeline, pnl0, pnl2, diffs)
    pilot_orders = summarize_pilot_order_impact(timeline, symbols, weights2, returns)

    payload = {
        "notes": {
            "scope": "Research-only shadow-impact audit for phase 0 vs phase 2; live defaults unchanged.",
            "fidelity_checks": "Fails on short history, target/return shape mismatch, non-finite targets, gross cap breach, or divergent aligned return matrices.",
            "live_order_mechanism": "Uses executor.targeting.weights_to_orders for pilot-style notional/min-order impact estimation.",
            "pilot_assumptions": {
                "equity": PILOT_EQUITY,
                "target_scale": PILOT_TARGET_SCALE,
                "gross_cap": PILOT_GROSS_CAP,
                "min_order_notional": PILOT_MIN_ORDER_NOTIONAL,
                "max_order_notional": PILOT_MAX_ORDER_NOTIONAL,
            },
        },
        "data_coverage": coverage,
        "aligned_history": {
            "bars": len(timeline),
            "symbols": len(symbols),
            "start": timeline[0].isoformat(),
            "end": timeline[-1].isoformat(),
        },
        "full_metrics": {
            "phase0": asdict(full_phase0),
            "phase2": asdict(full_phase2),
            "phase2_minus_phase0_return": full_phase2.ret - full_phase0.ret,
            "phase2_minus_phase0_sharpe": full_phase2.sharpe - full_phase0.sharpe,
        },
        "target_diff_summary": {
            "mean_l1_target_diff": float(np.mean(diffs["l1"])),
            "median_l1_target_diff": float(np.median(diffs["l1"])),
            "p95_l1_target_diff": float(np.quantile(diffs["l1"], 0.95)),
            "p99_l1_target_diff": float(np.quantile(diffs["l1"], 0.99)),
            "max_l1_target_diff": float(np.max(diffs["l1"])),
            "mean_phase0_gross": float(np.mean(diffs["gross0"])),
            "mean_phase2_gross": float(np.mean(diffs["gross2"])),
            "mean_sign_flips": float(np.mean(diffs["sign_flips"])),
            "p95_sign_flips": float(np.quantile(diffs["sign_flips"], 0.95)),
            "mean_cosine_similarity": float(np.nanmean(diffs["cosine"])),
        },
        "recent_windows": [
            {
                "label": row.label,
                "start": row.start,
                "end": row.end,
                "bars": row.bars,
                "phase0": asdict(row.phase0),
                "phase2": asdict(row.phase2),
                "phase2_minus_phase0_return": row.phase2_minus_phase0_return,
                "phase2_minus_phase0_sharpe": row.phase2_minus_phase0_sharpe,
                "mean_l1_target_diff": row.mean_l1_target_diff,
                "p95_l1_target_diff": row.p95_l1_target_diff,
                "mean_sign_flips": row.mean_sign_flips,
            }
            for row in recent
        ],
        "top_symbol_diffs": [asdict(row) for row in symbols_ranked],
        "hour_diff_profile": [asdict(row) for row in hours_ranked],
        "pilot_order_impact": [
            {
                "label": row.label,
                "start": row.start,
                "end": row.end,
                "bars": row.bars,
                "mean_orders_per_hour": row.mean_orders_per_hour,
                "p95_orders_per_hour": row.p95_orders_per_hour,
                "mean_abs_notional_per_hour": row.mean_abs_notional_per_hour,
                "p95_abs_notional_per_hour": row.p95_abs_notional_per_hour,
                "max_abs_notional_per_order": row.max_abs_notional_per_order,
                "max_abs_increase_notional_per_order": row.max_abs_increase_notional_per_order,
                "max_abs_reduce_notional_per_order": row.max_abs_reduce_notional_per_order,
                "reduce_only_share": row.reduce_only_share,
                "mean_ignored_l1_due_to_caps": row.mean_ignored_l1_due_to_caps,
                "p95_ignored_l1_due_to_caps": row.p95_ignored_l1_due_to_caps,
            }
            for row in pilot_orders
        ],
    }

    out_path = OUT_DIR / "v16a_phase_shadow_audit.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("== Full metrics ==")
    print(
        f"phase 0 sharpe={full_phase0.sharpe:.3f} return={full_phase0.ret:.3f} maxdd={full_phase0.max_dd:.3f}"
    )
    print(
        f"phase 2 sharpe={full_phase2.sharpe:.3f} return={full_phase2.ret:.3f} maxdd={full_phase2.max_dd:.3f}"
    )
    print("\n== Target diff ==")
    print(
        f"mean L1={np.mean(diffs['l1']):.5f} p95 L1={np.quantile(diffs['l1'], 0.95):.5f} "
        f"mean flips={np.mean(diffs['sign_flips']):.3f} mean cosine={np.nanmean(diffs['cosine']):.3f}"
    )
    print("\n== Recent windows ==")
    for row in recent:
        print(
            f"{row.label}: p2-p0 sharpe={row.phase2_minus_phase0_sharpe:+.3f} "
            f"return={row.phase2_minus_phase0_return:+.3f} mean_l1={row.mean_l1_target_diff:.5f}"
        )
    print("\n== Top symbol diffs ==")
    for row in symbols_ranked[:8]:
        print(
            f"{row.symbol}: mean_abs_diff={row.mean_abs_diff:.5f} p95={row.p95_abs_diff:.5f} "
            f"share={row.contribution_share:.3f} flips={row.sign_flip_hours}"
        )
    print("\n== Pilot order impact ==")
    for row in pilot_orders:
        print(
            f"{row.label}: mean_orders/h={row.mean_orders_per_hour:.3f} "
            f"p95_notional/h={row.p95_abs_notional_per_hour:.2f} "
            f"max_increase={row.max_abs_increase_notional_per_order:.2f} "
            f"max_reduce={row.max_abs_reduce_notional_per_order:.2f} "
            f"ignored_l1={row.mean_ignored_l1_due_to_caps:.4f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
