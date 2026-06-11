"""Unified backtest metrics computation.

Replaces the previously scattered metric calculation in:
- core/metrics.py (trade-driven PerformanceMetrics)
- executor/portfolio_backtest.py (inline dict-based calculate_hourly_metrics)
- research scripts (ad-hoc summary stats)
"""

from __future__ import annotations

import numpy as np

from .result import BacktestMetrics

EPS = 1e-12


def compute_metrics(
    returns: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    periods_per_year: float = 365 * 24,  # default: hourly bars
    weights: np.ndarray | None = None,
) -> BacktestMetrics:
    """Compute all standard metrics from a return stream.

    When ``weights`` is provided, portfolio-level metrics (gross/net exposure,
    avg turnover) are also computed. Without weights, those fields are None.

    Args:
        returns: 1-D array of per-period log-returns (length: N).
        initial_equity: Starting capital for equity curve reconstruction.
        periods_per_year: Number of periods in a year for annualization.
        weights: Optional 2-D array of target weights (N × M) for
            portfolio-level metrics.

    Returns:
        BacktestMetrics for the return stream.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("returns must be a 1-D array")
    n = returns.size
    if n < 2:
        return BacktestMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            max_dd_duration_bars=0,
            avg_drawdown=0.0,
            ulcer_index=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            tail_ratio=0.0,
        )

    # ── Equity curve ────────────────────────────────────────────
    equity = initial_equity * np.cumprod(1.0 + returns)
    total_return = float(equity[-1] / equity[0] - 1.0)
    annualized_return = float(
        (1.0 + total_return) ** (periods_per_year / max(n, 1)) - 1.0
    )
    volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

    # ── Drawdowns ────────────────────────────────────────────────
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max  # positive from peak
    max_dd = float(drawdown.max())
    avg_dd = float(drawdown[drawdown > 0].mean()) if (drawdown > 0).any() else 0.0
    ulcer = float(np.sqrt(np.mean(drawdown**2)))

    # ── Max DD duration (bars between peak and recovery) ─────────
    dd_duration = _max_dd_duration(equity)

    # ── Risk-adjusted ratios ─────────────────────────────────────
    sharpe = annualized_return / volatility if volatility > EPS else 0.0

    downside = returns[returns < 0]
    downside_std = float(
        np.std(downside, ddof=1) * np.sqrt(periods_per_year)
    ) if len(downside) > 1 else 0.0
    sortino = annualized_return / downside_std if downside_std > EPS else 0.0

    calmar = annualized_return / max_dd if max_dd > EPS else 0.0

    # ── Tail ratio (P95 gain / |P5 loss|) ───────────────────────
    tail = _tail_ratio(returns)

    # ── Portfolio-level (weight-driven) ──────────────────────────
    avg_gross: float | None = None
    avg_net: float | None = None
    avg_turnover: float | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 2 and w.shape[0] == n:
            avg_gross = float(np.nanmean(np.sum(np.abs(w), axis=1)))
            avg_net = float(np.nanmean(np.sum(w, axis=1)))
            avg_turnover = float(
                np.nanmean(np.sum(np.abs(np.diff(w, axis=0)), axis=1))
            )

    # ── Monthly returns ──────────────────────────────────────────
    monthly: dict[str, float] | None = None

    # Equities array for downstream use
    equity_list = [float(e) for e in equity]

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_dd,
        max_dd_duration_bars=dd_duration,
        avg_drawdown=avg_dd,
        ulcer_index=ulcer,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        tail_ratio=tail,
        avg_gross_exposure=avg_gross,
        avg_net_exposure=avg_net,
        avg_turnover=avg_turnover,
        equity_curve=equity_list,
        drawdown_curve=None,  # caller can add from drawdown array if needed
        monthly_returns=monthly,
    )


def _max_dd_duration(equity: np.ndarray) -> int:
    """Longest consecutive bars between a peak and its recovery."""
    running_max = np.maximum.accumulate(equity)
    in_dd = running_max > equity  # True when underwater

    max_streak = 0
    current_streak = 0
    for flag in in_dd:
        if flag:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 0
    max_streak = max(max_streak, current_streak)
    return int(max_streak)


def _tail_ratio(returns: np.ndarray) -> float:
    """P95 gain / |P5 loss| — asymmetry of extreme returns."""
    if returns.size < 20:
        return 0.0
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(gains) < 5 or len(losses) < 5:
        return 0.0
    p95_gain = float(np.percentile(gains, 95))
    p5_loss = float(np.percentile(losses, 5))
    if abs(p5_loss) < EPS:
        return 0.0
    return p95_gain / abs(p5_loss)


# ── Backward-compat wrapper ──────────────────────────────────────


def calculate_hourly_metrics(
    returns: np.ndarray, *, initial_equity: float = 10_000.0
) -> dict[str, float | np.ndarray]:
    """Legacy hourly metrics, kept for backward compatibility.

    New code should use ``compute_metrics()`` which returns a typed
    ``BacktestMetrics`` dataclass instead of a plain dict.
    """
    m = compute_metrics(returns, initial_equity=initial_equity)
    equity = initial_equity * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    return {
        "return": m.total_return,
        "ann_return": m.annualized_return,
        "volatility": m.volatility,
        "sharpe": m.sharpe_ratio,
        "max_dd": m.max_drawdown,
        "ulcer": m.ulcer_index,
        "equity": equity,
        "drawdown": drawdown,
    }
