# Strategy iteration checkpoint — 2026-04-28

This note consolidates the current research state after the 1h-native CTA iteration.
It is a checkpoint, not a production deployment decision.

## Objective

Improve the existing v10g trend-following system while avoiding self-deceptive
backtests. The priority is stable, controllable profitability rather than the
highest headline return.

## Baselines

| Strategy / validation view | Sharpe | Return | Max DD | Notes |
|---|---:|---:|---:|---|
| Original v10g reference | ~1.64 | +199% | 10.9% | Useful reference, but optimistic because signal and close execution happen on the same bar. |
| Shifted v10g conservative engine view | ~1.21 | +122% | 9.9% | Uses a one-bar signal lag and engine-derived held positions. |
| Fast-exit top2 1h overlay only | ~1.30 | +113% | 10.4% | Diversifying 1h sleeve; not a replacement for v10g. |

The original v10g still has the highest total return, but its validation is less
conservative. The conservative shifted view is the fairer baseline for new work.

## Current best comprehensive candidate

`joint-v10g-fast-overlay-badscore-v1`

Configuration:

- Core sleeve: shifted v10g 6h trend strategy.
- Overlay sleeve: 1h fast-exit top2 overlay.
  - Select only the strongest two 1h overlay signals.
  - Fast exit profile: `min_hold_bars=8`, `max_hold_bars=48`, `atr_stop_mult=4.0`, `signal_reversal_threshold=0.22`.
- Allocation: fixed 50% core / 50% overlay.
- Regime risk gate: `badscore2_050`.
  - Scale total exposure to 50% when at least two of these past-only conditions are true:
    1. market volatility is above its expanding median;
    2. market trend efficiency is below its expanding median;
    3. mean cross-asset correlation is above its expanding 66% quantile.
- Fees: 4 bp per turnover unit in the research harness.
- Validation style: engine-derived position targets, symbol-level netting, gross cap, and next-hour mark-to-market.

Performance:

| Candidate | Sharpe | Return | Max DD | Interpretation |
|---|---:|---:|---:|---|
| 50/50 + old volatility gate | ~1.64 | +108% | 7.8% | Previous robust candidate. |
| 50/50 + fixed badscore gate | ~1.95 | +123% | 5.9% | Current best balanced candidate. |
| 30/70 + fixed badscore gate | ~1.92 | +120% | 6.8% | Slightly more overlay-heavy; later-period performance is better but drawdown is higher. |

Why the 50/50 badscore candidate is preferred now:

- It has the best overall Sharpe among the stable fixed-weight candidates.
- It has the cleanest drawdown profile.
- It avoids an aggressive fitted overlay multiplier.
- It uses a fixed, interpretable regime rule rather than dynamic gate selection.

## Robustness checks performed

- Used one-bar-lagged v10g signals for conservative baseline comparison.
- Rejected pure 1h replacement attempts; 1h works better as an overlay.
- Rejected dynamic gate selection: walk-forward selection added noise and did not beat fixed rules.
- Checked fee sensitivity: the badscore candidate remains better than the old volatility gate at higher fee assumptions, though performance degrades as expected.
- Checked multiple OOS start dates: fixed badscore gate generally improves risk-adjusted results versus the simple volatility gate.
- Found and fixed research-harness alignment bugs before trusting results.

## Known limitations

- This is not yet a production-grade shared-cash / shared-margin backtest.
- The current joint validation combines engine-derived held position targets, but it is still a research approximation.
- 2022-2023 remains the weakest regime; the badscore gate improves drawdown and risk-adjusted performance but does not fully solve that period.
- Funding/carry showed high theoretical Sharpe only under a delta-hedged assumption; it is a separate future sleeve, not part of this candidate.

## Reproduction

A repo-native research script is available:

```bash
uv run python scripts/backtest/joint_badscore_research.py
```

It writes:

- `backtest-results/metrics_joint_badscore_research.json`
- `backtest-results/backtest_joint_badscore_research.png`

These files are intentionally ignored as generated backtest outputs.

## Productionization status

Phase 1 is implemented in reusable executor modules:

- `executor.targeting` defines `StrategyProfile`, `PortfolioTarget`, `SleeveTarget`, `TargetWeightStrategy`, and target-to-order delta utilities.
- `executor.portfolio_backtest` provides a target-weight simulation path.
- `executor.profiles.v16a_badscore_overlay` contains the reusable v16a profile and target construction logic.
- `scripts/backtest/joint_badscore_research.py` is now a thin reproduction CLI over those modules.

The reproduction command still matches the checkpoint metrics:

| Candidate | Sharpe | Return | Max DD |
|---|---:|---:|---:|
| Shifted v10g sleeve | ~1.21 | +122% | 9.9% |
| Fast-exit top2 1h overlay | ~1.30 | +113% | 10.4% |
| v16a Badscore Overlay | ~1.95 | +123% | 5.9% |

Phase 2 has started in `LiveEngine`:

- The live CLI accepts `STRATEGY_PROFILE`, defaulting to `v10g-engine-6h`.
- Unknown/non-wired live profiles fail fast instead of silently falling back.
- `LiveEngine` can accept an injected target-weight strategy and reconcile it into market-order deltas.
- Target reconciliation normalizes `BTCUSDT`-style research symbols to live `BTC` symbols, applies `MIN_ORDER_NOTIONAL`, and splits sign flips into reduce-only close plus a separate new-side order.

## Recommended next step

Build the actual online v16a target provider before enabling the profile in live/testnet:

1. compute and carry forward the 6h core sleeve and 1h overlay sleeve from live/cache data;
2. compute the past-only `badscore2_050` gate online;
3. run shadow/dry-run target generation against live data without sending orders;
4. only then allow `STRATEGY_PROFILE=v16a-badscore-overlay` in testnet.
