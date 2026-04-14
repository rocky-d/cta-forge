# Failed Experiments Archive (2026-04-12)

All tested against v10g baseline (Sharpe 1.58, +187%, DD 12.9%) over 6.5yr max-range.

| Script | Idea | Sharpe | Result |
|---|---|---|---|
| v11_expanded | 42 coins (19→42) | 1.20 | More noise, lower Sharpe |
| v11b_selective | 42 coins, higher threshold | 0.86 | Worse — filtered good signals |
| v11c_curated | 26 OG coins only | 1.13 | Lowest DD (10.4%) but lower return |
| v12_funding | Funding rate signal filter | 1.52 | FR dampened good trends in bull markets |
| v12b_funding | FR with conservative params | 1.46 | Still worse than baseline |
| v13_corr_filter | Correlation-based position filter | 1.06 | Crypto too correlated, killed opportunities |
| v14_dynpos | Dynamic max pos + scaled entry | 1.05 | Scaled entry missed best prices |

Conclusion: v10g is near-optimal within this trend-following framework.
Future alpha requires fundamentally different strategies (mean reversion, arb, etc).

## Round 3: v15 series — Structural fixes (all failed OOS validation)

| Script | Idea | Sharpe (IS) | OOS Result |
|---|---|---|---|
| v15a_no_maxhold | Remove max_hold_bars limit | 1.48 | Worse than v10g across all folds |
| v15b_wider_stop | Relax stop tightening (3.5/4.0 ATR) | 1.39 | Worse — more whipsaws |
| v15c_combined | a+b + asymmetric vol scaling | 1.17 | Worst of all variants |
| v15d_higher_vol | target_vol 0.12→0.20 | 1.64 | Full-sample looks great, OOS FAILED |
| walk_forward_vol | WF validation script v1 (buggy) | — | — |
| walk_forward_vol2 | WF validation script v2 (correct) | — | vol=OFF best OOS Sharpe (1.36) |
| analyze_2024 | 2024 performance diagnostic | — | Diagnostic only |

Walk-forward validation (3-fold time-series split):
- vol=OFF: OOS avg Sharpe 1.36, Ret +15.4%, DD 6.4% ← best risk-adjusted
- vol=0.12 (v10g): OOS avg Sharpe 1.15, Ret +18.5%, DD 9.7% ← reasonable middle
- vol=0.20 (v15d): OOS avg Sharpe 1.14, Ret +23.0%, DD 10.3% ← Fold3 OOS negative

Conclusion: v15d "improvement" was overfitting (amplified exposure in known bull markets).
v10g target_vol=0.12 remains the robust choice.
