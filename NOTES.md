# cta-forge — NOTES.md

## Project Overview

Crypto CTA (Commodity Trading Advisor) trend-following strategy engine.
Monorepo with 5 microservices + shared library, targeting Binance USDS-M perpetual futures.

## Architecture

```
libs/cta-core/       — shared types, protocols, constants
services/
  data-server/       — Binance kline fetcher, parquet store, REST API (:8001)
  alpha-server/      — factor computation (TSMOM, Donchian, VolRegime) (:8002)
  strategy-server/   — signal composition, allocation, risk (:8003)
  engine/            — backtest engine (:8004)
  reporter/          — metrics calculation, chart generation (:8005)
scripts/backtest/    — standalone backtest scripts (v10 sweep, long-term)
backtest-results/    — output charts, metrics JSON, sweep results
```

## Strategy: v10g (current best)

Multi-factor trend-following with adaptive features:

Signals:
- Multi-timeframe momentum (lookbacks: 20/60/120 bars)
- Ensemble ADX filter (thresholds: 22/27/32, averaged)
- Donchian breakout confirmation (20-bar channel)
- Volume confirmation (20-bar avg ratio)
- DI directional filter (+DI/-DI)
- BTC correlation filter (dampen alt signals contradicting BTC trend)
- Adaptive lookback weighting (high vol → favor short lookback)
- Signal persistence (require 2 consecutive bars same direction)

Position management:
- Risk parity sizing (inverse vol contribution)
- Vol targeting (12% annualized, EMA of recent returns)
- ATR trailing stop (5.0x, tightens to 3.0x after 2.0x ATR profit)
- Max hold 100 bars (forced exit)
- Min hold 16 bars (prevent whipsaw)
- Partial take-profit at 2.5x ATR
- Max 5 concurrent positions
- Rebalance every 4 bars
- 8% drawdown circuit breaker (halves new position size)

## Backtest Results (v10g, 2021-01 → 2026-04, 12 symbols)

| Metric        | Value   |
|---------------|---------|
| Period        | 1888 days (5.2 years) |
| Symbols       | 12 (BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK, ADA, DOT, ATOM, NEAR) |
| Return        | +67.5%  |
| Ann. Return   | +10.5%  |
| Sharpe        | 0.89    |
| Sortino       | 1.10    |
| Max DD        | 11.3%   |
| Calmar        | 0.92    |
| Profit Factor | 1.36    |
| Win Rate      | 61.3%   |
| Trades        | 899     |
| Ulcer Index   | 0.046   |

Yearly: 2021 +17.3% | 2022 +6.8% | 2023 +7.5% | 2024 +20.8% | 2025 +5.7% | 2026 YTD -2.9%

## Walk-Forward Validation (v10g, 893d window, 6 folds)

- Avg OOS Sharpe: 2.04 (±1.58)
- OOS positive: 5/6 windows
- Composite score: 1.35 (rank #1 out of 10 configs)

## Iteration History

| Version | Full Sharpe | Trades | Max DD | Key Change |
|---------|-------------|--------|--------|------------|
| v1      | -1.23       | 4690   | 19.8%  | Baseline (4 raw factors) |
| v2      | -0.20       | 464    | 34.6%  | Signal threshold, min hold, ATR sizing |
| v3-v5   | 0.2–0.8     | ~200   | ~15%   | Parameter tuning, Donchian confirm |
| v6      | 1.96        | 88     | 6.3%   | Cross-sectional signal, vol filter |
| v7      | 1.05 (OOS)  | —      | —      | IS/OOS validation |
| v8      | 1.37        | 462    | 13.6%  | Walk-forward 5-window |
| v9      | 1.11        | 430    | 9.8%   | Ensemble ADX, lowest OOS variance |
| v10g    | 1.14 (893d) | 471    | 14.7%  | Adaptive + risk parity + DD breaker + persistence |
| v10g-LT | 0.89 (5yr)  | 899    | 11.3%  | 5-year validation, no re-tuning |

## Design Principles

- Protocol + composition over ABC/inheritance
- No: ccxt, loguru, pandas (use polars/numpy), LangChain
- Ruff lint + format, ty type check, pytest
- Factor auto-discovery via registry
- Parquet storage: `{data_dir}/{symbol}/{interval}.parquet` (zstd)
- Signal range: normalized to [-1, +1]

## Phase 4: Live Trading (Hyperliquid Testnet)

Architecture:
- `services/exchange/` — new microservice (:8006), HL SDK wrapper
  - `ExchangeAdapter` Protocol interface (structural subtyping)
  - `HyperliquidAdapter` implementation (safe init, unified account, async executor)
  - REST routes: /account, /market/{symbol}, /order, /cancel, /leverage
- `services/engine/live.py` — LiveEngine (v10g strategy, 6h candle-aligned loop)
- `services/engine/run_live.py` — CLI entry point

Preflight checks (must all pass before trading):
1. Exchange connectivity
2. Minimum equity ($100)
3. Stale positions → auto-flatten
4. Stale open orders → auto-cancel
5. Market data spot check

Risk controls:
- 15% max drawdown → hard stop (flatten all)
- 8% drawdown → DD breaker (50% position size reduction)
- ATR trailing stops (4.5x)
- Max 5 concurrent positions
- 20% max per-position equity

Testnet wallet: `0xe9bfE8DF9277ACafA19667bA64aD617413D70B71`
Testnet balance: ~$2009 USDC (unified account, spot = perp collateral)

Ops:
- Use tmux for long-running engine process
- Env vars: HL_PRIVATE_KEY, HL_ACCOUNT_ADDRESS, HL_NETWORK, DRY_RUN

Commits:
- `3616ad2` feat: exchange-server
- `b71a928` feat: live engine (v10g)
- `7545cf1` feat: preflight checks

## Next Steps

- Run engine continuously in tmux (6h rebalance loop)
- Telegram notifications on trades
- State persistence (survive restarts)
- Exchange-server test suite
- Further parameter stability analysis (Monte Carlo?)
- Transaction cost sensitivity analysis
