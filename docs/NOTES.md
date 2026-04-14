# cta-forge -- NOTES.md

## Project Overview

Crypto CTA (Commodity Trading Advisor) trend-following strategy engine.
Monorepo with 5 microservices + 2 shared libraries, targeting Binance USDS-M perpetual futures.

## Architecture

```
libs/
  core/              -- shared protocols, constants, metrics
  exchange/          -- exchange connectivity (Hyperliquid adapter)
services/
  data-service/      -- Binance kline fetcher, parquet store, REST API
  alpha-service/     -- factor computation (TSMOM, Donchian, VolRegime, V10GComposite)
  strategy-service/  -- signal composition, allocation, risk
  executor/          -- backtest & live execution engine
  report-service/    -- metrics calculation, chart generation
scripts/backtest/    -- standalone backtest scripts
backtest-results/    -- output charts, metrics JSON
docs/                -- project documentation
```

Ports and service URLs have sensible defaults in `core/constants.py`, overridable via environment variables.

## Strategy: v10g

Multi-factor trend-following with adaptive features.

Signals:
- Multi-timeframe momentum (lookbacks: 20/60/120 bars)
- Ensemble ADX filter (thresholds: 22/27/32, averaged)
- Donchian breakout confirmation (20-bar channel)
- Volume confirmation (20-bar avg ratio)
- DI directional filter (+DI/-DI)
- BTC correlation filter (dampen alt signals contradicting BTC trend)
- Adaptive lookback weighting (high vol favors short lookback)
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

## Design Principles

- Protocol + composition over ABC/inheritance
- No: ccxt, loguru, pandas (use polars/numpy), LangChain
- Ruff lint + format, ty type check, pytest
- Factor auto-discovery via registry
- Parquet storage: `{data_dir}/{symbol}/{interval}.parquet` (zstd)
- Signal range: normalized to [-1, +1]

## Live Trading (Hyperliquid)

Architecture:
- `libs/exchange/` -- HL SDK wrapper library
  - `ExchangeAdapter` Protocol interface (structural subtyping)
  - `HyperliquidAdapter` implementation (safe init, unified account, async executor)
- `services/executor/src/executor/live.py` -- LiveEngine (v10g strategy, 6h candle-aligned loop)
- `services/executor/src/executor/run_live.py` -- CLI entry point

Preflight checks (must all pass before trading):
1. Exchange connectivity
2. Minimum equity ($100)
3. Stale positions -> auto-flatten
4. Stale open orders -> auto-cancel
5. Market data spot check

Risk controls:
- 15% max drawdown -> hard stop (flatten all)
- 8% drawdown -> DD breaker (50% position size reduction)
- ATR trailing stops (4.5x)
- Max 5 concurrent positions
- 20% max per-position equity

Ops:
- Env vars: HL_PRIVATE_KEY, HL_ACCOUNT_ADDRESS, HL_NETWORK, DRY_RUN, TG_BOT_TOKEN, TG_CHAT_ID
- State persistence: engine-state.json (auto-generated, gitignored)

## Next Steps

- Deploy to cloud server (systemd service + CI/CD)
- Transaction cost sensitivity analysis
