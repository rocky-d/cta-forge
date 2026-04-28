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
  executor/          -- backtest engine, target-weight portfolio layer & live execution
  report-service/    -- metrics calculation, chart generation (3-panel backtest chart)
scripts/backtest/    -- CLI entry point (thin wrapper, imports from services)
backtest-results/    -- output charts, metrics JSON
docs/                -- project documentation
```

Ports and service URLs have sensible defaults in `core/constants.py`, overridable via environment variables.

## Strategy: v10g

Multi-factor trend-following with adaptive features.

Research checkpoint: the current best robust iteration candidate is documented in
[`STRATEGY_ITERATION_2026-04-28.md`](STRATEGY_ITERATION_2026-04-28.md). It now
has a reusable target-weight profile/backtest path, but is not yet the default
live deployment profile.

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
- `services/executor/src/executor/decision.py` -- V10GDecisionEngine (stateless decision logic, shared by live + backtest)
- `services/executor/src/executor/targeting.py` -- target-weight portfolio abstractions and target-to-order delta calculation
- `services/executor/src/executor/portfolio_backtest.py` -- target-weight portfolio simulation utilities
- `services/executor/src/executor/profiles/v16a_badscore_overlay.py` -- reusable v16a Badscore Overlay target profile
- `services/executor/src/executor/live.py` -- LiveEngine (current default: v10g strategy, 6h candle-aligned loop; target-mode reconciliation is available behind explicit strategy injection)
- `services/executor/src/executor/notify.py` -- Notifier stack (Telegram, Lark/Feishu, multi-backend)
- `services/executor/src/executor/run_live.py` -- CLI entry point

Preflight checks (must all pass before trading):
1. Exchange connectivity
2. Minimum equity ($100)
3. Position reconcile (state vs exchange, adopt orphaned positions)
4. Stale open orders -> auto-cancel
5. Market data spot check

Risk controls:
- 15% max drawdown -> hard stop (flatten all)
- 8% drawdown -> DD breaker (50% position size reduction)
- ATR trailing stops (5.0x, tightens to 3.0x after 2.0x ATR profit)
- Max 5 concurrent positions
- 15% max per-position equity

Important: V10GDecisionEngine.tick() mutates state internally (deletes closed
positions, adjusts partial qty, updates best_price). Callers must snapshot
positions before tick() if they need pre-tick state for settlement.

Ops:
- Env vars: HL_PRIVATE_KEY, HL_ACCOUNT_ADDRESS, HL_NETWORK, DRY_RUN, TG_BOT_TOKEN, TG_CHAT_ID, LARK_WEBHOOK_URL, DATA_DIR, STRATEGY_PROFILE, MIN_ORDER_NOTIONAL, V16A_MAX_STALENESS_HOURS
- `STRATEGY_PROFILE` default is `v10g-engine-6h`. `v16a-badscore-overlay` is only allowed with `DRY_RUN=true` for shadow validation until explicitly promoted.
- State persistence: engine-state.json (auto-generated, gitignored)
- Data cache: parquet files in DATA_DIR (live + backtest share via ParquetStore)
- Deployment: GitHub Actions workflow_dispatch -> GHCR -> SSH EC2 (Tokyo t3.small)

## CI/CD

- Lint: ruff check + ruff format + ty check (3 parallel jobs)
- Test: pytest (unit + integration)
- Deploy: workflow_dispatch -> check-ci gate -> build Docker -> push GHCR -> SSH deploy EC2
- All actions on Node.js 22+ (checkout@v6, setup-uv@v7, docker/login-action@v4)

## Backtest

Architecture:
- `services/executor/src/executor/backtest.py` -- V10GDecisionEngine action-based backtest module
  - `fetch_bars()` -- ParquetStore cache + Binance incremental fetch
  - `precompute()`, `build_timeline()`, `align_data()`, `compute_signals()` -- data pipeline
  - `run_backtest()` -- V10GDecisionEngine loop (same engine as live trading)
  - `run_full_backtest()` -- single orchestration entry point
- `services/executor/src/executor/portfolio_backtest.py` -- target-weight portfolio backtest path used by v16a
- `scripts/backtest/joint_badscore_research.py` -- thin v16a reproduction CLI using the reusable profile/backtest modules
- `services/report-service/src/report_service/plot.py` -- `plot_backtest()` three-panel chart
  (equity + BTC/ETH indexed overlay, drawdown, monthly returns)

Usage:
- CLI: `uv run python scripts/backtest/v10g_maxrange.py` (thin wrapper, imports from services)
- REST API:
  1. `POST :8004/backtest` -> JSON (metrics, equity_curve, btc/eth prices, yearly)
  2. `POST :8005/plot/backtest` -> PNG (three-panel chart with BTC/ETH overlay)

Output: `backtest-results/backtest_v10g_engine.png`, `backtest-results/metrics_v10g_engine.json`
