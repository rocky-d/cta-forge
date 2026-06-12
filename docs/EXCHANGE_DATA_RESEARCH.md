# Exchange Historical Kline Data — Research Notes

Last update: 2026-06-12

Research scope: publicly accessible REST API for hourly (1h) OHLCV kline data
on perpetual futures markets, suitable for multi-year backtesting.

---

## 1. Binance (baseline — already in use)

| Aspect | Detail |
|--------|--------|
| Endpoint | `GET /fapi/v1/klines` |
| Base URL | `https://fapi.binance.com` |
| Auth | Public (no API key needed for market data) |
| Interval | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M |
| Limit per request | 1500 candles |
| Rate limit | 2400 req/min (IP-based, weight=2 for klines) |
| Earliest data | Depends on symbol; BTCUSDT perp since ~2019-09-08; most alts since listing date |
| Data format | `[open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]` |
| Immutability | NOT guaranteed — Binance may backfill or correct historical bars |
| Summary | **Best-in-class for backtesting.** Long history, high rate limit, 1500 bars/req, no auth needed. Only weakness: data is mutable (history can change between fetches). |

---

## 2. OKX

| Aspect | Detail |
|--------|--------|
| Endpoint | `GET /api/v5/market/history-candles` (historical, > 2 days old) |
|          | `GET /api/v5/market/candles` (recent, ≤ 1440 bars) |
| Base URL | `https://www.okx.com` |
| Auth | Public (no API key) |
| Interval | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 2d, 3d, 5d, 1w, 1M, 3M |
| Limit per request | 100 candles (candles endpoint) / 100 candles (history-candles endpoint) |
| Rate limit | 20 req / 2s (IP-based) |
| Earliest data | Perp swaps available since ~2019-03-30 per Tardis.dev; API returns data from listing date |
| Data format | `[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]` |
| Key limitation | **100 bars per request is very low** — fetching 58k 1h bars requires 580 requests. At 20req/2s, that's ~58s of back-to-back requests per symbol, ~19min for 19 symbols. Practical but slow. Also, history-candles is a separate endpoint for data older than ~2 days, adding complexity. |
| Summary | **Viable but slow.** Data range is comparable to Binance (since 2019 for early perps). Main friction: 100-bar pagination makes bulk download tedious. Requires careful rate limit handling. |

---

## 3. Bybit

| Aspect | Detail |
|--------|--------|
| Endpoint | `GET /v5/market/kline` |
| Base URL | `https://api.bybit.com` |
| Auth | Public |
| Interval | 1, 3, 5, 15, 30, 60, 120, 240, 360, 720 (min), D, W, M |
| Category | `linear` for USDT perpetuals, `inverse` for coin-margined |
| Limit per request | 1000 candles (default 200) |
| Rate limit | 50 req / 5s (IP-based for market data) |
| Earliest data | Depends on symbol listing date; Bybit perps launched ~2018-11 for BTC, with alts added progressively |
| Data format | `[startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]` |
| Key limitation | Limit 1000 is solid but still requires ~58 requests per symbol for 58k bars × 19 symbols = ~1100 requests. At 50req/5s, that's 109s per symbol, ~35min for 19 symbols. |
| Summary | **Good alternative.** 1000 bars/req is second only to Binance. Data from 2018 covers our full backtest window. Rate limit is workable. Clean API with `category` parameter separating linear/inverse. |

---

## 4. Hyperliquid

| Aspect | Detail |
|--------|--------|
| Endpoint | `POST /info` (type: `candleSnapshot`) |
| Base URL | `https://api.hyperliquid.xyz` |
| Auth | Public (POST with JSON body, no signature needed for info queries) |
| Interval | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w |
| Limit per request | 5000 candles |
| Rate limit | ~100 req/min (RPC-based) |
| Earliest data | HL mainnet launched ~2023-05. Earliest available data ~mid-2023. |
| Data format | `[{t, T, o, h, l, c, v, n, i, s}, ...]` — all values as strings |
| Key limitation | **5000 1h candles = ~208 days of history.** Total history only goes back to 2023 (~3 years). Cannot cover 2019-2022 period required for backtesting. |
| Summary | **Not viable for full backtest.** Data only from mid-2023. Even if that were enough, 5000 candle limit means at 1h granularity you can only look back ~7 months at a time before needing to paginate — and pagination requires knowing exact start time per symbol. Third-party providers (Dwellir, Allium) offer full HL history but at cost. |

---

## 5. Comparison Matrix

| | Binance | OKX | Bybit | Hyperliquid |
|---|---|---|---|---|
| Bars/req | **1500** | 100 | 1000 | 5000 |
| 1h history depth | 2019-09+ | 2019-03+ | 2018-11+ | **2023-05+** |
| Est. time (19 sym, 58k bars) | ~1min | ~19min | ~35min | N/A (no pre-2023) |
| Auth required | No | No | No | No |
| API complexity | Simple GET | 2 endpoints (recent + history) | Simple GET | POST with JSON |
| Data mutability | Yes (bars may change) | Unknown | Unknown | Unknown |

---

## 6. Practical Conclusions

**For the current backtest scope (2019→present, 19 symbols, 1h OHLCV):**

1. **Binance remains the best single source.** No other exchange matches its combination of historical depth + 1500-bar pages + rate limits.

2. **Bybit is the strongest alternative** if we needed redundancy. Similar depth, 1000 bars/req, clean API. Adding it as a second source would allow cross-verification.

3. **OKX is workable but painful.** The 100-bar limit means ~19min of sequential requests per full refresh for 19 symbols. Fine for one-off but painful for routine use.

4. **Hyperliquid is not viable for 2019-2022 backtesting.** Data only from 2023. It could serve as an out-of-sample validation source (2023→present) but not as a primary historical source.

5. **Data mutability is universal.** No exchange guarantees immutable historical klines. The only way to control for this is local parquet snapshots with versioning — which we already have but don't snapshot across runs. This is a process decision, not a source selection issue.

**Recommendation:** Keep Binance as primary. If cross-validation is desired, add Bybit fetcher for the same symbols. Don't bother with OKX unless Bybit data proves unreliable. Hyperliquid is irrelevant for pre-2023 backtests.
