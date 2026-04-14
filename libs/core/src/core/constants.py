"""Constants and configuration defaults for cta-forge."""

# Default service ports
DATA_SERVICE_PORT = 8001
ALPHA_SERVICE_PORT = 8002
STRATEGY_SERVICE_PORT = 8003
EXECUTOR_PORT = 8004
REPORT_SERVICE_PORT = 8005

# Default service URLs (local development)
DATA_SERVICE_URL = f"http://localhost:{DATA_SERVICE_PORT}"
ALPHA_SERVICE_URL = f"http://localhost:{ALPHA_SERVICE_PORT}"
STRATEGY_SERVICE_URL = f"http://localhost:{STRATEGY_SERVICE_PORT}"
EXECUTOR_URL = f"http://localhost:{EXECUTOR_PORT}"
REPORT_SERVICE_URL = f"http://localhost:{REPORT_SERVICE_PORT}"

# Binance API
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_KLINES_ENDPOINT = "/fapi/v1/klines"
BINANCE_EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
BINANCE_FUNDING_RATE_ENDPOINT = "/fapi/v1/fundingRate"
BINANCE_KLINE_LIMIT = 1500

# Default strategy parameters
DEFAULT_TIMEFRAME = "6h"
DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_LONG_RATIO = 0.7
DEFAULT_SHORT_RATIO = 0.3
DEFAULT_MAX_DRAWDOWN = 0.15
DEFAULT_TRAILING_STOP_ATR_MULT = 2.0

# Data
PARQUET_COMPRESSION = "zstd"

# ── v10g strategy parameters ─────────────────────────────────────
# Source of truth for the v10g CTA strategy (previously in engine/live.py).

V10G_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "ADA",
    "DOT",
    "ATOM",
    "NEAR",
]
V10G_TIMEFRAME_HOURS = 6
V10G_ADX_PERIODS = [22, 27, 32]  # ensemble ADX
V10G_ADX_THRESHOLD = 25
V10G_SIGNAL_THRESHOLD = 0.35
V10G_MIN_HOLD_BARS = 12
V10G_TRAILING_STOP_ATR = 4.5
V10G_RISK_PER_TRADE = 0.015  # 1.5% of equity
V10G_MAX_POSITIONS = 5
V10G_REBALANCE_EVERY = 4  # bars between rebalance checks

# v10g risk limits (decimal fractions, NOT percentages)
V10G_MAX_DRAWDOWN = 0.15  # hard stop: flatten everything
V10G_DD_BREAKER = 0.08  # reduce position sizes by 50%
