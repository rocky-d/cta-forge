"""Constants and configuration defaults for cta-forge."""

# Default service ports
DATA_SERVER_PORT = 8001
ALPHA_SERVER_PORT = 8002
STRATEGY_SERVER_PORT = 8003
ENGINE_PORT = 8004
REPORTER_PORT = 8005

# Default service URLs (local development)
DATA_SERVER_URL = f"http://localhost:{DATA_SERVER_PORT}"
ALPHA_SERVER_URL = f"http://localhost:{ALPHA_SERVER_PORT}"
STRATEGY_SERVER_URL = f"http://localhost:{STRATEGY_SERVER_PORT}"
ENGINE_URL = f"http://localhost:{ENGINE_PORT}"
REPORTER_URL = f"http://localhost:{REPORTER_PORT}"

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
