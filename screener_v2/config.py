"""
Every threshold the v2 screener uses, as a named constant.

Tuning must never require touching a logic file — if you find yourself
wanting to change a number in compression.py/strength.py/entry.py, the
number belongs here instead.

Ranking weights are deliberately all 1.0 at the start. Do not hand-tune
them against recent winners; tune only through calibration.py's extension
bucket table (spec §11, §12).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Feature flag. Default OFF — the existing momentum screener stays live and
# untouched until v2 validates through a parallel run (spec §13).
# ---------------------------------------------------------------------------
def screener_v2_enabled() -> bool:
    """Read at call time, not import time, so the flag can be flipped by
    restarting the process without a code change — and so tests can patch
    the environment without reimporting the package."""
    return os.environ.get("SCREENER_V2_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Layer 1 — universe
# ---------------------------------------------------------------------------
MIN_AVG_DOLLAR_VOLUME = 5_000_000.0   # 20-day average, in the listing's own currency
MIN_MARKET_CAP = 1_000_000_000.0      # 1B floor
MIN_PRICE = 5.0                       # avoids sub-dollar noise
MIN_SESSIONS = 260                    # ~1 trading year of history required
UNIVERSE_TTL_SECONDS = 7 * 24 * 3600  # recomputed weekly

# Crypto-proxy and crypto-treasury names: volume and social signal here are
# contaminated by promotion rather than reflecting genuine positioning, so
# compression/accumulation readings on them don't mean what they mean
# elsewhere. Do not delete this list without replacing the reasoning — it is
# here on purpose, not as leftover debris.
EXCLUDED_SYMBOLS = frozenset({
    "MSTR", "MARA", "RIOT", "CLSK", "HUT", "HIVE", "BITF", "CIFR", "WULF",
    "COIN", "GLXY.TO", "HUT.TO", "BITF.TO", "HIVE.TO", "DMGI.V", "SMLR",
    "BTBT", "CAN", "GREE", "SDIG", "BTDR", "IREN", "APLD",
})

# ---------------------------------------------------------------------------
# Layer 2 — compression
# ---------------------------------------------------------------------------
BB_PERIOD = 20
BB_NUM_STD = 2.0
BBW_RANK_LOOKBACK = 126               # ~6 months of sessions
MAX_BBW_PCT_RANK = 15.0               # bottom decile-and-a-half of its own 6 months

ATR_PERIOD = 14                       # Wilder's
ATR_MEDIAN_LOOKBACK = 60

ADX_PERIOD = 14
MAX_ADX = 20.0                        # no established trend yet
ADX_SLOPE_LOOKBACK = 5

SMA_FAST = 20
SMA_REGIME = 200

BASE_LOOKBACK = 40                    # sessions used for base_high / base_low
MAX_BASE_WIDTH_PCT = 25.0             # a base too wide to place a sane stop under

# THE VETO (spec §2.2). Absolute. There is no override path, no score that
# outweighs it, and no catalyst that excuses it. Above this, the candidate is
# watchlist-only.
MAX_EXTENSION_ATR = 1.5

# ---------------------------------------------------------------------------
# Layer 3 — direction
# ---------------------------------------------------------------------------
RS_WINDOWS = (5, 20, 60, 120)
BENCHMARK_US = "SPY"
BENCHMARK_CA = "^GSPTSE"
RS_SLOW_MAX = 0.0                     # rs_60 <= 0  — flat-or-weak base
RS_FAST_MIN = 0.0                     # rs_20 >  0  — turning up
AD_SLOPE_LOOKBACK = 20
PRICE_FLAT_MAX_PCT = 3.0              # |20-session % change| under this counts as flat

# ---------------------------------------------------------------------------
# Layer 4 — catalysts (cache reads only at scan time)
# ---------------------------------------------------------------------------
CATALYST_CACHE_TTL_SECONDS = 36 * 3600   # a nightly job that misses one night is still usable
INSIDER_CLUSTER_LOOKBACK_DAYS = 90
INSIDER_CLUSTER_MIN_BUYERS = 2           # >=2 distinct insiders buying
EARNINGS_BLACKOUT_DAYS = 5               # inside this, no order path — watchlist only
IV_PERCENTILE_LOW = 25.0                 # bottom quartile

# ---------------------------------------------------------------------------
# Layer 5 — entry
# ---------------------------------------------------------------------------
STOP_ATR_BUFFER = 0.5                 # stop = base_low - 0.5 * ATR(14)
MAX_RISK_PCT = 8.0                    # stop further than this from entry -> skip
RISK_BUDGET_PER_TRADE = 500.0         # currency units risked per position

# ---------------------------------------------------------------------------
# Ranking weights (spec §11). All start at 1.0.
# ---------------------------------------------------------------------------
W_COMPRESSION = 1.0
W_RS_INFLECTION = 1.0
W_ACCUMULATING = 1.0
W_INSIDER_CLUSTER = 1.0
W_IV_LOW = 1.0
W_ADX_SLOPE = 1.0
W_EXTENSION_PENALTY = 1.0

RS_INFLECTION_CLAMP = 20.0
ADX_SLOPE_CLAMP = 10.0
ACCUMULATING_POINTS = 25.0
INSIDER_CLUSTER_POINTS = 25.0
IV_LOW_POINTS = 15.0

# ---------------------------------------------------------------------------
# Scan mechanics
# ---------------------------------------------------------------------------
BARS_LOOKBACK_DAYS = 400              # calendar days fetched; ~275 sessions
BATCH_DOWNLOAD_TIMEOUT_SECONDS = 90   # whole-universe batch, larger than main.py's per-call cap
BATCH_CHUNK_SIZE = 100                # symbols per yfinance batch request
MAX_CANDIDATES = 25                   # top N returned by a scan
SCAN_TTL_SECONDS = 3600               # a full scan re-downloads the universe; don't do it per request
CATALYST_REFRESH_INTERVAL_SECONDS = 24 * 3600   # the "nightly" job cadence

# ---------------------------------------------------------------------------
# Layer 6 — logging
# ---------------------------------------------------------------------------
# Render's web-service disk is ephemeral (wiped on deploy), so a JSONL log
# here survives restarts but NOT deploys. calibration.py prefers the Postgres
# sink whenever DATABASE_URL is configured and falls back to this path.
LOG_PATH = os.environ.get(
    "SCREENER_V2_LOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screener_v2_log.jsonl"),
)
