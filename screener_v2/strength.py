"""
Layer 3 — direction.

Compression tells you a move is coming; it does not tell you which way.
This layer supplies the bias, and it does so by INFLECTION rather than
MAXIMUM: relative strength turning up from a flat-or-weak base, not
relative strength already at its high.

That distinction is the whole point. A name at maximum short-term RS is
precisely what this screen exists to avoid — it is the same "already
moved" condition the momentum screen selected on, wearing a different
label. So the gate is deliberately two-sided:

    rs_60 <= 0   the six-month base was flat or weak (nothing to chase)
    rs_20  > 0   the last month turned up (something changed)

Accumulation is scored, never gated: a rising Williams A/D line against
flat price inside a compressed base is the pre-breakout signature, but it
is a bonus, not a requirement.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import config, indicators as ind
from .schemas import StrengthState

logger = logging.getLogger("stock-analyzer")

_CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")


def benchmark_for(symbol: str) -> str:
    """SPY for US listings, the TSX composite for Canadian ones.

    Comparing a TSX name against SPY would price in the CAD/USD move and
    the index composition difference as if they were stock-specific
    strength, which is exactly the kind of false signal this layer is
    supposed to filter out."""
    upper = (symbol or "").upper()
    return config.BENCHMARK_CA if upper.endswith(_CA_SUFFIXES) else config.BENCHMARK_US


def evaluate(
    symbol: str,
    bars: Optional[pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
) -> StrengthState:
    """Relative-strength inflection plus the accumulation flag.

    Always returns a StrengthState — never raises."""
    state = StrengthState(benchmark=benchmark_for(symbol))
    try:
        if not ind.has_ohlcv(bars):
            state.reason = "no usable bars"
            return state
        if not ind.has_ohlcv(benchmark_bars):
            # Without a benchmark there is no relative strength to measure,
            # and guessing one would be worse than declining to answer.
            state.reason = f"benchmark {state.benchmark} unavailable"
            return state

        close = bars["Close"]
        bench_close = benchmark_bars["Close"]

        rs_values = {}
        for window in config.RS_WINDOWS:
            stock_ret = ind.pct_change_over(close, window)
            bench_ret = ind.pct_change_over(bench_close, window)
            # pct_change_over already returns percent, so the difference is
            # in percentage points — the spec's trailing "* 100" is applied
            # inside the helper, not again here.
            rs_values[window] = (
                None if stock_ret is None or bench_ret is None else round(stock_ret - bench_ret, 4)
            )

        state.rs_5 = rs_values.get(5)
        state.rs_20 = rs_values.get(20)
        state.rs_60 = rs_values.get(60)
        state.rs_120 = rs_values.get(120)

        ad_line = ind.williams_ad(bars)
        state.ad_slope_20 = ind.normalized_ad_slope(ad_line, config.AD_SLOPE_LOOKBACK)
        move_20 = ind.pct_change_over(close, config.AD_SLOPE_LOOKBACK)
        state.price_flat = move_20 is not None and abs(move_20) < config.PRICE_FLAT_MAX_PCT
        state.accumulating = bool(
            state.ad_slope_20 is not None and state.ad_slope_20 > 0 and state.price_flat
        )

        if state.rs_60 is None or state.rs_20 is None:
            state.reason = "not enough history for the 20/60-session comparison"
            return state
        if not state.rs_60 <= config.RS_SLOW_MAX:
            state.reason = (
                f"already strong over 60 sessions (RS {state.rs_60:+.1f}pp vs "
                f"{state.benchmark}) — this screen wants the inflection, not the maximum"
            )
            return state
        if not state.rs_20 > config.RS_FAST_MIN:
            state.reason = (
                f"no upturn yet — 20-session RS {state.rs_20:+.1f}pp vs {state.benchmark}"
            )
            return state

        state.ok = True
        state.reason = (
            f"RS turning up: {state.rs_60:+.1f}pp over 60 sessions, "
            f"{state.rs_20:+.1f}pp over 20, vs {state.benchmark}"
        )
        return state

    except Exception as e:
        logger.info(f"screener_v2: strength evaluation failed for {symbol}: {e}")
        state.ok = False
        state.reason = "strength evaluation failed"
        return state
