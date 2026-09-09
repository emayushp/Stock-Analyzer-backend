"""
Layer 2 — compression state. This is what replaces the momentum preset.

The momentum screen asked "what already moved". This asks "what is coiled
and hasn't moved yet": band width in the bottom of its own six-month range,
volatility contracting rather than expanding, no established trend, price
above its 200-day.

Two rules here are worth stating outright because they are easy to erode:

1. A gate needs an affirmative value to pass. Any indicator that comes back
   None — not enough history, an undefined ADX on a no-movement tape,
   whatever — fails its gate. "We couldn't measure it" is never "it's fine".

2. The extension veto is absolute (spec §2.2). If extension_atr > 1.5 the
   name is watchlist-only: no score outweighs it, no catalyst excuses it,
   and there is no override parameter on this function. If you are adding
   one, you are removing the entire point of the rebuild — the veto is the
   mechanism that stops the screen buying names that already ran.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import config, indicators as ind
from .schemas import CompressionState

logger = logging.getLogger("stock-analyzer")


def evaluate(symbol: str, bars: Optional[pd.DataFrame]) -> CompressionState:
    """Score one symbol's compression state.

    Always returns a CompressionState — never raises — so one malformed
    symbol cannot take down a whole scan."""
    state = CompressionState()
    try:
        if not ind.has_ohlcv(bars):
            state.reason = "no usable bars"
            return state
        if len(bars) < config.MIN_SESSIONS:
            state.reason = f"only {len(bars)} sessions, needs {config.MIN_SESSIONS}"
            return state

        close = bars["Close"]
        state.close = ind.latest(close)

        bbw_series = ind.bollinger_band_width(close, config.BB_PERIOD, config.BB_NUM_STD)
        state.bbw = ind.latest(bbw_series)
        state.bbw_pct_rank = ind.percentile_rank(bbw_series, config.BBW_RANK_LOOKBACK)

        atr_series = ind.atr(bars, config.ATR_PERIOD)
        atr_pct_series = ind.atr_pct(bars, config.ATR_PERIOD)
        state.atr = ind.latest(atr_series)
        state.atr_pct = ind.latest(atr_pct_series)
        median_window = atr_pct_series.dropna().tail(config.ATR_MEDIAN_LOOKBACK)
        if len(median_window) >= config.ATR_MEDIAN_LOOKBACK:
            state.atr_median_60 = round(float(median_window.median()), 6)

        adx_series = ind.adx(bars, config.ADX_PERIOD)
        state.adx_14 = ind.latest(adx_series)
        state.adx_slope_5 = ind.change_over(adx_series, config.ADX_SLOPE_LOOKBACK)

        state.sma20 = ind.latest(ind.sma(close, config.SMA_FAST))
        state.sma200 = ind.latest(ind.sma(close, config.SMA_REGIME))
        state.extension_atr = ind.extension_in_atr(state.close, state.sma20, state.atr)

        state.base_high, state.base_low, state.base_width_pct = ind.base_levels(
            bars, config.BASE_LOOKBACK
        )

        reason = _first_failure(state)
        if reason:
            state.reason = reason
            return state

        # Every setup gate passed. The only thing left is the price, and
        # that is where the veto lives.
        if state.extension_atr is None:
            state.reason = "extension unavailable — cannot clear the veto"
            return state
        if state.extension_atr > config.MAX_EXTENSION_ATR:
            state.veto = True
            state.reason = (
                f"extended {state.extension_atr:.2f} ATR above the 20-day "
                f"(limit {config.MAX_EXTENSION_ATR}) — watchlist only"
            )
            return state

        state.ok = True
        state.reason = "compressed, untrending, above the 200-day, not extended"
        return state

    except Exception as e:
        logger.info(f"screener_v2: compression evaluation failed for {symbol}: {e}")
        state.ok = False
        state.veto = False
        state.reason = "compression evaluation failed"
        return state


def _first_failure(state: CompressionState) -> str:
    """The first setup gate that isn't affirmatively satisfied, or "".

    Ordered cheapest-to-explain first so the logged reason is the most
    informative one rather than whichever happened to be checked last."""
    if state.bbw_pct_rank is None:
        return "band width percentile unavailable"
    if state.bbw_pct_rank > config.MAX_BBW_PCT_RANK:
        return (
            f"band width at the {state.bbw_pct_rank:.0f}th percentile of its own "
            f"6 months (needs <= {config.MAX_BBW_PCT_RANK:.0f})"
        )

    if state.atr_pct is None or state.atr_median_60 is None:
        return "ATR history unavailable"
    if not state.atr_pct < state.atr_median_60:
        return (
            f"volatility expanding — ATR {state.atr_pct:.2f}% vs a 60-day median "
            f"of {state.atr_median_60:.2f}%"
        )

    if state.adx_14 is None:
        return "ADX unavailable"
    if not state.adx_14 < config.MAX_ADX:
        return f"trend already established — ADX {state.adx_14:.1f} (needs < {config.MAX_ADX:.0f})"

    if state.close is None or state.sma200 is None:
        return "200-day average unavailable"
    if not state.close > state.sma200:
        return "below the 200-day"

    if state.base_width_pct is None:
        return "base levels unavailable"
    if state.base_width_pct > config.MAX_BASE_WIDTH_PCT:
        return (
            f"base is {state.base_width_pct:.0f}% wide (max {config.MAX_BASE_WIDTH_PCT:.0f}%) "
            "— no sane stop placement"
        )

    return ""
