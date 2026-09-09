"""
Layer 5 — entry.

Selection ends with a target price, never a market order (spec §2.3). The
edge being targeted by this whole rebuild is a better entry price and a
cheaper stop, and that edge is realized here or not at all: a good name
bought at the wrong price is the same trade the momentum screen was making.

Three rejections live here, and each one is the screen declining a trade
it could easily have taken:

  * `entry_target >= close` — the target must sit BELOW current price.
    Anything else is chasing, dressed up as a limit order.
  * `r_pct > MAX_RISK_PCT` — the stop is too far away to size sanely.
  * earnings blackout — a timing rule, not a score. The name stays on the
    watchlist; it just doesn't get an order that walks into a print.

On the order payload: the IBKR connector produces drafts only and supports
MARKET and LIMIT. A protective stop must be attached natively in IBKR as a
bracket — emitting an unlinked SELL LIMIT plus SELL STOP against the same
shares triggers a margin rejection — so the stop travels as a labelled
field, never as a second order.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from . import config
from .schemas import CompressionState, EntryPlan, OrderDraft

logger = logging.getLogger("stock-analyzer")


def plan(
    symbol: str,
    compression: CompressionState,
    earnings_blackout: bool = False,
) -> EntryPlan:
    """Turn a compressed setup into a resting limit order, or explain why
    there isn't one. Always returns an EntryPlan — never raises."""
    result = EntryPlan()
    try:
        close = compression.close
        base_high = compression.base_high
        base_low = compression.base_low
        sma20 = compression.sma20
        atr_value = compression.atr

        if close is None or base_high is None or base_low is None or atr_value is None:
            result.reason = "missing levels — cannot place a target"
            return result

        if close > base_high:
            # Already through the base: wait for the retest rather than
            # paying up for the breakout bar.
            entry_target = base_high
            result.already_broke_out = True
        else:
            # Still inside the base: wait for the pullback to the tighter
            # of the two supports.
            entry_target = base_high if sma20 is None else max(sma20, base_low)

        stop = base_low - config.STOP_ATR_BUFFER * atr_value

        entry_target = round(float(entry_target), 2)
        stop = round(float(stop), 2)

        # Re-check the chase rule AFTER rounding: a target that only cleared
        # it by a fraction of a cent shouldn't sneak through on rounding.
        if entry_target >= close:
            result.entry_target = entry_target
            result.stop = stop
            result.reason = (
                f"target {entry_target} is not below the last price {close} — "
                "would be chasing"
            )
            return result

        risk_per_share = round(entry_target - stop, 4)
        if risk_per_share <= 0:
            result.entry_target = entry_target
            result.stop = stop
            result.reason = "stop sits at or above the entry target"
            return result

        r_pct = round(risk_per_share / entry_target * 100.0, 3)
        result.entry_target = entry_target
        result.stop = stop
        result.risk_per_share = risk_per_share
        result.r_pct = r_pct

        if r_pct > config.MAX_RISK_PCT:
            result.reason = (
                f"stop is {r_pct:.1f}% away (max {config.MAX_RISK_PCT:.0f}%) — "
                "too far to size sanely"
            )
            return result

        shares = int(math.floor(config.RISK_BUDGET_PER_TRADE / risk_per_share))
        result.shares = shares
        if shares < 1:
            result.reason = (
                f"risk budget {config.RISK_BUDGET_PER_TRADE:.0f} buys less than one "
                f"share at {risk_per_share:.2f} of risk per share"
            )
            return result

        if earnings_blackout:
            # Deliberately AFTER the sizing math so the watchlist entry
            # still shows what the trade would have been.
            result.reason = (
                f"earnings within {config.EARNINGS_BLACKOUT_DAYS} sessions — "
                "watchlist only, no order"
            )
            return result

        result.order = OrderDraft(
            symbol=symbol,
            quantity=shares,
            limit_price=entry_target,
            stop_level_for_manual_attachment=stop,
        )
        result.ok = True
        result.reason = (
            f"resting limit at {entry_target}, stop {stop} ({r_pct:.1f}% risk), "
            f"{shares} shares"
        )
        return result

    except Exception as e:
        logger.info(f"screener_v2: entry planning failed for {symbol}: {e}")
        result.ok = False
        result.order = None
        result.reason = "entry planning failed"
        return result
