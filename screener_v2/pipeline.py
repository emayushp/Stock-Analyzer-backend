"""
Orchestration: universe -> compression -> strength -> catalysts -> entry,
then ranking, then the emission log.

The ranking function is the one place where it would be easiest to quietly
undo the whole rebuild, so it is worth being explicit about what it may
see. `score_candidate` takes exactly three arguments — the compression
state, the strength state and the catalysts — and there is no path from a
recent return into any of them. `perf_1d` is attached to the Candidate
AFTER scoring, for the calibration log only.

Order of evaluation matters too. A vetoed name stops at layer 2: it gets no
strength read, no catalyst read and no score, because "no score, no
catalyst, no override" is the veto's actual meaning, not just a slogan.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from . import calibration, catalysts as catalysts_layer, compression, config
from . import entry as entry_layer, indicators as ind, strength, universe
from .schemas import Candidate, Catalysts, CompressionState, EntryPlan, ScanResult, StrengthState

logger = logging.getLogger("stock-analyzer")


def clamp(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return low if low > 0 else 0.0
    return max(low, min(high, float(value)))


def score_candidate(
    compression_state: CompressionState,
    strength_state: StrengthState,
    catalysts,
) -> float:
    """Composite rank for a candidate that already cleared every gate.

    Deliberately narrow inputs: compression tightness, the size of the RS
    inflection, the accumulation and insider flags, a low-IV bonus, the ADX
    upturn, and a penalty for whatever extension remains. No recent return,
    no AI verdict, no sentiment (spec §2.1, §2.4)."""
    bbw_rank = compression_state.bbw_pct_rank
    compression_points = (100.0 - bbw_rank) if bbw_rank is not None else 0.0

    inflection = 0.0
    if strength_state.rs_20 is not None and strength_state.rs_60 is not None:
        inflection = clamp(strength_state.rs_20 - strength_state.rs_60, 0.0, config.RS_INFLECTION_CLAMP)

    iv_low = (
        catalysts.iv_percentile is not None
        and catalysts.iv_percentile <= config.IV_PERCENTILE_LOW
    )

    score = (
        config.W_COMPRESSION * compression_points
        + config.W_RS_INFLECTION * inflection
        + config.W_ACCUMULATING * (config.ACCUMULATING_POINTS if strength_state.accumulating else 0.0)
        + config.W_INSIDER_CLUSTER * (config.INSIDER_CLUSTER_POINTS if catalysts.insider_cluster_buy else 0.0)
        + config.W_IV_LOW * (config.IV_LOW_POINTS if iv_low else 0.0)
        + config.W_ADX_SLOPE * clamp(compression_state.adx_slope_5, 0.0, config.ADX_SLOPE_CLAMP)
        - config.W_EXTENSION_PENALTY * clamp(compression_state.extension_atr, 0.0, config.MAX_EXTENSION_ATR) * 10.0
    )
    return round(score, 3)


def run(
    provider,
    symbols: List[str],
    market_cap_lookup: Optional[Callable[[List[str]], Dict[str, Optional[float]]]] = None,
    force_universe: bool = False,
    log_emissions: bool = True,
) -> ScanResult:
    """One full scan. Never raises: a symbol that blows up is counted and
    skipped, and a failure in the log sink never costs you the scan."""
    generated_at = datetime.now(timezone.utc).isoformat()
    result = ScanResult(generated_at=generated_at)

    members = universe.get_universe(provider, symbols, market_cap_lookup, force=force_universe)
    result.universe_size = len(members)
    if not members:
        # Distinguish "no price data at all" from "data fine, nothing
        # qualified". They look identical in an empty result and send you
        # to completely different places when debugging.
        loaded = getattr(provider, "loaded_symbols", lambda: [])()
        result.note = (
            "No price data loaded for any symbol — the market data feed is "
            "unavailable, so this is not a screening result."
            if not loaded else
            "Universe is empty — nothing cleared the liquidity, size and history floors."
        )
        return result

    benchmark_bars = {
        name: provider.daily_bars(name, 0)
        for name in (config.BENCHMARK_US, config.BENCHMARK_CA)
    }

    candidates: List[Candidate] = []
    watchlist: List[Candidate] = []

    for member in members:
        symbol = member.listing_used
        try:
            bars = provider.daily_bars(symbol, 0)
            if not ind.has_ohlcv(bars):
                result.bars_missing += 1
                continue
            result.scanned += 1

            compression_state = compression.evaluate(symbol, bars)

            if compression_state.veto:
                # Stops here on purpose. No strength read, no catalyst read,
                # no score — the veto is not a tiebreaker.
                result.vetoed_extension += 1
                watchlist.append(Candidate(
                    symbol=member.symbol, listing_used=symbol, generated_at=generated_at,
                    score=None, watchlist_only=True, watchlist_reason=compression_state.reason,
                    compression=compression_state, strength=StrengthState(),
                    catalysts=Catalysts(available=False),
                    entry=EntryPlan(reason="vetoed on extension — no order"),
                    perf_1d=ind.pct_change_over(bars["Close"], 1),
                ))
                continue

            if not compression_state.ok:
                result.failed_compression += 1
                continue

            strength_state = strength.evaluate(
                symbol, bars, benchmark_bars.get(strength.benchmark_for(symbol))
            )
            if not strength_state.ok:
                result.failed_strength += 1
                continue

            # Cache read only. A cold cache yields available=False and the
            # candidate is emitted anyway.
            catalyst_state = catalysts_layer.read(symbol)
            if catalyst_state.available:
                result.catalysts_available = True

            entry_plan = entry_layer.plan(
                symbol, compression_state, earnings_blackout=catalyst_state.earnings_blackout
            )

            candidate = Candidate(
                symbol=member.symbol, listing_used=symbol, generated_at=generated_at,
                compression=compression_state, strength=strength_state,
                catalysts=catalyst_state, entry=entry_plan,
            )
            candidate.score = score_candidate(compression_state, strength_state, catalyst_state)
            # Attached after scoring, and never read by it (spec §2.1).
            candidate.perf_1d = ind.pct_change_over(bars["Close"], 1)

            if entry_plan.ok:
                candidates.append(candidate)
            elif catalyst_state.earnings_blackout:
                candidate.watchlist_only = True
                candidate.watchlist_reason = entry_plan.reason
                watchlist.append(candidate)
            else:
                result.failed_entry += 1

        except Exception as e:
            logger.info(f"screener_v2: scan failed for {member.listing_used}: {e}")
            continue

    candidates.sort(key=lambda c: (c.score if c.score is not None else -1e9), reverse=True)
    watchlist.sort(key=lambda c: (c.score if c.score is not None else -1e9), reverse=True)

    result.candidates = candidates[: config.MAX_CANDIDATES]
    result.watchlist_only = watchlist[: config.MAX_CANDIDATES]
    result.note = _build_note(result)

    if log_emissions:
        try:
            calibration.log_candidates(result.candidates + result.watchlist_only)
        except Exception as e:
            logger.error(f"screener_v2: emission logging failed: {e}")

    return result


def _build_note(result: ScanResult) -> str:
    catalyst_note = (
        "Catalyst cache is warm."
        if result.catalysts_available
        else "Catalyst cache is cold — candidates are emitted without catalyst context "
             "rather than fetching inline."
    )
    return (
        f"Scanned {result.scanned} of {result.universe_size} universe names. "
        f"{result.failed_compression} weren't compressed, {result.failed_strength} had no "
        f"relative-strength inflection, {result.vetoed_extension} were vetoed as too "
        f"extended, {result.failed_entry} had no sane entry. "
        "Selection produces resting limit targets, never market orders, and nothing here "
        f"is ranked on recent performance. {catalyst_note}"
    )
