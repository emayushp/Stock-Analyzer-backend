"""
Layer 1 — universe. Recomputed weekly, cached, read by the scan.

Liquidity, size, price and history floors, plus the excluded-symbols list.
Nothing here looks at returns.

Interlisted resolution (spec §5): a Canadian name can trade as both SHOP
and SHOP.TO, and the screener universe does not reliably hand back the TSX
form. Both listings are measured and the one with the higher dollar volume
becomes the analysis symbol, recorded on the output as `listing_used` so
every downstream number can be traced to the listing it came from.

That expansion costs one extra batched request per hundred symbols, once a
week. It is deliberately NOT done on the daily scan path.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import config, indicators as ind
from .schemas import UniverseMember

logger = logging.getLogger("stock-analyzer")

_CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")

# (built_at, members)
_UNIVERSE_CACHE: Tuple[float, Optional[List[UniverseMember]]] = (0.0, None)


def alternate_listings(symbol: str) -> List[str]:
    """Every listing worth measuring for one requested symbol.

    A suffixed ticker gets its bare form tried alongside it, and a bare
    ticker gets the TSX form tried — the universe list can hand back either
    for an interlisted name."""
    upper = (symbol or "").strip().upper()
    if not upper:
        return []
    for suffix in _CA_SUFFIXES:
        if upper.endswith(suffix):
            return [upper, upper[: -len(suffix)]]
    return [upper, f"{upper}.TO"]


def avg_dollar_volume(bars: Optional[pd.DataFrame], sessions: int = 20) -> Optional[float]:
    """Mean close x volume over the trailing window, in the listing's own
    currency. Currencies are deliberately not converted: the floor is a
    liquidity sanity check, not a cross-market comparison."""
    try:
        if not ind.has_ohlcv(bars):
            return None
        window = bars.tail(sessions)
        if len(window) < sessions:
            return None
        dollar = (window["Close"] * window["Volume"]).dropna()
        if dollar.empty:
            return None
        return round(float(dollar.mean()), 2)
    except Exception:
        return None


def _measure(symbol: str, bars: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    if not ind.has_ohlcv(bars):
        return None
    price = ind.latest(bars["Close"])
    adv = avg_dollar_volume(bars)
    if price is None or adv is None:
        return None
    return {"price": price, "adv": adv, "sessions": float(len(bars))}


def build(
    provider,
    symbols: List[str],
    market_cap_lookup: Optional[Callable[[List[str]], Dict[str, Optional[float]]]] = None,
) -> List[UniverseMember]:
    """Apply the layer 1 gates and return the surviving members.

    `provider` must already hold bars for the expanded listing set — call
    `prefetch_symbols()` to get that list. `market_cap_lookup` is injected
    rather than imported so this module never reaches the network itself;
    when it isn't supplied the size gate is skipped and each member is
    flagged `market_cap_known=False` rather than silently treated as if it
    had passed a check that never ran."""
    requested = [s.strip().upper() for s in symbols if s and s.strip()]
    resolved: Dict[str, Dict[str, float]] = {}

    for symbol in requested:
        if symbol in config.EXCLUDED_SYMBOLS:
            continue
        best_listing, best_stats = None, None
        for listing in alternate_listings(symbol):
            if listing in config.EXCLUDED_SYMBOLS:
                continue
            stats = _measure(listing, provider.daily_bars(listing, 0))
            if stats is None:
                continue
            if best_stats is None or stats["adv"] > best_stats["adv"]:
                best_listing, best_stats = listing, stats
        if best_listing is None or best_stats is None:
            continue
        # Two requested symbols can resolve onto the same listing (SHOP and
        # SHOP.TO both landing on SHOP); keep the more liquid measurement.
        existing = resolved.get(best_listing)
        if existing is None or best_stats["adv"] > existing["adv"]:
            resolved[best_listing] = best_stats

    caps: Dict[str, Optional[float]] = {}
    if market_cap_lookup is not None:
        try:
            caps = market_cap_lookup(sorted(resolved.keys())) or {}
        except Exception as e:
            logger.error(f"screener_v2: market cap lookup failed, size gate skipped: {e}")
            caps = {}

    members: List[UniverseMember] = []
    for listing, stats in sorted(resolved.items()):
        if stats["price"] < config.MIN_PRICE:
            continue
        if stats["adv"] < config.MIN_AVG_DOLLAR_VOLUME:
            continue
        if stats["sessions"] < config.MIN_SESSIONS:
            continue
        market_cap = caps.get(listing)
        if market_cap is not None and market_cap < config.MIN_MARKET_CAP:
            continue
        members.append(UniverseMember(
            symbol=listing,
            listing_used=listing,
            price=stats["price"],
            avg_dollar_volume=stats["adv"],
            market_cap=market_cap,
            market_cap_known=market_cap is not None,
            sessions=int(stats["sessions"]),
        ))
    return members


def prefetch_symbols(symbols: List[str]) -> List[str]:
    """Every listing the universe build needs bars for."""
    out: List[str] = []
    for symbol in symbols:
        for listing in alternate_listings(symbol):
            if listing not in out:
                out.append(listing)
    return out


def get_universe(
    provider,
    symbols: List[str],
    market_cap_lookup: Optional[Callable[[List[str]], Dict[str, Optional[float]]]] = None,
    force: bool = False,
) -> List[UniverseMember]:
    """Cached weekly universe. `force=True` rebuilds it now."""
    global _UNIVERSE_CACHE
    built_at, cached = _UNIVERSE_CACHE
    if not force and cached is not None and time.time() - built_at < config.UNIVERSE_TTL_SECONDS:
        return cached

    members = build(provider, symbols, market_cap_lookup)
    if members:
        _UNIVERSE_CACHE = (time.time(), members)
    elif cached is not None:
        # A build that returns nothing is far more likely to be a data
        # outage than a market where no symbol is liquid enough. Keep
        # serving the last good universe rather than emptying the screen.
        logger.error("screener_v2: universe build returned nothing — keeping the previous one")
        return cached
    return members


def cached_universe() -> Optional[List[UniverseMember]]:
    return _UNIVERSE_CACHE[1]


def clear_cache() -> None:
    global _UNIVERSE_CACHE
    _UNIVERSE_CACHE = (0.0, None)
