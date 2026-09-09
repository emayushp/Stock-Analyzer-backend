"""
The wiring seam between the app and the v2 screener.

Everything the screener needs from the outside world — the symbol list, a
market-cap lookup, the Stocklake fetchers — arrives through `configure()`
rather than through an import. That keeps the whole package free of any
dependency on the app module (which will import this one), and it means
every scheduled job can be exercised in a test with plain callables.

Job cadence, all driven by the app's existing background thread:
    refresh_universe()   weekly   layer 1, includes interlisted resolution
    refresh_catalysts()  nightly  layer 4 cache; the scan never fetches
    run_scan()           hourly   cached, so a request never triggers a
                                  full universe download
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import catalysts, config, pipeline, universe
from .provider import YFinanceProvider
from .schemas import ScanResult

logger = logging.getLogger("stock-analyzer")

_lock = threading.Lock()

_symbols: List[str] = []
_market_cap_lookup: Optional[Callable[[List[str]], Dict[str, Optional[float]]]] = None
_insider_fetcher: Optional[Callable[[str], Any]] = None
_stock_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None
_options_fetcher: Optional[Callable[[str], Dict[str, Any]]] = None

# (generated_at_epoch, result)
_SCAN_CACHE: Tuple[float, Optional[ScanResult]] = (0.0, None)
_last_catalyst_refresh = 0.0


def configure(
    symbols: List[str],
    market_cap_lookup: Optional[Callable[[List[str]], Dict[str, Optional[float]]]] = None,
    insider_fetcher: Optional[Callable[[str], Any]] = None,
    stock_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    options_fetcher: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> None:
    global _symbols, _market_cap_lookup, _insider_fetcher, _stock_fetcher, _options_fetcher
    _symbols = list(symbols or [])
    _market_cap_lookup = market_cap_lookup
    _insider_fetcher = insider_fetcher
    _stock_fetcher = stock_fetcher
    _options_fetcher = options_fetcher
    logger.info(f"screener_v2: configured with {len(_symbols)} universe symbols")


def is_configured() -> bool:
    return bool(_symbols)


def _load_provider(for_universe: bool) -> YFinanceProvider:
    """Bars for everything a scan (or a universe rebuild) needs.

    The universe rebuild additionally pulls the alternate listing of every
    symbol so interlisted names can be resolved by dollar volume; the daily
    scan does not pay that cost."""
    provider = YFinanceProvider()
    wanted = universe.prefetch_symbols(_symbols) if for_universe else list(_symbols)
    cached = universe.cached_universe()
    if not for_universe and cached:
        wanted = [m.listing_used for m in cached]
    wanted = list(dict.fromkeys(wanted + [config.BENCHMARK_US, config.BENCHMARK_CA]))
    loaded = provider.prefetch(wanted)
    logger.info(f"screener_v2: loaded bars for {loaded}/{len(wanted)} symbols")
    return provider


def refresh_universe() -> int:
    """Weekly job. Returns the universe size."""
    if not is_configured():
        return 0
    provider = _load_provider(for_universe=True)
    members = universe.get_universe(provider, _symbols, _market_cap_lookup, force=True)
    logger.info(f"screener_v2: universe rebuilt, {len(members)} members")
    return len(members)


def refresh_catalysts(limit: Optional[int] = None) -> int:
    """Nightly job. Populates the catalyst cache for the names most likely
    to matter tomorrow.

    Deliberately scoped to the current candidate + watchlist set rather
    than the whole universe: the per-symbol insider call has no batch form,
    and a compressed base persists for weeks, so yesterday's survivors
    overwhelmingly cover today's. Anything not covered simply reads as
    `catalysts: unavailable`, which is a designed state, not a failure."""
    global _last_catalyst_refresh
    if not is_configured():
        return 0

    cached_scan = _SCAN_CACHE[1]
    if cached_scan is not None:
        symbols = [c.listing_used for c in cached_scan.candidates + cached_scan.watchlist_only]
    else:
        members = universe.cached_universe() or []
        symbols = [m.listing_used for m in members]
    if limit:
        symbols = symbols[:limit]
    if not symbols:
        return 0

    count = catalysts.refresh_cache(
        symbols,
        insider_fetcher=_insider_fetcher,
        stock_fetcher=_stock_fetcher,
        options_fetcher=_options_fetcher,
    )
    _last_catalyst_refresh = time.time()
    logger.info(f"screener_v2: catalyst cache refreshed for {count} symbols")
    return count


def run_scan(force: bool = False) -> ScanResult:
    """A cached scan. `force=True` re-downloads and re-scans now."""
    global _SCAN_CACHE
    generated_at, cached = _SCAN_CACHE
    if not force and cached is not None and time.time() - generated_at < config.SCAN_TTL_SECONDS:
        return cached

    with _lock:
        generated_at, cached = _SCAN_CACHE
        if not force and cached is not None and time.time() - generated_at < config.SCAN_TTL_SECONDS:
            return cached

        if not is_configured():
            return ScanResult(note="screener_v2 is not configured with a universe.")

        provider = _load_provider(for_universe=universe.cached_universe() is None)
        result = pipeline.run(provider, _symbols, _market_cap_lookup)
        _SCAN_CACHE = (time.time(), result)
        return result


def cached_scan() -> Optional[ScanResult]:
    return _SCAN_CACHE[1]


def status() -> Dict[str, Any]:
    generated_at, cached = _SCAN_CACHE
    return {
        "enabled": config.screener_v2_enabled(),
        "configured": is_configured(),
        "universe_symbols_requested": len(_symbols),
        "universe_members": len(universe.cached_universe() or []),
        "scan_cached_at": generated_at or None,
        "scan_candidates": len(cached.candidates) if cached else 0,
        "scan_watchlist": len(cached.watchlist_only) if cached else 0,
        "last_catalyst_refresh": _last_catalyst_refresh or None,
        "catalysts": catalysts.cache_status(),
    }


def reset_for_tests() -> None:
    global _SCAN_CACHE, _last_catalyst_refresh
    _SCAN_CACHE = (0.0, None)
    _last_catalyst_refresh = 0.0
    universe.clear_cache()
    catalysts.clear_cache()
