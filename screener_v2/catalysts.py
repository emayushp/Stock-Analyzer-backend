"""
Layer 4 — catalysts. CACHE READS ONLY at scan time.

This module imports nothing that can reach the network, and that is on
purpose: it is the structural guarantee behind invariant §2.5. A scan that
hits a cold cache emits `catalysts: unavailable` and carries on. It does
not fetch, it does not block, and it does not fail — a previous production
crash came from exactly that inline-fetch pattern.

Population is a separate, scheduled call (`refresh_cache`) that takes its
fetchers as arguments. Injecting them rather than importing them keeps the
network in the caller, keeps this module unit-testable with plain dicts,
and avoids a circular import with the app module that owns the MCP client.

Catalysts are bonuses, never gates — with one exception that is a TIMING
rule rather than a score: inside the earnings blackout window a candidate
becomes watchlist-only, so the screen can never put on a position that
walks into a print unintentionally.

Coverage note: `iv_percentile` and `vol_oi_ratio` have no configured data
source on this deployment, so they stay None and contribute nothing. The
fields exist because the spec's scoring model includes them; they are not
quietly faked from something else.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .schemas import Catalysts

logger = logging.getLogger("stock-analyzer")

# symbol -> (stored_at_epoch, payload dict)
_CATALYST_CACHE: Dict[str, Tuple[float, dict]] = {}

_BUY_MARKERS = ("BUY", "PURCHASE", "P-", "ACQUIS")
_SELL_MARKERS = ("SELL", "SALE", "S-", "DISPOS")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def _parse_date(raw: Optional[str]):
    """ISO timestamp -> date, or None. Naive timestamps are read as UTC so
    a non-UTC offset can't shift the calendar day around midnight."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).date()
    except Exception:
        return None


def classify_transaction(raw_type: Optional[str]) -> Optional[str]:
    """"buy", "sell", or None for anything this can't place.

    The upstream feed merges SEC Form 4 with BaFin/AFM/CNMV filings and a
    Yahoo gap-fill, so the wording varies by source ("SELL", "S-Sale",
    "P-Purchase"). Matching on markers rather than an exact vocabulary is
    what keeps a new source from silently reading as zero activity — and
    anything still unrecognised is dropped rather than guessed, so it can
    never flip the sign of the net."""
    if not raw_type:
        return None
    text = str(raw_type).strip().upper()
    if any(marker in text for marker in _SELL_MARKERS):
        return "sell"
    if any(marker in text for marker in _BUY_MARKERS):
        return "buy"
    return None


def summarize_insiders(payload: Any, today=None) -> Dict[str, Any]:
    """Cluster-buy flag, distinct buyer count and net shares over the
    trailing window, from a get_insider_activity payload."""
    result = {"insider_cluster_buy": False, "insider_distinct_buyers_90d": 0, "insider_net_shares_90d": None}
    try:
        if not isinstance(payload, dict) or "error" in payload:
            return result
        rows = payload.get("transactions")
        if not isinstance(rows, list):
            return result

        today = today or datetime.now(timezone.utc).date()
        buyers, net, seen_any = set(), 0.0, False
        for row in rows:
            if not isinstance(row, dict):
                continue
            traded_on = _parse_date(row.get("date"))
            if traded_on is None:
                continue
            if (today - traded_on).days > config.INSIDER_CLUSTER_LOOKBACK_DAYS:
                continue
            side = classify_transaction(row.get("type"))
            if side is None:
                continue
            try:
                shares = float(row.get("shares") or 0)
            except (TypeError, ValueError):
                continue
            seen_any = True
            if side == "buy":
                net += shares
                name = str(row.get("name") or "").strip().upper()
                if name:
                    buyers.add(name)
            else:
                net -= shares

        result["insider_distinct_buyers_90d"] = len(buyers)
        result["insider_cluster_buy"] = len(buyers) >= config.INSIDER_CLUSTER_MIN_BUYERS
        result["insider_net_shares_90d"] = round(net, 2) if seen_any else None
        return result
    except Exception as e:
        logger.info(f"screener_v2: could not summarize insider payload: {e}")
        return result


def summarize_earnings(payload: Any, today=None) -> Dict[str, Any]:
    """Next earnings date, whether it's an estimate, and days until it.

    A missing date stays None and is never rendered as "not scheduled" —
    the upstream calendar returns empty far more often than it returns a
    genuine "no event", and the two must not be conflated."""
    result = {"next_earnings_date": None, "earnings_is_estimate": None, "days_to_earnings": None}
    try:
        if not isinstance(payload, dict) or "error" in payload:
            return result
        earnings_on = _parse_date(payload.get("earnings_date"))
        if earnings_on is None:
            return result
        today = today or datetime.now(timezone.utc).date()
        result["next_earnings_date"] = earnings_on.strftime("%Y-%m-%d")
        result["earnings_is_estimate"] = bool(payload.get("earnings_is_estimate"))
        result["days_to_earnings"] = (earnings_on - today).days
        return result
    except Exception as e:
        logger.info(f"screener_v2: could not summarize earnings payload: {e}")
        return result


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def read(symbol: str) -> Catalysts:
    """Read one symbol's catalysts from cache. NEVER fetches.

    A missing or stale entry is an ordinary outcome, reported as
    `available=False`, not an error and not a reason to go to the network."""
    try:
        entry = _CATALYST_CACHE.get((symbol or "").strip().upper())
        if entry is None:
            return Catalysts(available=False)
        stored_at, payload = entry
        if time.time() - stored_at > config.CATALYST_CACHE_TTL_SECONDS:
            return Catalysts(available=False)

        days = payload.get("days_to_earnings")
        blackout = days is not None and 0 <= days <= config.EARNINGS_BLACKOUT_DAYS
        return Catalysts(
            available=True,
            as_of=datetime.fromtimestamp(stored_at, timezone.utc).isoformat(),
            next_earnings_date=payload.get("next_earnings_date"),
            earnings_is_estimate=payload.get("earnings_is_estimate"),
            days_to_earnings=days,
            earnings_blackout=blackout,
            insider_cluster_buy=bool(payload.get("insider_cluster_buy")),
            insider_distinct_buyers_90d=int(payload.get("insider_distinct_buyers_90d") or 0),
            insider_net_shares_90d=payload.get("insider_net_shares_90d"),
            iv_percentile=payload.get("iv_percentile"),
            vol_oi_ratio=payload.get("vol_oi_ratio"),
        )
    except Exception as e:
        logger.info(f"screener_v2: catalyst cache read failed for {symbol}: {e}")
        return Catalysts(available=False)


def store(symbol: str, payload: dict) -> None:
    _CATALYST_CACHE[(symbol or "").strip().upper()] = (time.time(), payload)


def refresh_cache(
    symbols: List[str],
    insider_fetcher: Optional[Callable[[str], Any]] = None,
    stock_fetcher: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    options_fetcher: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> int:
    """The scheduled population job. Returns how many symbols were cached.

    Every fetcher is optional and injected: whatever isn't supplied simply
    leaves its fields empty. Each symbol is stored independently, so one
    upstream failure costs that symbol's catalysts and nothing else."""
    wanted = sorted({(s or "").strip().upper() for s in symbols if s and s.strip()})
    if not wanted:
        return 0

    stock_payloads: Dict[str, Any] = {}
    if stock_fetcher is not None:
        try:
            stock_payloads = stock_fetcher(wanted) or {}
        except Exception as e:
            logger.error(f"screener_v2: catalyst stock fetch failed: {e}")

    cached = 0
    for symbol in wanted:
        payload: Dict[str, Any] = {}
        try:
            payload.update(summarize_earnings(stock_payloads.get(symbol)))

            if insider_fetcher is not None:
                try:
                    payload.update(summarize_insiders(insider_fetcher(symbol)))
                except Exception as e:
                    logger.info(f"screener_v2: insider fetch failed for {symbol}: {e}")

            if options_fetcher is not None:
                try:
                    options = options_fetcher(symbol) or {}
                    payload["iv_percentile"] = options.get("iv_percentile")
                    payload["vol_oi_ratio"] = options.get("vol_oi_ratio")
                except Exception as e:
                    logger.info(f"screener_v2: options fetch failed for {symbol}: {e}")

            store(symbol, payload)
            cached += 1
        except Exception as e:
            logger.info(f"screener_v2: catalyst refresh failed for {symbol}: {e}")
    return cached


def cache_status() -> Dict[str, Any]:
    now = time.time()
    fresh = [s for s, (at, _) in _CATALYST_CACHE.items() if now - at <= config.CATALYST_CACHE_TTL_SECONDS]
    return {
        "cached_symbols": len(_CATALYST_CACHE),
        "fresh_symbols": len(fresh),
        "ttl_seconds": config.CATALYST_CACHE_TTL_SECONDS,
    }


def clear_cache() -> None:
    _CATALYST_CACHE.clear()
