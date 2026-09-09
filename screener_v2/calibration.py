"""
Layer 6 — the emission log and the review query it exists to serve.

Every emitted candidate is written at emission time with the full state
that produced it, `extension_atr` above all. A backfill then attaches
forward returns, and `review()` buckets by extension and reports mean and
median forward return per bucket.

That table is the point of this module. The veto threshold of 1.5 ATR is a
starting guess; the bucket table is the only thing that should ever move
it. Do not retune it from intuition, and do not retune the ranking weights
against last month's winners either — both are explicitly listed as
anti-patterns (spec §14).

STORAGE lives in log_store.py: Postgres whenever DATABASE_URL is
configured, JSONL at config.LOG_PATH otherwise. That matters here only
because Render wipes the filesystem on every deploy, so a file-backed log
cannot survive the 30-session parallel run this table exists to serve.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import log_store
from .schemas import CalibrationReport, Candidate, ExtensionBucket

logger = logging.getLogger("stock-analyzer")

# Storage is log_store's job; these are re-exported so callers and tests
# have one import for the whole log.
active_sink = log_store.active_sink
read_rows = log_store.read_rows
append_rows = log_store.append

# The review table's rows. These stop at the veto threshold on purpose:
# anything at or above it was never orderable, so it has no fill to bucket.
BUCKETS = (
    ("<0", None, 0.0),
    ("0-0.5", 0.0, 0.5),
    ("0.5-1.0", 0.5, 1.0),
    ("1.0-1.5", 1.0, 1.5),
)


def row_id(symbol: str, generated_at: str) -> str:
    return f"{(symbol or '').upper()}@{generated_at}"


def _flatten(candidate: Candidate) -> Dict[str, Any]:
    """One log row. Field list follows spec §12 exactly."""
    compression_state, strength_state = candidate.compression, candidate.strength
    catalysts, entry_plan = candidate.catalysts, candidate.entry
    return {
        "kind": "emission",
        "id": row_id(candidate.listing_used, candidate.generated_at),
        "symbol": candidate.symbol,
        "listing_used": candidate.listing_used,
        "timestamp": candidate.generated_at,
        "watchlist_only": candidate.watchlist_only,

        "bbw_pct_rank": compression_state.bbw_pct_rank,
        "atr_pct": compression_state.atr_pct,
        "adx_14": compression_state.adx_14,
        "adx_slope_5": compression_state.adx_slope_5,
        "extension_atr": compression_state.extension_atr,

        "rs_5": strength_state.rs_5,
        "rs_20": strength_state.rs_20,
        "rs_60": strength_state.rs_60,
        "rs_120": strength_state.rs_120,
        "accumulating": strength_state.accumulating,

        "catalysts_available": catalysts.available,
        "next_earnings_date": catalysts.next_earnings_date,
        "earnings_is_estimate": catalysts.earnings_is_estimate,
        "days_to_earnings": catalysts.days_to_earnings,
        "earnings_blackout": catalysts.earnings_blackout,
        "insider_cluster_buy": catalysts.insider_cluster_buy,
        "insider_distinct_buyers_90d": catalysts.insider_distinct_buyers_90d,
        "insider_net_shares_90d": catalysts.insider_net_shares_90d,
        "iv_percentile": catalysts.iv_percentile,
        "vol_oi_ratio": catalysts.vol_oi_ratio,

        "entry_target": entry_plan.entry_target,
        "stop": entry_plan.stop,
        "r_pct": entry_plan.r_pct,
        "score": candidate.score,

        # Stored for analysis only — never an input (spec §2.1).
        "perf_1d": candidate.perf_1d,
    }


def log_candidates(candidates: List[Candidate]) -> int:
    """Append every emitted candidate. Watchlist rows are logged too: what
    the veto excluded is exactly the comparison the bucket table needs."""
    return append_rows([_flatten(c) for c in candidates])


def record_fill(
    symbol: str,
    generated_at: str,
    actual_fill_price: float,
    extension_atr_at_fill: Optional[float] = None,
) -> bool:
    """Attach a fill to an earlier emission.

    Append-only: this writes a new row rather than rewriting the emission,
    and review() merges the two by id."""
    return append_rows([{
        "kind": "fill",
        "id": row_id(symbol, generated_at),
        "symbol": (symbol or "").upper(),
        "filled_at": datetime.now(timezone.utc).isoformat(),
        "actual_fill_price": actual_fill_price,
        "extension_atr_at_fill": extension_atr_at_fill,
    }]) > 0


def _merged() -> Dict[str, Dict[str, Any]]:
    """Emissions keyed by id, with fill and forward-return rows folded in."""
    merged: Dict[str, Dict[str, Any]] = {}
    for row in read_rows():
        row_key = row.get("id")
        if not row_key:
            continue
        kind = row.get("kind")
        if kind == "emission":
            merged.setdefault(row_key, {}).update(row)
        elif kind in ("fill", "forward"):
            merged.setdefault(row_key, {}).update(
                {k: v for k, v in row.items() if k not in ("kind", "id")}
            )
    return merged


def backfill_forward_returns(
    forward_return: Callable[[str, str, int], Optional[float]],
    horizons: tuple = (5, 10, 20),
) -> int:
    """Attach forward returns to emissions that don't have them yet.

    `forward_return(symbol, emission_timestamp, horizon_sessions)` returns
    the percentage move over that horizon, or None if it hasn't happened
    yet. Rows are only written when at least one horizon resolves, so an
    unresolved row is retried on the next run rather than being marked
    done with nulls."""
    written = 0
    new_rows: List[Dict[str, Any]] = []
    for row_key, row in _merged().items():
        symbol = row.get("listing_used") or row.get("symbol")
        timestamp = row.get("timestamp")
        if not symbol or not timestamp:
            continue
        payload: Dict[str, Any] = {}
        for horizon in horizons:
            field = f"fwd_return_{horizon}d"
            if row.get(field) is not None:
                continue
            try:
                value = forward_return(symbol, timestamp, horizon)
            except Exception as e:
                logger.info(f"screener_v2: forward return lookup failed for {symbol}: {e}")
                value = None
            if value is not None:
                payload[field] = value
        if payload:
            payload.update({"kind": "forward", "id": row_key})
            new_rows.append(payload)
            written += 1
    append_rows(new_rows)
    return written


def _bucket_for(extension: Optional[float]) -> Optional[str]:
    if extension is None:
        return None
    for name, low, high in BUCKETS:
        if low is None:
            if extension < high:
                return name
        elif low <= extension < high:
            return name
    return None  # at or above the veto threshold — outside the table by design


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None}
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"mean": round(sum(ordered) / n, 4), "median": round(median, 4)}


def review(prefer_fills: bool = True) -> CalibrationReport:
    """The extension bucket table (spec §12).

    Buckets by `extension_atr_at_fill` where a fill was recorded, falling
    back to the emission-time `extension_atr` otherwise — during a parallel
    run nothing is filled yet, and an empty table would tell you nothing
    about whether 1.5 is the right threshold."""
    merged = _merged()
    grouped: Dict[str, Dict[str, List[float]]] = {name: {} for name, _, _ in BUCKETS}
    with_returns = 0

    for row in merged.values():
        extension = row.get("extension_atr_at_fill") if prefer_fills else None
        if extension is None:
            extension = row.get("extension_atr")
        bucket = _bucket_for(extension)
        if bucket is None:
            continue
        horizons_present = False
        for horizon in (5, 10, 20):
            value = row.get(f"fwd_return_{horizon}d")
            if value is None:
                continue
            grouped[bucket].setdefault(str(horizon), []).append(float(value))
            horizons_present = True
        if horizons_present:
            with_returns += 1
        else:
            grouped[bucket].setdefault("_n", [])
            grouped[bucket]["_n"].append(0.0)

    buckets: List[ExtensionBucket] = []
    for name, _, _ in BUCKETS:
        data = grouped.get(name, {})
        counted = max(
            len(data.get("5", [])), len(data.get("10", [])),
            len(data.get("20", [])), len(data.get("_n", [])),
        )
        five, ten, twenty = _stats(data.get("5", [])), _stats(data.get("10", [])), _stats(data.get("20", []))
        buckets.append(ExtensionBucket(
            bucket=name, n=counted,
            mean_fwd_5d=five["mean"], median_fwd_5d=five["median"],
            mean_fwd_10d=ten["mean"], median_fwd_10d=ten["median"],
            mean_fwd_20d=twenty["mean"], median_fwd_20d=twenty["median"],
        ))

    return CalibrationReport(
        buckets=buckets,
        logged_rows=len(merged),
        rows_with_forward_returns=with_returns,
        sink=active_sink(),
        note=(
            "Mean and median forward return by extension-at-entry. This is the "
            "measurement that sets the veto threshold — move it on this table, not "
            "on intuition. Hit rate is deliberately not reported: a lower hit rate "
            "than the momentum screen is expected and is not a failure."
        ),
    )
