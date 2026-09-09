"""
HTTP surface for the v2 screener. Deliberately thin: everything here is
argument handling and flag checks, so no v2 logic ends up in the app's
large screener file (spec §14) and none of the layers depend on FastAPI.

The feature flag is checked per request rather than at import, so both
screeners can run in parallel during validation and the flag can be flipped
with a restart instead of a code change.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from . import calibration, config, runtime
from .schemas import CalibrationReport, ScanResult

logger = logging.getLogger("stock-analyzer")

router = APIRouter(prefix="/api/v2/screener", tags=["screener-v2"])


def _require_flag() -> None:
    if not config.screener_v2_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Screener v2 is off. Set SCREENER_V2_ENABLED=true to run it "
                "alongside the existing screener."
            ),
        )


@router.get("/status")
def status():
    """Flag, cache and job state. Intentionally readable with the flag off
    — checking whether v2 is enabled shouldn't require enabling it."""
    return runtime.status()


@router.get("", response_model=ScanResult)
def scan(force: bool = Query(False, description="Run a scan now instead of serving the cache")):
    """Cache-only by default.

    A full scan downloads bars for the whole universe and takes minutes, so
    it must never happen inside a request — that is the same rule the v1
    screener already follows after a live-fetch-on-read path took the app
    down. The background job fills the cache within a minute of startup and
    refreshes it hourly; until then this reports that plainly rather than
    making the caller wait for a scan it didn't ask for.

    `force=true` is the deliberate escape hatch and will block for minutes."""
    _require_flag()
    if not runtime.is_configured():
        raise HTTPException(status_code=503, detail="Screener v2 has no universe configured yet.")
    try:
        if force:
            return runtime.run_scan(force=True)
        cached = runtime.cached_scan()
        if cached is not None:
            return cached
        return ScanResult(
            generated_at=datetime.now(timezone.utc).isoformat(),
            note=(
                "The first scan hasn't run yet. The background job builds it within a "
                "minute of startup and refreshes it hourly — check back shortly."
            ),
        )
    except Exception as e:
        logger.error(f"screener_v2: scan failed: {e}")
        raise HTTPException(status_code=502, detail="The v2 scan couldn't complete right now.")


@router.get("/calibration", response_model=CalibrationReport)
def calibration_report(
    prefer_fills: bool = Query(True, description="Bucket by extension at fill where a fill was recorded"),
):
    """The extension bucket table. Readable with the flag off: the whole
    point of the parallel run is to look at this before flipping it."""
    try:
        return calibration.review(prefer_fills=prefer_fills)
    except Exception as e:
        logger.error(f"screener_v2: calibration review failed: {e}")
        raise HTTPException(status_code=502, detail="Couldn't build the calibration report.")


@router.post("/fill")
def record_fill(
    symbol: str = Query(..., description="Symbol as emitted (listing_used)"),
    generated_at: str = Query(..., description="The candidate's emission timestamp"),
    actual_fill_price: float = Query(...),
    extension_atr_at_fill: float = Query(None),
):
    """Attach a fill to an emitted candidate so the bucket table can be
    keyed on extension at entry rather than extension at emission."""
    _require_flag()
    ok = calibration.record_fill(symbol, generated_at, actual_fill_price, extension_atr_at_fill)
    if not ok:
        raise HTTPException(status_code=502, detail="Couldn't write the fill to the log.")
    return {"recorded": True, "symbol": symbol.upper(), "generated_at": generated_at}


@router.post("/jobs/universe")
def run_universe_job():
    """Manual trigger for the weekly universe rebuild."""
    _require_flag()
    return {"universe_members": runtime.refresh_universe()}


@router.post("/jobs/catalysts")
def run_catalyst_job(limit: int = Query(None, ge=1, le=200)):
    """Manual trigger for the nightly catalyst cache population."""
    _require_flag()
    return {"symbols_cached": runtime.refresh_catalysts(limit=limit)}
