"""
Pydantic models for the v2 screener.

Definition order matters and is enforced by tools/check_class_order.py:
every model referenced as a field type is defined above the model that
references it. A NameError here surfaces at import time, which on this
deployment means a dead worker rather than a handled error.

Note on `perf_1d` (and anything like it): recent return is STORED on the
output for later analysis and never read by a gate or by the ranking
function (spec §2.1). If you are about to reference it in compression.py,
strength.py, entry.py or the score, stop.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class UniverseMember(BaseModel):
    """One symbol that cleared layer 1."""
    symbol: str
    listing_used: str            # the ticker actually analyzed (see interlisted resolution)
    price: Optional[float] = None
    avg_dollar_volume: Optional[float] = None
    market_cap: Optional[float] = None
    market_cap_known: bool = True
    sessions: int = 0


class CompressionState(BaseModel):
    """Layer 2 output. `ok` False means one of the gates rejected it;
    `veto` True specifically means the extension veto fired, which is
    absolute and makes the name watchlist-only rather than merely
    rejected."""
    ok: bool = False
    veto: bool = False
    reason: str = ""

    close: Optional[float] = None
    bbw: Optional[float] = None
    bbw_pct_rank: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    atr_median_60: Optional[float] = None
    adx_14: Optional[float] = None
    adx_slope_5: Optional[float] = None
    sma20: Optional[float] = None
    sma200: Optional[float] = None
    extension_atr: Optional[float] = None
    base_high: Optional[float] = None
    base_low: Optional[float] = None
    base_width_pct: Optional[float] = None


class StrengthState(BaseModel):
    """Layer 3 output. The gate is the RS inflection; accumulation is a
    scored bonus, never a gate."""
    ok: bool = False
    reason: str = ""

    benchmark: Optional[str] = None
    rs_5: Optional[float] = None
    rs_20: Optional[float] = None
    rs_60: Optional[float] = None
    rs_120: Optional[float] = None
    ad_slope_20: Optional[float] = None
    price_flat: bool = False
    accumulating: bool = False


class Catalysts(BaseModel):
    """Layer 4 output, read from the nightly cache only.

    `available` False is the honest cold-cache state: the scan still emits
    the candidate, it just carries no catalyst contribution. It is never a
    reason to fetch inline (spec §2.5)."""
    available: bool = False
    as_of: Optional[str] = None

    next_earnings_date: Optional[str] = None
    earnings_is_estimate: Optional[bool] = None
    days_to_earnings: Optional[int] = None
    earnings_blackout: bool = False

    insider_cluster_buy: bool = False
    insider_distinct_buyers_90d: int = 0
    insider_net_shares_90d: Optional[float] = None

    iv_percentile: Optional[float] = None
    vol_oi_ratio: Optional[float] = None


class OrderDraft(BaseModel):
    """A LIMIT draft only.

    The IBKR connector supports MARKET and LIMIT and produces drafts, never
    live orders. Bracket/OCO/stop structures must be attached natively in
    IBKR — emitting an unlinked SELL LIMIT plus SELL STOP against the same
    shares triggers a margin rejection, so the protective stop travels as a
    labelled field for manual attachment rather than as an order."""
    action: str = "BUY"
    order_type: str = "LIMIT"
    symbol: str
    quantity: int
    limit_price: float
    tif: str = "GTC"
    stop_level_for_manual_attachment: float
    note: str = (
        "Draft only. Attach the stop natively in IBKR as a bracket — do not "
        "submit it as a separate unlinked order against these shares."
    )


class EntryPlan(BaseModel):
    """Layer 5 output. Selection ends with a resting target, never a market
    fill on the selection bar (spec §2.3)."""
    ok: bool = False
    reason: str = ""

    entry_target: Optional[float] = None
    stop: Optional[float] = None
    risk_per_share: Optional[float] = None
    r_pct: Optional[float] = None
    shares: Optional[int] = None
    already_broke_out: bool = False
    order: Optional[OrderDraft] = None


class Candidate(BaseModel):
    """One emitted candidate. `watchlist_only` names carry no order draft:
    either the extension veto fired or they are inside the earnings
    blackout window."""
    symbol: str
    listing_used: str
    generated_at: str
    score: Optional[float] = None
    watchlist_only: bool = False
    watchlist_reason: str = ""

    compression: CompressionState
    strength: StrengthState
    catalysts: Catalysts
    entry: EntryPlan

    # Stored for calibration analysis only — never an input to any gate or
    # to the ranking function. See the module docstring.
    perf_1d: Optional[float] = None


class ScanResult(BaseModel):
    candidates: List[Candidate] = []
    watchlist_only: List[Candidate] = []

    universe_size: int = 0
    scanned: int = 0
    bars_missing: int = 0
    failed_compression: int = 0
    failed_strength: int = 0
    failed_entry: int = 0
    vetoed_extension: int = 0

    catalysts_available: bool = False
    generated_at: str = ""
    note: str = ""


class ExtensionBucket(BaseModel):
    """One row of calibration.py's review table — the direct measurement of
    whether the veto threshold is set correctly (spec §12)."""
    bucket: str
    n: int = 0
    mean_fwd_5d: Optional[float] = None
    median_fwd_5d: Optional[float] = None
    mean_fwd_10d: Optional[float] = None
    median_fwd_10d: Optional[float] = None
    mean_fwd_20d: Optional[float] = None
    median_fwd_20d: Optional[float] = None


class CalibrationReport(BaseModel):
    buckets: List[ExtensionBucket] = []
    logged_rows: int = 0
    rows_with_forward_returns: int = 0
    sink: str = ""
    note: str = ""
