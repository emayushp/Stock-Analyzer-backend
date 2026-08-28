"""
Stock Market Analysis API
Technical analysis (RSI + MACD + Volume), volatility-based price targets,
company fundamentals, and AI news sentiment (via Stocklake) for US & Canadian equities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

import auth as auth_lib
import db as db_lib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-analyzer")


def _sanitize_non_finite(obj):
    """Replace NaN/Infinity with None, recursively. A handful of yfinance-
    derived calculations (RSI on a flat/degenerate price series, ratios with
    a near-zero denominator, etc.) can legitimately produce NaN or inf, and
    Starlette's default JSONResponse uses allow_nan=False — a single such
    value anywhere in a response raises ValueError and turns into a raw 500
    for the whole request, not just a missing field. Since we already treat
    "couldn't compute this" as None everywhere else in the app, do the same
    here instead of leaving this as a crash."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


class SanitizingJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_sanitize_non_finite(content))


app = FastAPI(title="Stock Market Analysis API", version="2.0.0", default_response_class=SanitizingJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response cache. Beyond making repeat lookups instant, this materially cuts
# how often we hit Yahoo — which is what triggered the rate-limit blocking
# we ran into earlier. Keyed by ticker, 10 minute TTL.
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 600

# ---------------------------------------------------------------------------
# Screener universe. Deliberately a fixed list of liquid, well-known names
# rather than "the whole market" — scanning thousands of tickers would mean
# thousands of Yahoo requests and near-certain rate-limit blocking. These are
# fetched in ONE batched request, then scored on technicals only (no FinBERT,
# which would be far too slow across this many names).
# ---------------------------------------------------------------------------
SCREENER_UNIVERSE = [
    # US Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRM", "ORCL",
    "ADBE", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "NOW", "INTU", "AMAT",
    "MU", "INTC", "AMD", "PYPL", "UBER", "ABNB",
    "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "WDAY",
    "TEAM", "SNOW", "NET", "DDOG", "ZS", "MDB", "ADSK",
    "NXPI", "MRVL", "ON", "SWKS", "MCHP", "KEYS", "TER", "GRMN", "HPQ",
    "DELL", "NTAP", "STX", "WDC", "AKAM", "VRSN", "GDDY", "EBAY", "ETSY", "DOCU", "OKTA",
    # US Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "C",
    "SCHW", "USB", "PNC", "TFC", "COF", "AIG", "MET", "PRU", "TRV", "ALL",
    "BX", "KKR", "APO", "ICE", "CME", "NDAQ", "SPGI", "MCO", "AON",
    "AJG", "WTW", "BEN", "IVZ", "STT",
    # US Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "CVS",
    "MDT", "GILD", "AMGN", "VRTX", "REGN", "ISRG", "SYK", "BSX", "ELV",
    "CI", "HUM", "ZTS", "DHR",
    "CNC", "MOH", "HCA", "EW", "IDXX", "IQV", "A", "MTD", "WAT", "RMD",
    "ALGN", "DXCM", "BAX", "BDX", "COO",
    # US Consumer
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "BKNG", "CMG", "YUM", "DG", "DLTR", "EL", "CL", "KMB",
    "GIS",
    "SYY", "ADM", "HSY", "MKC", "CLX", "CHD", "KDP", "MNST", "STZ", "TAP",
    "PM", "MO", "ROST", "ULTA", "LULU", "DPZ", "DRI",
    "AZO", "ORLY", "BBY", "GPC", "TSCO", "YETI", "DECK", "POOL",
    # US Energy
    "XOM", "CVX", "COP", "SLB", "PSX", "MPC", "VLO", "OXY", "WMB", "KMI",
    "DVN", "FANG", "EOG", "HAL", "BKR", "TRGP", "OKE",
    # US Industrials
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX",
    "DE", "MMM", "EMR", "ETN", "ITW", "PH", "GD", "NOC", "UNP", "CSX", "NSC", "FDX",
    "WM", "RSG", "CTAS", "FAST", "PCAR", "CMI", "ROK", "DOV", "XYL", "AME",
    "IEX", "TT", "CARR", "OTIS", "JCI", "LHX", "TDG", "HWM",
    # US Communication
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "MTCH", "WBD", "PARA",
    # US Real Estate
    "AMT", "PLD", "EQIX", "PSA", "O", "SPG",
    "DLR", "WELL", "AVB", "EQR", "VTR", "ESS", "MAA", "EXR", "SBAC", "CCI",
    # US Utilities
    "NEE", "DUK", "SO", "D", "AEP",
    "EXC", "XEL", "ED", "WEC", "ES", "PEG", "FE", "AEE", "CMS", "DTE",
    # US Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG", "ALB", "VMC",
    # Canadian Banks & Financials
    "RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO", "MFC.TO", "SLF.TO", "GWO.TO",
    "IFC.TO", "POW.TO", "FFH.TO", "EQB.TO",
    # Canadian Energy
    "ENB.TO", "SU.TO", "CNQ.TO", "TRP.TO", "PPL.TO", "CVE.TO",
    "IMO.TO", "ARX.TO", "OVV.TO", "TOU.TO", "WCP.TO",
    # Canadian Materials
    "ABX.TO", "FNV.TO", "WPM.TO", "NTR.TO",
    "K.TO", "CCO.TO", "TECK-B.TO", "AEM.TO", "LUN.TO", "IVN.TO",
    # Canadian Industrials & Transport
    "CNR.TO", "CP.TO", "WCN.TO", "TFII.TO", "STN.TO", "TIH.TO", "WSP.TO",
    # Canadian Consumer
    "ATD.TO", "L.TO", "QSR.TO", "DOL.TO", "MG.TO", "SAP.TO", "BYD.TO",
    # Canadian Tech
    "SHOP.TO", "CSU.TO", "CLS.TO", "OTEX.TO", "DSG.TO", "KXS.TO", "GIB-A.TO",
    # Canadian Telecom
    "BCE.TO", "T.TO", "RCI-B.TO",
    # Canadian REITs
    "REI-UN.TO", "CAR-UN.TO", "GRT-UN.TO", "AP-UN.TO",
    # Canadian Utilities
    "FTS.TO", "EMA.TO", "AQN.TO", "CU.TO",
]

# A curated set of TSX / TSX-V names that commonly trade under CAD $20 — used
# by the Brief's "Canadian Stocks Under $20" section. Same rationale as above:
# a fixed, known list rather than scanning the whole exchange.
CANADIAN_UNDER_20_UNIVERSE = [
    "AC.TO", "BB.TO", "CGX.TO", "BTE.TO", "NPI.TO", "TOU.TO",
    "WPM.TO", "KEY.TO", "PEY.TO", "CVE.TO", "BTO.TO", "IMG.TO", "ELD.TO",
    "AGI.TO", "FM.TO", "TA.TO", "H.TO", "GIB-A.TO", "DOO.TO",
]

_SCREENER_CACHE: Dict[str, Tuple[float, Any]] = {}
_SCREENER_TTL_SECONDS = 3600  # 1 hour — screening is not a minute-to-minute activity

_UNDER20_CACHE: Dict[str, Tuple[float, Any]] = {}
_UNDER20_TTL_SECONDS = 3600

# The "most actives" list shifts through the trading day but not second to
# second — 15 minutes keeps this responsive without hammering Yahoo's
# screener endpoint on every app open.
_HIGH_VOLUME_CACHE: Dict[str, Tuple[float, Any]] = {}
_HIGH_VOLUME_TTL_SECONDS = 900

# USD/CAD barely moves minute to minute — cache aggressively.
_FX_CACHE: Dict[str, Tuple[float, float]] = {}
_FX_TTL_SECONDS = 3600

# Ticker/company-name search results change rarely — cache aggressively.
_SEARCH_CACHE: Dict[str, Tuple[float, list]] = {}
_SEARCH_CACHE_TTL_SECONDS = 3600



def cache_get(ticker: str) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(ticker)
    if not entry:
        return None
    stored_at, payload = entry
    if time.time() - stored_at > _CACHE_TTL_SECONDS:
        _CACHE.pop(ticker, None)
        return None
    return payload


def cache_set(ticker: str, payload: Dict[str, Any]) -> None:
    _CACHE[ticker] = (time.time(), payload)
    # Keep the cache from growing without bound on a small instance.
    if len(_CACHE) > 100:
        oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _CACHE.pop(oldest, None)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class TechnicalAnalysis(BaseModel):
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    volume_ratio: Optional[float]
    volume_note: str
    divergence: Optional[str]
    divergence_note: str
    divergence_bars_ago: Optional[int] = None
    signal: str
    conviction: str
    reasoning: str
    source_ticker: Optional[str] = None
    track_record_edge: Optional[float] = None
    track_record_events: Optional[int] = None
    track_record_note: str = ""


class Fundamentals(BaseModel):
    market_cap: Optional[str]
    pe_ratio: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    position_in_range: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    sector: Optional[str]
    industry: Optional[str]


class PriceTargets(BaseModel):
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    note: str


class Headline(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    sentiment: str
    sentiment_score: float
    context: str


class EarningsQuarter(BaseModel):
    date: str
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    surprise_pct: Optional[float]


class EarningsInfo(BaseModel):
    next_earnings_date: Optional[str]
    recent_quarters: List[EarningsQuarter]
    note: str


class SentimentAnalysis(BaseModel):
    bullish_score: float
    overall_impact: str
    headline_count: int
    headlines: List[Headline]


class CorrelatedPair(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation: float


class DataQuality(BaseModel):
    reliability: str            # "Good" | "Fair" | "Poor"
    avg_daily_volume: Optional[int]
    instrument_type: Optional[str]
    is_derivative: bool
    warnings: List[str]
    note: str


class ConcentrationReport(BaseModel):
    highly_correlated_pairs: List[CorrelatedPair]
    largest_position_pct: Optional[float]
    top_three_pct: Optional[float]
    effective_positions: Optional[float]
    sector_concentration: Dict[str, float]
    warnings: List[str]
    note: str


class QualityScore(BaseModel):
    score: Optional[int]
    grade: str
    factors: List[str]
    concerns: List[str]
    note: str


class MarketRegime(BaseModel):
    regime: str
    index_used: str
    index_vs_200ma: Optional[float]
    volatility_percentile: Optional[float]
    note: str
    # Supplementary context from Stocklake's get_market_pulse — VIX, CNN-style
    # fear/greed, and market-wide RSI breadth. Deliberately additive, not a
    # replacement for index_vs_200ma/volatility_percentile above: those two
    # are what apply_regime_check actually reads to move conviction, computed
    # from a full 2-year daily history this app controls end to end. Market
    # pulse is a live snapshot with no historical series behind it — same
    # reason sector intelligence (see SectorIntelligence's docstring) stayed
    # informational instead of replacing anything: there's nothing to backtest
    # it against. None on any fetch failure — this whole block is optional.
    vix: Optional[float] = None
    fear_greed_value: Optional[int] = None
    fear_greed_label: Optional[str] = None
    breadth_oversold_pct: Optional[float] = None
    breadth_overbought_pct: Optional[float] = None


class SectorIntelligence(BaseModel):
    """
    Display-only context from Stocklake's get_sector_intelligence — an
    AI-assessed real-time snapshot per sector (cycle stage, rotation
    signal, breadth, recent performance), refreshed ~every 4 hours.

    Deliberately NOT a backtest-validated conviction modifier, and — unlike
    insider activity — genuinely can't become one the same way: Stocklake
    only exposes up to 3 PRIOR signal states per sector (history_count),
    nowhere near the 1-5 years of daily history the app's own
    VARIANT_DEFS/aggregate_variants harness needs to test whether a filter
    actually has an edge. There's no way to backtest this at all with what
    Stocklake provides, so it's shown as context only, same tier as news
    sentiment and insider activity, with no path to graduating into
    apply_regime_check the way a historically-backtestable signal could.
    """
    sector: Optional[str] = None
    signal: Optional[str] = None
    cycle_stage: Optional[str] = None
    rotation_signal: Optional[str] = None
    drivers: Optional[str] = None
    alert: Optional[str] = None
    confidence: Optional[int] = None
    avg_perf_1w_pct: Optional[float] = None
    avg_perf_1m_pct: Optional[float] = None
    sma200_breadth_pct: Optional[float] = None


class InsiderActivity(BaseModel):
    """
    Display-only context from Stocklake's get_insider_activity — SEC Form 4
    insider transactions plus institutional-holdings flow. Deliberately NOT
    a conviction modifier: unlike the five that are (track record, quality,
    regime, divergence, earnings), this has never been run through the
    backtest harness, so there's no evidence yet that it should move
    conviction one way or the other. See DECISION_LOGIC.md §16 on why the
    app doesn't wire new signals into the live conviction on intuition alone.
    """
    signal: Optional[str] = None
    signal_score: Optional[float] = None
    signal_score_band: Optional[str] = None
    insider_signal: Optional[str] = None
    institutional_signal: Optional[str] = None
    summary: Optional[str] = None
    insider_buys: Optional[int] = None
    insider_sells: Optional[int] = None
    institutional_ownership_pct: Optional[float] = None
    total_holders: Optional[int] = None


class AnalystConsensus(BaseModel):
    """
    Wall Street's own consensus (rating, price target, analyst count) from
    Stocklake's get_stock — already fetched by analyze() for every ticker
    (it's part of the same payload quality/fundamentals/earnings pull from),
    just never surfaced before now. Informational only, same posture as
    insider activity and sector intelligence: an analyst consensus is
    someone else's opinion, not a backtested signal, so it stays out of
    this app's own conviction scoring.
    """
    rating: Optional[str] = None
    rating_score: Optional[float] = None
    target: Optional[float] = None
    analyst_count: Optional[int] = None


class AnalysisResponse(BaseModel):
    ticker: str
    company_name: str
    current_price: Optional[float]
    price_change: Optional[float]
    price_change_pct: Optional[float]
    currency: str
    market: str
    price_history: List[float]
    fundamentals: Fundamentals
    quality: "QualityScore"
    data_quality: "DataQuality"
    regime: "MarketRegime"
    earnings: EarningsInfo
    technical_analysis: TechnicalAnalysis
    price_targets: PriceTargets
    sentiment_analysis: SentimentAnalysis
    insider_activity: Optional[InsiderActivity] = None
    sector_intelligence: Optional[SectorIntelligence] = None
    analyst_consensus: Optional[AnalystConsensus] = None
    generated_at: str
    cached: bool = False


class AIDecisionResponse(BaseModel):
    conviction_score: Optional[int] = None
    conviction_label: Optional[str] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    rationale: Optional[str] = None


class DecisionHistoryEntry(BaseModel):
    """
    One past AI Decision for this ticker, already resolved against a later
    price by the client (which is the only side that persists this history —
    see AIDecisionRequest). Sent back on the next request so the model can
    see whether its own past calls on this exact ticker actually panned out,
    instead of reasoning fresh from zero every time.
    """
    decided_at: str
    conviction_score: int
    price_then: float
    price_now: float
    hit_take_profit: bool = False
    hit_stop_loss: bool = False


class AIDecisionRequest(BaseModel):
    # Present only when the person already holds this ticker — switches the
    # question from "is this worth a fresh entry" to "is this worth
    # continuing to hold at this cost basis."
    shares: Optional[float] = None
    cost_basis: Optional[float] = None
    # Past decisions for this ticker the client has already resolved against
    # a later price. Bounded client-side; capped again here defensively.
    history: List[DecisionHistoryEntry] = []


# --- Screener / portfolio / digest -----------------------------------------
class ScreenerHit(BaseModel):
    ticker: str
    price: Optional[float]
    change_pct: Optional[float]
    rsi: Optional[float]
    macd_histogram: Optional[float]
    volume_ratio: Optional[float]
    signal: str
    conviction: str
    reasoning: str


class ScreenerResponse(BaseModel):
    buy_candidates: List[ScreenerHit]
    sell_candidates: List[ScreenerHit]
    scanned_count: int
    universe_note: str
    generated_at: str
    cached: bool = False


class StocklakeScreenerHit(BaseModel):
    """
    One row from Stocklake's get_screener — a different data source, a
    different (AI-scored) methodology, and a much larger universe than
    this app's own fixed-list Screener above. Not this app's RSI+MACD
    rule, not backtest-validated by this app. See StocklakeScreenerResponse.
    """
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    price: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_forward: Optional[float] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    sma200_trend: Optional[str] = None
    analyst_rating: Optional[str] = None
    rating: Optional[float] = None
    ai_verdict: Optional[str] = None
    ai_headline: Optional[str] = None
    ai_score: Optional[int] = None
    ai_score_band: Optional[str] = None


class StocklakeScreenerResponse(BaseModel):
    count: int
    preset: Optional[str] = None
    results: List[StocklakeScreenerHit]
    note: str


class MarketPulseResponse(BaseModel):
    """
    Standalone version of the same market-pulse snapshot get_market_regime()
    already folds into the Analyze page's regime card — pulled out onto its
    own page (Stocklake-first plan, P5c) since VIX/fear-greed/breadth are
    market-wide context worth their own screen, not just a line under one
    ticker's regime note.
    """
    vix: Optional[float] = None
    fear_greed_value: Optional[int] = None
    fear_greed_label: Optional[str] = None
    breadth_oversold_pct: Optional[float] = None
    breadth_overbought_pct: Optional[float] = None
    generated_at: str
    note: str


class UpcomingEarning(BaseModel):
    ticker: str
    earnings_date: str
    is_estimate: bool


class EarningsCalendarRequest(BaseModel):
    tickers: List[str]


class EarningsCalendarResponse(BaseModel):
    upcoming: List[UpcomingEarning]
    note: str


class SectorRotationResponse(BaseModel):
    sectors: List[SectorIntelligence]
    note: str


class HighVolumeResponse(BaseModel):
    stocks: List[ScreenerHit]
    universe_note: str
    generated_at: str
    cached: bool = False


class PortfolioHolding(BaseModel):
    ticker: str
    shares: float
    cost_basis: float  # average price paid per share
    account: Optional[str] = None  # e.g. "TFSA", "FHSA", "Non-Registered" — display/grouping only


class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]


class HoldingResult(BaseModel):
    ticker: str
    shares: float
    cost_basis: float
    account: Optional[str] = None
    current_price: Optional[float]
    currency: str
    market_value: Optional[float]
    unrealized_pl: Optional[float]
    unrealized_pl_pct: Optional[float]
    day_pl: Optional[float]
    day_pl_pct: Optional[float]
    rsi: Optional[float]
    macd_histogram: Optional[float]
    volume_ratio: Optional[float]
    signal: str
    conviction: str
    reasoning: str
    context: str
    error: Optional[str] = None


class PortfolioResponse(BaseModel):
    holdings: List[HoldingResult]
    total_cost: float
    total_value: float
    total_pl: float
    total_pl_pct: float
    total_day_pl: float
    total_day_pl_pct: float
    usd_cad_rate: Optional[float]
    concentration: Optional["ConcentrationReport"] = None
    summary: str
    generated_at: str


class DigestRequest(BaseModel):
    holdings: List[PortfolioHolding] = []
    watchlist: List[str] = []
    extra_scan_tickers: List[str] = []


class DigestResponse(BaseModel):
    portfolio: Optional[PortfolioResponse]
    watchlist_signals: List[ScreenerHit]
    opportunities: List[ScreenerHit]
    warnings: List[ScreenerHit]
    under20_buys: List[ScreenerHit]
    under20_avoid: List[ScreenerHit]
    headline: str
    note: str
    generated_at: str


class SymbolMatch(BaseModel):
    symbol: str
    name: str
    exchange: Optional[str]
    quote_type: Optional[str]


class SearchResponse(BaseModel):
    query: str
    results: List[SymbolMatch]


class HistoryPoint(BaseModel):
    date: str
    close: float


class HistoryResponse(BaseModel):
    ticker: str
    range: str
    points: List[HistoryPoint]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_market(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".TO"):
        return "TSX"
    if t.endswith(".V"):
        return "TSX-V"
    if t.endswith(".CN"):
        return "CSE"
    if t.endswith(".NE"):
        return "NEO"
    return "US"


def _yf_history_with_timeout(stock, period: str, timeout: Optional[float] = None):
    """
    Bounds one yf.Ticker(...).history() call to `timeout` seconds (defaults
    to the same BATCH_DOWNLOAD_TIMEOUT_SECONDS batch_score() uses — resolved
    at call time, not as a default-argument value, since that constant is
    defined later in this file than this function). Same reasoning as
    batch_score()'s guard (deliberately not a `with ThreadPoolExecutor(...)
    as pool:` block — that waits for the call to finish on exit even after
    .result() has already timed out): the single-ticker /api/analyze path
    had no timeout at all before this, unlike the batch path, which is
    exactly the gap the Stocklake-first plan's P1 flagged for this endpoint.
    """
    if timeout is None:
        timeout = BATCH_DOWNLOAD_TIMEOUT_SECONDS
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(
            stock.history, period=period, interval="1d", auto_adjust=True
        ).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)


def fetch_price_history(ticker: str):
    """6 months of daily bars — enough for a 26/9 MACD, 14-day RSI and 20-day volume avg.

    Retries with a longer lookback if that window comes back thin or empty —
    thinly-traded listings (CAD-hedged CDRs on NEO/.NE in particular) sometimes
    return a sparse/empty 6-month window from Yahoo even though a longer window
    has plenty of history. Same retry already proven in batch_score()'s own
    fallback path, applied here so the single-ticker /api/analyze endpoint gets
    the same fix, not just the portfolio/screener batch path.

    NOTE (Stocklake-first plan, P1): this is still yfinance-only — swapping it
    for Stocklake's get_stock_history needs that tool's real response schema
    verified against a live call first (this session's Stocklake MCP
    connector was down while this was written), so it's deliberately not
    guessed at here. What ships now is the reliability half: a call that used
    to be able to hang indefinitely is bounded, same as every yfinance call
    batch_score() makes.
    """
    stock = yf.Ticker(ticker)
    hist = _yf_history_with_timeout(stock, "6mo")
    if hist is None or hist.empty or len(hist) < 30:
        current_len = 0 if hist is None else len(hist)
        longer = _yf_history_with_timeout(stock, "2y")
        if longer is not None and not longer.empty and len(longer) > current_len:
            hist = longer
    return stock, hist


def format_market_cap(value: Optional[float]) -> Optional[str]:
    if not value:
        return None
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:,.0f}"


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Standard Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Standard MACD: fast EMA minus slow EMA, plus a signal EMA of that line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_atr(hist: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range — how much this stock typically moves per day."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def compute_volume_ratio(hist: pd.DataFrame) -> Optional[float]:
    """Today's volume vs the trailing 20-day average."""
    if "Volume" not in hist or len(hist) < 21:
        return None
    recent = float(hist["Volume"].iloc[-1])
    avg = float(hist["Volume"].tail(21).iloc[:-1].mean())
    if avg <= 0:
        return None
    return round(recent / avg, 2)


def describe_volume(volume_ratio: Optional[float]) -> str:
    if volume_ratio is None:
        return "Volume data unavailable."
    if volume_ratio >= 1.5:
        return f"Volume is {volume_ratio}x its 20-day average — heavy participation behind this move."
    if volume_ratio >= 1.0:
        return f"Volume is {volume_ratio}x its 20-day average — roughly normal participation."
    return f"Volume is only {volume_ratio}x its 20-day average — light participation, so this move carries less conviction."


def generate_signal(
    rsi: Optional[float], macd_hist: Optional[float], volume_ratio: Optional[float]
) -> Tuple[str, str, str]:
    """
    Transparent scoring model:
      RSI < 30 (oversold) -> +1   |   RSI > 70 (overbought) -> -1
      MACD histogram > 0  -> +1   |   MACD histogram < 0    -> -1
    score >= 1 -> BUY, score <= -1 -> SELL, else HOLD.

    Volume does not change the direction — it modifies *conviction*, which is
    how volume is actually used in practice: it tells you how much to trust
    the move, not which way it's going.
    """
    if rsi is None or macd_hist is None:
        return "HOLD", "Low", "Insufficient indicator data to generate a confident signal."

    score = 0
    reasons = []

    if rsi < 30:
        score += 1
        reasons.append(f"RSI ({rsi:.1f}) indicates the stock is oversold")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI ({rsi:.1f}) indicates the stock is overbought")
    else:
        reasons.append(f"RSI ({rsi:.1f}) is in a neutral range")

    if macd_hist > 0:
        score += 1
        reasons.append("MACD histogram is positive, showing bullish momentum")
    elif macd_hist < 0:
        score -= 1
        reasons.append("MACD histogram is negative, showing bearish momentum")
    else:
        reasons.append("MACD histogram is flat")

    signal = "BUY" if score >= 1 else "SELL" if score <= -1 else "HOLD"

    if signal == "HOLD":
        conviction = "Neutral"
    elif volume_ratio is None:
        conviction = "Moderate"
    elif volume_ratio >= 1.5:
        conviction = "High"
        reasons.append("above-average volume supports the move")
    elif volume_ratio >= 1.0:
        conviction = "Moderate"
    else:
        conviction = "Low"
        reasons.append("below-average volume weakens conviction")

    return signal, conviction, "; ".join(reasons) + "."


def compute_technicals(hist: pd.DataFrame) -> TechnicalAnalysis:
    """
    Below 30 rows this degrades rather than blocks: RSI needs 14 warm-up bars
    (min_periods=14) so it comes back None on a very new listing, and
    generate_signal() already treats a None RSI/MACD as "not enough data for
    a signal" (HOLD/Low, see below) rather than crashing — so a recently
    listed security (a new NEO/.NE CDR is the common real case) still shows
    whatever price/chart data actually exists instead of erroring out
    entirely. Only truly empty history has nothing to compute.
    """
    if hist is None or hist.empty:
        raise HTTPException(
            status_code=422,
            detail="No price history available for this ticker yet.",
        )

    close = hist["Close"]
    rsi_series = compute_rsi(close, length=14)
    macd_line, signal_line, histogram = compute_macd(close)

    rsi_clean = rsi_series.dropna()
    rsi_val = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else None
    macd_val = float(macd_line.iloc[-1]) if not macd_line.dropna().empty else None
    macd_signal_val = float(signal_line.iloc[-1]) if not signal_line.dropna().empty else None
    macd_hist_val = float(histogram.iloc[-1]) if not histogram.dropna().empty else None

    volume_ratio = compute_volume_ratio(hist)
    signal, conviction, reasoning = generate_signal(rsi_val, macd_hist_val, volume_ratio)
    if len(hist) < 30:
        reasoning = (
            f"Only {len(hist)} trading day(s) of price history available, likely a recently "
            f"listed security — indicators need more history to be reliable. " + reasoning
        )

    # Look for a divergence in the recent past. Only the last ~30 bars are
    # considered "current" — an older one has usually already played out.
    divergence_kind = None
    bars_ago = None
    try:
        div_series = detect_divergences(close, rsi_series)
        recent = div_series.tail(30)
        hits = [(i, v) for i, v in enumerate(recent.values) if v]
        if hits:
            idx, divergence_kind = hits[-1]
            bars_ago = len(recent) - 1 - idx
    except Exception as e:
        logger.warning(f"Divergence detection failed: {e}")

    # Divergence is deliberately NOT folded into conviction here — this
    # function has no idea yet whether the underlying data is even reliable
    # enough to trust a divergence read on. analyze() applies it later
    # (apply_divergence), at the same point as the other conviction checks,
    # after data reliability has been decided — see the note there on why a
    # Poor-reliability ticker skips this the same way it skips track record.
    return TechnicalAnalysis(
        rsi=round(rsi_val, 2) if rsi_val is not None else None,
        macd=round(macd_val, 4) if macd_val is not None else None,
        macd_signal=round(macd_signal_val, 4) if macd_signal_val is not None else None,
        macd_histogram=round(macd_hist_val, 4) if macd_hist_val is not None else None,
        volume_ratio=volume_ratio,
        volume_note=describe_volume(volume_ratio),
        divergence=divergence_kind,
        divergence_note=describe_divergence(divergence_kind, bars_ago),
        divergence_bars_ago=bars_ago,
        signal=signal,
        conviction=conviction,
        reasoning=reasoning,
    )


def compute_price_targets(
    hist: pd.DataFrame, signal: str, current_price: Optional[float]
) -> PriceTargets:
    """
    ATR-based volatility sizing — a standard technique, not a prediction of
    where price will actually go.
    """
    default_note = (
        "These are calculated reference levels, not a guarantee — no formula can "
        "promise a maximum profit, since that depends on where the price actually goes."
    )

    if current_price is None or hist is None or len(hist) < 20:
        return PriceTargets(
            entry=None, stop_loss=None, take_profit=None, risk_reward_ratio=None,
            support=None, resistance=None,
            note="Not enough price history to calculate reliable levels.",
        )

    support = round(float(hist["Low"].tail(20).min()), 2)
    resistance = round(float(hist["High"].tail(20).max()), 2)

    atr_series = compute_atr(hist, length=14).dropna()
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else None

    if not atr_val:
        return PriceTargets(
            entry=None, stop_loss=None, take_profit=None, risk_reward_ratio=None,
            support=support, resistance=resistance,
            note="Not enough history to size a volatility-based stop yet. " + default_note,
        )

    stop_multiplier, target_multiplier = 1.5, 3.0

    if signal == "BUY":
        entry = current_price
        stop_loss = round(entry - stop_multiplier * atr_val, 2)
        take_profit = round(entry + target_multiplier * atr_val, 2)
        note = (
            "Entry at current price; stop-loss and take-profit are sized off recent "
            "volatility (ATR) for roughly a 2:1 reward-to-risk setup. " + default_note
        )
    else:
        # SELL and HOLD both skip new-entry numbers. A short-sale stop/take-profit
        # setup here would put stop-loss numerically ABOVE take-profit (correct
        # for shorting, since you profit from the price falling) — but that reads
        # as backwards for a long-only investor, which is how this app is used.
        entry = stop_loss = take_profit = None
        if signal == "SELL":
            note = (
                "No new long entry is suggested — momentum has turned negative. If "
                "you're already holding, support below is a downside level to watch; "
                "reclaiming resistance would suggest the negative momentum is fading. "
                + default_note
            )
        else:
            note = (
                "No new entry is suggested while the signal is HOLD. Support and "
                "resistance below are levels worth watching. " + default_note
            )

    risk_reward_ratio = None
    if entry is not None and stop_loss is not None and take_profit is not None:
        risk = abs(entry - stop_loss)
        if risk > 0:
            risk_reward_ratio = round(abs(take_profit - entry) / risk, 2)

    return PriceTargets(
        entry=entry, stop_loss=stop_loss, take_profit=take_profit,
        risk_reward_ratio=risk_reward_ratio, support=support,
        resistance=resistance, note=note,
    )


# ---------------------------------------------------------------------------
# TRACK-RECORD-INFORMED CONVICTION
#
# Closes the loop between the backtesting tools and the live signal. Every
# other conviction modifier in this app (volume, data quality, earnings
# proximity) is a heuristic about whether to trust a signal. This one is
# different: it's the stock's own measured history, using the exact same
# event-anchored, baseline-compared methodology already validated for the
# on-demand backtest — reused directly, not reimplemented, to avoid any
# drift between what this shows and what the Backtest card reports.
#
# Deliberately conservative by design: this can nudge conviction one notch
# up or down, and never touches signal direction. A stock's history not
# supporting today's BUY doesn't mean the BUY is wrong — RSI/MACD crossing
# a threshold is still the same simple, auditable rule it always was. This
# only adjusts how much weight to put behind it.
# ---------------------------------------------------------------------------
_TRACK_RECORD_CACHE: Dict[str, Tuple[float, Any]] = {}
_TRACK_RECORD_TTL_SECONDS = 86400  # 24h — backtest results don't meaningfully shift within a day
TRACK_RECORD_MIN_EVENTS = 8
TRACK_RECORD_STRONG_EDGE = 1.0  # percentage points, net of trading costs

CONVICTION_LEVELS = ["Low", "Moderate", "High"]


def _nudge_conviction(conviction: str, steps: int) -> str:
    """Move conviction up/down by `steps` notches, clamped to Low..High.
    No-op for "Neutral" (HOLD's fixed conviction isn't on this ladder). The
    one shared implementation of a rule several conviction modifiers in this
    file need — track record, earnings proximity, and (below) quality,
    regime, and divergence."""
    if conviction not in CONVICTION_LEVELS:
        return conviction
    idx = CONVICTION_LEVELS.index(conviction)
    idx = max(0, min(len(CONVICTION_LEVELS) - 1, idx + steps))
    return CONVICTION_LEVELS[idx]


def get_ticker_track_record(ticker: str) -> Optional["BacktestResponse"]:
    """
    Cache-only — deliberately never triggers a fresh 2-year backtest inline
    during a live analyze() call. That fetch is exactly the kind of added
    latency that's caused real reliability problems in this app before, and
    a slow or timed-out live signal is worse than one simply missing its
    track-record annotation until the cache is warm.

    The cache gets populated by the manual "Run backtest" button (which
    shares this same cache), or by a background task analyze() kicks off
    after it has already returned its response (see _warm_track_record_cache
    below) — never by this function blocking to compute it, and never on
    analyze()'s own critical path.
    """
    cached = _TRACK_RECORD_CACHE.get(ticker)
    if cached and time.time() - cached[0] < _TRACK_RECORD_TTL_SECONDS:
        return cached[1]
    return None


_TRACK_RECORD_IN_PROGRESS: set = set()


def _get_or_run_backtest(
    ticker: str, period: str = "2y", skip_if_in_progress: bool = False
) -> Optional["BacktestResponse"]:
    """
    The one place that actually computes a backtest and (for period="2y")
    writes it to _TRACK_RECORD_CACHE — shared by the manual "Run backtest"
    endpoint and the background warm task below, so a change to the cache
    contract only has to happen once.

    `_TRACK_RECORD_IN_PROGRESS` guards against two near-simultaneous
    background warms for the same not-yet-cached ticker (e.g. two users
    opening it at once) both running the same expensive 2-year fetch and
    backtest concurrently. It's opt-in via `skip_if_in_progress` — the
    background warm sets it, since skipping there just means "the cache
    will still get warm, slightly later." The manual endpoint leaves it
    off: a user who clicked "Run backtest" needs a real result back, not a
    silent None, even in the rare case a background warm for the same
    ticker happens to be running at that exact moment — it still marks
    itself in-progress meanwhile, so at least that direction (a concurrent
    background warm skipping in favor of this one) is covered.

    This is a plain set, not a lock — under CPython's GIL the check-then-add
    below isn't perfectly atomic, so a very tight race could still let two
    callers through, but that only costs one redundant backtest, not a
    correctness problem, and a real lock would be more machinery than that
    failure mode justifies.
    """
    ticker = ticker.strip().upper()
    cached = _TRACK_RECORD_CACHE.get(ticker) if period == "2y" else None
    if cached and time.time() - cached[0] < _TRACK_RECORD_TTL_SECONDS:
        return cached[1]
    if skip_if_in_progress and ticker in _TRACK_RECORD_IN_PROGRESS:
        return None  # another caller is already computing this one
    _TRACK_RECORD_IN_PROGRESS.add(ticker)
    try:
        result = run_backtest(ticker, period)
        if period == "2y":
            _TRACK_RECORD_CACHE[ticker] = (time.time(), result)
        return result
    finally:
        _TRACK_RECORD_IN_PROGRESS.discard(ticker)


# Caps how many of these can run at once. Each one fetches 2 years of price
# history over the network and runs a full backtest — real CPU and I/O, not
# free. Without a cap, a burst of analyze() calls (e.g. a Portfolio with
# many holdings, or several tabs open at once) could queue up enough of
# these to start competing with foreground requests for the same limited
# worker threads — including, worst case, an AI Decision request that
# arrives right after, since /decision always follows an analyze() call for
# the same ticker. A non-blocking acquire means a warm that can't get a
# slot is simply skipped rather than queued: that ticker just stays cold
# until a later analyze() call tries again, which is a fine trade-off for
# something that was already fully optional.
_TRACK_RECORD_WARM_SEMAPHORE = threading.Semaphore(2)


def _warm_track_record_cache(ticker: str) -> None:
    """
    Runs as a FastAPI BackgroundTask, after analyze() has already sent its
    response — so a slow or failed backtest here can never add latency to,
    or break, the live signal a user is actually waiting on.

    Without this, get_ticker_track_record() above is genuinely cache-only:
    nothing else ever writes to _TRACK_RECORD_CACHE except the manual "Run
    backtest" button, so apply_track_record() — the one conviction modifier
    that's a stock's own measured history rather than a heuristic — would
    silently never fire for a ticker nobody has happened to backtest by
    hand in the last 24h. This closes that gap: the FIRST analyze() call
    for a ticker still won't have a track record to show (there was nothing
    to read yet), but every one after it within the cache's 24h window will.
    """
    if not _TRACK_RECORD_WARM_SEMAPHORE.acquire(blocking=False):
        logger.info(f"Background track-record warm skipped for {ticker}: too many warms already in flight")
        return
    try:
        _get_or_run_backtest(ticker, "2y", skip_if_in_progress=True)
    except Exception as e:
        logger.info(f"Background track-record warm skipped for {ticker}: {e}")
    finally:
        _TRACK_RECORD_WARM_SEMAPHORE.release()


def apply_track_record(technical: TechnicalAnalysis, ticker: str) -> TechnicalAnalysis:
    """
    This function must never be able to crash a live analyze() request — the
    signal itself is more important than the track-record annotation on top
    of it. Everything below is wrapped; any failure here is logged and the
    signal is returned exactly as it was, unaffected.
    """
    try:
        return _apply_track_record_inner(technical, ticker)
    except Exception as e:
        logger.warning(f"Track record adjustment failed for {ticker}, skipping: {e}")
        return technical


def _apply_track_record_inner(technical: TechnicalAnalysis, ticker: str) -> TechnicalAnalysis:
    if technical.signal not in ("BUY", "SELL"):
        return technical

    bt = get_ticker_track_record(ticker)
    if bt is None:
        return technical  # not cached yet — no panel shown, live signal unaffected

    side = bt.buy if technical.signal == "BUY" else bt.sell
    ten = next((h for h in side.horizons if h.days == 10), None)

    if side.event_count < TRACK_RECORD_MIN_EVENTS or ten is None or ten.net_edge is None:
        technical.track_record_events = side.event_count
        technical.track_record_note = (
            f"Only {side.event_count} past {technical.signal} signals on this stock — too few "
            "to check a track record yet. Conviction unaffected."
        )
        return technical

    edge = ten.net_edge
    technical.track_record_edge = edge
    technical.track_record_events = side.event_count

    if edge >= TRACK_RECORD_STRONG_EDGE:
        before = technical.conviction
        technical.conviction = _nudge_conviction(technical.conviction, 1)
        raised = technical.conviction != before
        technical.track_record_note = (
            f"This stock's own {technical.signal} signals have beaten random days by "
            f"{edge:+.2f}% over the following two weeks, across {side.event_count} past "
            "signals — a track record that supports this call."
            + (" Conviction raised accordingly." if raised else " Already at High conviction, so no further raise.")
        )
    elif edge <= -TRACK_RECORD_STRONG_EDGE:
        before = technical.conviction
        technical.conviction = _nudge_conviction(technical.conviction, -1)
        lowered = technical.conviction != before
        technical.track_record_note = (
            f"Worth knowing: this stock's own {technical.signal} signals have historically "
            f"UNDERPERFORMED random days by {abs(edge):.2f}% over the following two weeks, "
            f"across {side.event_count} past signals. The signal itself hasn't changed, but "
            "its track record here argues for caution."
            + (" Conviction lowered accordingly." if lowered else " Already at Low conviction.")
        )
    else:
        technical.track_record_note = (
            f"This stock's own {technical.signal} signals have shown a modest "
            f"{edge:+.2f}% edge over {side.event_count} past signals — not a strong enough "
            "pattern either way to move conviction."
        )
    return technical


def apply_earnings_proximity(
    technical: TechnicalAnalysis, earnings: EarningsInfo
) -> TechnicalAnalysis:
    """
    Downgrade conviction when earnings are imminent.

    This is a guardrail, not a prediction — and the distinction matters. It
    doesn't claim to know which way the stock will go. It reflects something
    plainly true: a scheduled event capable of gapping the price 10% overrides
    whatever RSI and MACD are saying about the prior few weeks. The signal
    isn't wrong so much as about to be irrelevant.
    """
    if not earnings.next_earnings_date:
        return technical

    try:
        next_date = datetime.strptime(earnings.next_earnings_date, "%Y-%m-%d").date()
        days_until = (next_date - datetime.now(timezone.utc).date()).days
    except Exception:
        return technical

    if days_until < 0 or days_until > 7:
        return technical

    when = "tomorrow" if days_until == 1 else "today" if days_until == 0 else f"in {days_until} days"

    # Earnings being imminent is worth surfacing either way — even when
    # conviction is already at the floor (or, for HOLD, never on the ladder
    # to begin with) and the nudge below is a no-op, that's still something
    # worth knowing about. Just don't claim a conviction change — or a prior
    # adjustment — that didn't happen.
    before = technical.conviction
    technical.conviction = _nudge_conviction(technical.conviction, -1)
    if technical.conviction != before:
        technical.reasoning += (
            f" Note: earnings are due {when}, which typically overrides technical signals — "
            "conviction lowered accordingly."
        )
    elif before in CONVICTION_LEVELS:
        technical.reasoning += (
            f" Note: earnings are due {when}, which typically overrides technical signals — "
            "already reflected in the conviction above."
        )
    else:
        technical.reasoning += (
            f" Note: earnings are due {when}, which typically overrides technical signals — "
            "worth factoring in even though this is a HOLD."
        )
    return technical


# ---------------------------------------------------------------------------
# Three more conviction modifiers, added to close a real gap: quality score,
# market regime, and RSI divergence were each already computed with their own
# well-reasoned notes explaining exactly how they should affect a signal ("a
# BUY on a Poor business deserves more scepticism", "treat BUY signals with
# real caution" in a falling market, divergence as "one of the few genuinely
# anticipatory technical patterns") — but none of that reasoning ever reached
# technical.conviction. It only ever reached the optional, opt-in AI Decision
# as passive JSON context. Since the mechanical signal (not the AI Decision)
# is what actually drives Screener, Portfolio, and Brief for every user on
# every ticker, that's where this evidence was going unused. These three
# functions apply it, using the exact same nudge-one-notch pattern (and the
# same shared _nudge_conviction helper, defined above) as
# apply_track_record/apply_earnings_proximity — never touching signal
# direction, and (except track record, which is symmetric by design) mostly
# downgrade-only: the goal is tempering false confidence, not manufacturing
# extra confidence a check hasn't earned.
# ---------------------------------------------------------------------------


def apply_divergence(technical: TechnicalAnalysis, bars_ago: Optional[int]) -> TechnicalAnalysis:
    """
    A divergence within the last 10 bars — bars_ago in 0..9, the same window
    the app's own backtest "divergence_confirm" variant tests via
    rolling(10) — either corroborates the current signal (bullish divergence
    + BUY, bearish + SELL) or contradicts it (the move looks like it's
    running out of steam right as the signal fires). HOLD is untouched, and
    the reasoning sentence is only appended when the nudge actually moved
    conviction — e.g. it's already a no-op at the High/Low ceiling/floor, and
    silently claiming a change that didn't happen would be misleading.
    """
    if technical.signal not in ("BUY", "SELL") or not technical.divergence or bars_ago is None or bars_ago >= 10:
        return technical
    supports = (technical.signal == "BUY" and technical.divergence == "bullish") or (
        technical.signal == "SELL" and technical.divergence == "bearish"
    )
    before = technical.conviction
    technical.conviction = _nudge_conviction(technical.conviction, 1 if supports else -1)
    if technical.conviction != before:
        technical.reasoning += (
            f" {'Supported' if supports else 'Undercut'} by the recent {technical.divergence} "
            "divergence — conviction adjusted accordingly."
        )
    return technical


def apply_quality_check(technical: TechnicalAnalysis, quality: "QualityScore") -> TechnicalAnalysis:
    """Skepticism toward a signal that fights the business's own fundamentals.
    Doesn't touch HOLD, and never upgrades conviction — a Strong-quality BUY
    doesn't get extra credit for being Strong, since the technicals already
    speak for themselves; a Poor-quality BUY or a Strong-quality SELL does
    get more scepticism, because those are the two combinations most likely
    to be chasing noise (a technical bounce in a weak business) or panicking
    (a short-term dip in a strong one)."""
    if technical.signal not in ("BUY", "SELL") or quality.grade in ("Unknown", None):
        return technical
    penalize = (technical.signal == "BUY" and quality.grade == "Poor") or (
        technical.signal == "SELL" and quality.grade == "Strong"
    )
    if not penalize:
        return technical
    before = technical.conviction
    technical.conviction = _nudge_conviction(technical.conviction, -1)
    if technical.conviction == before:
        return technical
    if technical.signal == "BUY":
        technical.reasoning += (
            " Business quality is Poor — conviction lowered; a technical bounce in a weak "
            "business is a riskier bet than the same setup in a stronger one."
        )
    else:
        technical.reasoning += (
            " Business quality is Strong — conviction lowered; a short-term technical dip in "
            "a strong business is less likely to be the start of a real decline."
        )
    return technical


def apply_regime_check(technical: TechnicalAnalysis, regime: "MarketRegime") -> TechnicalAnalysis:
    """
    get_market_regime()'s own thresholds — not its label string — decide
    this, so the two mostly stay in sync as regime wording evolves:
    index_vs_200ma < -3 / > 3 is the same "Falling"/"Trending up" cutoff the
    regime label itself is built from. Volatility is closer but not exact:
    get_market_regime itself isn't perfectly symmetric at the boundary — its
    "Trending up" branch treats volatility_percentile >= 70 as volatile,
    while its "Falling" branch requires strictly > 70 — so a single cutoff
    here can't match both exactly. >= 70 is used deliberately, since it's
    the more conservative (inclusive) of the two and never misses a case
    the regime label itself already calls "volatile". A counter-trend call
    (BUY against a falling market, SELL against a rising one) or an
    elevated-volatility regime (which produces more false signals in either
    direction, per that function's own notes) each earn one downgrade —
    combined into a single nudge rather than stacking, since both
    conditions describe the same underlying "hostile regime," not two
    independent penalties. Downgrade-only and HOLD-exempt, same asymmetry
    as the quality check above.
    """
    if technical.signal not in ("BUY", "SELL"):
        return technical
    counter_trend = regime.index_vs_200ma is not None and (
        (technical.signal == "BUY" and regime.index_vs_200ma < -3)
        or (technical.signal == "SELL" and regime.index_vs_200ma > 3)
    )
    volatile = regime.volatility_percentile is not None and regime.volatility_percentile >= 70
    if not (counter_trend or volatile):
        return technical
    before = technical.conviction
    technical.conviction = _nudge_conviction(technical.conviction, -1)
    if technical.conviction == before:
        return technical
    if counter_trend:
        technical.reasoning += (
            f" Market regime is '{regime.regime}' — {technical.signal} conviction lowered; "
            "counter-trend calls fare worse against the broader market's own trend."
        )
    else:
        technical.reasoning += (
            f" Market regime is '{regime.regime}', with volatility elevated enough that "
            "momentum signals produce more false positives — conviction lowered."
        )
    return technical


def _parse_future_stocklake_earnings_date(raw: Optional[str], ticker: str) -> Optional[str]:
    """
    Parses get_stock()/get_stocks()'s earnings_date field into a plain
    YYYY-MM-DD, or None if it's missing, unparseable, or already in the
    past. Shared by fetch_earnings() (single ticker) and the earnings
    calendar endpoint (batch) so both agree on what "upcoming" means.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        # Every live response observed so far uses a +00:00 offset, but
        # nothing guarantees that always holds — normalize to UTC (and
        # treat a naive timestamp as already UTC) before taking .date(),
        # so a non-UTC offset can't shift the calendar date by a day in
        # either direction right around midnight.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed_date = parsed.astimezone(timezone.utc).date()
        if parsed_date >= datetime.now(timezone.utc).date():
            return parsed_date.strftime("%Y-%m-%d")
        return None
    except Exception as e:
        logger.info(f"Couldn't parse Stocklake earnings_date for {ticker}: {e}")
        return None


def fetch_earnings(
    stock: yf.Ticker, ticker: str, stocklake_data: Optional[dict] = None
) -> EarningsInfo:
    """
    Next earnings date plus the last few quarters' EPS estimate vs. actual.
    Coverage varies a lot by ticker — especially thinner for smaller Canadian
    names — so this degrades gracefully to "not available" rather than error.

    The next-earnings-date half now prefers Stocklake's get_stock() response
    (pre-fetched once by analyze() and passed in as `stocklake_data`, shared
    with compute_quality_score's fundamentals — see fetch_stocklake_stock)
    over yfinance's own calendar, since that yfinance path has a known
    reliability problem: it HTML-scrapes Yahoo's calendar page, which
    yfinance's own source notes Yahoo stopped keeping current ("reverting to
    scraping HTML" after the proper endpoint broke) — in practice it was
    returning nothing for most tickers, well-covered large-caps included.
    yfinance's `calendar` still runs as the fallback whenever Stocklake
    doesn't have this ticker (a real, common case — confirmed live even
    some well-known large-caps aren't in Stocklake's ~3,500-symbol
    universe) or has no earnings_date on it.

    recent_quarters (historical EPS actual-vs-estimate) has no Stocklake
    equivalent — Stocklake's earnings data is calendar/forward-looking
    only — so that half is always yfinance's get_earnings_history(),
    unchanged. yfinance's `calendar`/`get_earnings_history()` both hit
    Yahoo's quoteSummary JSON API, same as fundamentals/info elsewhere in
    this file, and are far more reliable than the scraped alternative.
    """
    fallback = EarningsInfo(
        next_earnings_date=None, recent_quarters=[],
        note="Earnings data isn't available for this ticker.",
    )

    next_date = _parse_future_stocklake_earnings_date(
        stocklake_data.get("earnings_date") if stocklake_data else None, ticker
    )

    def fetch_next_date_yf():
        cal = stock.calendar or {}
        # yfinance builds these dates with datetime.fromtimestamp() (naive,
        # server-local time) — compare against a local-time "today" too,
        # rather than UTC, so the two sides can't disagree near midnight.
        today = datetime.now().date()
        upcoming = [d for d in (cal.get("Earnings Date") or []) if d and d >= today]
        return min(upcoming).strftime("%Y-%m-%d") if upcoming else None

    def fetch_quarters():
        df = stock.get_earnings_history()
        if df is None or df.empty:
            return []
        result = []
        for idx, row in df.sort_index(ascending=False).head(4).iterrows():
            if pd.isna(idx):
                continue  # Yahoo occasionally omits a quarter's date; skip rather than abort the rest
            est = row.get("epsEstimate")
            act = row.get("epsActual")
            surprise = row.get("surprisePercent")
            result.append(
                EarningsQuarter(
                    date=idx.strftime("%Y-%m-%d"),
                    eps_estimate=round(float(est), 2) if pd.notna(est) else None,
                    eps_actual=round(float(act), 2) if pd.notna(act) else None,
                    surprise_pct=round(float(surprise), 2) if pd.notna(surprise) else None,
                )
            )
        return result

    quarters = []
    # Two independent quoteSummary round trips — run concurrently rather than
    # doubling this function's latency, matching the pattern already used
    # for the high-volume screener's US/CA fetches. The date fetch is only
    # submitted at all when Stocklake didn't already supply one.
    with ThreadPoolExecutor(max_workers=2) as pool:
        date_future = None if next_date is not None else pool.submit(fetch_next_date_yf)
        quarters_future = pool.submit(fetch_quarters)
        if date_future is not None:
            try:
                next_date = date_future.result()
            except Exception as e:
                logger.info(f"Earnings calendar fetch failed: {e}")
        try:
            quarters = quarters_future.result()
        except Exception as e:
            logger.info(f"Earnings history fetch failed: {e}")

    if next_date is None and not quarters:
        return fallback

    return EarningsInfo(
        next_earnings_date=next_date,
        recent_quarters=quarters,
        note="Beat = actual EPS came in above analyst estimates; miss = below. Large surprises often move price sharply on the report date.",
    )


def build_fundamentals(info: dict, current_price: Optional[float]) -> Fundamentals:
    def num(key: str) -> Optional[float]:
        v = info.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    high_52 = num("fiftyTwoWeekHigh")
    low_52 = num("fiftyTwoWeekLow")

    # Where the price sits in its 52-week band, 0 = at the low, 100 = at the high.
    position = None
    if high_52 and low_52 and current_price and high_52 > low_52:
        position = round(((current_price - low_52) / (high_52 - low_52)) * 100, 1)
        position = max(0.0, min(100.0, position))

    # yfinance is inconsistent about whether this is a fraction or a percent.
    div = num("dividendYield")
    if div is not None and div < 1:
        div = div * 100
    if div is not None:
        div = round(div, 2)

    pe = num("trailingPE")

    return Fundamentals(
        market_cap=format_market_cap(num("marketCap")),
        pe_ratio=round(pe, 2) if pe else None,
        fifty_two_week_high=round(high_52, 2) if high_52 else None,
        fifty_two_week_low=round(low_52, 2) if low_52 else None,
        position_in_range=position,
        dividend_yield=div,
        beta=round(num("beta"), 2) if num("beta") else None,
        sector=info.get("sector"),
        industry=info.get("industry"),
    )


# ---------------------------------------------------------------------------
# News sentiment via Stocklake
#
# Used to run a local FinBERT model (torch + transformers — the single
# heaviest thing in this app's deploy image) over yfinance headlines, with
# a hand-rolled regex table guessing WHY each headline mattered. Replaced
# with one call to Stocklake's AI news pipeline (get_stock_news): it fetches
# the news itself, and each article already carries a real sentiment label
# and a generated summary explaining the story — no local inference, no
# pattern-matching guesswork standing in for an actual explanation.
#
# Stocklake is an MCP server, not a plain REST API — there's no bare HTTP
# endpoint to hit, so this speaks the MCP protocol directly (initialize,
# then a single tools/call) using the official SDK. That's a real per-call
# cost (a fresh session handshake every time, not just a GET), so this
# degrades exactly like every other optional integration in this file:
# missing key, timeout, or any failure at all -> the same neutral
# placeholder that used to mean "no headlines," never a broken Analyze page.
#
# Known trade-off, not (yet) fixed: unlike the old local FinBERT call, this
# runs synchronously on analyze()'s own request thread — a slow-but-not-
# failing Stocklake response can occupy one of this app's worker threads
# for close to the full timeout below. The AI Decision feature hit this
# same shape of problem (a slow external AI call blocking the fast, already-
# computed rest of the page) and was split into its own endpoint
# specifically to avoid it — see analyze_decision()'s own docstring. The
# same treatment would fix this properly; not done here since it changes
# the response shape and the frontend's loading state, which is more than
# a like-for-like swap. The timeout below is kept deliberately tight as a
# partial mitigation in the meantime.
# ---------------------------------------------------------------------------
STOCKLAKE_MCP_URL = "https://api.stocklake.dev/mcp"
STOCKLAKE_TIMEOUT_SECONDS = 8

# Stocklake's ai_sentiment values ("POSITIVE"/"NEGATIVE"/"NEUTRAL"/"mixed",
# inconsistently cased) collapse onto the three buckets the rest of this
# app — and the frontend's color lookup — already knows about. "mixed"
# (genuinely two-sided coverage) reads as neutral for the purposes of the
# directional bullish_score below, the same as an article with no clear lean.
_STOCKLAKE_SENTIMENT_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral", "mixed": "neutral"}


async def _stocklake_call_tool(session: ClientSession, tool: str, arguments: dict, timeout: timedelta) -> Any:
    """Calls one Stocklake tool within an already-initialized session and
    returns its parsed JSON payload. Raises on a tool-level error or a
    response with no parseable text content — callers decide how to degrade."""
    result = await session.call_tool(tool, arguments, read_timeout_seconds=timeout)
    if result.isError:
        raise RuntimeError(f"Stocklake tool '{tool}' returned an error for arguments {arguments}")
    for block in result.content:
        if isinstance(block, TextContent):
            return json.loads(block.text)
    raise RuntimeError(f"Stocklake tool '{tool}' returned no parseable content")


async def _stocklake_fetch(api_key: str, calls: Dict[str, Tuple[str, dict]], timeout: timedelta) -> Dict[str, Any]:
    """
    Opens exactly ONE MCP session (one handshake) and runs every call in
    `calls` against it sequentially, returning {key: payload_or_exception}.
    One tool failing doesn't take the others down with it — each result is
    either the parsed payload or the Exception raised trying to get it;
    callers decide per-key how to degrade.

    This is the reason to route every Stocklake integration through here
    rather than each opening its own session: a session handshake is real,
    non-trivial latency (see the trade-off note above), and an analyze()
    call that wants sentiment + earnings + fundamentals from Stocklake
    should pay that handshake cost once, not three times.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    results: Dict[str, Any] = {}
    async with streamablehttp_client(STOCKLAKE_MCP_URL, headers=headers, timeout=timeout) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for key, (tool, arguments) in calls.items():
                try:
                    results[key] = await _stocklake_call_tool(session, tool, arguments, timeout)
                except Exception as e:
                    results[key] = e
    return results


def fetch_stocklake_context(ticker: str, want: Dict[str, Tuple[str, dict]]) -> Dict[str, Any]:
    """
    Sync entrypoint for every Stocklake-backed feature in this file — call
    it once per analyze() request with every tool call you want this time
    (e.g. {"news": ("get_stock_news", {...}), "earnings": ("get_earnings_calendar", {...})})
    and get back {key: payload_or_None} for all of them from a single MCP
    session. A key maps to None if STOCKLAKE_API_KEY isn't configured, the
    whole session failed outright (e.g. connection refused), or that
    specific call raised — never raises itself, so a caller can always just
    check for None and fall back, the same pattern every other optional
    integration in this file already uses.

    Timeout budget scales with how many calls are requested (up to
    STOCKLAKE_TIMEOUT_SECONDS per call, sequential within the one session).
    Deliberately NOT +1 for the handshake on top of that: today's only
    caller (fetch_sentiment) makes exactly one call, and that single-call
    case should stay at exactly STOCKLAKE_TIMEOUT_SECONDS worst case — the
    same tight budget the trade-off note above this section calls out —
    not silently double it while making room for callers that don't exist
    yet.
    """
    if not want:
        return {}
    api_key = os.environ.get("STOCKLAKE_API_KEY")
    if not api_key:
        return {key: None for key in want}

    timeout = timedelta(seconds=STOCKLAKE_TIMEOUT_SECONDS)
    outer_budget = STOCKLAKE_TIMEOUT_SECONDS * len(want)
    try:
        results = asyncio.run(
            asyncio.wait_for(_stocklake_fetch(api_key, want, timeout), timeout=outer_budget)
        )
    except Exception as e:
        logger.info(f"Stocklake session failed for {ticker}: {e}")
        return {key: None for key in want}

    return {key: (None if isinstance(v, Exception) else v) for key, v in results.items()}


def fetch_sentiment(payload: Any) -> SentimentAnalysis:
    """
    Pure mapping — takes whatever get_stock_news already returned (via
    analyze()'s single combined Stocklake call, see the "stock" + "news"
    fetch_stocklake_context call there), rather than fetching it itself.
    This used to open its own dedicated MCP session; now it shares the one
    session analyze() opens for the "stock" (earnings/fundamentals) call
    too — the whole reason fetch_stocklake_context/_stocklake_fetch exist
    is to pay that handshake cost once per request, not once per feature.
    """
    fallback = SentimentAnalysis(
        bullish_score=0.5,
        overall_impact="Neutral (no recent news available)",
        headline_count=0,
        headlines=[],
    )

    # payload is whatever get_stock_news's JSON happened to decode to — a
    # dict on the happy path, but fetch_stocklake_context only guarantees
    # valid JSON, not a particular shape (or that the key was even present,
    # if STOCKLAKE_API_KEY isn't configured or the session failed). A
    # malformed/missing response degrades the same as no news at all,
    # rather than raising out of this function.
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    if not articles:
        return fallback

    scored_headlines = []
    directional_total = 0.0

    for a in articles:
        raw_sentiment = (a.get("ai_sentiment") or "neutral").strip().lower()
        sentiment = _STOCKLAKE_SENTIMENT_MAP.get(raw_sentiment, "neutral")
        # signal_score is Stocklake's own 0-100 "how strong is this idea"
        # magnitude — used here as the per-article weight, the same role
        # FinBERT's classification confidence used to play. `or 50.0` would
        # wrongly treat a genuine, deliberate 0 the same as "missing" (0.0
        # is falsy in Python) — signal_score can legitimately be exactly 0,
        # per Stocklake's own docs, so only a true null/absent value should
        # fall back to the neutral 50 default.
        raw_score = a.get("signal_score")
        magnitude = (raw_score if raw_score is not None else 50.0) / 100.0

        if sentiment == "positive":
            directional_total += magnitude
        elif sentiment == "negative":
            directional_total -= magnitude

        scored_headlines.append(
            Headline(
                title=a.get("title") or "",
                publisher=None,
                link=None,
                sentiment=sentiment,
                sentiment_score=round(magnitude, 3),
                context=a.get("ai_summary") or "",
            )
        )

    avg_directional = directional_total / len(articles)
    bullish_score = round((avg_directional + 1) / 2, 3)

    if bullish_score > 0.6:
        overall_impact = "Positive — recent news is likely to support the price"
    elif bullish_score < 0.4:
        overall_impact = "Negative — recent news may pressure the price"
    else:
        overall_impact = "Neutral — recent news is unlikely to move the price significantly"

    return SentimentAnalysis(
        bullish_score=bullish_score,
        overall_impact=overall_impact,
        headline_count=len(scored_headlines),
        headlines=scored_headlines,
    )


# ---------------------------------------------------------------------------
# Stocklake's per-symbol get_stock — one call, two consumers below
# (fetch_earnings's next_earnings_date, compute_quality_score's
# fundamentals) via analyze()'s own shared fetch, so this only runs once
# per request rather than once per consumer.
# ---------------------------------------------------------------------------
_STOCKLAKE_CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")


def _stocklake_symbol(ticker: str) -> str:
    """
    Stocklake's universe doesn't recognize this app's Canadian exchange
    suffixes — verified live: get_stock("SHOP.TO") 404s, get_stock("SHOP")
    correctly resolves to Shopify's primary listing. Stripping the suffix
    is only safe when paired with the country check in fetch_stocklake_stock
    below — a bare symbol can just as easily collide with an unrelated
    company that happens to share the same letters.
    """
    for suffix in _STOCKLAKE_CA_SUFFIXES:
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def validate_stocklake_stock(ticker: str, symbol: str, data: Any, yf_name: str = "") -> Optional[dict]:
    """
    Pure validator for an already-fetched get_stock() response (see
    analyze()'s single combined Stocklake call — this used to fetch its
    own data via a second, independent MCP session, which defeated the
    entire point of sharing one session per request with the news call).

    Fails closed on Stocklake's own no-data response: get_stock embeds
    "not found" as a normal {"error": {...}} payload rather than an MCP-
    level error (confirmed live — even well-known tickers aren't always in
    Stocklake's ~3,500-symbol universe), so that shape is checked for
    explicitly rather than trusting any dict-shaped response.

    Fails closed on a stripped-suffix lookup, with two independent checks
    since neither alone is conclusive: if get_stock("SHOP") resolves for a
    ticker requested as "SHOP.TO", (1) the response's own `country` field
    must say "Canada" (confirmed live against SHOP's real data), and (2)
    when a yfinance company name is available for cross-check, it must
    share at least one significant word with Stocklake's own `name` field
    — a same-country company that happens to share the stripped ticker
    letters (a real, if narrow, residual risk once the suffix is dropped)
    is still caught by the name mismatch even though the country matches.
    Either check failing treats the lookup as not found rather than
    risking a silent wrong-company substitution.
    """
    if not isinstance(data, dict) or "error" in data:
        return None
    if symbol == ticker:
        return data  # no suffix was stripped, so no cross-company risk to check

    if data.get("country") != "Canada":
        return None

    sl_name = (data.get("name") or "").lower()
    yf_name_l = (yf_name or "").lower()
    if yf_name_l and sl_name:
        strip = str.maketrans("", "", ".,")
        yf_words = {w for w in yf_name_l.translate(strip).split() if len(w) > 2}
        sl_words = {w for w in sl_name.translate(strip).split() if len(w) > 2}
        if yf_words and sl_words and not (yf_words & sl_words):
            return None
    return data


def _merge_stocklake_fundamentals(info: dict, stocklake_data: Optional[dict]) -> dict:
    """
    yfinance's `info` dict is what compute_quality_score was built against
    (and what build_fundamentals also reads), but it's inconsistently
    populated across tickers. This fills in ONLY the specific fields
    compute_quality_score needs when yfinance's own value is missing, from
    an already-fetched Stocklake get_stock() response (see
    fetch_stocklake_stock) — never overrides a real yfinance value that's
    already present, only patches genuine gaps.

    debt_to_equity needs a unit conversion: yfinance expresses it as
    already-scaled percentage points (e.g. 45.3 meaning 45.3%), Stocklake
    as a plain ratio (e.g. 1.403 meaning 140.3%) — confirmed live against
    SHOP's real data. Multiplying by 100 aligns it to the scale
    compute_quality_score's own thresholds (< 50, < 150) are calibrated
    for; profit margin, ROE, and revenue growth are already expressed as
    the same 0-1 fraction convention on both sides, so those pass through
    unconverted.

    A negative debt_to_equity is excluded rather than converted and merged:
    it means negative shareholder equity (a distress signal — accumulated
    deficits or heavy buybacks), but compute_quality_score's `d2e < 50`
    branch would score any negative number as "Low debt" and award full
    points, rewarding the exact pattern it's meant to penalize. Leaving it
    unfilled scores that factor the same as if it were simply unavailable,
    rather than feeding the scorer a number it would misread.
    """
    if not stocklake_data:
        return info
    merged = dict(info)
    field_map = {
        "profitMargins": "profit_margins",
        "returnOnEquity": "return_on_equity",
        "revenueGrowth": "revenue_growth",
    }
    for yf_key, sl_key in field_map.items():
        if merged.get(yf_key) is None and stocklake_data.get(sl_key) is not None:
            merged[yf_key] = stocklake_data[sl_key]
    d2e = stocklake_data.get("debt_to_equity")
    if merged.get("debtToEquity") is None and d2e is not None and d2e >= 0:
        merged["debtToEquity"] = d2e * 100
    return merged


def map_insider_activity(payload: Any, symbol_validated: bool) -> Optional[InsiderActivity]:
    """
    Maps an already-fetched get_insider_activity() response into the app's
    own model. `symbol_validated` should be whether validate_stocklake_stock
    passed for this same ticker's get_stock() call in the same shared
    session — get_insider_activity is looked up by the same
    (possibly-suffix-stripped) symbol, so it carries the same wrong-company
    substitution risk that validator already checked for the stock call;
    reusing that result here avoids running an equivalent check twice.
    """
    if not symbol_validated or not isinstance(payload, dict) or "error" in payload:
        return None
    return InsiderActivity(
        signal=payload.get("signal"),
        signal_score=payload.get("signal_score"),
        signal_score_band=payload.get("signal_score_band"),
        insider_signal=payload.get("insider_signal"),
        institutional_signal=payload.get("inst_signal"),
        summary=payload.get("summary"),
        insider_buys=payload.get("insider_buys"),
        insider_sells=payload.get("insider_sells"),
        institutional_ownership_pct=payload.get("inst_ownership"),
        total_holders=payload.get("total_holders"),
    )


def map_analyst_consensus(stocklake_data: Optional[dict]) -> Optional[AnalystConsensus]:
    """
    Maps the analyst_rating/analyst_rating_score/analyst_target/analyst_count
    fields off an already-validated get_stock() payload — the same call
    validate_stocklake_stock() already checked for wrong-company
    substitution, so no separate check needed here. Returns None rather
    than an all-null consensus when the payload has nothing to show, same
    as every other Stocklake mapper in this app.
    """
    if not stocklake_data:
        return None
    if all(
        stocklake_data.get(k) is None
        for k in ("analyst_rating", "analyst_target", "analyst_count")
    ):
        return None
    return AnalystConsensus(
        rating=stocklake_data.get("analyst_rating"),
        rating_score=stocklake_data.get("analyst_rating_score"),
        target=stocklake_data.get("analyst_target"),
        analyst_count=stocklake_data.get("analyst_count"),
    )


# yfinance's `info["sector"]` doesn't use the same names as Stocklake's
# fixed 11-sector (GICS-style) taxonomy for four of the eleven — the other
# seven pass through unchanged. Passing the yfinance name straight through
# for these four would ask Stocklake for a sector name it doesn't have.
_STOCKLAKE_SECTOR_MAP = {
    "financial services": "Financials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
}


def map_sector_intelligence(payload: Any, sector_name: Optional[str]) -> Optional[SectorIntelligence]:
    """Maps an already-fetched get_sector_intelligence() response into the
    app's own model. Degrades to None on a missing sector name, an error
    payload, or a response for a different sector than requested (the
    single-sector call shouldn't ever return a mismatch, but this is a
    cheap, cheap-to-check guard against silently mislabeling one sector's
    data as another's)."""
    if not sector_name or not isinstance(payload, dict) or "error" in payload:
        return None
    if payload.get("sector") and payload["sector"].lower() != sector_name.lower():
        return None
    stats = payload.get("stats") or {}
    return SectorIntelligence(
        sector=payload.get("sector") or sector_name,
        signal=payload.get("signal"),
        cycle_stage=payload.get("cycle_stage"),
        rotation_signal=payload.get("rotation_signal"),
        drivers=payload.get("drivers"),
        alert=payload.get("alert"),
        confidence=payload.get("confidence"),
        avg_perf_1w_pct=stats.get("avg_perf_1w_pct"),
        avg_perf_1m_pct=stats.get("avg_perf_1m_pct"),
        sma200_breadth_pct=stats.get("sma200_breadth_pct"),
    )


# Stocklake's fixed 11-sector (GICS-style) taxonomy — the same list
# _STOCKLAKE_SECTOR_MAP's docstring above references. Used to scan every
# sector at once for the rotation dashboard (Stocklake-first plan, P5b),
# rather than the single sector a ticker's own ANalyze page happens to ask for.
STOCKLAKE_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Materials",
    "Utilities", "Real Estate", "Communication Services",
]

_SECTOR_ROTATION_CACHE: Tuple[float, Optional[List[dict]]] = (0.0, None)
_SECTOR_ROTATION_TTL_SECONDS = 3600  # matches sector intelligence's own ~4h refresh cadence loosely


def fetch_sector_rotation() -> List[SectorIntelligence]:
    """
    All 11 sectors' current cycle stage/rotation signal in one shared MCP
    session (same reasoning as _stocklake_batch_stocks: one handshake for
    11 sequential get_sector_intelligence calls, not 11 separate ones).
    Cached for an hour — this is a dedicated dashboard page, not a hot
    path like Screener/Brief, so a plain request-time cache (rather than
    the full proactive background-refresh loop those get) is proportionate.
    """
    global _SECTOR_ROTATION_CACHE
    cached_at, cached_data = _SECTOR_ROTATION_CACHE
    if cached_data is not None and time.time() - cached_at < _SECTOR_ROTATION_TTL_SECONDS:
        return [SectorIntelligence(**s) for s in cached_data]

    if not os.environ.get("STOCKLAKE_API_KEY"):
        return []

    want = {sector: ("get_sector_intelligence", {"sector": sector}) for sector in STOCKLAKE_SECTORS}
    context = fetch_stocklake_context("__sector_rotation__", want)

    sectors: List[SectorIntelligence] = []
    for sector in STOCKLAKE_SECTORS:
        mapped = map_sector_intelligence(context.get(sector), sector)
        if mapped is not None:
            sectors.append(mapped)

    if sectors:
        _SECTOR_ROTATION_CACHE = (time.time(), [s.model_dump() for s in sectors])
    return sectors


# ---------------------------------------------------------------------------
# Batched multi-ticker scoring
#
# yf.download() pulls many tickers in a SINGLE request, which is the only
# practical way to score dozens of names without tripping Yahoo's rate limits.
# Technicals only here — running FinBERT across 37 tickers' worth of headlines
# would be far too slow and memory-hungry for a per-request call.
# ---------------------------------------------------------------------------
def _cap_conviction_for_liquidity(avg_vol: Optional[float], conviction: str) -> Tuple[str, Optional[str]]:
    """
    Shared thresholds behind the batch-scoring reliability cap below, used
    both by the yfinance path (average pulled from its own OHLCV history)
    and the Stocklake path (average taken directly from its own avg_volume
    field) — kept as one function so the two data sources can never drift
    to different "thin trading" cutoffs.

    Returns (possibly-capped conviction, a short note if it was capped).
    """
    if conviction == "Neutral":
        return conviction, None
    if avg_vol is None or math.isnan(avg_vol):
        if conviction != "Low":
            return "Low", "Conviction capped: volume data unavailable, so liquidity can't be assessed."
        return conviction, None

    if avg_vol < 50_000:
        if conviction != "Low":
            return "Low", f"Conviction capped: ~{avg_vol:,.0f} shares/day is very thin trading."
        return conviction, None
    if avg_vol < 250_000 and conviction == "High":
        return "Moderate", f"Conviction capped: ~{avg_vol:,.0f} shares/day is light trading."
    return conviction, None


def _batch_reliability_cap(df: pd.DataFrame, conviction: str) -> Tuple[str, Optional[str]]:
    """
    A cheap proxy for assess_data_quality() — full reliability scoring needs
    `info` (for the derivative/ETF check) and a staleness scan, neither of
    which batch scoring can afford per-ticker across dozens of tickers in
    one request. This uses only what's already in `df` from the same
    yf.download() call: 60-day average volume. Deliberately more
    conservative than §4's exact point thresholds, since it's the only
    signal this path has — a thin name here gets capped without the
    derivative/staleness checks that could otherwise offset it.

    Returns (possibly-capped conviction, a short note if it was capped).
    """
    if conviction == "Neutral" or "Volume" not in df or len(df) < 20:
        return conviction, None
    try:
        avg_vol = float(df["Volume"].tail(60).mean())
    except Exception:
        return conviction, None
    return _cap_conviction_for_liquidity(avg_vol, conviction)


def _score_history(t: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Shared technical-scoring logic for one ticker's OHLCV history, used by
    both the batched pass and the single-ticker retry below. Returns None
    (rather than an error dict) only when there's truly no usable data —
    the caller decides what that means in context.

    Below 30 rows this degrades rather than returns None: RSI/MACD come back
    None (same graceful path as compute_technicals) and the caller still
    gets a price to show current-value-vs-cost-basis for a recently listed
    holding, rather than a hard "not enough history" error blocking the row."""
    df = df.dropna(how="all")
    if df.empty:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    rsi_series = compute_rsi(close, length=14).dropna()
    _, _, histogram = compute_macd(close)
    hist_clean = histogram.dropna()

    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
    macd_hist_val = float(hist_clean.iloc[-1]) if not hist_clean.empty else None
    volume_ratio = compute_volume_ratio(df)

    signal, conviction, reasoning = generate_signal(rsi_val, macd_hist_val, volume_ratio)
    conviction, cap_note = _batch_reliability_cap(df, conviction)
    if cap_note:
        reasoning += " " + cap_note
    if len(df) < 30:
        reasoning = f"Only {len(df)} trading day(s) available, likely a recently listed security. " + reasoning

    price = round(float(close.iloc[-1]), 2)
    change_pct = None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev:
            change_pct = round((price - prev) / prev * 100, 2)

    return {
        "ticker": t,
        "price": price,
        "change_pct": change_pct,
        "rsi": round(rsi_val, 2) if rsi_val is not None else None,
        "macd_histogram": round(macd_hist_val, 4) if macd_hist_val is not None else None,
        "volume_ratio": volume_ratio,
        "signal": signal,
        "conviction": conviction,
        "reasoning": reasoning,
    }


BATCH_DOWNLOAD_TIMEOUT_SECONDS = 30
SINGLE_TICKER_RETRY_TIMEOUT_SECONDS = 15
STOCKLAKE_BATCH_CHUNK_SIZE = 25  # get_stocks' own hard per-call limit


def _score_from_stocklake(ticker: str, data: dict) -> Optional[Dict[str, Any]]:
    """
    Maps one get_stocks() entry onto this app's OWN generate_signal() rule
    — same RSI/MACD-histogram thresholds as the yfinance path, applied to
    Stocklake's precomputed indicators instead. Deliberately not Stocklake's
    own `rating`/`signals.overall`/`ai_verdict` fields: this keeps the BUY/
    SELL/HOLD call the same mechanical, backtested rule regardless of which
    data source served it, rather than quietly swapping in an unvalidated
    third-party verdict. See DECISION_LOGIC.md §16 on why Stocklake's richer
    signals stay informational elsewhere in this app rather than live inputs.
    """
    price = data.get("price")
    if price is None:
        return None

    indicators = data.get("indicators") or {}
    rsi_val = indicators.get("rsi")
    macd_hist_val = (indicators.get("macd") or {}).get("histogram")
    volume = data.get("volume")
    avg_volume = data.get("avg_volume")
    volume_ratio = (
        round(volume / avg_volume, 2) if volume is not None and avg_volume else None
    )

    signal, conviction, reasoning = generate_signal(rsi_val, macd_hist_val, volume_ratio)
    conviction, cap_note = _cap_conviction_for_liquidity(avg_volume, conviction)
    if cap_note:
        reasoning += " " + cap_note

    change_pct = data.get("change_pct")
    return {
        "ticker": ticker,
        "price": round(float(price), 2),
        "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
        "rsi": round(float(rsi_val), 2) if rsi_val is not None else None,
        "macd_histogram": round(float(macd_hist_val), 4) if macd_hist_val is not None else None,
        "volume_ratio": volume_ratio,
        "signal": signal,
        "conviction": conviction,
        "reasoning": reasoning,
    }


def _stocklake_batch_stocks(tickers: List[str]) -> Dict[str, dict]:
    """
    Raw get_stocks() payload for a list of tickers, chunked to Stocklake's
    25-symbol batch cap with every chunk sharing ONE MCP session (same
    reasoning as fetch_stocklake_context's own docstring: a full-universe
    scan can need 15+ chunked calls, and paying a separate handshake per
    chunk would be slow and reintroduce the kind of unbounded-latency risk
    the Stocklake-first plan exists to route around).

    Skips any ticker whose Stocklake lookup symbol differs from the ticker
    itself (i.e. suffix-stripped Canadian listings like .TO/.V/.CN/.NE) —
    validating that a stripped-suffix lookup is actually the same company
    needs the country/name check analyze() does per-ticker (see
    validate_stocklake_stock), which isn't affordable at batch scale. Used
    by both stocklake_batch_score() (technicals for the signal engine) and
    the earnings-calendar endpoint (each ticker's own earnings_date field) —
    one shared, verified fetch path instead of two.
    """
    if not os.environ.get("STOCKLAKE_API_KEY"):
        return {}

    direct_tickers = [t for t in tickers if _stocklake_symbol(t) == t]
    if not direct_tickers:
        return {}

    chunks = [
        direct_tickers[i : i + STOCKLAKE_BATCH_CHUNK_SIZE]
        for i in range(0, len(direct_tickers), STOCKLAKE_BATCH_CHUNK_SIZE)
    ]
    want = {f"chunk_{i}": ("get_stocks", {"symbols": chunk}) for i, chunk in enumerate(chunks)}
    context = fetch_stocklake_context("__batch_stocks__", want)

    results: Dict[str, dict] = {}
    direct_set = set(direct_tickers)
    for key in want:
        payload = context.get(key)
        if not isinstance(payload, dict) or "error" in payload:
            continue
        for symbol, data in (payload.get("symbols") or {}).items():
            ticker = symbol.upper()
            if ticker not in direct_set or not isinstance(data, dict):
                continue
            results[ticker] = data
    return results


def stocklake_batch_score(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Primary path for batch_score() below: Stocklake's precomputed
    fundamentals+indicators (get_stocks), scored through this app's own
    generate_signal() rule rather than trusting Stocklake's own verdict —
    added after yfinance was observed rate-limited/blocked from this
    deployment for 30+ minutes straight on 2026-08-28, taking Brief and
    Screener down with it while every Stocklake call kept succeeding.

    Anything _stocklake_batch_stocks() skipped or Stocklake's own batch
    call didn't return is left for batch_score()'s yfinance fallback below
    instead of risking a silently wrong company's data on a BUY/SELL
    screener row.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for ticker, data in _stocklake_batch_stocks(tickers).items():
        scored = _score_from_stocklake(ticker, data)
        if scored is not None:
            results[ticker] = scored
    return results


def batch_score(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return {}

    # Stocklake first (see stocklake_batch_score's docstring for why) —
    # whatever it can't or shouldn't cover falls through to the yfinance
    # path below unchanged, so this degrades exactly like every other
    # Stocklake integration in this app: no key, an error, or a skipped
    # ticker all just mean "use the fallback," never a broken response.
    results: Dict[str, Dict[str, Any]] = dict(stocklake_batch_score(tickers))
    tickers = [t for t in tickers if t not in results]
    if not tickers:
        return results

    # yf.download()'s own per-request timeout only bounds a single HTTP
    # call — when Yahoo stalls instead of erroring (seen in production as
    # rate-limit/crumb failures around the same window), yfinance's internal
    # retries can still sit well past that. Wrapping it in its own executor
    # with a hard wall-clock cap turns a request that would otherwise hang
    # indefinitely (worst for /api/digest, which calls this up to 4x in one
    # HTTP request) into a fast, clear per-ticker error instead. Deliberately
    # NOT a `with ThreadPoolExecutor(...) as guard:` block — that waits for
    # the submitted task to finish on exit regardless of whether .result()
    # already timed out, which would silently undo the timeout entirely.
    guard = ThreadPoolExecutor(max_workers=1)
    try:
        raw = guard.submit(
            yf.download,
            tickers,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        ).result(timeout=BATCH_DOWNLOAD_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        logger.error(f"Batch download timed out after {BATCH_DOWNLOAD_TIMEOUT_SECONDS}s for {len(tickers)} tickers")
        results.update({t: {"error": "Market data is slow to respond right now — try again shortly."} for t in tickers})
        return results
    except Exception as e:
        logger.error(f"Batch download failed: {e}")
        results.update({t: {"error": "Could not fetch market data."} for t in tickers})
        return results
    finally:
        guard.shutdown(wait=False)

    if raw is None or raw.empty:
        results.update({t: {"error": "No market data returned."} for t in tickers})
        return results

    for t in tickers:
        try:
            # yf.download() always returns a per-ticker column level (a
            # MultiIndex keyed by ticker), even when only one ticker was
            # requested, since multi_level_index isn't set to False above.
            if t not in raw.columns.get_level_values(0):
                results[t] = {"error": "No data found for this ticker."}
                continue
            scored = _score_history(t, raw[t])
            results[t] = scored if scored is not None else {"error": "Not enough price history."}
        except Exception as e:
            logger.warning(f"Scoring failed for {t}: {e}")
            results[t] = {"error": "Could not analyze this ticker."}

    # The batched multi-ticker download sometimes comes back sparse for
    # thinly-traded listings — CAD-hedged CDRs on NEO (.NE) in particular —
    # even when Yahoo has plenty of history for them individually via a
    # single-ticker request. Retry the failures with a longer lookback
    # before giving up, concurrently rather than one at a time (this runs on
    # every uncached call — /api/digest's watchlist/scan-list scoring and
    # portfolio analysis both hit batch_score() live, not just the cached
    # screener), and capped so a large batch of persistent failures can't
    # turn into dozens of extra round trips on a single request.
    failed = [t for t in tickers if results.get(t, {}).get("error")][:15]

    def retry_one(t: str):
        hist = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True)
        return _score_history(t, hist)

    if failed:
        # Same reasoning as the guard above: a plain `with ... as pool:`
        # blocks on exit until every submitted future finishes, which would
        # silently cancel out the per-future timeout below the moment one
        # retry stalls. shutdown(wait=False) lets the function return as
        # soon as the last non-stalled future is collected.
        pool = ThreadPoolExecutor(max_workers=min(len(failed), 8))
        try:
            future_to_ticker = {pool.submit(retry_one, t): t for t in failed}
            for future in future_to_ticker:
                t = future_to_ticker[future]
                try:
                    scored = future.result(timeout=SINGLE_TICKER_RETRY_TIMEOUT_SECONDS)
                    if scored is not None:
                        results[t] = scored
                except Exception as e:
                    logger.info(f"Single-ticker retry failed for {t}: {e}")
        finally:
            pool.shutdown(wait=False)

    return results


CONVICTION_RANK = {"High": 3, "Moderate": 2, "Low": 1, "Neutral": 0}


def sort_hits(hits: List[ScreenerHit]) -> List[ScreenerHit]:
    """Strongest conviction first; RSI as a tiebreak toward the more stretched names."""
    return sorted(
        hits,
        key=lambda h: (CONVICTION_RANK.get(h.conviction, 0), abs((h.rsi or 50) - 50)),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Stocklake-first plan, P1: request handlers below read ONLY from these
# caches — run_screener()/run_under20_screener()/run_high_volume_screener()
# with force=True (the actual live fetch) are called exclusively by the
# background refresh loop now. A request can still ask for `force=true`,
# but that only nudges an out-of-band refresh along in the background and
# still returns whatever's cached right now — it can no longer be the thing
# that blocks on a live vendor call.
# ---------------------------------------------------------------------------
_MANUAL_REFRESH_LOCKS: Dict[str, threading.Lock] = {
    "screener": threading.Lock(),
    "under20": threading.Lock(),
    "high_volume": threading.Lock(),
}


def _kick_background_refresh(key: str, fn) -> None:
    """Fire-and-forget refresh for a user's force=true request. Guarded so a
    burst of these (or one arriving mid-cycle of the scheduled background
    refresh) can't pile up concurrent live fetches against the same cache —
    a kick that can't get the lock is just a no-op; that request's response
    still reflects whichever refresh is already in flight once it lands."""
    lock = _MANUAL_REFRESH_LOCKS[key]
    if not lock.acquire(blocking=False):
        return

    def run():
        try:
            fn()
        except Exception as e:
            logger.error(f"Manual refresh kick failed ({key}): {e}")
        finally:
            lock.release()

    threading.Thread(target=run, daemon=True, name=f"refresh-kick-{key}").start()


def screener_from_cache(force: bool = False) -> ScreenerResponse:
    if force:
        _kick_background_refresh("screener", lambda: run_screener(force=True))
    cached = _SCREENER_CACHE.get("universe")
    if cached:
        return ScreenerResponse(**{**cached[1], "cached": True})
    return ScreenerResponse(
        buy_candidates=[], sell_candidates=[], scanned_count=0,
        universe_note="Still warming up — the first scan hasn't finished yet. Try again in a moment.",
        generated_at=datetime.now(timezone.utc).isoformat(), cached=False,
    )


def under20_from_cache(force: bool = False) -> ScreenerResponse:
    if force:
        _kick_background_refresh("under20", lambda: run_under20_screener(force=True))
    cached = _UNDER20_CACHE.get("under20")
    if cached:
        return ScreenerResponse(**{**cached[1], "cached": True})
    return ScreenerResponse(
        buy_candidates=[], sell_candidates=[], scanned_count=0,
        universe_note="Still warming up — the first scan hasn't finished yet. Try again in a moment.",
        generated_at=datetime.now(timezone.utc).isoformat(), cached=False,
    )


def high_volume_from_cache(force: bool = False) -> HighVolumeResponse:
    if force:
        _kick_background_refresh("high_volume", lambda: run_high_volume_screener(force=True))
    cached = _HIGH_VOLUME_CACHE.get("high_volume")
    if cached:
        return HighVolumeResponse(**{**cached[1], "cached": True})
    return HighVolumeResponse(
        stocks=[],
        universe_note="Still warming up — the first scan hasn't finished yet. Try again in a moment.",
        generated_at=datetime.now(timezone.utc).isoformat(), cached=False,
    )


def run_screener(force: bool = False) -> ScreenerResponse:
    if not force:
        cached = _SCREENER_CACHE.get("universe")
        if cached and time.time() - cached[0] < _SCREENER_TTL_SECONDS:
            payload = cached[1]
            return ScreenerResponse(**{**payload, "cached": True})

    scored = batch_score(SCREENER_UNIVERSE)

    buys, sells = [], []
    for t, r in scored.items():
        if r.get("error"):
            continue
        hit = ScreenerHit(**r)
        if hit.signal == "BUY":
            buys.append(hit)
        elif hit.signal == "SELL":
            sells.append(hit)

    response = ScreenerResponse(
        buy_candidates=sort_hits(buys),
        sell_candidates=sort_hits(sells),
        scanned_count=len([r for r in scored.values() if not r.get("error")]),
        universe_note=(
            f"Scanned a fixed list of {len(SCREENER_UNIVERSE)} liquid US and Canadian "
            "large-caps — not the entire market. These are mechanical RSI/MACD signal "
            "hits, not vetted recommendations, and no news sentiment is included here."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )

    _SCREENER_CACHE["universe"] = (time.time(), response.model_dump())
    return response


def run_under20_screener(force: bool = False) -> ScreenerResponse:
    if not force:
        cached = _UNDER20_CACHE.get("under20")
        if cached and time.time() - cached[0] < _UNDER20_TTL_SECONDS:
            payload = cached[1]
            return ScreenerResponse(**{**payload, "cached": True})

    scored = batch_score(CANADIAN_UNDER_20_UNIVERSE)

    buys, sells = [], []
    for t, r in scored.items():
        if r.get("error") or r.get("price") is None or r["price"] >= 20:
            continue
        hit = ScreenerHit(**r)
        if hit.signal == "BUY":
            buys.append(hit)
        elif hit.signal == "SELL":
            sells.append(hit)

    response = ScreenerResponse(
        buy_candidates=sort_hits(buys),
        sell_candidates=sort_hits(sells),
        scanned_count=len([r for r in scored.values() if not r.get("error")]),
        universe_note=(
            f"Scanned a fixed list of {len(CANADIAN_UNDER_20_UNIVERSE)} TSX/TSX-V names "
            "commonly priced under CAD $20 — not the whole exchange. 'Avoid' here means "
            "negative momentum right now, not a judgment on the business. Low-priced "
            "stocks are often more volatile; size positions accordingly."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )

    _UNDER20_CACHE["under20"] = (time.time(), response.model_dump())
    return response


def fetch_high_volume_tickers(us_count: int = 15, ca_count: int = 5) -> List[str]:
    """
    Today's actual highest-volume tickers, live from Yahoo — both US and
    Canadian markets, deliberately NOT limited to SCREENER_UNIVERSE, so names
    outside this app's fixed scan list still show up here when they're
    genuinely trading heavy volume today.

    Two separate queries, not one combined sort: Yahoo's predefined
    "most_actives" screen only covers US-region tickers, and US mega-cap
    share volume (hundreds of millions of shares/day) would swamp every
    Canadian name if ranked together on raw share volume. The Canadian
    query uses its own, much lower volume/market-cap thresholds — TSX/TSX-V
    liquidity is a different scale entirely.
    """
    def fetch_us() -> List[str]:
        result = yf.screen("most_actives", count=us_count)
        quotes = (result or {}).get("quotes", [])
        # yf.screen() internally defaults 'count' to 25 for custom queries
        # regardless of what's passed (see fetch_ca below) — slice locally
        # rather than trust the API to honor the requested size.
        return [q["symbol"] for q in quotes if q.get("symbol")][:us_count]

    def fetch_ca() -> List[str]:
        ca_query = yf.EquityQuery("and", [
            yf.EquityQuery("eq", ["region", "ca"]),
            yf.EquityQuery("gte", ["intradaymarketcap", 300_000_000]),
            yf.EquityQuery("gt", ["dayvolume", 300_000]),
        ])
        # Custom (non-predefined) queries are documented to take `size`, but
        # yf.screen() still fills an unset `count` with its own default (25)
        # and sends both fields — pass count too and slice locally either way,
        # since which one Yahoo's endpoint actually honors isn't documented.
        result = yf.screen(ca_query, sortField="dayvolume", sortAsc=False, size=ca_count, count=ca_count)
        quotes = (result or {}).get("quotes", [])
        return [q["symbol"] for q in quotes if q.get("symbol")][:ca_count]

    tickers: List[str] = []
    # Same reasoning as batch_score()'s guard: no `with ... as pool:` here,
    # since that blocks on exit until both futures finish regardless of
    # whether .result() already timed out — this call now feeds the
    # background refresh loop, and a stalled yf.screen() must not be able to
    # wedge that loop for good.
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        us_future = pool.submit(fetch_us)
        ca_future = pool.submit(fetch_ca)
        try:
            tickers.extend(us_future.result(timeout=BATCH_DOWNLOAD_TIMEOUT_SECONDS))
        except Exception as e:
            logger.warning(f"US high-volume screen fetch failed: {e}")
        try:
            tickers.extend(ca_future.result(timeout=BATCH_DOWNLOAD_TIMEOUT_SECONDS))
        except Exception as e:
            logger.warning(f"Canadian high-volume screen fetch failed: {e}")
    finally:
        pool.shutdown(wait=False)

    return tickers


def run_high_volume_screener(force: bool = False) -> HighVolumeResponse:
    if not force:
        cached = _HIGH_VOLUME_CACHE.get("high_volume")
        if cached and time.time() - cached[0] < _HIGH_VOLUME_TTL_SECONDS:
            payload = cached[1]
            return HighVolumeResponse(**{**payload, "cached": True})

    tickers = fetch_high_volume_tickers()
    if not tickers:
        return HighVolumeResponse(
            stocks=[],
            universe_note="Could not fetch today's high-volume list from Yahoo right now — try again shortly.",
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )

    scored = batch_score(tickers)
    # Preserve Yahoo's own volume-descending order rather than re-sorting by
    # conviction — the point of this panel is the volume ranking itself.
    stocks = [
        ScreenerHit(**scored[t])
        for t in tickers
        if scored.get(t) and not scored[t].get("error") and scored[t].get("price") is not None
    ]

    if not stocks:
        # batch_score() failed for every ticker (rate limit / network blip on
        # the price-history call, separate from the screen call above) —
        # don't cache an empty "success" that would hide the failure for the
        # full TTL, same as the empty-ticker-list guard above.
        return HighVolumeResponse(
            stocks=[],
            universe_note="Could not score today's high-volume tickers right now — try again shortly.",
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )

    # One region's fetch can fail while the other succeeds (fetch_high_volume_tickers
    # logs but doesn't raise) — say so rather than claiming full US+CA coverage
    # when the result is actually one-sided.
    has_us = any(not t.endswith((".TO", ".V", ".CN", ".NE")) for t in tickers)
    has_ca = any(t.endswith((".TO", ".V", ".CN", ".NE")) for t in tickers)
    if has_us and has_ca:
        markets_note = "Today's highest-volume US and Canadian tickers"
    elif has_us:
        markets_note = "Today's highest-volume US tickers (Canadian data was unavailable this refresh)"
    else:
        markets_note = "Today's highest-volume Canadian tickers (US data was unavailable this refresh)"

    response = HighVolumeResponse(
        stocks=stocks,
        universe_note=(
            f"{markets_note} on Yahoo Finance, refreshed live every 15 minutes — not the "
            "fixed scan list used elsewhere, so names can appear here even if you've never "
            "added or scanned them before."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )
    _HIGH_VOLUME_CACHE["high_volume"] = (time.time(), response.model_dump())
    return response


def describe_holding_context(pl_pct: Optional[float], signal: str) -> str:
    """
    Frames what the indicators say against the position's P/L, without telling
    the person what to do — the decision depends on their goals and tax
    situation, which this app knows nothing about.
    """
    if pl_pct is None:
        return "Position data incomplete."

    if signal == "SELL" and pl_pct > 0:
        return f"Up {pl_pct}% and momentum indicators have turned negative — worth reviewing."
    if signal == "SELL" and pl_pct <= 0:
        return f"Down {abs(pl_pct)}% with negative momentum — indicators aren't showing a turnaround yet."
    if signal == "BUY" and pl_pct > 0:
        return f"Up {pl_pct}% and momentum is still positive."
    if signal == "BUY" and pl_pct <= 0:
        return f"Down {abs(pl_pct)}% but momentum indicators have turned positive."
    if pl_pct > 0:
        return f"Up {pl_pct}%, indicators neutral — no momentum signal either way."
    return f"Down {abs(pl_pct)}%, indicators neutral — no momentum signal either way."


def get_usd_cad_rate() -> Optional[float]:
    """
    Live USD→CAD rate via Yahoo's CAD=X pair, cached for an hour. Falls back
    to the last known rate if a fresh fetch fails, rather than breaking the
    whole portfolio view over a single failed FX lookup.
    """
    cached = _FX_CACHE.get("USDCAD")
    if cached and time.time() - cached[0] < _FX_TTL_SECONDS:
        return cached[1]
    try:
        hist = yf.Ticker("CAD=X").history(period="5d", interval="1d")
        rate = float(hist["Close"].dropna().iloc[-1]) if hist is not None and not hist.empty else None
        if rate:
            _FX_CACHE["USDCAD"] = (time.time(), rate)
            return rate
    except Exception as e:
        logger.warning(f"FX rate fetch failed: {e}")
    return cached[1] if cached else None


def analyze_portfolio(holdings: List[PortfolioHolding]) -> PortfolioResponse:
    if not holdings:
        return PortfolioResponse(
            holdings=[], total_cost=0, total_value=0, total_pl=0, total_pl_pct=0,
            total_day_pl=0, total_day_pl_pct=0, usd_cad_rate=None,
            summary="No holdings added yet.",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    tickers = [h.ticker.strip().upper() for h in holdings]
    scored = batch_score(tickers)
    usd_cad_rate = get_usd_cad_rate()

    results: List[HoldingResult] = []
    # These totals are normalized to CAD so a mixed US/Canadian portfolio adds
    # up correctly. Each individual holding below still shows its own native
    # currency (USD price stays in USD) — only the combined totals convert.
    total_cost = total_value = 0.0
    total_day_pl_cad = 0.0

    for h in holdings:
        t = h.ticker.strip().upper()
        r = scored.get(t, {})
        currency = "CAD" if t.endswith((".TO", ".V", ".CN", ".NE")) else "USD"
        fx = usd_cad_rate if (currency == "USD" and usd_cad_rate) else 1.0

        cost_native = h.shares * h.cost_basis
        total_cost += cost_native * fx

        if r.get("error") or r.get("price") is None:
            results.append(
                HoldingResult(
                    ticker=t, shares=h.shares, cost_basis=h.cost_basis, account=h.account,
                    current_price=None, currency=currency, market_value=None,
                    unrealized_pl=None, unrealized_pl_pct=None,
                    day_pl=None, day_pl_pct=None,
                    rsi=None, macd_histogram=None, volume_ratio=None,
                    signal="HOLD", conviction="Neutral",
                    reasoning="Could not analyze this holding.",
                    context="Data unavailable.",
                    error=r.get("error", "No data."),
                )
            )
            continue

        price = r["price"]
        value_native = h.shares * price
        total_value += value_native * fx
        pl_native = value_native - cost_native
        pl_pct = round((pl_native / cost_native) * 100, 2) if cost_native else None

        change_pct = r.get("change_pct")
        day_pl_native = None
        if change_pct is not None and change_pct > -100 and price:
            prev_close = price / (1 + change_pct / 100)
            day_pl_native = round(h.shares * (price - prev_close), 2)
            total_day_pl_cad += day_pl_native * fx

        results.append(
            HoldingResult(
                ticker=t, shares=h.shares, cost_basis=h.cost_basis, account=h.account,
                current_price=price, currency=currency,
                market_value=round(value_native, 2),
                unrealized_pl=round(pl_native, 2), unrealized_pl_pct=pl_pct,
                day_pl=day_pl_native, day_pl_pct=change_pct,
                rsi=r.get("rsi"), macd_histogram=r.get("macd_histogram"),
                volume_ratio=r.get("volume_ratio"),
                signal=r.get("signal", "HOLD"),
                conviction=r.get("conviction", "Neutral"),
                reasoning=r.get("reasoning", ""),
                context=describe_holding_context(pl_pct, r.get("signal", "HOLD")),
            )
        )

    total_pl = total_value - total_cost
    total_pl_pct = round((total_pl / total_cost) * 100, 2) if total_cost else 0.0

    total_day_pl = round(total_day_pl_cad, 2)
    total_value_yesterday = total_value - total_day_pl
    total_day_pl_pct = round((total_day_pl / total_value_yesterday) * 100, 2) if total_value_yesterday else 0.0

    sells = [r.ticker for r in results if r.signal == "SELL"]
    buys = [r.ticker for r in results if r.signal == "BUY"]

    parts = []
    if sells:
        parts.append(f"{len(sells)} holding(s) showing bearish momentum: {', '.join(sells)}")
    if buys:
        parts.append(f"{len(buys)} showing bullish momentum: {', '.join(buys)}")
    if not parts:
        parts.append("No strong momentum signals across your holdings right now")

    # Sector lookup for concentration analysis. Best-effort — a failure here
    # shouldn't take down the whole portfolio view.
    sectors: Dict[str, str] = {}
    for r in results:
        if r.error:
            continue
        try:
            sectors[r.ticker] = (yf.Ticker(r.ticker).info or {}).get("sector") or "Unknown"
        except Exception:
            sectors[r.ticker] = "Unknown"

    try:
        concentration = analyze_concentration(results, sectors)
    except Exception as e:
        logger.info(f"Concentration analysis skipped: {e}")
        concentration = None

    return PortfolioResponse(
        holdings=results,
        concentration=concentration,
        total_cost=round(total_cost, 2),
        total_value=round(total_value, 2),
        total_pl=round(total_pl, 2),
        total_pl_pct=total_pl_pct,
        total_day_pl=total_day_pl,
        total_day_pl_pct=total_day_pl_pct,
        usd_cad_rate=round(usd_cad_rate, 4) if usd_cad_rate else None,
        summary=". ".join(parts) + ".",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Symbol search — lets people type a company name ("Apple") instead of
# needing to already know the ticker ("AAPL"). Uses yfinance's own Search
# module rather than a separate paid API. Cached, since a name-to-symbol
# mapping essentially never changes.
# ---------------------------------------------------------------------------
def search_symbols(query: str) -> List[SymbolMatch]:
    key = query.strip().lower()
    if not key:
        return []

    cached = _SEARCH_CACHE.get(key)
    if cached and time.time() - cached[0] < _SEARCH_CACHE_TTL_SECONDS:
        raw = cached[1]
    else:
        try:
            raw = yf.Search(query.strip(), max_results=8).quotes or []
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            raw = []
        _SEARCH_CACHE[key] = (time.time(), raw)
        if len(_SEARCH_CACHE) > 300:
            oldest = min(_SEARCH_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _SEARCH_CACHE.pop(oldest, None)

    matches = []
    for item in raw:
        symbol = item.get("symbol")
        if not symbol:
            continue
        quote_type = (item.get("quoteType") or "").upper()
        # Stick to things this app can actually analyze.
        if quote_type not in ("EQUITY", "ETF", ""):
            continue
        name = item.get("shortname") or item.get("longname") or symbol
        matches.append(
            SymbolMatch(
                symbol=symbol,
                name=name,
                exchange=item.get("exchange"),
                quote_type=quote_type or None,
            )
        )
    return matches


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Serves the web app. /health remains the machine-readable status check."""
    web_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(web_path):
        return FileResponse(web_path)
    return {"status": "ok", "service": "Stock Market Analysis API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "sentiment_source_configured": bool(os.environ.get("STOCKLAKE_API_KEY")),
        "cached_tickers": len(_CACHE),
    }


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = ""):
    """Look up tickers by company name or partial symbol, e.g. 'Apple' -> AAPL."""
    q = q.strip()
    if len(q) < 2:
        return SearchResponse(query=q, results=[])
    return SearchResponse(query=q, results=search_symbols(q))


AI_DECISION_MODEL = "claude-sonnet-5"
_DECISION_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class AIDecisionOutput(BaseModel):
    """
    Strict schema for the structured-output request itself (all fields
    required, so the response always parses cleanly) — not the API response
    model. See AIDecisionResponse below for what /api/analyze/{ticker}/decision
    actually returns, which is nullable throughout for the disabled/failed case.
    """
    conviction_score: int = Field(ge=0, le=100, description=(
        "0-100. Weigh the technical signal, its own historical track record on "
        "THIS ticker, fundamentals, quality score, market regime, earnings timing, "
        "and news sentiment together into one number. Don't default to 50 out of "
        "caution — commit to a real number reflecting the actual weight of evidence. "
        "A signal with a weak/small-sample track record, or one fighting the market "
        "regime, or contradicted by sentiment, should score meaningfully lower than "
        "one where everything agrees and the track record is real."
    ))
    conviction_label: str = Field(description="2-4 words, e.g. 'Strong setup', 'Mixed signals', 'Weak case', 'High risk'.")
    entry: Optional[float] = Field(description=(
        "Suggested entry price, adjusted from mechanical_price_targets.entry only if "
        "something in the context gives a specific reason to. mechanical_price_targets.entry "
        "is null whenever the signal is HOLD/SELL with no suggested entry — in that case "
        "you MUST return null here too. Also null whenever existing_position is present "
        "(the person already holds shares — there's no new-entry question). Never invent a "
        "price level with no baseline to anchor to."
    ))
    stop_loss: Optional[float] = Field(description=(
        "Suggested stop-loss price, same rule as entry: adjust mechanical_price_targets."
        "stop_loss (e.g. widen ahead of earnings, tighten if data reliability is poor) "
        "or return it unchanged — but null if the baseline is null."
    ))
    take_profit: Optional[float] = Field(description=(
        "Suggested take-profit price, same rule as entry: adjust mechanical_price_targets."
        "take_profit (e.g. pull it inside a nearby resistance/support level) or return it "
        "unchanged — but null if the baseline is null."
    ))
    risk_reward_ratio: Optional[float] = Field(description=(
        "Recomputed from YOUR final entry/stop_loss/take_profit above, not copied from the "
        "baseline. Null if those are null."
    ))
    rationale: str = Field(description=(
        "One sentence, under 20 words: the single biggest factor behind the score, "
        "and 'kept mechanical baseline' or a note on whatever price adjustment was made "
        "(or that no price levels apply, if entry is null)."
    ))


def generate_ai_decision(
    ticker: str,
    company_name: str,
    current_price: Optional[float],
    price_change_pct: Optional[float],
    currency: str,
    technical: TechnicalAnalysis,
    fundamentals: Fundamentals,
    quality: QualityScore,
    data_quality: DataQuality,
    regime: MarketRegime,
    earnings: EarningsInfo,
    price_targets: PriceTargets,
    sentiment: SentimentAnalysis,
    insider: Optional[InsiderActivity] = None,
    research: Optional[dict] = None,
    held_shares: Optional[float] = None,
    held_cost_basis: Optional[float] = None,
    history: Optional[List[DecisionHistoryEntry]] = None,
) -> Optional[AIDecisionOutput]:
    """
    The one place in the app that actually reasons across everything it
    already knows about a stock — technicals, fundamentals, quality, market
    regime, earnings timing, news sentiment, and (the part no dashboard card
    can show on its own) whether this exact signal has actually had an edge
    on THIS ticker historically. Returns a decision artifact (a conviction
    score plus concrete price levels) instead of prose to read — see
    AIDecisionOutput. Called from the separate /decision endpoint below (see
    _DECISION_CACHE), not inline in /api/analyze, so a slow Claude call never
    blocks data that's already computed.

    held_shares/held_cost_basis switch the question from "is this worth a
    fresh entry" to "is this worth continuing to hold" — see the system
    prompt. history closes the loop on the model's own past calls for this
    ticker: whether they'd have hit take-profit or stop-loss by now.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    is_held = held_shares is not None and held_cost_basis is not None
    payload = {
        "ticker": ticker,
        "company_name": company_name,
        "price": current_price,
        "price_change_pct": price_change_pct,
        "currency": currency,
        "technical_signal": {
            "signal": technical.signal,
            "conviction": technical.conviction,
            "rsi": technical.rsi,
            "macd_histogram": technical.macd_histogram,
            "volume_ratio": technical.volume_ratio,
            "reasoning": technical.reasoning,
            "divergence": technical.divergence,
        },
        "signal_track_record": {
            "historical_edge_vs_baseline": technical.track_record_edge,
            "sample_size": technical.track_record_events,
            "note": technical.track_record_note,
        },
        "data_reliability": {
            "reliability": data_quality.reliability,
            "is_derivative_or_thinly_traded": data_quality.is_derivative,
            "warnings": data_quality.warnings,
        },
        "fundamentals": fundamentals.model_dump(),
        "quality_score": quality.model_dump(),
        "market_regime": regime.model_dump(),
        "earnings": earnings.model_dump(),
        "mechanical_price_targets": price_targets.model_dump(),
        "news_sentiment": {
            "bullish_score": sentiment.bullish_score,
            "overall_impact": sentiment.overall_impact,
            "headline_count": sentiment.headline_count,
            "headlines": [
                {"title": h.title, "sentiment": h.sentiment, "context": h.context}
                for h in sentiment.headlines
            ],
        },
    }
    if insider is not None:
        # Unlike everything above, this was never folded into
        # technical.conviction (see InsiderActivity's own docstring on why
        # — no backtest evidence behind it yet) — genuinely fresh evidence
        # for the model to weigh on its own, not something to guard against
        # double-counting.
        payload["insider_institutional_activity"] = insider.model_dump()
    if research:
        # Stocklake's own AI-generated take (get_stock_research) — a second,
        # independently-produced opinion alongside this model's own
        # reasoning, not a substitute for it. Only the synthesis piece is
        # included; the raw indicators/news/signals get_stock_research also
        # returns are already covered above from this app's own pipeline.
        payload["independent_ai_research"] = research
    if is_held:
        unrealized_pl_pct = round((current_price - held_cost_basis) / held_cost_basis * 100, 2) \
            if current_price and held_cost_basis else None
        payload["existing_position"] = {
            "shares": held_shares,
            "cost_basis": held_cost_basis,
            "unrealized_pl_pct": unrealized_pl_pct,
        }
    if history:
        payload["past_ai_decisions_for_this_ticker"] = [h.model_dump() for h in history[:5]]

    held_instructions = (
        "\n\nThe person already holds this position (see existing_position) — the question "
        "is whether to keep holding, trim, or exit, not whether to buy fresh. You MUST return "
        "null for entry: there is no new-entry question here. stop_loss/take_profit, if "
        "returned, are protective levels from the current price going forward, not an entry "
        "setup. conviction_score should directly answer 'how justified is continuing to hold "
        "at this cost basis' — weigh their unrealized P/L as context (e.g. a large unrealized "
        "loss with a now-bearish signal is a different situation than a small one), not as "
        "something to be defensive about."
    ) if is_held else ""
    history_instructions = (
        "\n\npast_ai_decisions_for_this_ticker shows your own previous calls on this exact "
        "ticker, each already resolved against a later price by the caller. If past high-"
        "conviction scores did NOT hit take-profit before hitting stop-loss (or the price "
        "moved against the score), calibrate this new score more conservatively rather than "
        "repeating the same optimism — and say so in the rationale if it materially changed "
        "your score. Past scores having been right isn't itself a reason for extra confidence "
        "now; only the current evidence is."
    ) if history else ""

    try:
        response = client.with_options(timeout=25.0).messages.parse(
            model=AI_DECISION_MODEL,
            max_tokens=800,
            system=(
                "You compute a decision score and price levels for someone deciding "
                "whether to buy, hold, or sell a stock — a concrete artifact to act on, "
                "not a paragraph to read and interpret. You're given everything this app "
                "already computed for the ticker: a technical signal, that exact signal's "
                "own historical backtest track record on THIS ticker, fundamentals, a "
                "quality score, the broader market regime, earnings timing, recent news "
                "sentiment, and the existing mechanical (ATR-based) price targets — plus, "
                "when present, insider/institutional trading activity and a second, "
                "independently-generated AI research take from a different provider "
                "(Stocklake) for you to weigh against your own reasoning, not defer to. "
                "No information beyond what's in the JSON, and never invent facts, news, "
                "or numbers not present there.\n\n"
                "technical_signal.conviction is not a raw indicator reading — it has already "
                "been nudged, one notch at a time, by signal_track_record, quality_score, "
                "market_regime, divergence, and earnings timing: track_record and divergence "
                "can raise OR lower it (agreement raises, conflict lowers), while quality, "
                "regime, and earnings only ever lower it, whenever THOSE specific values are "
                "the ones that triggered it — e.g. quality_score only pulled conviction down if "
                "grade is Poor on a BUY or Strong on a SELL; market_regime only pulled it down "
                "on a counter-trend call or elevated volatility; earnings only pulled it down "
                "if a report lands within about a week. Use these fields to understand WHY "
                "conviction is what it is, not as fresh, independent evidence to move your own "
                "conviction_score down (or up) again for the same reason already covered above "
                "— that would be counting it twice. Only let them shift conviction_score "
                "further if you see something in their specific values that the mechanical "
                "one-notch nudge wouldn't have captured (e.g. several of them pointing the same "
                "way at once, or a severity the nudge logic can't express).\n\n"
                "Only adjust entry/stop_loss/take_profit away from the mechanical baseline "
                "when something in the context gives a specific, statable reason (earnings "
                "gap risk, poor data reliability, a support/resistance level that makes the "
                "baseline unrealistic) — otherwise return the baseline values unchanged. If "
                "mechanical_price_targets.entry is null (a HOLD/SELL signal with no suggested "
                "entry), you MUST return null for entry, stop_loss, take_profit, and "
                "risk_reward_ratio too — there is no baseline to adjust from, and inventing "
                "price levels here would be a fabricated trade setup, not an analysis. "
                "Never issue an instruction ('buy this now') — the numbers themselves are "
                "the output; rationale is one sentence explaining them, not advice."
                + held_instructions + history_instructions
            ),
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            output_format=AIDecisionOutput,
        )
        decision = response.parsed_output
        if decision is None:
            return None
        # Match compute_price_targets()'s own convention (round(x, 2)) rather
        # than showing whatever raw float precision the model returned.
        for f in ("entry", "stop_loss", "take_profit", "risk_reward_ratio"):
            val = getattr(decision, f)
            if val is not None:
                setattr(decision, f, round(val, 2))
        return decision
    except Exception as e:
        logger.warning(f"AI decision generation failed for {ticker}: {e}")
        return None


@app.get("/api/analyze/{ticker}", response_model=AnalysisResponse)
def analyze(ticker: str, background_tasks: BackgroundTasks):
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

    cached = cache_get(ticker)
    if cached:
        return AnalysisResponse(**{**cached, "cached": True})

    try:
        stock, hist = fetch_price_history(ticker)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise HTTPException(
            status_code=502,
            detail="Couldn't reach the market data provider right now (it may be temporarily rate-limiting requests). Please try again in a minute.",
        )

    if hist is None or hist.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker '{ticker}'. Check the symbol (e.g. AAPL, SHOP.TO, XYZ.V).",
        )

    technical = compute_technicals(hist)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    company_name = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency") or ("CAD" if ticker.endswith((".TO", ".V", ".CN", ".NE")) else "USD")

    close = hist["Close"]
    current_price = round(float(close.iloc[-1]), 2)

    price_change = price_change_pct = None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev:
            price_change = round(current_price - prev, 2)
            price_change_pct = round((current_price - prev) / prev * 100, 2)

    price_history = [round(float(v), 2) for v in close.tail(30).tolist()]

    # One combined Stocklake session for this whole request — get_stock()
    # (reused below by both the quality score's fundamentals and, further
    # down, fetch_earnings's next-earnings-date) and get_stock_news()
    # (reused by fetch_sentiment, further down still) — rather than each
    # feature opening its own MCP session. See fetch_stocklake_context's
    # own docstring for why paying the handshake once per request matters.
    stocklake_symbol = _stocklake_symbol(ticker)
    yf_sector = info.get("sector")
    stocklake_sector = _STOCKLAKE_SECTOR_MAP.get((yf_sector or "").lower(), yf_sector)
    stocklake_calls = {
        "stock": ("get_stock", {"symbol": stocklake_symbol}),
        "news": ("get_stock_news", {"symbol": ticker, "days": 14, "limit": 5}),
        "insider": ("get_insider_activity", {"symbol": stocklake_symbol}),
    }
    if stocklake_sector:
        stocklake_calls["sector"] = ("get_sector_intelligence", {"sector": stocklake_sector})
    stocklake_context = fetch_stocklake_context(ticker, stocklake_calls)
    stocklake_stock_data = validate_stocklake_stock(
        ticker, stocklake_symbol, stocklake_context.get("stock"),
        yf_name=info.get("longName") or info.get("shortName") or "",
    )
    insider_activity = map_insider_activity(stocklake_context.get("insider"), stocklake_stock_data is not None)
    sector_intelligence = map_sector_intelligence(stocklake_context.get("sector"), stocklake_sector)
    analyst_consensus = map_analyst_consensus(stocklake_stock_data)

    fundamentals = build_fundamentals(info, current_price)
    quality = compute_quality_score(_merge_stocklake_fundamentals(info, stocklake_stock_data))
    data_quality = assess_data_quality(ticker, hist, info)

    # For a CAD-hedged / CDR product, try to substitute a reliable signal from
    # the actual underlying stock rather than just penalizing conviction on a
    # signal that's fundamentally noise. Fails closed: if no trustworthy
    # underlying is found, falls through to the conviction penalty below —
    # this stays graceful degradation, never a wrong substitution.
    underlying_hist = None
    if data_quality.is_derivative:
        try:
            underlying = resolve_underlying_ticker(ticker, info, data_quality.avg_daily_volume)
        except Exception as e:
            logger.info(f"Underlying resolution failed for {ticker}: {e}")
            underlying = None

        if underlying:
            try:
                _u_stock, underlying_hist = fetch_price_history(underlying["ticker"])
            except Exception as e:
                logger.info(f"Underlying fetch failed for {underlying['ticker']}: {e}")
                underlying_hist = None

            if underlying_hist is not None and not underlying_hist.empty and len(underlying_hist) >= 30:
                u_ticker = underlying["ticker"]
                technical = compute_technicals(underlying_hist)
                technical.source_ticker = u_ticker
                technical.reasoning = (
                    f"This is {u_ticker}'s own signal — {ticker} trades too thin on its own "
                    f"for RSI/MACD to be dependable, so its numbers are borrowed from the "
                    f"primary listing instead. {technical.reasoning}"
                )
                data_quality.warnings.insert(
                    0,
                    f"{ticker} is a CAD-hedged product tracking {u_ticker}. Its own trading "
                    f"is too thin for reliable signals, so the reading above is {u_ticker}'s "
                    "own — computed from a listing with far more volume.",
                )
            else:
                underlying_hist = None  # resolution found something but data fetch failed

    # A signal computed on thin data shouldn't carry the same weight as one
    # computed on a liquid name — unless it was already substituted above.
    if underlying_hist is None:
        if data_quality.reliability == "Poor":
            technical.conviction = "Low"
            technical.reasoning += " Conviction capped at Low: this listing trades too thinly for the indicators to be dependable."
            # Skip track-record and divergence too: both are read off this
            # same thin, unreliable price data, so neither would be
            # trustworthy either — and applying them after the cap above
            # would just contradict it in the reasoning text.
        else:
            if data_quality.reliability == "Fair" and technical.conviction == "High":
                technical.conviction = "Moderate"
            technical = apply_track_record(technical, ticker)
            technical = apply_divergence(technical, technical.divergence_bars_ago)
    else:
        # Signal was substituted from a reliable underlying — that
        # underlying's own track record (and this same divergence, read off
        # the underlying's own reliable price data) are the relevant ones.
        technical = apply_track_record(technical, underlying["ticker"])
        technical = apply_divergence(technical, technical.divergence_bars_ago)

    technical = apply_quality_check(technical, quality)

    regime = get_market_regime(canadian=ticker.endswith((".TO", ".V", ".CN", ".NE")))
    technical = apply_regime_check(technical, regime)

    earnings = fetch_earnings(stock, ticker, stocklake_stock_data)
    technical = apply_earnings_proximity(technical, earnings)
    price_targets = compute_price_targets(hist, technical.signal, current_price)

    sentiment = fetch_sentiment(stocklake_context.get("news"))

    response = AnalysisResponse(
        ticker=ticker,
        company_name=company_name,
        current_price=current_price,
        price_change=price_change,
        price_change_pct=price_change_pct,
        currency=currency,
        market=detect_market(ticker),
        price_history=price_history,
        fundamentals=fundamentals,
        quality=quality,
        data_quality=data_quality,
        regime=regime,
        earnings=earnings,
        technical_analysis=technical,
        price_targets=price_targets,
        sentiment_analysis=sentiment,
        insider_activity=insider_activity,
        sector_intelligence=sector_intelligence,
        analyst_consensus=analyst_consensus,
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )

    cache_set(ticker, response.model_dump())

    # Warm the track-record cache for whichever ticker apply_track_record
    # actually used above (see the same underlying-substitution / Poor-
    # reliability skip logic a few lines up) — never blocks this response,
    # only helps the NEXT analyze() call on this ticker within 24h. See
    # _warm_track_record_cache for why this exists.
    if underlying_hist is not None or data_quality.reliability != "Poor":
        track_record_ticker = underlying["ticker"] if underlying_hist is not None else ticker
        background_tasks.add_task(_warm_track_record_cache, track_record_ticker)

    return response


@app.post("/api/analyze/{ticker}/decision", response_model=AIDecisionResponse)
def analyze_decision(ticker: str, request: AIDecisionRequest = AIDecisionRequest()):
    """
    The AI Decision, split out from /api/analyze itself. It reuses that
    endpoint's cached technicals/fundamentals/etc. rather than refetching,
    but runs as its own request so a slow Claude call never adds latency to
    (or, worse, drops) the price/technical/fundamental data that's already
    computed and ready. Call this only after /api/analyze/{ticker} has
    populated the cache for the same ticker.

    POST (not GET) because the request body carries optional context that
    changes the actual question being asked: shares/cost_basis switch it
    from "worth a fresh entry" to "worth continuing to hold," and history
    is the client's own record of how this ticker's past AI Decisions here
    turned out, fed back so the model can calibrate against its own track
    record rather than reasoning fresh every time.
    """
    ticker = ticker.strip().upper()
    cached = cache_get(ticker)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="No cached analysis for this ticker yet — call /api/analyze/{ticker} first.",
        )

    # Cached per the exact question asked, not just per ticker — held vs.
    # fresh-entry framing and a changing history are genuinely different
    # requests, so the same content hash pattern used elsewhere in this app
    # (see the earlier AI brief) applies here too. Reuse across repeat views
    # of the *same* question within the analysis cache window — otherwise
    # reopening the Analyze screen re-triggers a Claude call every time even
    # though nothing underneath changed.
    cache_key_material = {
        "ticker": ticker,
        "shares": request.shares,
        "cost_basis": request.cost_basis,
        "history": [h.model_dump() for h in request.history[:5]],
    }
    cache_key = ticker + ":" + hashlib.sha256(
        json.dumps(cache_key_material, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    cached_decision = _DECISION_CACHE.get(cache_key)
    if cached_decision and time.time() - cached_decision[0] < _CACHE_TTL_SECONDS:
        return AIDecisionResponse(**cached_decision[1])

    a = AnalysisResponse(**cached)

    # A fresh Stocklake call, not reused from /api/analyze's own session:
    # this endpoint runs as its own request (see this function's docstring
    # on why), so there's no in-flight session left to piggyback on. Only
    # the ai_summary block is kept — the rest of get_stock_research
    # duplicates data this app's own pipeline already has above.
    stocklake_symbol = _stocklake_symbol(a.ticker)
    research_payload = fetch_stocklake_context(
        a.ticker, {"research": ("get_stock_research", {"symbol": stocklake_symbol})}
    ).get("research")
    research = None
    if isinstance(research_payload, dict) and "error" not in research_payload:
        # Same wrong-company risk validate_stocklake_stock guards against
        # for the main analyze() call: a suffix-stripped symbol can resolve
        # to an unrelated company that happens to share the same letters.
        # get_stock_research's own "stock" block doesn't carry the country
        # field that function checks, so this falls back to the name-overlap
        # half of that check alone — good enough for "don't hand the model
        # research about a different company," even if slightly more
        # conservative than the full check would be.
        research_name = ((research_payload.get("stock") or {}).get("name") or "").lower()
        yf_name = (a.company_name or "").lower()
        name_overlaps = True
        if stocklake_symbol != a.ticker and research_name and yf_name:
            strip = str.maketrans("", "", ".,")
            yf_words = {w for w in yf_name.translate(strip).split() if len(w) > 2}
            sl_words = {w for w in research_name.translate(strip).split() if len(w) > 2}
            name_overlaps = bool(yf_words & sl_words) if (yf_words and sl_words) else True
        if name_overlaps:
            research = research_payload.get("ai_summary")

    decision = generate_ai_decision(
        a.ticker, a.company_name, a.current_price, a.price_change_pct, a.currency,
        a.technical_analysis, a.fundamentals, a.quality, a.data_quality, a.regime,
        a.earnings, a.price_targets, a.sentiment_analysis,
        insider=a.insider_activity, research=research,
        held_shares=request.shares, held_cost_basis=request.cost_basis,
        history=request.history,
    )
    if decision is None:
        return AIDecisionResponse()

    decision_dict = decision.model_dump()
    _DECISION_CACHE[cache_key] = (time.time(), decision_dict)
    if len(_DECISION_CACHE) > 100:
        oldest = min(_DECISION_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _DECISION_CACHE.pop(oldest, None)
    return AIDecisionResponse(**decision_dict)


@app.get("/api/screener", response_model=ScreenerResponse)
def screener(force: bool = False):
    """Scan the fixed universe for current BUY / SELL technical signals.
    Reads the background-refreshed cache only — see screener_from_cache()."""
    return screener_from_cache(force)


@app.get("/api/screener/under20", response_model=ScreenerResponse)
def screener_under20(force: bool = False):
    """Canadian stocks under CAD $20, screened for current momentum signals.
    Reads the background-refreshed cache only — see under20_from_cache()."""
    return under20_from_cache(force)


@app.get("/api/screener/high-volume", response_model=HighVolumeResponse)
def screener_high_volume(force: bool = False):
    """Today's highest-volume US and Canadian tickers — not limited to SCREENER_UNIVERSE.
    Reads the background-refreshed cache only — see high_volume_from_cache()."""
    return high_volume_from_cache(force)


@app.get("/api/screener/stocklake", response_model=StocklakeScreenerResponse)
def screener_stocklake(
    preset: Optional[str] = None,
    sector: Optional[str] = None,
    min_ai_score: Optional[int] = None,
    sma_trend: Optional[str] = None,
    sort_by: Optional[str] = None,
    limit: int = 20,
):
    """
    A second, independent discovery tool alongside the fixed-universe
    Screener above — scans Stocklake's full ~3,500-symbol universe
    (filtered server-side) using Stocklake's own AI-scored methodology,
    rather than this app's small fixed list and RSI+MACD rule. A genuinely
    different tool, not a replacement: nothing returned here has been
    through this app's own backtest-validated pipeline.
    """
    if not os.environ.get("STOCKLAKE_API_KEY"):
        raise HTTPException(status_code=503, detail="Stocklake isn't configured for this deployment.")

    args: Dict[str, Any] = {"limit": max(1, min(25, limit))}
    if preset:
        args["preset"] = preset
    if sector:
        args["sector"] = sector
    if min_ai_score is not None:
        args["min_ai_score"] = min_ai_score
    if sma_trend:
        args["sma_trend"] = sma_trend
    if sort_by:
        args["sort_by"] = sort_by

    payload = fetch_stocklake_context(
        "__stocklake_screener__", {"screener": ("get_screener", args)}
    ).get("screener")
    if not isinstance(payload, dict) or "error" in payload:
        raise HTTPException(status_code=502, detail="Couldn't reach Stocklake's screener right now.")

    results = [StocklakeScreenerHit(**r) for r in payload.get("results", [])]
    return StocklakeScreenerResponse(
        count=payload.get("count", len(results)),
        preset=payload.get("preset"),
        results=results,
        note=(
            "Discovery via Stocklake's own AI-scored screener — a different "
            "methodology and a much larger universe than this app's own Screener. "
            "Not backtest-validated by this app."
        ),
    )


@app.get("/api/market-pulse", response_model=MarketPulseResponse)
def market_pulse():
    """
    Standalone market-wide snapshot (Stocklake-first plan, P5c) — the same
    _fetch_market_pulse() data get_market_regime() already folds into every
    Analyze page, exposed here on its own so it doesn't require picking a
    ticker to see how jumpy or calm the broad market is right now.
    """
    pulse = _fetch_market_pulse()
    if not pulse:
        raise HTTPException(
            status_code=503,
            detail="Market pulse isn't available right now — Stocklake may be unconfigured or unreachable.",
        )
    fear_greed = pulse.get("fear_greed") or {}
    breadth = pulse.get("breadth") or {}
    return MarketPulseResponse(
        vix=pulse.get("vix"),
        fear_greed_value=fear_greed.get("value"),
        fear_greed_label=fear_greed.get("description"),
        breadth_oversold_pct=breadth.get("oversold_pct"),
        breadth_overbought_pct=breadth.get("overbought_pct"),
        generated_at=datetime.now(timezone.utc).isoformat(),
        note=(
            "Market-wide snapshot via Stocklake — VIX, fear/greed, and the share "
            "of stocks technically oversold or overbought right now. Refreshed "
            "roughly hourly; informational only, not tied to any one ticker's signal."
        ),
    )


@app.post("/api/earnings-calendar", response_model=EarningsCalendarResponse)
def earnings_calendar(request: EarningsCalendarRequest):
    """
    Upcoming earnings dates across a list of tickers (Stocklake-first plan,
    P5a) — the frontend passes holdings + watchlist, the same tickers Brief
    already scans. Reuses the exact earnings_date/earnings_is_estimate
    fields fetch_earnings() already relies on for a single ticker, just
    batched via _stocklake_batch_stocks() instead of a per-ticker call, so
    nothing here is a new, unverified Stocklake schema — same fields, same
    parsing rule (_parse_future_stocklake_earnings_date), just scanned
    across many tickers instead of one.

    Deliberately not the dedicated get_earnings_calendar/get_earnings_
    intelligence tools — this app hasn't verified their response shape
    against a live call yet, so it's not guessed at here. Full historical
    surprise data (beat/miss vs. estimate) also isn't available this way;
    that stays a per-ticker-only feature on the Analyze page.
    """
    tickers = [t.strip().upper() for t in request.tickers if t and t.strip()]
    if not tickers:
        return EarningsCalendarResponse(upcoming=[], note="No tickers to check.")

    raw = _stocklake_batch_stocks(tickers)
    upcoming: List[UpcomingEarning] = []
    for ticker, data in raw.items():
        date = _parse_future_stocklake_earnings_date(data.get("earnings_date"), ticker)
        if date:
            upcoming.append(UpcomingEarning(
                ticker=ticker, earnings_date=date,
                is_estimate=bool(data.get("earnings_is_estimate")),
            ))
    upcoming.sort(key=lambda e: e.earnings_date)

    return EarningsCalendarResponse(
        upcoming=upcoming,
        note=(
            "Upcoming earnings dates via Stocklake, checked against the tickers you "
            "hold or watch. Suffix-stripped Canadian tickers and anything outside "
            "Stocklake's ~3,500-symbol universe can't be checked this way."
        ),
    )


@app.get("/api/sector-rotation", response_model=SectorRotationResponse)
def sector_rotation():
    """
    All 11 sectors' cycle stage and rotation signal side by side (Stocklake-
    first plan, P5b) — the same get_sector_intelligence data the Analyze
    page's sector card already shows for one ticker's sector, scanned
    across every sector at once so money moving between sectors is visible
    without having to check ticker by ticker.
    """
    if not os.environ.get("STOCKLAKE_API_KEY"):
        raise HTTPException(status_code=503, detail="Stocklake isn't configured for this deployment.")
    sectors = fetch_sector_rotation()
    if not sectors:
        raise HTTPException(status_code=502, detail="Couldn't reach Stocklake's sector intelligence right now.")
    return SectorRotationResponse(
        sectors=sectors,
        note=(
            "All 11 sectors' current cycle stage and rotation signal via Stocklake, "
            "refreshed roughly every hour. Informational only — not backtestable "
            "(no historical time series available) and not factored into any signal."
        ),
    )


HISTORY_RANGES = {
    "1D": ("1d", "5m"),
    "1W": ("5d", "15m"),
    "1M": ("1mo", "1d"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}


@app.get("/api/history/{ticker}", response_model=HistoryResponse)
def history(ticker: str, range: str = "3M"):
    """Price history for the chart, at a selectable timeframe (like Yahoo Finance's range buttons)."""
    ticker = ticker.strip().upper()
    range_key = range.strip().upper()
    period, interval = HISTORY_RANGES.get(range_key, ("3mo", "1d"))

    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        logger.error(f"History fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't fetch price history right now.")

    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail=f"No history found for '{ticker}'.")

    close = hist["Close"].dropna()
    fmt = "%Y-%m-%d %H:%M" if interval in ("5m", "15m") else "%Y-%m-%d"
    points = [
        HistoryPoint(date=idx.strftime(fmt), close=round(float(v), 2))
        for idx, v in close.items()
    ]
    return HistoryResponse(ticker=ticker, range=range_key, points=points)


@app.post("/api/portfolio/analyze", response_model=PortfolioResponse)
def portfolio_analyze(request: PortfolioRequest):
    """Score a set of holdings: P/L plus current technical signal for each."""
    return analyze_portfolio(request.holdings)


# ---------------------------------------------------------------------------
# Claude client — shared by any AI-generated feature. Lazily constructed so
# the app runs fine with it entirely unset; every caller must treat None as
# "feature disabled" and degrade gracefully, never raise.
# ---------------------------------------------------------------------------
_anthropic_client: Optional[anthropic.Anthropic] = None
_anthropic_unavailable_logged = False


def _get_anthropic_client() -> Optional[anthropic.Anthropic]:
    global _anthropic_client, _anthropic_unavailable_logged
    if _anthropic_client is not None:
        return _anthropic_client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not _anthropic_unavailable_logged:
            logger.info("ANTHROPIC_API_KEY not set — AI features disabled, rest of the app is unaffected.")
            _anthropic_unavailable_logged = True
        return None
    _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


@app.post("/api/digest", response_model=DigestResponse)
def digest(request: DigestRequest, force: bool = False):
    """
    The morning brief: portfolio status, watchlist signals, and screener hits
    in one call. Technicals only — see the note field. The fixed-universe and
    under-$20 scans are read from a cache kept warm by a background refresh
    loop (see _background_refresh_loop) rather than fetched live here, so
    this endpoint can't be the thing that blocks on a slow or down vendor.
    ?force=true nudges an out-of-band refresh along but still returns
    whatever's cached right now, same as /api/screener.

    Returns the FULL scanned list for opportunities/warnings, not a capped
    top-N — the frontend decides how much to show by default and how much
    to reveal on request. This also means "refresh" and "show all" are
    always looking at the same data instead of two separate fetches that
    can drift out of sync with each other.
    """
    portfolio = analyze_portfolio(request.holdings) if request.holdings else None

    watchlist_signals: List[ScreenerHit] = []
    if request.watchlist:
        scored = batch_score(request.watchlist)
        for r in scored.values():
            if not r.get("error"):
                watchlist_signals.append(ScreenerHit(**r))
        watchlist_signals = sort_hits(watchlist_signals)

    screen = screener_from_cache(force)
    under20 = under20_from_cache(force)

    # Custom tickers the person added to the scan — scored live, uncached
    # (unlike the shared, cached fixed universe), then merged in below.
    extra_hits: List[ScreenerHit] = []
    extra_tickers = [t.strip().upper() for t in request.extra_scan_tickers if t and t.strip()]
    if extra_tickers:
        scored_extra = batch_score(extra_tickers)
        for r in scored_extra.values():
            if not r.get("error"):
                extra_hits.append(ScreenerHit(**r))

    # Don't repeat names the person already holds or watches under "opportunities".
    known = {h.ticker.strip().upper() for h in request.holdings} | {
        t.strip().upper() for t in request.watchlist
    }

    def merge(fixed_list: List[ScreenerHit], signal_name: str) -> List[ScreenerHit]:
        extra_matches = [h for h in extra_hits if h.signal == signal_name]
        combined = {h.ticker: h for h in fixed_list}
        for h in extra_matches:
            combined[h.ticker] = h  # a custom addition overrides a stale fixed-universe entry
        return sort_hits([h for h in combined.values() if h.ticker not in known])

    opportunities = merge(screen.buy_candidates, "BUY")
    warnings = merge(screen.sell_candidates, "SELL")
    under20_buys = [h for h in under20.buy_candidates if h.ticker not in known]
    under20_avoid = [h for h in under20.sell_candidates if h.ticker not in known]

    bits = []
    if portfolio and portfolio.holdings:
        direction = "up" if portfolio.total_pl >= 0 else "down"
        bits.append(
            f"Portfolio {direction} {abs(portfolio.total_pl_pct)}% "
            f"({portfolio.total_pl:+,.2f})"
        )
        flagged = [h.ticker for h in portfolio.holdings if h.signal == "SELL"]
        if flagged:
            bits.append(f"{len(flagged)} holding(s) flagged bearish")
    if opportunities:
        bits.append(f"{len(opportunities)} buy signal(s) in the scanned universe")
    headline = " · ".join(bits) if bits else "No significant signals this morning."

    return DigestResponse(
        portfolio=portfolio,
        watchlist_signals=watchlist_signals,
        opportunities=opportunities,
        warnings=warnings,
        under20_buys=under20_buys,
        under20_avoid=under20_avoid,
        headline=headline,
        note=(
            "This brief uses RSI, MACD and volume only — no news sentiment, which is "
            "too slow to run across this many tickers. Signals are mechanical, drawn "
            "from a fixed list of large-caps plus any tickers you've added to the scan, "
            "and are not recommendations. Tap any ticker for the full analysis including "
            "news sentiment."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ===========================================================================
# BACKTESTING
#
# Measures whether this app's BUY/SELL signals have actually preceded gains
# historically. Three design decisions that keep the numbers honest:
#
# 1. NO LOOK-AHEAD. RSI and MACD at day T are computed only from prices up to
#    day T (EMAs are causal by construction), so nothing "knows the future."
#
# 2. DISTINCT EVENTS, NOT SIGNAL-DAYS. If BUY stays on for 6 days straight,
#    that's ONE event, not six. Counting signal-days inflates the sample and
#    double-counts a single decision.
#
# 3. BASELINE COMPARISON. A 55% win rate means nothing on its own — in a
#    rising market, a random day might win 58% of the time. Every stat is
#    reported against the all-days baseline, and "edge" is the difference.
#    Edge is the number that matters; win rate alone is misleading.
# ===========================================================================

BACKTEST_HORIZONS = [5, 10, 20]


# Realistic round-trip trading cost, in percent. Covers commission plus the
# bid-ask spread you actually cross on both entry and exit. Deliberately
# conservative for a retail account — an "edge" smaller than this is not an
# edge, it's a transfer to your broker.
ROUND_TRIP_COST_PCT = 0.30


class HorizonResult(BaseModel):
    days: int
    win_rate: Optional[float]
    avg_return: Optional[float]
    baseline_win_rate: Optional[float]
    baseline_avg_return: Optional[float]
    edge: Optional[float]
    net_edge: Optional[float]


class SignalBacktest(BaseModel):
    signal: str
    event_count: int
    horizons: List[HorizonResult]
    plain_summary: str = ""
    plain_result: str = ""


class BacktestResponse(BaseModel):
    ticker: str
    period: str
    trading_days: int
    buy: SignalBacktest
    sell: SignalBacktest
    buy_and_hold_return: Optional[float]
    verdict: str
    caveats: str
    generated_at: str


def _score_row(rsi_val: float, macd_hist_val: float) -> str:
    """Same scoring rule as the live signal, applied historically."""
    score = 0
    if rsi_val < 30:
        score += 1
    elif rsi_val > 70:
        score -= 1
    if macd_hist_val > 0:
        score += 1
    elif macd_hist_val < 0:
        score -= 1
    return "BUY" if score >= 1 else "SELL" if score <= -1 else "HOLD"



def _plain_signal_summary(signal_name: str, event_count: int, horizons: List[HorizonResult],
                          period: str) -> Tuple[str, str]:
    """
    Turns the statistics into two plain sentences: what happened, and whether
    it means anything. No jargon — no "edge", no "baseline", no percentages
    of percentages.
    """
    years = {"1y": "year", "2y": "2 years", "5y": "5 years"}.get(period, period)
    word = "buy" if signal_name == "BUY" else "sell"

    if event_count == 0:
        return (f"The app never said {word.upper()} on this stock in the last {years}.", "")

    ten = next((h for h in horizons if h.days == 10), None)
    if ten is None or ten.win_rate is None:
        return (f"The app said {word.upper()} {event_count} times in the last {years}.",
                "There isn't enough recent data to see how those turned out yet.")

    direction = "higher" if signal_name == "BUY" else "lower"
    summary = (
        f"The app said {word.upper()} {event_count} times in the last {years}. "
        f"Two weeks later, the price was {direction} {ten.win_rate:.0f} times out of 100. "
        f"On a random day picked out of a hat, it would have been {direction} "
        f"{ten.baseline_win_rate:.0f} times out of 100."
    )

    gap = ten.edge if ten.edge is not None else 0
    net = ten.net_edge if ten.net_edge is not None else gap

    if event_count < 10:
        result = (
            f"That said, {event_count} is a small number of times to judge anything by. "
            "Treat this as a hint, not a finding."
        )
    elif net > 0.5:
        result = (
            f"So the signal did genuinely better than guessing — and it still holds up "
            f"after the cost of buying and selling. This one looks useful on this stock."
        )
    elif gap > 0:
        result = (
            "So the signal did slightly better than guessing, but the gap is small enough "
            "that commission and the buy/sell spread would eat it. Not worth trading on."
        )
    else:
        result = (
            "So the signal actually did worse than guessing. On this stock, following it "
            "would have cost you money rather than made you money."
        )
    return summary, result


def _plain_variant(description: str, edge: Optional[float], base_edge: Optional[float],
                   events: int) -> str:
    if edge is None or events == 0:
        return f"{description}: not enough signals to tell."
    if events < 10:
        return f"{description}: only {events} signals — too few to judge."
    if base_edge is None:
        base_edge = 0.0
    diff = edge - base_edge
    if diff > 0.5:
        return f"{description}: did better than the app's current rules here."
    if diff > 0:
        return f"{description}: barely different from the current rules."
    return f"{description}: did worse than the current rules."


def _plain_aggregate(description: str, wins: int, tested: int, mean_edge: Optional[float],
                     p_value: Optional[float], mean_signals: Optional[float],
                     is_baseline: bool) -> str:
    if is_baseline:
        return "This is what the app does today — everything else is compared against it."
    if tested == 0:
        return f"{description}: couldn't be tested."
    if mean_signals is not None and mean_signals < 10:
        return (
            f"{description}: filters out so many trades that only about "
            f"{mean_signals:.0f} signals are left per stock — not enough to judge."
        )
    if p_value is not None and p_value < 0.05 and (mean_edge or 0) > 0:
        return (
            f"{description}: did better on {wins} of {tested} stocks. That's a strong "
            "enough pattern that it's unlikely to be a fluke — this one looks real."
        )
    if wins > tested / 2:
        return (
            f"{description}: did better on {wins} of {tested} stocks. That sounds "
            "promising, but when you try this many different ideas, one of them looks "
            "good by luck almost every time. Not enough to act on."
        )
    return f"{description}: did better on only {wins} of {tested} stocks. No sign it helps."


def run_backtest(ticker: str, period: str = "2y") -> BacktestResponse:
    ticker = ticker.strip().upper()

    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as e:
        logger.error(f"Backtest fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't fetch history for the backtest.")

    if hist is None or hist.empty or len(hist) < 120:
        raise HTTPException(
            status_code=422,
            detail="Not enough price history to backtest meaningfully (need ~6 months minimum).",
        )

    close = hist["Close"].dropna()
    rsi = compute_rsi(close, length=14)
    _, _, macd_hist = compute_macd(close)

    df = pd.DataFrame({"close": close, "rsi": rsi, "macd_hist": macd_hist}).dropna()
    if len(df) < 60:
        raise HTTPException(
            status_code=422,
            detail="Not enough usable history after indicator warm-up to backtest.",
        )

    df["signal"] = [
        _score_row(r, m) for r, m in zip(df["rsi"].values, df["macd_hist"].values)
    ]
    # A signal "event" is the first day of a run — not every day it stays on.
    df["is_event"] = df["signal"] != df["signal"].shift(1)

    for h in BACKTEST_HORIZONS:
        df[f"fwd{h}"] = df["close"].shift(-h) / df["close"] - 1

    def summarize(signal_name: str) -> SignalBacktest:
        # Count an event only when a signal run STARTS, treating a brief
        # interruption as part of the same decision rather than a new one.
        is_active = (df["signal"] == signal_name)
        min_gap = 5
        event_positions = []
        last_active = -(min_gap + 1)
        for i, active in enumerate(is_active.values):
            if active:
                if i - last_active > min_gap:
                    event_positions.append(i)
                last_active = i

        event_mask = pd.Series(False, index=df.index)
        if event_positions:
            event_mask.iloc[event_positions] = True
        events = df[event_mask]

        horizons = []
        for h in BACKTEST_HORIZONS:
            col = f"fwd{h}"
            sample = events[col].dropna()
            baseline = df[col].dropna()

            if len(sample) == 0 or len(baseline) == 0:
                horizons.append(
                    HorizonResult(
                        days=h, win_rate=None, avg_return=None,
                        baseline_win_rate=None, baseline_avg_return=None,
                        edge=None, net_edge=None,
                    )
                )
                continue

            # For SELL, "winning" means the price fell — the signal was right.
            if signal_name == "SELL":
                win_rate = float((sample < 0).mean()) * 100
                baseline_win_rate = float((baseline < 0).mean()) * 100
            else:
                win_rate = float((sample > 0).mean()) * 100
                baseline_win_rate = float((baseline > 0).mean()) * 100

            avg_return = float(sample.mean()) * 100
            baseline_avg = float(baseline.mean()) * 100
            # Edge: for BUY, beating the baseline means higher returns. For
            # SELL, the signal claims the price will fall, so a LOWER return
            # than baseline is the correct direction — flip the sign.
            edge = (avg_return - baseline_avg) if signal_name == "BUY" else (baseline_avg - avg_return)

            horizons.append(
                HorizonResult(
                    days=h,
                    win_rate=round(win_rate, 1),
                    avg_return=round(avg_return, 2),
                    baseline_win_rate=round(baseline_win_rate, 1),
                    baseline_avg_return=round(baseline_avg, 2),
                    edge=round(edge, 2),
                    net_edge=round(edge - ROUND_TRIP_COST_PCT, 2),
                )
            )
        plain_summary, plain_result = _plain_signal_summary(
            signal_name, int(len(events)), horizons, period
        )
        return SignalBacktest(
            signal=signal_name, event_count=int(len(events)), horizons=horizons,
            plain_summary=plain_summary, plain_result=plain_result,
        )

    buy = summarize("BUY")
    sell = summarize("SELL")

    buy_and_hold = round(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2)

    # ---- Verdict: deliberately conservative. Small samples and thin edges
    # get called out rather than dressed up.
    def edge_at(bt: SignalBacktest, days: int) -> Optional[float]:
        for h in bt.horizons:
            if h.days == days:
                return h.edge
        return None

    buy_edge = edge_at(buy, 10)
    sell_edge = edge_at(sell, 10)
    min_events = 10

    parts = []
    if buy.event_count < min_events:
        parts.append(
            f"Only {buy.event_count} BUY signals fired in this period — too few to conclude anything reliable."
        )
    elif buy_edge is None:
        parts.append("Not enough forward data to judge BUY signals.")
    elif buy_edge > ROUND_TRIP_COST_PCT + 0.5:
        parts.append(
            f"BUY signals beat the baseline by {buy_edge:.2f}% over 10 days across "
            f"{buy.event_count} events — about {buy_edge - ROUND_TRIP_COST_PCT:.2f}% after "
            "realistic trading costs. A real edge in this sample."
        )
    elif buy_edge > 0:
        parts.append(
            f"BUY signals edged the baseline by {buy_edge:.2f}% over 10 days, but roughly "
            f"{ROUND_TRIP_COST_PCT:.2f}% goes to commission and spread — leaving about "
            f"{buy_edge - ROUND_TRIP_COST_PCT:+.2f}% net. Not worth trading on."
        )
    else:
        parts.append(
            f"BUY signals actually UNDERPERFORMED just picking a random day, by {abs(buy_edge):.2f}% over 10 days. On this stock, they haven't worked."
        )

    if sell.event_count < min_events:
        parts.append(f"Only {sell.event_count} SELL signals — too few to judge.")
    elif sell_edge is None:
        parts.append("Not enough forward data to judge SELL signals.")
    elif sell_edge > 1.0:
        parts.append(f"SELL signals correctly anticipated weakness, beating baseline by {sell_edge:.2f}%.")
    elif sell_edge > 0:
        parts.append(f"SELL signals showed a marginal {sell_edge:.2f}% edge — weak.")
    else:
        parts.append(
            f"SELL signals were counterproductive here — price tended to do {abs(sell_edge):.2f}% BETTER than baseline after them."
        )

    verdict = " ".join(parts)

    caveats = (
        "A few things worth knowing before you trust this. It only looks at one stock over "
        "one stretch of time — a good result here says nothing about other stocks, or about "
        "next month. It assumes you buy at the closing price on the day the signal appears "
        "and sell a fixed number of days later, which is tidier than real life. It ignores "
        "taxes. And it can't see the reasons behind any of the moves: earnings, news and "
        "interest rates drove a lot of what happened, not the chart patterns. The number "
        "that matters is the comparison against random days — a signal that wins 60% of the "
        "time sounds great until you learn that random days won 58% of the time."
    )

    return BacktestResponse(
        ticker=ticker,
        period=period,
        trading_days=int(len(df)),
        buy=buy,
        sell=sell,
        buy_and_hold_return=buy_and_hold,
        verdict=verdict,
        caveats=caveats,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/backtest/{ticker}", response_model=BacktestResponse)
def backtest(ticker: str, period: str = "2y"):
    """
    How well have this app's own BUY/SELL signals performed on this ticker
    historically, measured against a do-nothing baseline?
    """
    if period not in ("1y", "2y", "5y"):
        period = "2y"

    # Shares a cache (and, via _get_or_run_backtest, an in-progress guard)
    # with the live signal's track-record check — if this ticker was just
    # analyzed, or is being warmed in the background right now, this reuses
    # that work instead of redundantly refetching and recomputing.
    return _get_or_run_backtest(ticker, period)


# ===========================================================================
# SIGNAL VARIANT COMPARISON (ablation study)
#
# Rather than assuming a filter helps, this runs candidate rule-sets against
# each other on real history and reports each one's edge. Keep what wins.
#
# IMPORTANT STATISTICAL WARNING, enforced in the response text below: testing
# many variants and picking the best is itself a way to overfit. With 5
# variants, one will look best by chance alone maybe 20-30% of the time. That
# is why this deliberately tests only a handful of *pre-specified* ideas with
# a mechanical reason to work — not dozens of parameter tweaks — and why the
# response insists on confirming a winner across several unrelated tickers.
# ===========================================================================


class VariantResult(BaseModel):
    plain: str = ""
    name: str
    description: str
    buy_events: int
    buy_edge_10d: Optional[float]
    buy_win_rate: Optional[float]
    sell_events: int
    sell_edge_10d: Optional[float]
    reliable: bool


class VariantComparison(BaseModel):
    ticker: str
    period: str
    trading_days: int
    baseline_avg_10d: Optional[float]
    variants: List[VariantResult]
    recommendation: str
    warning: str
    generated_at: str


_BENCHMARK_CACHE: Dict[str, Tuple[float, Any]] = {}
_BENCHMARK_TTL_SECONDS = 3600


def _get_benchmark(symbol: str, period: str) -> Optional[pd.Series]:
    """Index data for relative strength. Cached — a basket run would otherwise
    refetch the same index once per ticker."""
    key = f"{symbol}:{period}"
    cached = _BENCHMARK_CACHE.get(key)
    if cached and time.time() - cached[0] < _BENCHMARK_TTL_SECONDS:
        return cached[1]
    try:
        bh = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        series = bh["Close"].dropna() if bh is not None and not bh.empty else None
    except Exception as e:
        logger.info(f"Benchmark fetch skipped: {e}")
        series = None
    _BENCHMARK_CACHE[key] = (time.time(), series)
    return series


def _weekly_trend_flag(close: pd.Series) -> pd.Series:
    """
    Is the weekly trend up? Forward-filled to daily.

    LOOK-AHEAD TRAP handled here: the current week is incomplete, so using its
    close would mean knowing Friday's price on Monday. Shifted by one completed
    week so only finished bars are used.
    """
    try:
        wk = close.resample("W").last()
        wk_ema = wk.ewm(span=10, adjust=False).mean()
        up = (wk > wk_ema).shift(1)
        return up.reindex(close.index, method="ffill")
    except Exception:
        return pd.Series(True, index=close.index)


def _relative_strength(close: pd.Series, benchmark: Optional[pd.Series], window: int = 60) -> pd.Series:
    """Trailing return minus the benchmark's. Positive = outperforming."""
    if benchmark is None or benchmark.empty:
        return pd.Series(np.nan, index=close.index)
    s = close.pct_change(window)
    b = benchmark.pct_change(window).reindex(close.index, method="ffill")
    return s - b


def _prepare_indicator_frame(ticker: str, period: str) -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as e:
        logger.error(f"Variant fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't fetch history.")

    if hist is None or hist.empty or len(hist) < 260:
        raise HTTPException(
            status_code=422,
            detail="Need at least ~1 year of history to compare variants (the 200-day trend filter alone eats 200 days of warm-up).",
        )

    close = hist["Close"].dropna()
    volume = hist["Volume"].reindex(close.index)

    rsi = compute_rsi(close, length=14)
    _, _, macd_hist = compute_macd(close)

    df = pd.DataFrame({
        "close": close,
        "volume": volume,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "vol_avg20": volume.rolling(20).mean(),
    })
    df["vol_ratio"] = df["volume"] / df["vol_avg20"]
    df["weekly_up"] = _weekly_trend_flag(close)

    # Relative strength vs a market benchmark (S&P 500 for US names, TSX for
    # Canadian). One extra request per comparison — acceptable here since this
    # is an on-demand analysis, not something the screener runs in bulk.
    benchmark_symbol = "^GSPTSE" if ticker.upper().endswith((".TO", ".V", ".CN", ".NE")) else "^GSPC"
    benchmark = _get_benchmark(benchmark_symbol, period)
    df["rel_strength"] = _relative_strength(close, benchmark)

    # Base signal, same rule the live app uses.
    df["base_signal"] = [
        _score_row(r, m) if pd.notna(r) and pd.notna(m) else "HOLD"
        for r, m in zip(df["rsi"], df["macd_hist"])
    ]

    # RSI divergence, aligned to the same index.
    try:
        df["divergence"] = detect_divergences(close, rsi)
    except Exception:
        df["divergence"] = None

    for h in BACKTEST_HORIZONS:
        df[f"fwd{h}"] = df["close"].shift(-h) / df["close"] - 1

    return df.dropna(subset=["close", "rsi", "macd_hist"])


def _apply_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    """Returns a signal series. Filters only ever downgrade a signal to HOLD."""
    sig = df["base_signal"].copy()

    if variant == "baseline":
        return sig

    if variant == "trend200":
        # Don't buy below the long-term trend, don't sell above it.
        sig = sig.where(~((sig == "BUY") & (df["close"] < df["sma200"])), "HOLD")
        sig = sig.where(~((sig == "SELL") & (df["close"] > df["sma200"])), "HOLD")
        return sig

    if variant == "trend50":
        sig = sig.where(~((sig == "BUY") & (df["close"] < df["sma50"])), "HOLD")
        sig = sig.where(~((sig == "SELL") & (df["close"] > df["sma50"])), "HOLD")
        return sig

    if variant == "volume":
        # Require at-or-above-average participation behind the move.
        weak = df["vol_ratio"] < 1.0
        sig = sig.where(~(weak & (sig != "HOLD")), "HOLD")
        return sig

    if variant == "trend200_volume":
        sig = sig.where(~((sig == "BUY") & (df["close"] < df["sma200"])), "HOLD")
        sig = sig.where(~((sig == "SELL") & (df["close"] > df["sma200"])), "HOLD")
        weak = df["vol_ratio"] < 1.0
        sig = sig.where(~(weak & (sig != "HOLD")), "HOLD")
        return sig

    if variant == "weekly_agree":
        # Only take a signal the weekly trend agrees with.
        wk_up = df["weekly_up"].fillna(False).astype(bool)
        sig = sig.where(~((sig == "BUY") & ~wk_up), "HOLD")
        sig = sig.where(~((sig == "SELL") & wk_up), "HOLD")
        return sig

    if variant == "rel_strength":
        # Only buy names outperforming their index; only sell laggards.
        rs = df["rel_strength"]
        sig = sig.where(~((sig == "BUY") & (rs.notna()) & (rs <= 0)), "HOLD")
        sig = sig.where(~((sig == "SELL") & (rs.notna()) & (rs >= 0)), "HOLD")
        return sig

    if variant == "divergence_confirm":
        # Require a supporting RSI divergence within the last 10 bars.
        div = df.get("divergence")
        if div is None:
            return sig
        bull_recent = (div == "bullish").rolling(10, min_periods=1).max().fillna(0).astype(bool)
        bear_recent = (div == "bearish").rolling(10, min_periods=1).max().fillna(0).astype(bool)
        sig = sig.where(~((sig == "BUY") & ~bull_recent), "HOLD")
        sig = sig.where(~((sig == "SELL") & ~bear_recent), "HOLD")
        return sig

    return sig


def _edge_for(df: pd.DataFrame, sig: pd.Series, signal_name: str, horizon: int):
    """
    Returns (event_count, edge, win_rate).

    Subtle but important correctness point: filters convert scattered days to
    HOLD, which can split one continuous signal run into fragments
    (BUY→HOLD→BUY). Counting "signal differs from yesterday" would treat that
    as two events and INFLATE the sample size, making filtered variants look
    better-supported than they are — verified in testing, where a naive count
    made a filter appear to *add* signals.

    The fix: anchor events to runs of the UNFILTERED base signal. A filter can
    then only ever remove an event, never manufacture one. This keeps sample
    sizes comparable across variants, which is the whole point of an ablation.
    """
    col = f"fwd{horizon}"
    base = df["base_signal"] if "base_signal" in df.columns else df["signal"]

    base_active = (base == signal_name)
    base_run_start = base_active & ~base_active.shift(1, fill_value=False)

    # Keep an event only if the filtered signal still fires on that same day.
    event_mask = base_run_start & (sig == signal_name)

    sample = df.loc[event_mask, col].dropna()
    baseline = df[col].dropna()

    if len(sample) == 0 or len(baseline) == 0:
        return int(event_mask.sum()), None, None

    avg = float(sample.mean()) * 100
    base_avg = float(baseline.mean()) * 100
    edge = (avg - base_avg) if signal_name == "BUY" else (base_avg - avg)

    if signal_name == "SELL":
        win = float((sample < 0).mean()) * 100
    else:
        win = float((sample > 0).mean()) * 100

    return len(sample), round(edge, 2), round(win, 1)


VARIANT_DEFS = [
    ("baseline", "RSI + MACD only (what the app uses now)"),
    ("trend200", "Only trade with the 200-day trend"),
    ("trend50", "Only trade with the 50-day trend"),
    ("volume", "Require at-or-above-average volume"),
    ("trend200_volume", "200-day trend AND volume confirmation"),
    ("weekly_agree", "Require the weekly trend to agree"),
    ("rel_strength", "Only trade names beating their index"),
    ("divergence_confirm", "Require a supporting RSI divergence"),
]

MIN_RELIABLE_EVENTS = 10


def compare_variants(ticker: str, period: str = "5y") -> VariantComparison:
    ticker = ticker.strip().upper()
    df = _prepare_indicator_frame(ticker, period)

    baseline_avg = df["fwd10"].dropna()
    baseline_avg_10d = round(float(baseline_avg.mean()) * 100, 2) if len(baseline_avg) else None

    results: List[VariantResult] = []
    for name, description in VARIANT_DEFS:
        sig = _apply_variant(df, name)
        buy_n, buy_edge, buy_win = _edge_for(df, sig, "BUY", 10)
        sell_n, sell_edge, _ = _edge_for(df, sig, "SELL", 10)
        base_ref = next((r for r in results if r.name == "baseline"), None)
        results.append(
            VariantResult(
                plain=_plain_variant(description, buy_edge,
                                     base_ref.buy_edge_10d if base_ref else None, buy_n),
                name=name,
                description=description,
                buy_events=buy_n,
                buy_edge_10d=buy_edge,
                buy_win_rate=buy_win,
                sell_events=sell_n,
                sell_edge_10d=sell_edge,
                reliable=buy_n >= MIN_RELIABLE_EVENTS,
            )
        )

    # Recommendation, deliberately reluctant.
    usable = [r for r in results if r.reliable and r.buy_edge_10d is not None]
    base = next((r for r in results if r.name == "baseline"), None)

    if not usable:
        recommendation = (
            "No variant produced enough signals on this ticker to judge. Try a longer "
            "period, or a stock that moves more."
        )
    else:
        best = max(usable, key=lambda r: r.buy_edge_10d)
        base_edge = base.buy_edge_10d if base and base.buy_edge_10d is not None else 0.0
        improvement = best.buy_edge_10d - base_edge

        if best.buy_edge_10d <= 0:
            recommendation = (
                f"None of these ways of filtering the signals beat simply guessing on {ticker}. "
                "That's genuinely useful to know: on this stock, these chart patterns aren't "
                "telling you anything."
            )
        elif best.name == "baseline":
            recommendation = (
                "The app's current rules did best here. Adding extra filters made things worse, "
                "not better, on this stock."
            )
        elif improvement < 0.5:
            recommendation = (
                f"'{best.description}' came out slightly ahead, but by so little that it could "
                "easily be chance. Not a reason to change anything."
            )
        else:
            recommendation = (
                f"'{best.description}' did noticeably better than the current rules on this "
                "stock. Worth checking against other stocks before reading much into it — one "
                "stock proves nothing."
            )

    warning = (
        "One important catch. There are eight different filters being tried here, and when "
        "you try eight ideas, one of them usually looks good by pure luck — even if none of "
        "them actually work. So a single good result on a single stock means very little. "
        "Use the 'test across 12 stocks' button below to see whether anything holds up more "
        "widely. Also watch how many signals are left: a filter works by throwing trades "
        "away, so a great-looking result based on 8 trades is much weaker evidence than a "
        "modest one based on 60."
    )

    return VariantComparison(
        ticker=ticker,
        period=period,
        trading_days=int(len(df)),
        baseline_avg_10d=baseline_avg_10d,
        variants=results,
        recommendation=recommendation,
        warning=warning,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/backtest/{ticker}/variants", response_model=VariantComparison)
def backtest_variants(ticker: str, period: str = "5y"):
    """
    Head-to-head comparison of candidate signal rule-sets on real history.
    Use this to decide whether a filter is worth adopting — don't guess.
    """
    if period not in ("2y", "5y", "10y"):
        period = "5y"
    return compare_variants(ticker, period)


# ===========================================================================
# RSI DIVERGENCE
#
# Bullish: price makes a LOWER low while RSI makes a HIGHER low — the second
# decline had less selling force behind it. Bearish is the mirror.
#
# This is one of the few genuinely anticipatory technical patterns, because
# it describes momentum weakening BEFORE price confirms it.
#
# THE CRITICAL CORRECTNESS POINT: a swing low at bar i can only be identified
# once bars i+1..i+k exist. Recording a divergence AT the swing bar would mean
# using information that didn't exist yet — it backtests beautifully and is
# worthless live. So the signal is deliberately recorded at bar i+k, when it
# first becomes knowable. Verified in testing: swing at 129 -> signal at 132.
#
# Parameters (k=3, gap 8-120 bars) were validated against injected patterns
# and checked for false-positive rate on random walks (~1% of bars).
# ===========================================================================

DIVERGENCE_K = 3
DIVERGENCE_MIN_GAP = 8
DIVERGENCE_MAX_GAP = 120


def detect_divergences(close: pd.Series, rsi: pd.Series) -> pd.Series:
    """Returns a series of 'bullish' / 'bearish' / None, lagged to avoid look-ahead."""
    n = len(close)
    k = DIVERGENCE_K
    win = 2 * k + 1

    roll_min = close.rolling(win, center=True).min()
    roll_max = close.rolling(win, center=True).max()
    is_low = (close == roll_min) & roll_min.notna()
    is_high = (close == roll_max) & roll_max.notna()

    out = pd.Series([None] * n, index=close.index, dtype=object)
    lows = [i for i in range(n) if bool(is_low.iloc[i])]
    highs = [i for i in range(n) if bool(is_high.iloc[i])]

    for positions, kind in ((lows, "bullish"), (highs, "bearish")):
        for a, b in zip(positions, positions[1:]):
            gap = b - a
            if gap < DIVERGENCE_MIN_GAP or gap > DIVERGENCE_MAX_GAP:
                continue
            price_a, price_b = close.iloc[a], close.iloc[b]
            rsi_a, rsi_b = rsi.iloc[a], rsi.iloc[b]
            if pd.isna(rsi_a) or pd.isna(rsi_b):
                continue
            matched = (
                (kind == "bullish" and price_b < price_a and rsi_b > rsi_a)
                or (kind == "bearish" and price_b > price_a and rsi_b < rsi_a)
            )
            if matched and b + k < n:
                out.iloc[b + k] = kind
    return out


def describe_divergence(kind: Optional[str], bars_ago: Optional[int]) -> str:
    if not kind:
        return (
            "No recent RSI divergence detected. Divergences are uncommon — their absence "
            "is normal and isn't a signal in itself."
        )
    when = f"{bars_ago} trading day{'s' if bars_ago != 1 else ''} ago" if bars_ago is not None else "recently"
    if kind == "bullish":
        return (
            f"Bullish divergence spotted {when}: price made a lower low, but RSI made a "
            "higher low — the second decline carried less selling pressure. This is one of "
            "the few patterns that can hint at a turn before price shows it. It is not a "
            "guarantee: divergences can persist for a long time, or simply fail."
        )
    return (
        f"Bearish divergence spotted {when}: price made a higher high, but RSI made a "
        "lower high — the rally is running on weaker momentum. Often precedes a pullback, "
        "though strong trends can push through divergences for weeks."
    )


# ===========================================================================
# CROSS-TICKER AGGREGATE COMPARISON
#
# The single-ticker comparison can't settle anything on its own: with eight
# variants, something looks best by luck most of the time. The only way to
# tell a real filter from a lucky one is whether it wins CONSISTENTLY across
# unrelated stocks.
#
# So this runs the variants over a basket and applies a binomial test: if a
# filter beat the baseline on k of n tickers, how likely is that under the
# null hypothesis that it's a coin flip? A filter winning 9 of 10 is hard to
# explain by chance (p ≈ 0.011); winning 6 of 10 is not (p ≈ 0.377).
#
# The p-value is then Bonferroni-corrected for the number of variants tested,
# because testing eight hypotheses and reporting the best one without
# correction is precisely how false discoveries get published.
# ===========================================================================

import math

DEFAULT_BASKET = [
    "AAPL", "MSFT", "JNJ", "XOM", "JPM", "WMT",
    "NVDA", "PFE", "RY.TO", "ENB.TO", "SHOP.TO", "BCE.TO",
]


class AggregateVariantResult(BaseModel):
    plain: str = ""
    name: str
    description: str
    tickers_tested: int
    tickers_beating_baseline: int
    mean_edge: Optional[float]
    median_edge: Optional[float]
    mean_signals_per_ticker: Optional[float]
    p_value: Optional[float]
    significant: bool
    verdict: str


class AggregateComparison(BaseModel):
    tickers_requested: List[str]
    tickers_analyzed: List[str]
    tickers_failed: List[str]
    period: str
    results: List[AggregateVariantResult]
    conclusion: str
    method_note: str
    generated_at: str


def _binomial_p_at_least(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p). One-sided."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def aggregate_variants(tickers: List[str], period: str = "5y") -> AggregateComparison:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()][:15]
    if not tickers:
        tickers = DEFAULT_BASKET

    # variant name -> list of (edge, signal_count) per ticker
    per_variant: Dict[str, List[Tuple[float, int]]] = {name: [] for name, _ in VARIANT_DEFS}
    baseline_edges: Dict[str, float] = {}
    analyzed, failed = [], []

    for t in tickers:
        try:
            df = _prepare_indicator_frame(t, period)
        except Exception as e:
            logger.info(f"Aggregate: skipping {t} ({e})")
            failed.append(t)
            continue

        ticker_results = {}
        for name, _desc in VARIANT_DEFS:
            try:
                sig = _apply_variant(df, name)
                n_events, edge, _win = _edge_for(df, sig, "BUY", 10)
                ticker_results[name] = (edge, n_events)
            except Exception:
                ticker_results[name] = (None, 0)

        base_edge = ticker_results.get("baseline", (None, 0))[0]
        if base_edge is None:
            failed.append(t)
            continue

        analyzed.append(t)
        baseline_edges[t] = base_edge
        for name, (edge, n) in ticker_results.items():
            if edge is not None:
                per_variant[name].append((edge, n, base_edge))

    results: List[AggregateVariantResult] = []
    n_variants = max(len(VARIANT_DEFS) - 1, 1)  # baseline isn't a hypothesis

    for name, description in VARIANT_DEFS:
        rows = per_variant.get(name, [])
        if not rows:
            results.append(
                AggregateVariantResult(
                    plain=f"{description}: couldn't be tested.",
                    name=name, description=description, tickers_tested=0,
                    tickers_beating_baseline=0, mean_edge=None, median_edge=None,
                    mean_signals_per_ticker=None, p_value=None, significant=False,
                    verdict="No usable data.",
                )
            )
            continue

        edges = [r[0] for r in rows]
        counts = [r[1] for r in rows]
        wins = sum(1 for edge, _n, base in rows if edge > base)
        n = len(rows)
        mean_edge = float(np.mean(edges))
        median_edge = float(np.median(edges))
        mean_signals = float(np.mean(counts))

        if name == "baseline":
            p_value = None
            significant = False
            verdict = "Reference point — the rules the app uses today."
        else:
            raw_p = _binomial_p_at_least(wins, n)
            # Bonferroni: testing several filters inflates false positives.
            p_value = min(1.0, raw_p * n_variants)
            significant = p_value < 0.05 and mean_edge > 0

            if mean_signals < 10:
                verdict = (
                    f"Averages only {mean_signals:.0f} signals per ticker — too few to judge, "
                    "regardless of how the edge looks."
                )
            elif significant:
                verdict = (
                    f"Beat baseline on {wins}/{n} tickers with a mean edge of {mean_edge:+.2f}%. "
                    f"Survives correction for multiple testing (p={p_value:.3f}) — this looks real."
                )
            elif wins > n / 2 and mean_edge > 0:
                verdict = (
                    f"Beat baseline on {wins}/{n} tickers ({mean_edge:+.2f}% mean edge), but that's "
                    f"within what chance produces when testing this many filters (p={p_value:.2f}). "
                    "Not enough to act on."
                )
            else:
                verdict = (
                    f"Beat baseline on only {wins}/{n} tickers, mean edge {mean_edge:+.2f}%. "
                    "No evidence this filter helps."
                )

        results.append(
            AggregateVariantResult(
                plain=_plain_aggregate(description, wins, n, mean_edge,
                                       p_value, mean_signals, name == "baseline"),
                name=name, description=description, tickers_tested=n,
                tickers_beating_baseline=wins, mean_edge=round(mean_edge, 2),
                median_edge=round(median_edge, 2),
                mean_signals_per_ticker=round(mean_signals, 1),
                p_value=round(p_value, 4) if p_value is not None else None,
                significant=significant, verdict=verdict,
            )
        )

    winners = [r for r in results if r.significant]
    if not analyzed:
        conclusion = "No tickers could be analyzed. Try again, or use a different basket."
    elif not winners:
        conclusion = (
            f"Tested across {len(analyzed)} different stocks, none of the eight filters "
            "reliably beat what the app already does. This is the most common outcome, and "
            "it's a real answer rather than a failure: it means don't change anything. "
            "Chart-based signals mostly have very thin edges, and a filter that shone on one "
            "or two stocks was almost certainly luck."
        )
    else:
        best = max(winners, key=lambda r: r.mean_edge)
        conclusion = (
            f"One filter stands out: {best.description.lower()}. It beat the app's current "
            f"rules on {best.tickers_beating_baseline} of {best.tickers_tested} stocks — a "
            "wide enough pattern that it's unlikely to be luck, even allowing for the fact "
            "that eight ideas were tried. This is the one worth taking seriously."
        )

    method_note = (
        "How this works, in plain terms. Each filter is tried on every stock, and we count "
        "how many stocks it beat the app's current rules on. Then we ask: if this filter "
        "were useless and we were just flipping coins, how often would it look this good by "
        "accident? Because eight filters are being tried at once, the bar is set higher than "
        "it would be for a single one — otherwise something always wins by chance. A filter "
        "only gets recommended if it wins on most stocks, keeps enough trades to be worth "
        "measuring, and clears that raised bar. All of this looks backwards at what already "
        "happened, so it narrows down what's worth trying rather than proving anything about "
        "the future."
    )

    return AggregateComparison(
        tickers_requested=tickers,
        tickers_analyzed=analyzed,
        tickers_failed=failed,
        period=period,
        results=results,
        conclusion=conclusion,
        method_note=method_note,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/backtest/aggregate", response_model=AggregateComparison)
def backtest_aggregate(tickers: str = "", period: str = "5y"):
    """
    Run the variant comparison across a basket of tickers and report which
    filters hold up statistically. This is the endpoint that can actually
    settle whether a filter is worth adopting.

    tickers: comma-separated. Omit to use a diversified default basket.
    """
    if period not in ("2y", "5y", "10y"):
        period = "5y"
    ticker_list = [t for t in tickers.split(",") if t.strip()] if tickers else DEFAULT_BASKET
    return aggregate_variants(ticker_list, period)


# ===========================================================================
# PORTFOLIO CONCENTRATION / CORRELATION
#
# Analyzing each holding alone hides the most common real-world mistake:
# owning RY, TD and BMO isn't three positions, it's one leveraged bet on
# Canadian banks. Correlation makes that visible.
# ===========================================================================

def analyze_concentration(
    holdings: List[HoldingResult], sectors: Dict[str, str]
) -> ConcentrationReport:
    valid = [h for h in holdings if not h.error and h.market_value]
    if len(valid) < 1:
        return ConcentrationReport(
            highly_correlated_pairs=[], largest_position_pct=None, top_three_pct=None,
            effective_positions=None, sector_concentration={}, warnings=[],
            note="Add holdings to see concentration analysis.",
        )

    total = sum(h.market_value for h in valid)
    weights = {h.ticker: h.market_value / total for h in valid} if total else {}

    largest = max(weights.values()) * 100 if weights else None
    top3 = sum(sorted(weights.values(), reverse=True)[:3]) * 100 if weights else None

    # Effective number of positions (inverse Herfindahl). Ten equal holdings
    # gives 10; ten holdings where one is 90% gives barely above 1. This is a
    # far better diversification measure than simply counting positions.
    hhi = sum(w * w for w in weights.values())
    effective = round(1 / hhi, 1) if hhi > 0 else None

    sector_conc: Dict[str, float] = {}
    for h in valid:
        sec = sectors.get(h.ticker) or "Unknown"
        sector_conc[sec] = sector_conc.get(sec, 0) + weights.get(h.ticker, 0) * 100
    sector_conc = {k: round(v, 1) for k, v in sorted(sector_conc.items(), key=lambda kv: -kv[1])}

    # Correlation of daily returns over the past year.
    pairs: List[CorrelatedPair] = []
    tickers = [h.ticker for h in valid]
    if len(tickers) >= 2:
        try:
            raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
            closes = {}
            for t in tickers:
                try:
                    series = raw[t]["Close"] if len(tickers) > 1 else raw["Close"]
                    closes[t] = series.dropna()
                except Exception:
                    continue
            if len(closes) >= 2:
                rets = pd.DataFrame({t: s.pct_change() for t, s in closes.items()}).dropna()
                if len(rets) > 30:
                    corr = rets.corr()
                    seen = set()
                    for a in corr.columns:
                        for b in corr.columns:
                            if a == b or (b, a) in seen:
                                continue
                            seen.add((a, b))
                            c = corr.loc[a, b]
                            if pd.notna(c) and c >= 0.7:
                                pairs.append(CorrelatedPair(
                                    ticker_a=a, ticker_b=b, correlation=round(float(c), 2)
                                ))
                    pairs.sort(key=lambda p: -p.correlation)
                    pairs = pairs[:8]
        except Exception as e:
            logger.info(f"Correlation calc skipped: {e}")

    warnings = []
    if largest and largest > 30:
        warnings.append(
            f"Your largest position is {largest:.0f}% of the portfolio. A single bad "
            "outcome there would dominate your overall result."
        )
    if top3 and top3 > 70 and len(valid) > 3:
        warnings.append(
            f"Your top three positions are {top3:.0f}% of the portfolio — the rest are "
            "too small to matter much either way."
        )
    if effective and len(valid) >= 4 and effective < len(valid) * 0.5:
        warnings.append(
            f"You hold {len(valid)} positions, but weighting makes them behave more like "
            f"{effective:.0f}. Diversification is thinner than the position count suggests."
        )
    for p in pairs[:3]:
        warnings.append(
            f"{p.ticker_a} and {p.ticker_b} move together ({p.correlation:.0%} correlated) — "
            "they'll tend to fall at the same time, so they don't diversify each other."
        )
    for sec, pct in list(sector_conc.items())[:1]:
        if pct > 40 and sec != "Unknown":
            warnings.append(f"{pct:.0f}% of the portfolio sits in {sec}.")

    note = (
        "Correlation is measured on the past year of daily moves. Two stocks above 0.7 "
        "have historically risen and fallen together, so holding both gives less protection "
        "than it appears. Correlations also tend to rise sharply in a crash — exactly when "
        "diversification matters most — so treat these as a floor, not a ceiling."
    )

    return ConcentrationReport(
        highly_correlated_pairs=pairs,
        largest_position_pct=round(largest, 1) if largest else None,
        top_three_pct=round(top3, 1) if top3 else None,
        effective_positions=effective,
        sector_concentration=sector_conc,
        warnings=warnings,
        note=note,
    )


# ===========================================================================
# FUNDAMENTAL QUALITY SCORE
#
# The app already fetches this data and then ignores it. A BUY on a
# profitable, low-debt company is a different proposition from the identical
# chart pattern on one burning cash.
# ===========================================================================

def compute_quality_score(info: dict) -> QualityScore:
    def num(key):
        v = info.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    score = 0
    max_score = 0
    factors, concerns = [], []

    margin = num("profitMargins")
    if margin is not None:
        max_score += 2
        if margin > 0.15:
            score += 2; factors.append(f"Strong profit margin ({margin*100:.1f}%)")
        elif margin > 0.05:
            score += 1; factors.append(f"Modest profit margin ({margin*100:.1f}%)")
        elif margin > 0:
            factors.append(f"Thin profit margin ({margin*100:.1f}%)")
        else:
            concerns.append(f"Unprofitable ({margin*100:.1f}% margin)")

    d2e = num("debtToEquity")
    if d2e is not None:
        max_score += 2
        if d2e < 50:
            score += 2; factors.append(f"Low debt (debt/equity {d2e:.0f}%)")
        elif d2e < 150:
            score += 1; factors.append(f"Moderate debt (debt/equity {d2e:.0f}%)")
        else:
            concerns.append(f"High debt load (debt/equity {d2e:.0f}%)")

    roe = num("returnOnEquity")
    if roe is not None:
        max_score += 2
        if roe > 0.15:
            score += 2; factors.append(f"Strong return on equity ({roe*100:.1f}%)")
        elif roe > 0.05:
            score += 1; factors.append(f"Adequate return on equity ({roe*100:.1f}%)")
        else:
            concerns.append(f"Weak return on equity ({roe*100:.1f}%)")

    growth = num("revenueGrowth")
    if growth is not None:
        max_score += 2
        if growth > 0.10:
            score += 2; factors.append(f"Revenue growing {growth*100:.1f}%")
        elif growth > 0:
            score += 1; factors.append(f"Revenue growing slowly ({growth*100:.1f}%)")
        else:
            concerns.append(f"Revenue shrinking ({growth*100:.1f}%)")

    current = num("currentRatio")
    if current is not None:
        max_score += 1
        if current > 1.5:
            score += 1; factors.append(f"Comfortable liquidity (current ratio {current:.1f})")
        elif current < 1:
            concerns.append(f"Tight liquidity (current ratio {current:.1f})")

    if max_score == 0:
        return QualityScore(
            score=None, grade="Unknown", factors=[], concerns=[],
            note="Fundamental data isn't available for this ticker — common for ETFs, funds, and smaller listings.",
        )

    pct = int(round(score / max_score * 100))
    grade = "Strong" if pct >= 75 else "Decent" if pct >= 50 else "Weak" if pct >= 25 else "Poor"

    note = (
        "This grades the BUSINESS, not the stock price — a strong company can still be a bad "
        "buy if it's overpriced, and a weak one can rally hard. It's a sanity check against "
        "the chart: a BUY signal on a 'Poor' business deserves more scepticism than the same "
        "signal on a 'Strong' one. Based on the most recent reported figures, which lag."
    )

    return QualityScore(score=pct, grade=grade, factors=factors, concerns=concerns, note=note)


# ===========================================================================
# MARKET REGIME
#
# RSI and MACD behave very differently in trending versus choppy markets.
# Knowing which one you're in is more useful than another indicator.
# ===========================================================================

_REGIME_CACHE: Dict[str, Tuple[float, Any]] = {}
_REGIME_TTL_SECONDS = 3600
_MARKET_PULSE_CACHE: Tuple[float, Optional[dict]] = (0.0, None)


def _fetch_market_pulse() -> Optional[dict]:
    """
    Cached separately from _REGIME_CACHE (keyed by US/CA symbol) since
    market pulse is neither — one shared snapshot for the whole app, same
    1h TTL as the regime calc it augments, so a US regime cache miss and a
    CA regime cache miss within the same hour still only fetch this once.
    """
    global _MARKET_PULSE_CACHE
    cached_at, cached_data = _MARKET_PULSE_CACHE
    if cached_data is not None and time.time() - cached_at < 3600:
        return cached_data
    if not os.environ.get("STOCKLAKE_API_KEY"):
        return None
    payload = fetch_stocklake_context("__market_pulse__", {"pulse": ("get_market_pulse", {})}).get("pulse")
    if not isinstance(payload, dict) or "error" in payload:
        return None
    _MARKET_PULSE_CACHE = (time.time(), payload)
    return payload


def get_market_regime(canadian: bool = False) -> MarketRegime:
    symbol = "^GSPTSE" if canadian else "^GSPC"
    cached = _REGIME_CACHE.get(symbol)
    if cached and time.time() - cached[0] < _REGIME_TTL_SECONDS:
        return MarketRegime(**cached[1])

    fallback = MarketRegime(
        regime="Unknown", index_used=symbol, index_vs_200ma=None,
        volatility_percentile=None,
        note="Couldn't determine the current market regime.",
    )
    try:
        hist = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 200:
            return fallback
        close = hist["Close"].dropna()
        sma200 = close.rolling(200).mean()
        vs_200 = float((close.iloc[-1] / sma200.iloc[-1] - 1) * 100)

        rets = close.pct_change().dropna()
        vol20 = rets.rolling(20).std()
        current_vol = float(vol20.iloc[-1])
        vol_pct = float((vol20.dropna() < current_vol).mean() * 100)

        if vs_200 > 3 and vol_pct < 70:
            regime = "Trending up, calm"
            note = ("The index is comfortably above its 200-day average with unremarkable "
                    "volatility. Momentum signals tend to work better here, and BUY signals "
                    "have the broader market behind them.")
        elif vs_200 > 3:
            regime = "Trending up, volatile"
            note = ("The index is above its long-term average but moving sharply. Signals "
                    "fire more often and reverse more often — position sizes should reflect that.")
        elif vs_200 < -3 and vol_pct > 70:
            regime = "Falling, volatile"
            note = ("The index is below its 200-day average with elevated volatility — the "
                    "hardest conditions for momentum signals. Oversold readings can stay "
                    "oversold for a long time. Treat BUY signals with real caution.")
        elif vs_200 < -3:
            regime = "Falling, calm"
            note = ("The index is below its long-term average in an orderly decline. "
                    "Counter-trend BUY signals have a poor historical record in this setup.")
        else:
            regime = "Choppy / rangebound"
            note = ("The index is near its 200-day average with no clear direction. Choppy "
                    "markets produce the most false signals — trend-following logic whipsaws.")

        result = MarketRegime(
            regime=regime, index_used="S&P/TSX" if canadian else "S&P 500",
            index_vs_200ma=round(vs_200, 2),
            volatility_percentile=round(vol_pct, 0),
            note=note,
        )
        pulse = _fetch_market_pulse()
        if pulse:
            result.vix = pulse.get("vix")
            fear_greed = pulse.get("fear_greed") or {}
            result.fear_greed_value = fear_greed.get("value")
            result.fear_greed_label = fear_greed.get("description")
            breadth = pulse.get("breadth") or {}
            result.breadth_oversold_pct = breadth.get("oversold_pct")
            result.breadth_overbought_pct = breadth.get("overbought_pct")
        _REGIME_CACHE[symbol] = (time.time(), result.model_dump())
        return result
    except Exception as e:
        logger.info(f"Regime detection skipped: {e}")
        return fallback


@app.get("/api/regime", response_model=MarketRegime)
def market_regime(canadian: bool = False):
    """Current market conditions — context for how much to trust any signal."""
    return get_market_regime(canadian)


# Forward references (QualityScore, MarketRegime, ConcentrationReport are
# defined after the response models that reference them).
# (No model_rebuild() needed — every model's dependencies are now defined
# before the routes that use them, verified by static analysis above.)


# ===========================================================================
# QUOTES — lightweight batch price lookup, used by the decision journal to
# measure how logged decisions actually turned out.
# ===========================================================================

class Quote(BaseModel):
    ticker: str
    price: Optional[float]
    currency: str
    error: Optional[str] = None


class QuotesResponse(BaseModel):
    quotes: List[Quote]
    generated_at: str


_QUOTES_CACHE: Dict[str, Tuple[float, float]] = {}
_QUOTES_TTL_SECONDS = 300


@app.get("/api/quotes", response_model=QuotesResponse)
def quotes(tickers: str = ""):
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:40]
    if not symbols:
        return QuotesResponse(quotes=[], generated_at=datetime.now(timezone.utc).isoformat())

    results: List[Quote] = []
    fresh_needed = []
    now = time.time()

    for s in symbols:
        cached = _QUOTES_CACHE.get(s)
        if cached and now - cached[0] < _QUOTES_TTL_SECONDS:
            results.append(Quote(
                ticker=s, price=cached[1],
                currency="CAD" if s.endswith((".TO", ".V", ".CN", ".NE")) else "USD",
            ))
        else:
            fresh_needed.append(s)

    if fresh_needed:
        try:
            raw = yf.download(fresh_needed, period="5d", interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
            for s in fresh_needed:
                try:
                    # yf.download() always returns a per-ticker column level
                    # (a MultiIndex keyed by ticker), even for a single ticker.
                    series = raw[s]["Close"]
                    price = round(float(series.dropna().iloc[-1]), 2)
                    _QUOTES_CACHE[s] = (now, price)
                    results.append(Quote(
                        ticker=s, price=price,
                        currency="CAD" if s.endswith((".TO", ".V", ".CN", ".NE")) else "USD",
                    ))
                except Exception:
                    results.append(Quote(ticker=s, price=None, currency="USD", error="No data"))
        except Exception as e:
            logger.warning(f"Quotes fetch failed: {e}")
            for s in fresh_needed:
                results.append(Quote(ticker=s, price=None, currency="USD", error="Fetch failed"))

    return QuotesResponse(quotes=results, generated_at=datetime.now(timezone.utc).isoformat())


# ===========================================================================
# DATA QUALITY / SIGNAL RELIABILITY
#
# Prompted by a real case: Tesla showed BUY while a CAD-hedged Tesla product
# showed SELL. Both can't be describing Tesla. The hedged product trades a
# fraction of the volume, so RSI and MACD end up measuring the fund's own
# liquidity quirks — wide spreads, sparse trades, tracking error — rather
# than the underlying business.
#
# Technical indicators assume a liquid, continuously-priced market. When that
# assumption breaks, the numbers still compute and look authoritative. This
# flags when they shouldn't be trusted.
# ===========================================================================

DERIVATIVE_HINTS = (
    "HEDGED", "CAD-HEDGED", "ETF", "ETN", "TRUST", "INDEX", "2X", "3X",
    "BULL", "BEAR", "INVERSE", "LEVERAGED",
)


def assess_data_quality(
    ticker: str, hist: pd.DataFrame, info: dict
) -> DataQuality:
    warnings: List[str] = []
    quote_type = (info.get("quoteType") or "").upper() or None
    long_name = (info.get("longName") or info.get("shortName") or "").upper()

    avg_vol = None
    if "Volume" in hist and len(hist) >= 20:
        try:
            avg_vol = int(hist["Volume"].tail(60).mean())
        except Exception:
            avg_vol = None

    is_derivative = (
        quote_type in ("ETF", "MUTUALFUND")
        or any(hint in long_name for hint in DERIVATIVE_HINTS)
    )

    score = 0

    # Volume is the main driver — thin trading makes every indicator noisier.
    if avg_vol is None:
        warnings.append("Volume data unavailable, so signal confidence can't be assessed.")
        score += 2
    elif avg_vol < 50_000:
        warnings.append(
            f"Very thin trading (~{avg_vol:,} shares/day). At this volume, RSI and MACD are "
            "largely measuring the spread and a handful of trades, not real demand."
        )
        score += 3
    elif avg_vol < 250_000:
        warnings.append(
            f"Light trading (~{avg_vol:,} shares/day). Signals will be noisier than on a "
            "heavily-traded name."
        )
        score += 2
    elif avg_vol < 1_000_000:
        score += 1

    if is_derivative:
        warnings.append(
            "This looks like a fund or derivative product rather than a company's primary "
            "listing. Its price reflects the fund's own trading — spreads, tracking error, "
            "and hedging costs — layered on top of whatever it holds. If you're really "
            "interested in an underlying stock, analyze that stock's main listing instead: "
            "it has far more volume and much cleaner signals."
        )
        score += 2

    # Stale or gappy pricing
    if len(hist) >= 10:
        try:
            flat_days = int((hist["Close"].tail(20).diff() == 0).sum())
            if flat_days >= 5:
                warnings.append(
                    f"The price didn't move on {flat_days} of the last 20 sessions — a sign "
                    "of infrequent trading. Indicators built on flat data mean little."
                )
                score += 2
        except Exception:
            pass

    reliability = "Good" if score <= 1 else "Fair" if score <= 3 else "Poor"

    note = (
        "RSI, MACD and volume all assume a liquid, continuously-priced market. When that "
        "assumption doesn't hold, the formulas still produce confident-looking numbers — "
        "they're just describing the instrument's trading quirks rather than the business. "
        "A thin listing and its heavily-traded parent can easily disagree; when they do, "
        "the liquid one is the one telling you something real."
    )

    return DataQuality(
        reliability=reliability,
        avg_daily_volume=avg_vol,
        instrument_type=quote_type,
        is_derivative=is_derivative,
        warnings=warnings,
        note=note,
    )


# ===========================================================================
# HEDGED / CDR SIGNAL SUBSTITUTION
#
# Prompted by a real case: TSLA.NE (Tesla's CAD-hedged CDR on NEO) showed a
# different signal than TSLA itself. Both can't be describing Tesla — the CDR
# trades a small fraction of the volume, so its RSI/MACD mostly reflect the
# wrapper's own thin trading, not the business.
#
# CDR tickers do NOT reliably relate to their underlying ticker — confirmed
# via research: Chevron's CDR trades as CHEV (underlying CVX), Citigroup's as
# CITI (underlying C). Suffix-stripping would silently get these wrong. What
# IS reliable is Yahoo's own name for the instrument, which says "CDR (CAD
# Hedged)" and sometimes includes the real ticker in parentheses directly —
# e.g. "TESLA (TSLA) BMO CDR (CAD HEDGE". That's what this matches on.
# ===========================================================================

HEDGE_NAME_PATTERN = re.compile(
    r"(cad[\s\-]?hedge|usd[\s\-]?hedge|\bhedged?\b|\bcdr\b|depositary receipt|depository receipt)",
    re.IGNORECASE,
)
TICKER_HINT_PATTERN = re.compile(r"\(([A-Z]{1,5})\)")
_NOT_A_TICKER = {"CDR", "CAD", "USD", "ETF", "INC", "CORP", "LTD"}


def resolve_underlying_ticker(
    ticker: str, info: dict, thin_avg_volume: Optional[int]
) -> Optional[dict]:
    """
    For a CAD-hedged / CDR product, find the actual primary listing so a
    reliable signal can be shown instead of one computed on thin wrapper
    volume. Returns {"ticker": ..., "avg_volume": ...} or None if nothing
    trustworthy was found — this fails closed, not open.
    """
    name = info.get("longName") or info.get("shortName") or ""
    if not HEDGE_NAME_PATTERN.search(name):
        return None

    candidates: List[str] = []

    # Fast path: the name sometimes hands us the real ticker directly, e.g.
    # "TESLA (TSLA) BMO CDR (CAD HEDGE".
    for m in TICKER_HINT_PATTERN.finditer(name):
        token = m.group(1)
        if token not in _NOT_A_TICKER:
            candidates.append(token)

    # Fallback: search by the company name with the CDR qualifier cut off.
    cdr_match = re.search(r"\bcdr\b", name, re.IGNORECASE)
    base_name = name[: cdr_match.start()] if cdr_match else name
    base_name = TICKER_HINT_PATTERN.sub(" ", base_name)
    base_name = re.sub(r"[,.]", " ", base_name)
    base_name = re.sub(r"\s+", " ", base_name).strip()
    if base_name:
        try:
            candidates.extend(c.symbol.strip().upper() for c in search_symbols(base_name))
        except Exception as e:
            logger.info(f"Underlying search failed for '{base_name}': {e}")

    seen = set()
    for cand in candidates:
        cand = cand.strip().upper()
        if not cand or cand == ticker.strip().upper() or cand in seen:
            continue
        seen.add(cand)
        try:
            chist = yf.Ticker(cand).history(period="3mo", interval="1d")
            if chist is None or chist.empty or len(chist) < 30:
                continue
            cand_vol = int(chist["Volume"].tail(30).mean())
        except Exception:
            continue

        # Require the candidate to be meaningfully more liquid — otherwise
        # this isn't really "the main stock", just another thin listing.
        threshold = max((thin_avg_volume or 0) * 5, 200_000)
        if cand_vol > threshold:
            return {"ticker": cand, "avg_volume": cand_vol}

    return None


# ===========================================================================
# CONVICTION-LEVEL BACKTESTING
#
# The live app labels every signal High/Moderate/Low conviction, based on
# volume — and that label drives real behavior: it downgrades unreliable
# signals, it's the first thing shown on the readout. But it's never been
# tested. This checks whether "High conviction" BUY signals have actually
# outperformed "Low conviction" ones, or whether the label is decorative.
#
# Method mirrors the existing backtest: events are anchored to the START of
# each signal run (not every day it's active — the bug already found and
# fixed once this build), classified by conviction on the day the signal
# FIRED (matching how a person actually uses it — checked once, at signal
# onset, not re-checked daily), and compared against the same random-day
# baseline used everywhere else in this app.
# ===========================================================================

CONVICTION_MIN_EVENTS = 8


def classify_conviction_value(vol_ratio: Optional[float]) -> str:
    """Same thresholds as the live signal's generate_signal(), kept in sync deliberately."""
    if vol_ratio is None or (isinstance(vol_ratio, float) and math.isnan(vol_ratio)):
        return "Moderate"
    if vol_ratio >= 1.5:
        return "High"
    if vol_ratio >= 1.0:
        return "Moderate"
    return "Low"


class ConvictionBucket(BaseModel):
    conviction: str
    event_count: int
    avg_return_10d: Optional[float]
    win_rate_10d: Optional[float]
    edge_10d: Optional[float]
    plain: str


class ConvictionBacktest(BaseModel):
    ticker: str
    period: str
    buy_buckets: List[ConvictionBucket]
    sell_buckets: List[ConvictionBucket]
    verdict: str
    caveats: str
    generated_at: str


def _conviction_buckets_for(df: pd.DataFrame, signal_name: str) -> List[ConvictionBucket]:
    base = df["base_signal"]
    active = base == signal_name
    starts = active & ~active.shift(1, fill_value=False)
    event_index = df.index[starts]

    raw: Dict[str, list] = {"High": [], "Moderate": [], "Low": []}
    for idx in event_index:
        vr = df.loc[idx, "vol_ratio"] if "vol_ratio" in df.columns else None
        conv = classify_conviction_value(vr if pd.notna(vr) else None)
        fwd = df.loc[idx, "fwd10"]
        if pd.notna(fwd):
            raw[conv].append(float(fwd))

    baseline = df["fwd10"].dropna()
    base_avg = float(baseline.mean()) * 100 if len(baseline) else None

    buckets = []
    for conv in ["High", "Moderate", "Low"]:
        vals = raw[conv]
        n = len(vals)
        if n == 0:
            buckets.append(ConvictionBucket(
                conviction=conv, event_count=0, avg_return_10d=None,
                win_rate_10d=None, edge_10d=None,
                plain=f"{conv} conviction: no {signal_name} signals of this type in this period.",
            ))
            continue

        avg = float(np.mean(vals)) * 100
        if signal_name == "SELL":
            win = float(np.mean([v < 0 for v in vals])) * 100
            edge = (base_avg - avg) if base_avg is not None else None
        else:
            win = float(np.mean([v > 0 for v in vals])) * 100
            edge = (avg - base_avg) if base_avg is not None else None

        if n < CONVICTION_MIN_EVENTS:
            plain = f"{conv} conviction: only {n} signals — too few to judge."
        elif edge is not None and edge > 0.5:
            plain = f"{conv} conviction: beat random days by {edge:+.2f}% on average, across {n} signals."
        elif edge is not None and edge > 0:
            plain = f"{conv} conviction: barely beat random days ({edge:+.2f}%) across {n} signals — thin."
        else:
            plain = f"{conv} conviction: did no better than random days across {n} signals."

        buckets.append(ConvictionBucket(
            conviction=conv, event_count=n, avg_return_10d=round(avg, 2),
            win_rate_10d=round(win, 1),
            edge_10d=round(edge, 2) if edge is not None else None,
            plain=plain,
        ))
    return buckets


def backtest_conviction(ticker: str, period: str = "5y") -> ConvictionBacktest:
    ticker = ticker.strip().upper()
    df = _prepare_indicator_frame(ticker, period)

    buy_buckets = _conviction_buckets_for(df, "BUY")
    sell_buckets = _conviction_buckets_for(df, "SELL")

    def find(buckets, name):
        b = next((x for x in buckets if x.conviction == name), None)
        return b.edge_10d if b and b.event_count >= CONVICTION_MIN_EVENTS else None

    high_edge, low_edge = find(buy_buckets, "High"), find(buy_buckets, "Low")

    if high_edge is None or low_edge is None:
        verdict = (
            "Not enough signals in both the High and Low conviction buckets to compare them "
            "meaningfully on this stock — conviction naturally produces a small High-volume "
            "bucket, so this is common on quieter names."
        )
    elif high_edge > low_edge + 0.5:
        verdict = (
            f"Conviction is doing real work here: High-conviction BUY signals beat "
            f"Low-conviction ones by {high_edge - low_edge:.2f}% on average. The volume check "
            "is adding genuine information on this stock."
        )
    elif low_edge > high_edge + 0.5:
        verdict = (
            f"Conviction reads backwards on this stock: Low-conviction BUY signals actually "
            f"did better than High-conviction ones, by {low_edge - high_edge:.2f}%. Worth "
            "treating the conviction label skeptically here specifically."
        )
    else:
        verdict = (
            "High and Low conviction signals performed about the same here. Conviction isn't "
            "clearly adding information on this stock — it may just be noise dressed up as a "
            "confidence level."
        )

    caveats = (
        "This splits an already-modest number of signals into three smaller groups, so each "
        "bucket's sample is small — check the event count before trusting any single number. "
        "The High-conviction bucket is naturally the thinnest, since it requires a real volume "
        "spike. This looks at one stock over one period and describes the past, not a guarantee "
        "about what conviction will mean going forward."
    )

    return ConvictionBacktest(
        ticker=ticker, period=period, buy_buckets=buy_buckets, sell_buckets=sell_buckets,
        verdict=verdict, caveats=caveats, generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/backtest/{ticker}/conviction", response_model=ConvictionBacktest)
def backtest_conviction_endpoint(ticker: str, period: str = "5y"):
    """
    Does this app's own conviction label (High/Moderate/Low) actually predict
    better outcomes, or is it decorative? Tests the claim rather than assuming it.
    """
    if period not in ("2y", "5y", "10y"):
        period = "5y"
    return backtest_conviction(ticker, period)


# ---------------------------------------------------------------------------
# Cross-device sync (optional). Everything above this point works with zero
# database — holdings, watchlist, journal, etc. all live client-side in
# localStorage/AsyncStorage. This section adds an opt-in account layer so
# the same data can follow a person across devices: email+password auth
# (see auth.py) plus a generic per-user key-value store (see db.py) keyed
# by the exact same names the client already uses for its own local store,
# so syncing a key is just "push this value" / "pull all values," not a
# bespoke schema per feature. If DATABASE_URL/JWT_SECRET aren't set, every
# endpoint below 503s and the rest of the app is completely unaffected.
# ---------------------------------------------------------------------------
SYNC_ALLOWED_KEYS = {
    "ma_portfolio", "ma_watchlist", "ma_cash", "ma_journal", "ma_activity",
    "ma_scan_tickers", "ma_risk_pct", "ma_currency", "ma_ai_decisions",
    "ma_price_alerts",
}


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str


class MeResponse(BaseModel):
    email: str


class SyncSetRequest(BaseModel):
    value: Any


class SyncGetResponse(BaseModel):
    data: Dict[str, Any]


def _sync_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Cross-device sync isn't configured on this server yet.")


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency: verifies the bearer token and returns {"id", "email"}
    for the logged-in user, or raises. Extracts what's needed before closing
    the session rather than returning the ORM object itself, which would be
    detached (and its attributes unsafe to touch) once the session closes."""
    if db_lib.get_engine() is None:
        raise _sync_unavailable()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    user_id = auth_lib.decode_token(authorization[len("Bearer "):])
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please sign in again.")
    session = db_lib.get_session()
    try:
        user = session.get(db_lib.User, user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token. Please sign in again.")
        return {"id": user.id, "email": user.email}
    finally:
        session.close()


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(request: SignupRequest):
    if db_lib.get_engine() is None:
        raise _sync_unavailable()
    email = request.email.strip().lower()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    session = db_lib.get_session()
    try:
        if session.query(db_lib.User).filter(db_lib.User.email == email).first():
            raise HTTPException(status_code=409, detail="An account with this email already exists — try logging in instead.")
        user = db_lib.User(email=email, password_hash=auth_lib.hash_password(request.password))
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            # Two signups for the same email racing past the check above —
            # the unique index catches the loser here instead of both
            # succeeding, so turn it into the same 409 rather than a raw 500.
            session.rollback()
            raise HTTPException(status_code=409, detail="An account with this email already exists — try logging in instead.")
        session.refresh(user)
        token = auth_lib.create_token(user.id)
        if token is None:
            raise _sync_unavailable()
        return AuthResponse(token=token, email=user.email)
    finally:
        session.close()


@app.post("/api/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    if db_lib.get_engine() is None:
        raise _sync_unavailable()
    email = request.email.strip().lower()
    session = db_lib.get_session()
    try:
        user = session.query(db_lib.User).filter(db_lib.User.email == email).first()
        if user is None or not auth_lib.verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        token = auth_lib.create_token(user.id)
        if token is None:
            raise _sync_unavailable()
        return AuthResponse(token=token, email=user.email)
    finally:
        session.close()


@app.get("/api/auth/me", response_model=MeResponse)
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return MeResponse(email=current_user["email"])


@app.get("/api/sync", response_model=SyncGetResponse)
def sync_get_all(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Pulls every synced key for the logged-in user in one call — used both
    on login (to populate a fresh device) and on app start (to refresh)."""
    session = db_lib.get_session()
    try:
        rows = session.query(db_lib.UserData).filter(db_lib.UserData.user_id == current_user["id"]).all()
        return SyncGetResponse(data={row.key: row.value for row in rows})
    finally:
        session.close()


@app.put("/api/sync/{key}")
def sync_set_key(key: str, request: SyncSetRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Upserts one key's value — called on every local store.set() once the
    person is logged in, mirroring the client's own store.get/set(key, value)
    shape exactly."""
    if key not in SYNC_ALLOWED_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown sync key '{key}'.")
    session = db_lib.get_session()
    try:
        row = session.query(db_lib.UserData).filter(
            db_lib.UserData.user_id == current_user["id"], db_lib.UserData.key == key,
        ).first()
        if row:
            row.value = request.value
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
        else:
            # Two devices can race to create the same not-yet-existing key at
            # once (e.g. both pushing up local data right after login) — the
            # unique (user_id, key) constraint catches the loser here, which
            # retries as an update instead of raising.
            session.add(db_lib.UserData(user_id=current_user["id"], key=key, value=request.value))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                row = session.query(db_lib.UserData).filter(
                    db_lib.UserData.user_id == current_user["id"], db_lib.UserData.key == key,
                ).first()
                if row is None:
                    raise
                row.value = request.value
                row.updated_at = datetime.now(timezone.utc)
                session.commit()
        return {"ok": True}
    finally:
        session.close()


# ===========================================================================
# BACKGROUND REFRESH LOOP
#
# Stocklake-first plan, P0+P1: nothing above this point should ever be the
# thing a user request waits on. Screener, the under-$20 scan, high-volume,
# and market regime were all previously fetched live on whichever request
# happened to arrive after their cache went stale — meaning a slow or down
# vendor turned into a slow or hung *response*. This loop refreshes those
# same cache dicts proactively on a timer instead, so a request only ever
# reads whatever's currently cached; see run_screener()/run_under20_screener()/
# run_high_volume_screener()'s own force=True read-through-on-miss behavior
# below for the "cache is still empty" edge case (a cold start, before this
# loop's first cycle completes).
#
# Runs in a plain OS thread, not an asyncio task, because
# fetch_stocklake_context() calls asyncio.run() internally — that raises if
# invoked from a thread that already has a running event loop. uvicorn's
# main thread has one; a plain background thread doesn't.
# ===========================================================================

_BACKGROUND_REFRESH_ENABLED = os.environ.get("DISABLE_BACKGROUND_REFRESH", "").lower() != "true"
_BACKGROUND_REFRESH_TICK_SECONDS = 60


def _background_refresh_loop() -> None:
    # Small startup delay so this doesn't compete with the app's own
    # first-request warm-up for the same worker/network resources.
    time.sleep(5)
    last_run = {"screener": 0.0, "under20": 0.0, "high_volume": 0.0, "regime": 0.0}
    while True:
        now = time.time()
        if now - last_run["screener"] >= _SCREENER_TTL_SECONDS:
            try:
                run_screener(force=True)
            except Exception as e:
                logger.error(f"Background refresh failed (screener universe): {e}")
            last_run["screener"] = time.time()
        if now - last_run["under20"] >= _UNDER20_TTL_SECONDS:
            try:
                run_under20_screener(force=True)
            except Exception as e:
                logger.error(f"Background refresh failed (under-$20 universe): {e}")
            last_run["under20"] = time.time()
        if now - last_run["high_volume"] >= _HIGH_VOLUME_TTL_SECONDS:
            try:
                run_high_volume_screener(force=True)
            except Exception as e:
                logger.error(f"Background refresh failed (high-volume screener): {e}")
            last_run["high_volume"] = time.time()
        if now - last_run["regime"] >= _REGIME_TTL_SECONDS:
            try:
                get_market_regime(canadian=False)
                get_market_regime(canadian=True)
            except Exception as e:
                logger.error(f"Background refresh failed (market regime): {e}")
            last_run["regime"] = time.time()
        time.sleep(_BACKGROUND_REFRESH_TICK_SECONDS)


@app.on_event("startup")
def _start_background_refresh() -> None:
    if not _BACKGROUND_REFRESH_ENABLED:
        logger.info("Background refresh loop disabled via DISABLE_BACKGROUND_REFRESH.")
        return
    threading.Thread(target=_background_refresh_loop, daemon=True, name="background-refresh").start()
    logger.info("Background refresh loop started.")
