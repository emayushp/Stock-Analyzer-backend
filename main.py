"""
Stock Market Analysis API
Technical analysis (RSI + MACD + Volume), volatility-based price targets,
company fundamentals, and AI news sentiment (FinBERT) for US & Canadian equities.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Fewer threads = less memory overhead from torch's internal thread pool.
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-analyzer")

app = FastAPI(title="Stock Market Analysis API", version="2.0.0")

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
    # US large caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT",
    "XOM", "UNH", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "AMD", "NFLX",
    "CRM", "INTC", "DIS", "BA", "PFE",
    # Canadian large caps
    "RY.TO", "TD.TO", "ENB.TO", "CNR.TO", "BMO.TO", "BNS.TO", "CP.TO",
    "SHOP.TO", "SU.TO", "CNQ.TO", "ATD.TO", "MFC.TO",
]

# A curated set of TSX / TSX-V names that commonly trade under CAD $20 — used
# by the Brief's "Canadian Stocks Under $20" section. Same rationale as above:
# a fixed, known list rather than scanning the whole exchange.
CANADIAN_UNDER_20_UNIVERSE = [
    "AC.TO", "BB.TO", "CGX.TO", "BTE.TO", "MEG.TO", "NPI.TO", "TOU.TO",
    "WPM.TO", "KEY.TO", "PEY.TO", "CVE.TO", "BTO.TO", "IMG.TO", "ELD.TO",
    "AGI.TO", "FM.TO", "TA.TO", "H.TO", "GIB-A.TO", "DOO.TO",
]

_SCREENER_CACHE: Dict[str, Tuple[float, Any]] = {}
_SCREENER_TTL_SECONDS = 3600  # 1 hour — screening is not a minute-to-minute activity

_UNDER20_CACHE: Dict[str, Tuple[float, Any]] = {}
_UNDER20_TTL_SECONDS = 3600

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


_sentiment_pipeline = None


@app.on_event("startup")
def load_model():
    global _sentiment_pipeline
    logger.info("Loading FinBERT model (ProsusAI/finbert)... this can take a moment.")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    # int8 dynamic quantization keeps the resident memory footprint small.
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()

    _sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    logger.info("FinBERT loaded (quantized).")


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
    signal: str
    conviction: str
    reasoning: str


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
    earnings: EarningsInfo
    technical_analysis: TechnicalAnalysis
    price_targets: PriceTargets
    sentiment_analysis: SentimentAnalysis
    generated_at: str
    cached: bool = False


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


class PortfolioHolding(BaseModel):
    ticker: str
    shares: float
    cost_basis: float  # average price paid per share


class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]


class HoldingResult(BaseModel):
    ticker: str
    shares: float
    cost_basis: float
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
    summary: str
    generated_at: str


class DigestRequest(BaseModel):
    holdings: List[PortfolioHolding] = []
    watchlist: List[str] = []


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
    return "US"


def fetch_price_history(ticker: str):
    """6 months of daily bars — enough for a 26/9 MACD, 14-day RSI and 20-day volume avg."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo", interval="1d", auto_adjust=True)
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
    if hist is None or hist.empty or len(hist) < 30:
        raise HTTPException(
            status_code=422,
            detail="Not enough price history to compute reliable indicators (need 30+ trading days).",
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

    return TechnicalAnalysis(
        rsi=round(rsi_val, 2) if rsi_val is not None else None,
        macd=round(macd_val, 4) if macd_val is not None else None,
        macd_signal=round(macd_signal_val, 4) if macd_signal_val is not None else None,
        macd_histogram=round(macd_hist_val, 4) if macd_hist_val is not None else None,
        volume_ratio=volume_ratio,
        volume_note=describe_volume(volume_ratio),
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


def fetch_earnings(stock: yf.Ticker) -> EarningsInfo:
    """
    Next earnings date plus the last few quarters' EPS estimate vs. actual.
    Coverage varies a lot by ticker — especially thinner for smaller Canadian
    names — so this degrades gracefully to "not available" rather than error.
    """
    fallback = EarningsInfo(
        next_earnings_date=None, recent_quarters=[],
        note="Earnings data isn't available for this ticker.",
    )
    try:
        df = stock.get_earnings_dates(limit=8)
    except Exception as e:
        logger.info(f"Earnings fetch skipped: {e}")
        return fallback

    if df is None or df.empty:
        return fallback

    try:
        now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz is not None else pd.Timestamp.now()
        upcoming = df[df.index > now]
        past = df[df.index <= now].sort_index(ascending=False)

        next_date = upcoming.index.min().strftime("%Y-%m-%d") if not upcoming.empty else None

        quarters = []
        for idx, row in past.head(4).iterrows():
            est = row.get("EPS Estimate")
            act = row.get("Reported EPS")
            surprise = row.get("Surprise(%)")
            quarters.append(
                EarningsQuarter(
                    date=idx.strftime("%Y-%m-%d"),
                    eps_estimate=round(float(est), 2) if pd.notna(est) else None,
                    eps_actual=round(float(act), 2) if pd.notna(act) else None,
                    surprise_pct=round(float(surprise), 2) if pd.notna(surprise) else None,
                )
            )

        return EarningsInfo(
            next_earnings_date=next_date,
            recent_quarters=quarters,
            note="Beat = actual EPS came in above analyst estimates; miss = below. Large surprises often move price sharply on the report date.",
        )
    except Exception as e:
        logger.warning(f"Earnings parsing failed: {e}")
        return fallback


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
# Headline "why it matters" tagging
#
# Honest caveat about how this works: FinBERT gives sentiment (positive /
# negative / neutral), not an understanding of WHY a headline matters. Rather
# than pretend otherwise, this matches common, well-understood financial-news
# patterns (earnings beats, guidance cuts, layoffs, M&A, etc.) and explains
# those specifically. Headlines that don't match a pattern fall back to a
# sentiment-based explanation — genuinely less specific, and labeled as such.
# ---------------------------------------------------------------------------
HEADLINE_PATTERNS = [
    (r"\bbeat[s]?\b.{0,20}\bestimate|exceed[s]?.{0,20}expectation", "positive",
     "Earnings or revenue came in above what analysts expected — often supports the price short-term."),
    (r"\bmiss(?:es|ed)?\b.{0,20}\bestimate|below.{0,20}expectation|fell short", "negative",
     "Results came in below analyst expectations — often pressures the price short-term."),
    (r"\braises?\b.{0,20}\bguidance|\bguidance\b.{0,20}\braise|upgrad", "positive",
     "The company or an analyst raised expectations for future performance."),
    (r"\bcuts?\b.{0,20}\bguidance|\blowers?\b.{0,20}\bguidance|downgrad|slash", "negative",
     "Guidance was lowered or an analyst downgraded outlook — signals reduced confidence ahead."),
    (r"\blayoff|\bjob cuts|\brestructur", "negative",
     "Workforce reductions often signal cost pressure, though markets sometimes read this as improved efficiency."),
    (r"\bacqui(?:re|sition)|\bmerger|\bbuyout|\btakeover", "neutral",
     "M&A activity — impact depends heavily on price paid and strategic fit; can go either way."),
    (r"\blawsuit|\bsu(?:e|ing|it)\b|\bregulat|\bprobe|\binvestigat|\bfine\b", "negative",
     "Legal or regulatory scrutiny introduces uncertainty and potential costs."),
    (r"\brecall\b", "negative",
     "Product recalls carry direct costs and can dent brand trust."),
    (r"\bdividend\b.{0,20}\b(?:raise|increase|hike)", "positive",
     "A dividend increase signals management's confidence in sustained cash flow."),
    (r"\bdividend\b.{0,20}\b(?:cut|suspend|reduce)", "negative",
     "A dividend cut often signals real cash-flow strain."),
    (r"\bprice target\b.{0,20}\braise|\braise[sd]?\b.{0,20}\bprice target", "positive",
     "An analyst raised their price target — a vote of confidence, though targets are frequently wrong."),
    (r"\bprice target\b.{0,20}\bcut|\bcut[s]?\b.{0,20}\bprice target|\blower.{0,20}price target", "negative",
     "An analyst cut their price target — signals reduced confidence, though targets are frequently wrong."),
    (r"\blaunch|\bunveil|\bnew product", "positive",
     "New product news can drive near-term attention, though actual sales impact takes longer to show up."),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), tone, expl) for p, tone, expl in HEADLINE_PATTERNS]


def explain_headline(title: str, sentiment: str) -> str:
    for pattern, _tone, explanation in _COMPILED_PATTERNS:
        if pattern.search(title):
            return explanation
    # No specific pattern matched — fall back to the sentiment reading alone,
    # and say so, rather than implying a specific reason that isn't there.
    if sentiment == "positive":
        return "Reads positive in tone based on the headline's language — read the full article for the specific reason."
    if sentiment == "negative":
        return "Reads negative in tone based on the headline's language — read the full article for the specific reason."
    return "Tone reads as neutral — no strong positive or negative signal from the headline alone."


def extract_headline(item: dict) -> Optional[dict]:
    """yfinance's news schema has changed across versions — handle both shapes."""
    if "content" in item and isinstance(item["content"], dict):
        content = item["content"]
        title = content.get("title")
        publisher = (content.get("provider") or {}).get("displayName")
        link = (content.get("canonicalUrl") or {}).get("url") or (
            content.get("clickThroughUrl") or {}
        ).get("url")
        if title:
            return {"title": title, "publisher": publisher, "link": link}

    title = item.get("title")
    if title:
        return {"title": title, "publisher": item.get("publisher"), "link": item.get("link")}
    return None


def fetch_headlines(stock: yf.Ticker, limit: int = 5) -> List[dict]:
    try:
        raw_news = stock.news or []
    except Exception as e:
        logger.warning(f"Could not fetch news: {e}")
        return []

    headlines = []
    for item in raw_news:
        parsed = extract_headline(item)
        if parsed:
            headlines.append(parsed)
        if len(headlines) >= limit:
            break
    return headlines


def analyze_sentiment(headlines: List[dict]) -> SentimentAnalysis:
    if not headlines or _sentiment_pipeline is None:
        return SentimentAnalysis(
            bullish_score=0.5,
            overall_impact="Neutral (no recent news available)",
            headline_count=0,
            headlines=[],
        )

    titles = [h["title"] for h in headlines]
    results = _sentiment_pipeline(titles)

    scored_headlines = []
    directional_total = 0.0

    for h, r in zip(headlines, results):
        label = r["label"].lower()
        score = float(r["score"])

        if label == "positive":
            directional_total += score
        elif label == "negative":
            directional_total -= score

        scored_headlines.append(
            Headline(
                title=h["title"],
                publisher=h.get("publisher"),
                link=h.get("link"),
                sentiment=label,
                sentiment_score=round(score, 3),
                context=explain_headline(h["title"], label),
            )
        )

    avg_directional = directional_total / len(headlines)
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
# Batched multi-ticker scoring
#
# yf.download() pulls many tickers in a SINGLE request, which is the only
# practical way to score dozens of names without tripping Yahoo's rate limits.
# Technicals only here — running FinBERT across 37 tickers' worth of headlines
# would be far too slow and memory-hungry for a per-request call.
# ---------------------------------------------------------------------------
def batch_score(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    try:
        raw = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Batch download failed: {e}")
        return {t: {"error": "Could not fetch market data."} for t in tickers}

    if raw is None or raw.empty:
        return {t: {"error": "No market data returned."} for t in tickers}

    for t in tickers:
        try:
            # Multi-ticker downloads come back with a per-ticker column level;
            # a single-ticker download comes back flat.
            if len(tickers) == 1:
                df = raw
            else:
                if t not in raw.columns.get_level_values(0):
                    results[t] = {"error": "No data found for this ticker."}
                    continue
                df = raw[t]

            df = df.dropna(how="all")
            if df.empty or len(df) < 30:
                results[t] = {"error": "Not enough price history."}
                continue

            close = df["Close"].dropna()
            if close.empty:
                results[t] = {"error": "No closing prices available."}
                continue

            rsi_series = compute_rsi(close, length=14).dropna()
            _, _, histogram = compute_macd(close)
            hist_clean = histogram.dropna()

            rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
            macd_hist_val = float(hist_clean.iloc[-1]) if not hist_clean.empty else None
            volume_ratio = compute_volume_ratio(df)

            signal, conviction, reasoning = generate_signal(rsi_val, macd_hist_val, volume_ratio)

            price = round(float(close.iloc[-1]), 2)
            change_pct = None
            if len(close) >= 2:
                prev = float(close.iloc[-2])
                if prev:
                    change_pct = round((price - prev) / prev * 100, 2)

            results[t] = {
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
        except Exception as e:
            logger.warning(f"Scoring failed for {t}: {e}")
            results[t] = {"error": "Could not analyze this ticker."}

    return results


CONVICTION_RANK = {"High": 3, "Moderate": 2, "Low": 1, "Neutral": 0}


def sort_hits(hits: List[ScreenerHit]) -> List[ScreenerHit]:
    """Strongest conviction first; RSI as a tiebreak toward the more stretched names."""
    return sorted(
        hits,
        key=lambda h: (CONVICTION_RANK.get(h.conviction, 0), abs((h.rsi or 50) - 50)),
        reverse=True,
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


def analyze_portfolio(holdings: List[PortfolioHolding]) -> PortfolioResponse:
    if not holdings:
        return PortfolioResponse(
            holdings=[], total_cost=0, total_value=0, total_pl=0, total_pl_pct=0,
            total_day_pl=0, total_day_pl_pct=0,
            summary="No holdings added yet.",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    tickers = [h.ticker.strip().upper() for h in holdings]
    scored = batch_score(tickers)

    results: List[HoldingResult] = []
    total_cost = total_value = 0.0

    for h in holdings:
        t = h.ticker.strip().upper()
        r = scored.get(t, {})
        currency = "CAD" if t.endswith((".TO", ".V", ".CN")) else "USD"
        cost = h.shares * h.cost_basis
        total_cost += cost

        if r.get("error") or r.get("price") is None:
            results.append(
                HoldingResult(
                    ticker=t, shares=h.shares, cost_basis=h.cost_basis,
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
        value = h.shares * price
        total_value += value
        pl = value - cost
        pl_pct = round((pl / cost) * 100, 2) if cost else None

        change_pct = r.get("change_pct")
        day_pl = None
        if change_pct is not None and price:
            prev_close = price / (1 + change_pct / 100)
            day_pl = round(h.shares * (price - prev_close), 2)

        results.append(
            HoldingResult(
                ticker=t, shares=h.shares, cost_basis=h.cost_basis,
                current_price=price, currency=currency,
                market_value=round(value, 2),
                unrealized_pl=round(pl, 2), unrealized_pl_pct=pl_pct,
                day_pl=day_pl, day_pl_pct=change_pct,
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

    total_day_pl = round(sum(r.day_pl for r in results if r.day_pl is not None), 2)
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

    return PortfolioResponse(
        holdings=results,
        total_cost=round(total_cost, 2),
        total_value=round(total_value, 2),
        total_pl=round(total_pl, 2),
        total_pl_pct=total_pl_pct,
        total_day_pl=total_day_pl,
        total_day_pl_pct=total_day_pl_pct,
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
    return {"status": "ok", "service": "Stock Market Analysis API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _sentiment_pipeline is not None,
        "cached_tickers": len(_CACHE),
    }


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = ""):
    """Look up tickers by company name or partial symbol, e.g. 'Apple' -> AAPL."""
    q = q.strip()
    if len(q) < 2:
        return SearchResponse(query=q, results=[])
    return SearchResponse(query=q, results=search_symbols(q))


@app.get("/api/analyze/{ticker}", response_model=AnalysisResponse)
def analyze(ticker: str):
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
    currency = info.get("currency") or ("CAD" if ticker.endswith((".TO", ".V", ".CN")) else "USD")

    close = hist["Close"]
    current_price = round(float(close.iloc[-1]), 2)

    price_change = price_change_pct = None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev:
            price_change = round(current_price - prev, 2)
            price_change_pct = round((current_price - prev) / prev * 100, 2)

    price_history = [round(float(v), 2) for v in close.tail(30).tolist()]

    fundamentals = build_fundamentals(info, current_price)
    earnings = fetch_earnings(stock)
    price_targets = compute_price_targets(hist, technical.signal, current_price)

    headlines = fetch_headlines(stock, limit=5)
    sentiment = analyze_sentiment(headlines)

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
        earnings=earnings,
        technical_analysis=technical,
        price_targets=price_targets,
        sentiment_analysis=sentiment,
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )

    cache_set(ticker, response.model_dump())
    return response


@app.get("/api/screener", response_model=ScreenerResponse)
def screener(force: bool = False):
    """Scan the fixed universe for current BUY / SELL technical signals."""
    return run_screener(force=force)


@app.get("/api/screener/under20", response_model=ScreenerResponse)
def screener_under20(force: bool = False):
    """Canadian stocks under CAD $20, screened for current momentum signals."""
    return run_under20_screener(force=force)


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


@app.post("/api/digest", response_model=DigestResponse)
def digest(request: DigestRequest, force: bool = False):
    """
    The morning brief: portfolio status, watchlist signals, and screener hits
    in one call. Technicals only — see the note field. Pass ?force=true to
    bypass the screener's hourly cache and pull fresh signals.
    """
    portfolio = analyze_portfolio(request.holdings) if request.holdings else None

    watchlist_signals: List[ScreenerHit] = []
    if request.watchlist:
        scored = batch_score(request.watchlist)
        for r in scored.values():
            if not r.get("error"):
                watchlist_signals.append(ScreenerHit(**r))
        watchlist_signals = sort_hits(watchlist_signals)

    screen = run_screener(force=force)
    under20 = run_under20_screener(force=force)

    # Don't repeat names the person already holds or watches under "opportunities".
    known = {h.ticker.strip().upper() for h in request.holdings} | {
        t.strip().upper() for t in request.watchlist
    }
    opportunities = [h for h in screen.buy_candidates if h.ticker not in known][:8]
    warnings = [h for h in screen.sell_candidates if h.ticker not in known][:5]
    under20_buys = [h for h in under20.buy_candidates if h.ticker not in known][:8]
    under20_avoid = [h for h in under20.sell_candidates if h.ticker not in known][:5]

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
            "from a fixed list of large-caps, and are not recommendations. Tap any "
            "ticker for the full analysis including news sentiment."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
