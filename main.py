"""
Stock Market Analysis API
Provides technical analysis (RSI + MACD) and AI-driven news sentiment (FinBERT)
for US and Canadian equities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Fewer threads = less memory overhead from torch's internal thread pool.
# This machine only has 0.5 CPU on Render's smaller tiers anyway, so there's
# no performance cost to this.
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-analyzer")

app = FastAPI(title="Stock Market Analysis API", version="1.0.0")

# Allow the Expo app (running on your phone / emulator) to reach this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# FinBERT is loaded once, at process startup, so requests don't pay the
# multi-second model-load cost on every call.
# ---------------------------------------------------------------------------
_sentiment_pipeline = None


@app.on_event("startup")
def load_model():
    global _sentiment_pipeline
    logger.info("Loading FinBERT model (ProsusAI/finbert)... this can take a moment.")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    # Dynamic int8 quantization: the model's weights are compressed from
    # 32-bit floats to 8-bit integers after loading. This roughly quarters
    # the model's memory footprint with only a negligible accuracy cost —
    # important for fitting inside a 512MB hosting instance.
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
    signal: str
    reasoning: str


class Headline(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    sentiment: str
    sentiment_score: float


class SentimentAnalysis(BaseModel):
    bullish_score: float
    overall_impact: str
    headline_count: int
    headlines: List[Headline]


class PriceTargets(BaseModel):
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    note: str


class AnalysisResponse(BaseModel):
    ticker: str
    company_name: str
    current_price: Optional[float]
    currency: str
    market: str
    technical_analysis: TechnicalAnalysis
    price_targets: PriceTargets
    sentiment_analysis: SentimentAnalysis
    generated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_market(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".TO"):
        return "TSX (Canada)"
    if t.endswith(".V"):
        return "TSX-V (Canada)"
    if t.endswith(".CN"):
        return "CSE (Canada)"
    return "US"


def fetch_price_history(ticker: str):
    """6 months of daily bars — enough history for a 26/9 MACD and 14-day RSI."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo", interval="1d", auto_adjust=True)
    return stock, hist


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Standard Wilder's RSI, no external library needed."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Standard MACD: fast EMA minus slow EMA, plus a signal EMA of that line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(hist: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range — a standard measure of how much a stock typically
    moves per day. Used below to size stop-loss / take-profit distances off
    the stock's own volatility rather than an arbitrary percentage.
    """
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def compute_price_targets(hist: pd.DataFrame, signal: str, current_price: Optional[float]) -> PriceTargets:
    """
    Generates entry/stop-loss/take-profit levels using ATR-based volatility
    sizing (a standard, well-known technique — not a prediction of where the
    price will actually go). Support/resistance use the trailing 20-day
    price range as simple reference levels regardless of signal.
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

    if atr_val is None or atr_val == 0:
        return PriceTargets(
            entry=None, stop_loss=None, take_profit=None, risk_reward_ratio=None,
            support=support, resistance=resistance,
            note="Not enough price history to size a volatility-based stop yet. " + default_note,
        )

    # 1.5x ATR stop, 3x ATR target -> roughly a 2:1 reward-to-risk setup,
    # a common convention in technical trading.
    stop_multiplier, target_multiplier = 1.5, 3.0

    if signal == "BUY":
        entry = current_price
        stop_loss = round(entry - stop_multiplier * atr_val, 2)
        take_profit = round(entry + target_multiplier * atr_val, 2)
        note = (
            "Entry at current price; stop-loss and take-profit are sized off recent "
            "volatility (ATR) for roughly a 2:1 reward-to-risk setup. " + default_note
        )
    elif signal == "SELL":
        entry = current_price
        stop_loss = round(entry + stop_multiplier * atr_val, 2)
        take_profit = round(entry - target_multiplier * atr_val, 2)
        note = (
            "These levels apply to a short position, or as an exit reference if you're "
            "already holding the stock. " + default_note
        )
    else:  # HOLD
        entry, stop_loss, take_profit = None, None, None
        note = (
            "No new entry is suggested while the signal is HOLD. Support and resistance "
            "below are levels worth watching. " + default_note
        )

    risk_reward_ratio = None
    if entry is not None and stop_loss is not None and take_profit is not None:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk > 0:
            risk_reward_ratio = round(reward / risk, 2)

    return PriceTargets(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=risk_reward_ratio,
        support=support,
        resistance=resistance,
        note=note,
    )


def compute_technicals(hist: pd.DataFrame) -> TechnicalAnalysis:
    if hist is None or hist.empty or len(hist) < 30:
        raise HTTPException(
            status_code=422,
            detail="Not enough price history to compute reliable RSI/MACD (need 30+ trading days).",
        )

    close = hist["Close"]

    rsi_series = compute_rsi(close, length=14)
    macd_line, signal_line, histogram = compute_macd(close, fast=12, slow=26, signal=9)

    rsi_clean = rsi_series.dropna()
    rsi_val = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else None

    macd_val = float(macd_line.iloc[-1]) if not macd_line.dropna().empty else None
    macd_signal_val = float(signal_line.iloc[-1]) if not signal_line.dropna().empty else None
    macd_hist_val = float(histogram.iloc[-1]) if not histogram.dropna().empty else None

    signal, reasoning = generate_signal(rsi_val, macd_hist_val)

    return TechnicalAnalysis(
        rsi=round(rsi_val, 2) if rsi_val is not None else None,
        macd=round(macd_val, 4) if macd_val is not None else None,
        macd_signal=round(macd_signal_val, 4) if macd_signal_val is not None else None,
        macd_histogram=round(macd_hist_val, 4) if macd_hist_val is not None else None,
        signal=signal,
        reasoning=reasoning,
    )


def generate_signal(rsi: Optional[float], macd_hist: Optional[float]) -> Tuple[str, str]:
    """
    Transparent scoring model combining RSI and MACD histogram:
      - RSI < 30 (oversold)   -> +1 bullish point
      - RSI > 70 (overbought) -> -1 bearish point
      - MACD histogram > 0 (bullish momentum) -> +1
      - MACD histogram < 0 (bearish momentum) -> -1
    score >= 1  -> BUY
    score <= -1 -> SELL
    otherwise   -> HOLD
    """
    if rsi is None or macd_hist is None:
        return "HOLD", "Insufficient indicator data to generate a confident signal."

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

    if score >= 1:
        signal = "BUY"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, "; ".join(reasons) + "."


def extract_headline(item: dict) -> Optional[dict]:
    """
    yfinance's `Ticker.news` schema has changed across versions. This handles
    both the flat legacy format and the newer nested 'content' format.
    """
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
        return {
            "title": title,
            "publisher": item.get("publisher"),
            "link": item.get("link"),
        }
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
        label = r["label"].lower()  # "positive" | "negative" | "neutral"
        score = float(r["score"])

        if label == "positive":
            directional_total += score
        elif label == "negative":
            directional_total -= score
        # neutral contributes 0

        scored_headlines.append(
            Headline(
                title=h["title"],
                publisher=h.get("publisher"),
                link=h.get("link"),
                sentiment=label,
                sentiment_score=round(score, 3),
            )
        )

    avg_directional = directional_total / len(headlines)  # roughly in [-1, 1]
    bullish_score = round((avg_directional + 1) / 2, 3)  # normalize to [0, 1]

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
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Stock Market Analysis API"}


@app.get("/health")
def health():
    """Lightweight check — useful for uptime monitors that keep a free-tier server awake."""
    return {"status": "healthy", "model_loaded": _sentiment_pipeline is not None}


@app.get("/api/analyze/{ticker}", response_model=AnalysisResponse)
def analyze(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

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
    currency = info.get("currency") or (
        "CAD" if ticker.endswith((".TO", ".V", ".CN")) else "USD"
    )
    current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
    current_price_rounded = round(current_price, 2) if current_price is not None else None

    price_targets = compute_price_targets(hist, technical.signal, current_price_rounded)

    headlines = fetch_headlines(stock, limit=5)
    sentiment = analyze_sentiment(headlines)

    return AnalysisResponse(
        ticker=ticker,
        company_name=company_name,
        current_price=current_price_rounded,
        currency=currency,
        market=detect_market(ticker),
        technical_analysis=technical,
        price_targets=price_targets,
        sentiment_analysis=sentiment,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
