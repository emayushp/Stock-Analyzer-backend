"""
Pure indicator functions over an OHLCV bars DataFrame.

No network, no caching, no config reads — every threshold lives in
config.py and every gate lives in its layer module. Functions here take a
DataFrame (or a Series) and return a Series or a scalar, so they can be
unit-tested against a fixed hand-checked sample (tests/test_indicators.py).

Deliberately pandas-only: numpy ships with pandas, but declaring a direct
dependency on it would mean pinning a second version into requirements.txt
against a live deployment for no functional gain.

Conventions worth knowing before changing anything here:
  * Wilder smoothing (ATR, ADX) is an EWM with alpha = 1/period and
    adjust=False, which is the standard equivalent of Wilder's running
    average.
  * Bollinger bands use POPULATION standard deviation (ddof=0), the usual
    Bollinger convention. pandas defaults to sample (ddof=1), so this is
    passed explicitly everywhere it matters.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

OHLC_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def has_ohlcv(bars: Optional[pd.DataFrame]) -> bool:
    """True when `bars` looks like a usable OHLCV frame."""
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        return False
    return all(col in bars.columns for col in OHLC_COLUMNS)


def latest(series: Optional[pd.Series]) -> Optional[float]:
    """Last non-null value of a series as a plain float, or None.

    Every caller in this package goes through here rather than indexing
    with [-1] directly: a series that is empty, all-NaN, or shorter than
    its own warm-up period is an ordinary outcome for a thinly-traded or
    recently-listed symbol, not an error worth raising."""
    try:
        if series is None or len(series) == 0:
            return None
        cleaned = series.dropna()
        if cleaned.empty:
            return None
        value = float(cleaned.iloc[-1])
        return None if pd.isna(value) else value
    except Exception:
        return None


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running average — the smoothing ATR and ADX are defined in."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(high-low, |high-prev_close|, |low-prev_close|).

    On the first bar there is no previous close, so the two gap terms are
    NaN and the max collapses to high-low, which is the correct seed."""
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's Average True Range."""
    return wilder_rma(true_range(bars), period)


def atr_pct(bars: pd.DataFrame, period: int) -> pd.Series:
    """ATR as a percentage of close — comparable across price levels."""
    return atr(bars, period) / bars["Close"] * 100.0


def bollinger_band_width(close: pd.Series, period: int, num_std: float) -> pd.Series:
    """(upper - lower) / middle for BB(period, num_std).

    Reduces algebraically to 2 * num_std * stdev / mean, but is written out
    in band terms so it stays readable next to the spec."""
    middle = close.rolling(window=period, min_periods=period).mean()
    stdev = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * stdev
    lower = middle - num_std * stdev
    return (upper - lower) / middle


def percentile_rank(series: pd.Series, lookback: int) -> Optional[float]:
    """Where the most recent value sits within its own trailing `lookback`
    window, 0-100.

    0 means it is the lowest value in the window, 100 the highest. Defined
    as the share of OTHER values in the window that are strictly below the
    current one, which keeps the endpoints exact and the arithmetic simple
    enough to check by hand."""
    try:
        cleaned = series.dropna()
        if len(cleaned) < 2:
            return None
        window = cleaned.tail(lookback)
        if len(window) < 2:
            return None
        current = float(window.iloc[-1])
        below = int((window < current).sum())
        return round(below / (len(window) - 1) * 100.0, 2)
    except Exception:
        return None


def adx(bars: pd.DataFrame, period: int) -> pd.Series:
    """Standard Wilder ADX.

    +DM/-DM are the directional moves, each smoothed by Wilder's average
    and normalized by ATR to give +DI/-DI; DX is their normalized spread,
    and ADX is DX smoothed again over the same period."""
    high, low = bars["High"], bars["Low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_series = wilder_rma(true_range(bars), period)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / atr_series
    minus_di = 100.0 * wilder_rma(minus_dm, period) / atr_series

    # A zero DI sum means neither side moved at all that bar; NaN (not
    # pd.NA) keeps the series float64 so the second smoothing pass and the
    # eventual float() in latest() both stay well-typed.
    di_sum = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return wilder_rma(dx, period)


def slope_over(series: pd.Series, lookback: int) -> Optional[float]:
    """Least-squares slope per session over the last `lookback` points.

    Closed-form rather than a polyfit call, both to stay pandas-only and
    because the formula is checkable by hand in a test."""
    try:
        cleaned = series.dropna()
        if len(cleaned) < lookback or lookback < 2:
            return None
        window = cleaned.tail(lookback)
        n = len(window)
        sum_x = n * (n - 1) / 2.0
        sum_xx = (n - 1) * n * (2 * n - 1) / 6.0
        sum_y = float(window.sum())
        sum_xy = float(sum(i * float(v) for i, v in enumerate(window)))
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return None
        return (n * sum_xy - sum_x * sum_y) / denominator
    except Exception:
        return None


def williams_ad(bars: pd.DataFrame) -> pd.Series:
    """Williams Accumulation/Distribution, cumulative.

    Each bar contributes close - true_low on an up close, close - true_high
    on a down close, and nothing on an unchanged close, where true low/high
    extend the bar's own range to include the previous close."""
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    prev_close = close.shift(1)

    true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)

    contribution = pd.Series(0.0, index=bars.index)
    contribution = contribution.where(~(close > prev_close), close - true_low)
    contribution = contribution.where(~(close < prev_close), close - true_high)
    return contribution.fillna(0.0).cumsum()


def normalized_ad_slope(ad_line: pd.Series, lookback: int) -> Optional[float]:
    """A/D slope expressed in typical-daily-flow units.

    The raw slope is in price units and scales with both the price level
    and how active the name is, so it can't be compared across symbols.
    Dividing by the mean absolute daily change over the same window gives a
    dimensionless number: roughly +1 means the line advanced a full typical
    day's flow every session (steady one-way accumulation), 0 means the
    flow cancelled out."""
    try:
        raw = slope_over(ad_line, lookback)
        if raw is None:
            return None
        increments = ad_line.diff().dropna().tail(lookback)
        if increments.empty:
            return None
        scale = float(increments.abs().mean())
        if scale == 0 or pd.isna(scale):
            return None
        return round(raw / scale, 4)
    except Exception:
        return None


def change_over(series: pd.Series, sessions: int) -> Optional[float]:
    """Absolute change across the last `sessions` bars (now minus then).

    Distinct from slope_over: the ADX slope in the spec is defined as a
    plain difference over five sessions, not a fitted trend."""
    try:
        cleaned = series.dropna()
        if len(cleaned) < sessions + 1:
            return None
        return float(cleaned.iloc[-1]) - float(cleaned.iloc[-(sessions + 1)])
    except Exception:
        return None


def pct_change_over(series: pd.Series, sessions: int) -> Optional[float]:
    """Percentage change across the last `sessions` bars."""
    try:
        cleaned = series.dropna()
        if len(cleaned) < sessions + 1:
            return None
        start = float(cleaned.iloc[-(sessions + 1)])
        end = float(cleaned.iloc[-1])
        if start == 0:
            return None
        return (end - start) / start * 100.0
    except Exception:
        return None


def extension_in_atr(close: float, sma_fast: float, atr_value: float) -> Optional[float]:
    """How far price sits above its fast average, measured in ATRs.

    This is the input to the veto, so it is deliberately a plain function
    of three scalars with no fallback behaviour: if any input is missing or
    ATR is zero, the answer is None and the caller must treat that as
    "cannot evaluate," never as "not extended"."""
    try:
        if close is None or sma_fast is None or atr_value is None or atr_value == 0:
            return None
        return round((close - sma_fast) / atr_value, 4)
    except Exception:
        return None


def base_levels(bars: pd.DataFrame, lookback: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(base_high, base_low, base_width_pct) over the trailing window."""
    try:
        window = bars.tail(lookback)
        if len(window) < lookback:
            return None, None, None
        base_high = float(window["High"].max())
        base_low = float(window["Low"].min())
        if pd.isna(base_high) or pd.isna(base_low) or base_low <= 0:
            return None, None, None
        width = (base_high - base_low) / base_low * 100.0
        return round(base_high, 4), round(base_low, 4), round(width, 4)
    except Exception:
        return None, None, None
