"""
Hand-checked unit tests for indicators.py (spec build order step 1).

pytest is not a dependency of this project, so every test is a plain
function and `python3 screener_v2/tests/test_indicators.py` runs the lot.
pytest would also collect them unchanged if it were ever added.

Expected values below are computed by hand in the comments, not captured
from a previous run — a test that only records what the code already did
cannot catch the code being wrong.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from screener_v2 import indicators as ind  # noqa: E402


def _bars(highs, lows, closes, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "Open": closes,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes if volumes is not None else [1_000_000] * n,
    })


def approx(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


def test_sma():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ind.sma(s, 2)
    # min_periods=2, so index 0 is NaN; then (1+2)/2, (2+3)/2, (3+4)/2
    assert pd.isna(out.iloc[0])
    assert approx(out.iloc[1], 1.5) and approx(out.iloc[2], 2.5) and approx(out.iloc[3], 3.5)


def test_wilder_rma():
    s = pd.Series([1.0, 2.0, 3.0])
    out = ind.wilder_rma(s, 2)
    # alpha = 1/2, adjust=False: y0 = 1 (masked by min_periods=2),
    # y1 = .5*2 + .5*1 = 1.5, y2 = .5*3 + .5*1.5 = 2.25
    assert pd.isna(out.iloc[0])
    assert approx(out.iloc[1], 1.5) and approx(out.iloc[2], 2.25)


def test_true_range():
    bars = _bars(highs=[10, 11, 12], lows=[9, 9.5, 11], closes=[9.5, 10.5, 11.5])
    tr = ind.true_range(bars)
    # bar0: no prev close -> high-low = 1.0
    # bar1: max(11-9.5=1.5, |11-9.5|=1.5, |9.5-9.5|=0) = 1.5
    # bar2: max(12-11=1.0, |12-10.5|=1.5, |11-10.5|=0.5) = 1.5
    assert approx(tr.iloc[0], 1.0) and approx(tr.iloc[1], 1.5) and approx(tr.iloc[2], 1.5)


def test_bollinger_band_width():
    close = pd.Series([10.0, 12.0, 14.0])
    bbw = ind.bollinger_band_width(close, period=3, num_std=2.0)
    # mean = 12; population stdev = sqrt(((-2)^2 + 0 + 2^2)/3) = sqrt(8/3) = 1.632993
    # width = (upper - lower)/middle = 4 * 1.632993 / 12 = 0.5443311
    assert pd.isna(bbw.iloc[0]) and pd.isna(bbw.iloc[1])
    assert approx(bbw.iloc[2], 0.5443310539518174, tol=1e-9)

    flat = pd.Series([5.0, 5.0, 5.0])
    assert approx(ind.bollinger_band_width(flat, 3, 2.0).iloc[2], 0.0)


def test_percentile_rank():
    # current is the maximum -> 100; the minimum -> 0
    assert approx(ind.percentile_rank(pd.Series([1, 2, 3, 4, 5]), 5), 100.0)
    assert approx(ind.percentile_rank(pd.Series([5, 4, 3, 2, 1]), 5), 0.0)
    # current = 3, values strictly below it are 1 and 2 -> 2 of the other 4 -> 50
    assert approx(ind.percentile_rank(pd.Series([1, 2, 3, 4, 3]), 5), 50.0)
    # lookback truncates: only the last 3 ([3, 4, 3]) are in the window.
    # Current is 3 and nothing in the window is strictly below it, so this
    # reads 0 — a value tied with the window minimum genuinely IS at the
    # bottom of its own range, which is the direction a compression screen
    # wants ties to fall.
    assert approx(ind.percentile_rank(pd.Series([1, 2, 3, 4, 3]), 3), 0.0)
    # Ties count as "not below", so a tie at the top does NOT read 100:
    # window [3, 4, 4], current 4, one of the other two is below it -> 50.
    assert approx(ind.percentile_rank(pd.Series([3, 4, 4]), 3), 50.0)
    assert ind.percentile_rank(pd.Series([], dtype="float64"), 5) is None


def test_slope_over():
    assert approx(ind.slope_over(pd.Series([1.0, 3.0, 5.0, 7.0]), 4), 2.0)
    assert approx(ind.slope_over(pd.Series([7.0, 5.0, 3.0, 1.0]), 4), -2.0)
    assert approx(ind.slope_over(pd.Series([4.0, 4.0, 4.0]), 3), 0.0)
    assert ind.slope_over(pd.Series([1.0, 2.0]), 5) is None  # not enough points


def test_williams_ad():
    bars = _bars(highs=[10, 11, 11], lows=[9, 9.5, 9], closes=[9.5, 10.5, 9.8])
    ad = ind.williams_ad(bars)
    # bar0: no prev close -> 0
    # bar1: close 10.5 > prev 9.5 -> true_low = min(9.5, 9.5) = 9.5 -> +1.0
    # bar2: close 9.8 < prev 10.5 -> true_high = max(11, 10.5) = 11 -> -1.2
    assert approx(ad.iloc[0], 0.0)
    assert approx(ad.iloc[1], 1.0)
    assert approx(ad.iloc[2], -0.2, tol=1e-9)


def test_normalized_ad_slope():
    # A perfectly steady +1/session line: slope 1, mean |increment| 1 -> 1.0
    line = pd.Series([float(i) for i in range(20)])
    assert approx(ind.normalized_ad_slope(line, 20), 1.0, tol=1e-6)
    # A flat line has no flow at all to normalize against -> unavailable
    assert ind.normalized_ad_slope(pd.Series([5.0] * 20), 20) is None


def test_pct_change_over():
    assert approx(ind.pct_change_over(pd.Series([100.0, 110.0]), 1), 10.0)
    assert approx(ind.pct_change_over(pd.Series([100.0, 90.0]), 1), -10.0)
    assert ind.pct_change_over(pd.Series([100.0]), 5) is None


def test_extension_in_atr():
    assert approx(ind.extension_in_atr(110.0, 100.0, 5.0), 2.0)
    assert approx(ind.extension_in_atr(95.0, 100.0, 5.0), -1.0)
    # A zero or missing ATR must read as "cannot evaluate", never as "flat"
    assert ind.extension_in_atr(110.0, 100.0, 0.0) is None
    assert ind.extension_in_atr(110.0, None, 5.0) is None


def test_base_levels():
    bars = _bars(highs=[10, 12, 11], lows=[8, 9, 9.5], closes=[9, 11, 10])
    high, low, width = ind.base_levels(bars, 3)
    # high 12, low 8, width = (12-8)/8*100 = 50
    assert approx(high, 12.0) and approx(low, 8.0) and approx(width, 50.0)
    assert ind.base_levels(bars, 10) == (None, None, None)  # window longer than history


def test_adx_structural():
    # Wilder's ADX over 14 periods is not sensibly hand-computed, so this
    # asserts the properties that actually matter to layer 2: a clean
    # one-way trend reads high with +DI dominant, and a flat tape reads low.
    n = 60
    trend = _bars(
        highs=[100 + i for i in range(n)],
        lows=[99 + i for i in range(n)],
        closes=[99.5 + i for i in range(n)],
    )
    trending_adx = ind.latest(ind.adx(trend, 14))
    assert trending_adx is not None and trending_adx > 25, trending_adx

    # A four-bar oscillation: real range each bar, but direction reverses
    # constantly, so +DI and -DI stay close and ADX reads low.
    cycle = [100.0, 101.0, 100.0, 99.0]
    closes = [cycle[i % 4] for i in range(n)]
    chop = _bars(
        highs=[c + 0.5 for c in closes],
        lows=[c - 0.5 for c in closes],
        closes=closes,
    )
    choppy_adx = ind.latest(ind.adx(chop, 14))
    assert choppy_adx is not None and choppy_adx < 20, choppy_adx


def test_adx_undefined_when_no_directional_movement():
    # Successive inside bars produce no +DM and no -DM at all, so DX is a
    # genuine 0/0. Layer 2 depends on this reading as "cannot evaluate"
    # (None) rather than as a convenient 0 that would sail through the
    # adx < 20 gate — see compression.py, where a missing value fails the
    # gate rather than passing it.
    n = 60
    bars = _bars(
        highs=[100.5 if i % 2 else 100.4 for i in range(n)],
        lows=[99.5 if i % 2 else 99.6 for i in range(n)],
        closes=[100.1 if i % 2 else 99.9 for i in range(n)],
    )
    assert ind.latest(ind.adx(bars, 14)) is None


def test_latest_is_defensive():
    assert ind.latest(None) is None
    assert ind.latest(pd.Series([], dtype="float64")) is None
    assert ind.latest(pd.Series([float("nan")])) is None
    assert approx(ind.latest(pd.Series([1.0, float("nan")])), 1.0)


def test_has_ohlcv():
    assert ind.has_ohlcv(_bars([1], [1], [1]))
    assert not ind.has_ohlcv(None)
    assert not ind.has_ohlcv(pd.DataFrame())
    assert not ind.has_ohlcv(pd.DataFrame({"Close": [1.0]}))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
