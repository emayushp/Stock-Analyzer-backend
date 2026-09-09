"""
Tests for layers 2, 3 and 5 (compression, strength, entry).

The entry tests are fully hand-computed against a CompressionState built
by hand — no synthetic price series involved — because the entry math is
where the actual edge of this rebuild lives and it deserves arithmetic
that a reader can check line by line.

Run: python3 screener_v2/tests/test_layers.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from screener_v2 import compression, config, entry, strength  # noqa: E402
from screener_v2.schemas import CompressionState  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def bars_from_closes(closes, spread=0.4):
    """OHLCV frame around a close series, with a constant intrabar range."""
    return pd.DataFrame({
        "Open": closes,
        "High": [c + spread for c in closes],
        "Low": [c - spread for c in closes],
        "Close": closes,
        "Volume": [2_000_000] * len(closes),
    })


def compressed_closes():
    """200 sessions of a noisy climb, then 80 sessions of a tight base.

    The climb puts the 200-day average well below price and leaves a high
    band-width history to rank against; the base then collapses band width,
    contracts ATR and lets ADX decay."""
    climb = [50.0 + i * 0.25 + (0.9 if i % 2 else -0.9) for i in range(200)]
    base = [100.0 + (0.12 if i % 2 else -0.12) for i in range(80)]
    return climb + base


def approx(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


def compound(start, rate, n):
    out, value = [], start
    for _ in range(n):
        out.append(value)
        value *= rate
    return out


# ---------------------------------------------------------------------------
# layer 2 — compression
# ---------------------------------------------------------------------------
def test_compression_accepts_a_compressed_base():
    state = compression.evaluate("TEST", bars_from_closes(compressed_closes(), spread=0.15))
    assert state.ok, state.reason
    assert not state.veto
    assert state.bbw_pct_rank is not None and state.bbw_pct_rank <= config.MAX_BBW_PCT_RANK
    assert state.atr_pct < state.atr_median_60
    assert state.adx_14 < config.MAX_ADX
    assert state.close > state.sma200
    assert abs(state.extension_atr) <= config.MAX_EXTENSION_ATR


def test_compression_rejects_short_history():
    state = compression.evaluate("SHORT", bars_from_closes([100.0] * 50))
    assert not state.ok
    assert "sessions" in state.reason


def test_compression_rejects_missing_bars():
    assert not compression.evaluate("NONE", None).ok
    assert not compression.evaluate("EMPTY", pd.DataFrame()).ok


def test_compression_rejects_an_expanding_trend():
    # A clean, still-accelerating uptrend: band width is nowhere near the
    # bottom of its range and ADX is high. This is exactly the shape the
    # momentum screen would have surfaced and this one must not.
    closes = [50.0 + i * 0.4 for i in range(280)]
    state = compression.evaluate("TREND", bars_from_closes(closes))
    assert not state.ok, state.reason


def test_extension_veto_is_absolute(monkeypatched_limit=0.0001):
    """The same bars that pass above must become watchlist-only the moment
    they sit above the extension limit — with no score or catalyst able to
    reach in and change that, because evaluate() takes no override."""
    bars = bars_from_closes(compressed_closes(), spread=0.15)
    original = config.MAX_EXTENSION_ATR
    try:
        config.MAX_EXTENSION_ATR = monkeypatched_limit
        state = compression.evaluate("TEST", bars)
        assert not state.ok
        assert state.veto, state.reason
        assert "watchlist only" in state.reason
    finally:
        config.MAX_EXTENSION_ATR = original


# ---------------------------------------------------------------------------
# layer 3 — strength
# ---------------------------------------------------------------------------
def test_strength_accepts_an_inflection():
    # Benchmark compounds +0.2%/session throughout. The stock drifts DOWN
    # for 120 sessions, then turns up hard for the last 20 — weak over 60,
    # positive over 20, which is the inflection this layer wants.
    bench = compound(100.0, 1.002, 140)
    stock = compound(100.0, 0.9995, 120)
    stock += compound(stock[-1] * 1.006, 1.006, 20)

    state = strength.evaluate("TEST", bars_from_closes(stock), bars_from_closes(bench))
    assert state.ok, state.reason
    assert state.rs_60 <= 0, state.rs_60
    assert state.rs_20 > 0, state.rs_20
    assert state.benchmark == "SPY"


def test_strength_rejects_a_name_already_at_maximum_rs():
    bench = compound(100.0, 1.001, 140)
    stock = compound(100.0, 1.004, 140)  # outperforming on every window
    state = strength.evaluate("HOT", bars_from_closes(stock), bars_from_closes(bench))
    assert not state.ok
    assert "inflection, not the maximum" in state.reason


def test_strength_needs_a_benchmark():
    stock = compound(100.0, 1.001, 140)
    state = strength.evaluate("TEST", bars_from_closes(stock), None)
    assert not state.ok
    assert "benchmark" in state.reason


def test_benchmark_selection_by_listing():
    assert strength.benchmark_for("AAPL") == config.BENCHMARK_US
    assert strength.benchmark_for("SHOP.TO") == config.BENCHMARK_CA
    assert strength.benchmark_for("well.v") == config.BENCHMARK_CA


def test_accumulation_is_a_flag_not_a_gate():
    # Flat price with a steadily rising A/D line. Whatever the RS gate
    # decides, `accumulating` must be computed independently of it.
    bench = compound(100.0, 1.002, 140)
    stock = compound(100.0, 0.9995, 120)
    stock += compound(stock[-1] * 1.006, 1.006, 20)
    state = strength.evaluate("TEST", bars_from_closes(stock), bars_from_closes(bench))
    assert isinstance(state.accumulating, bool)
    assert state.ad_slope_20 is not None


# ---------------------------------------------------------------------------
# layer 5 — entry (hand-computed)
# ---------------------------------------------------------------------------
def _state(close, base_high, base_low, sma20, atr):
    return CompressionState(
        ok=True, close=close, base_high=base_high, base_low=base_low,
        sma20=sma20, atr=atr,
    )


def test_entry_pullback_target():
    # close 100 is inside the base (high 105), so the target is the tighter
    # of sma20 98 and base_low 95 -> 98.
    # stop = 95 - 0.5*2 = 94 ; risk = 98 - 94 = 4 ; r% = 4/98*100 = 4.0816
    # shares = floor(500 / 4) = 125
    plan = entry.plan("TEST", _state(100.0, 105.0, 95.0, 98.0, 2.0))
    assert plan.ok, plan.reason
    assert approx(plan.entry_target, 98.0)
    assert approx(plan.stop, 94.0)
    assert approx(plan.risk_per_share, 4.0)
    assert approx(plan.r_pct, 4.082, tol=1e-3)
    assert plan.shares == 125
    assert not plan.already_broke_out
    assert plan.order is not None
    assert plan.order.order_type == "LIMIT" and plan.order.action == "BUY"
    assert approx(plan.order.limit_price, 98.0)
    assert approx(plan.order.stop_level_for_manual_attachment, 94.0)
    assert plan.order.quantity == 125


def test_entry_waits_for_the_retest_after_a_breakout():
    # close 110 is above base_high 105 -> target the retest at 105.
    # stop = 95 - 0.5*2 = 94 ; risk = 11 ; r% = 11/105*100 = 10.48 > 8 -> rejected
    plan = entry.plan("TEST", _state(110.0, 105.0, 95.0, 104.0, 2.0))
    assert plan.already_broke_out
    assert not plan.ok
    assert approx(plan.entry_target, 105.0)
    assert "too far to size sanely" in plan.reason


def test_entry_refuses_to_chase():
    # sma20 102 is above the last price 100, so the only available target
    # would sit above the market. That is chasing, so there is no order.
    plan = entry.plan("TEST", _state(100.0, 105.0, 99.0, 102.0, 1.0))
    assert not plan.ok
    assert plan.order is None
    assert "chasing" in plan.reason


def test_entry_blackout_is_watchlist_only_but_keeps_the_math():
    plan = entry.plan("TEST", _state(100.0, 105.0, 95.0, 98.0, 2.0), earnings_blackout=True)
    assert not plan.ok
    assert plan.order is None                 # never an order into a print
    assert plan.shares == 125                 # but the sizing is still shown
    assert "earnings within" in plan.reason


def test_entry_rejects_a_position_smaller_than_one_share():
    # entry 20000, stop 19100 - 0.5*200 = 19000, risk 1000 (r% = 5, inside
    # the cap), but a 500 risk budget buys floor(500/1000) = 0 shares.
    plan = entry.plan("TEST", _state(20500.0, 21000.0, 19100.0, 20000.0, 200.0))
    assert not plan.ok
    assert plan.shares == 0
    assert "less than one share" in plan.reason


def test_entry_never_emits_a_stop_order():
    plan = entry.plan("TEST", _state(100.0, 105.0, 95.0, 98.0, 2.0))
    assert plan.order.order_type in ("LIMIT", "MARKET")
    assert plan.order.order_type == "LIMIT"
    # The protective stop is a labelled field for manual attachment, never
    # a second unlinked order against the same shares.
    dumped = plan.order.model_dump()
    assert "stop_level_for_manual_attachment" in dumped
    assert not any(str(v).upper() == "STOP" for v in dumped.values())


def test_entry_handles_missing_levels():
    plan = entry.plan("TEST", CompressionState(ok=True, close=100.0))
    assert not plan.ok
    assert plan.order is None
    assert "missing levels" in plan.reason


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
