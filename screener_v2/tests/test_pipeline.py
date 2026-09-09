"""
End-to-end pipeline tests against an in-memory provider — no network.

Covers the validation items from spec §13 that can be checked without a
30-session parallel run: cold-cache behaviour, the veto routing candidates
to the watchlist, the earnings blackout suppressing the order path, and
the ranking function's indifference to recent performance.

Run: python3 screener_v2/tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from screener_v2 import calibration, catalysts, config, pipeline, universe  # noqa: E402
from screener_v2.provider import DictProvider  # noqa: E402
from screener_v2.schemas import Catalysts, CompressionState, StrengthState  # noqa: E402


def bars_from_closes(closes, spread=0.15, volume=2_000_000):
    return pd.DataFrame({
        "Open": closes,
        "High": [c + spread for c in closes],
        "Low": [c - spread for c in closes],
        "Close": closes,
        "Volume": [volume] * len(closes),
    })


def compressed_closes():
    climb = [50.0 + i * 0.25 + (0.9 if i % 2 else -0.9) for i in range(200)]
    base = [100.0 + (0.12 if i % 2 else -0.12) for i in range(80)]
    return climb + base


def benchmark_closes():
    """Up over the medium term, down over the last month — so a flat stock
    reads as weak over 60 sessions and turning up over 20."""
    up, value = [], 100.0
    for _ in range(260):
        up.append(value)
        value *= 1.0015
    down = []
    for _ in range(20):
        value *= 0.999
        down.append(value)
    return up + down


def build_provider():
    provider = DictProvider()
    provider.inject("TEST", bars_from_closes(compressed_closes()))
    provider.inject(config.BENCHMARK_US, bars_from_closes(benchmark_closes()))
    provider.inject(config.BENCHMARK_CA, bars_from_closes(benchmark_closes()))
    return provider


def fresh_state(tmp_log):
    universe.clear_cache()
    catalysts.clear_cache()
    config.LOG_PATH = tmp_log
    if os.path.exists(tmp_log):
        os.remove(tmp_log)


TMP_LOG = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "screener_v2_test_log.jsonl"
)


# ---------------------------------------------------------------------------
def test_cold_catalyst_cache_still_emits():
    """Spec §13.5: a scan with an empty catalyst cache must complete and
    emit candidates marked unavailable — never crash, never fetch inline."""
    fresh_state(TMP_LOG)
    result = pipeline.run(build_provider(), ["TEST"])
    assert result.scanned == 1, result.note
    assert result.candidates, result.note
    candidate = result.candidates[0]
    assert candidate.catalysts.available is False
    assert result.catalysts_available is False
    assert "cold" in result.note
    assert candidate.entry.ok and candidate.entry.order is not None


def test_emitted_order_is_a_limit_draft_below_the_market():
    fresh_state(TMP_LOG)
    candidate = pipeline.run(build_provider(), ["TEST"]).candidates[0]
    order = candidate.entry.order
    assert order.order_type == "LIMIT"
    assert order.action == "BUY"
    assert order.limit_price < candidate.compression.close, "target must sit below the market"
    assert order.stop_level_for_manual_attachment < order.limit_price
    assert order.quantity >= 1


def test_earnings_blackout_moves_it_to_the_watchlist():
    fresh_state(TMP_LOG)
    catalysts.store("TEST", {
        "next_earnings_date": "2099-01-01",
        "earnings_is_estimate": False,
        "days_to_earnings": 2,          # inside the 5-session blackout
        "insider_cluster_buy": True,
        "insider_distinct_buyers_90d": 3,
        "insider_net_shares_90d": 12345.0,
    })
    result = pipeline.run(build_provider(), ["TEST"])
    assert not result.candidates, "a name inside the blackout must not get an order"
    assert result.watchlist_only
    watched = result.watchlist_only[0]
    assert watched.catalysts.available and watched.catalysts.earnings_blackout
    assert watched.entry.order is None
    assert watched.score is not None, "a blackout name is still ranked, just not orderable"


def test_veto_routes_to_watchlist_with_no_score_and_no_catalysts():
    fresh_state(TMP_LOG)
    catalysts.store("TEST", {"insider_cluster_buy": True, "days_to_earnings": 90})
    original = config.MAX_EXTENSION_ATR
    try:
        config.MAX_EXTENSION_ATR = 0.0001
        result = pipeline.run(build_provider(), ["TEST"])
        assert result.vetoed_extension == 1
        assert not result.candidates
        vetoed = result.watchlist_only[0]
        # "No score, no catalyst, no override" — literally.
        assert vetoed.score is None
        assert vetoed.catalysts.available is False
        assert vetoed.entry.order is None
        assert "watchlist only" in vetoed.watchlist_reason
    finally:
        config.MAX_EXTENSION_ATR = original


def test_ranking_ignores_recent_performance():
    """The same states must score identically regardless of what the last
    day did — the structural version of invariant §2.1."""
    compression_state = CompressionState(
        ok=True, bbw_pct_rank=5.0, adx_slope_5=3.0, extension_atr=0.4,
    )
    strength_state = StrengthState(ok=True, rs_20=4.0, rs_60=-2.0, accumulating=True)
    catalyst_state = Catalysts(available=True, insider_cluster_buy=True, iv_percentile=10.0)

    score = pipeline.score_candidate(compression_state, strength_state, catalyst_state)
    # 1*(100-5) + 1*clamp(4 - -2, 0, 20)=6 + 25 + 25 + 15 + clamp(3,0,10)=3
    #   - clamp(0.4,0,1.5)*10 = 4   ->  95 + 6 + 25 + 25 + 15 + 3 - 4 = 165
    assert abs(score - 165.0) < 1e-6, score

    # A tighter base must outrank a looser one, all else equal.
    looser = CompressionState(ok=True, bbw_pct_rank=14.0, adx_slope_5=3.0, extension_atr=0.4)
    assert pipeline.score_candidate(looser, strength_state, catalyst_state) < score

    # More extension must score strictly worse.
    extended = CompressionState(ok=True, bbw_pct_rank=5.0, adx_slope_5=3.0, extension_atr=1.4)
    assert pipeline.score_candidate(extended, strength_state, catalyst_state) < score


def test_emissions_are_logged_with_the_extension_field():
    fresh_state(TMP_LOG)
    pipeline.run(build_provider(), ["TEST"])
    rows = calibration.read_rows()
    assert rows, "every emitted candidate is logged"
    emission = [r for r in rows if r.get("kind") == "emission"][0]
    for field in ("extension_atr", "bbw_pct_rank", "rs_20", "rs_60", "entry_target",
                  "stop", "r_pct", "score", "perf_1d"):
        assert field in emission, f"missing {field} from the emission log"


def test_calibration_buckets_populate_from_emissions():
    fresh_state(TMP_LOG)
    pipeline.run(build_provider(), ["TEST"])
    rows = [r for r in calibration.read_rows() if r.get("kind") == "emission"]
    calibration.record_fill(
        rows[0]["listing_used"], rows[0]["timestamp"],
        actual_fill_price=100.0, extension_atr_at_fill=0.75,
    )
    calibration._append([{
        "kind": "forward", "id": rows[0]["id"],
        "fwd_return_5d": 2.5, "fwd_return_10d": 4.0, "fwd_return_20d": -1.0,
    }])
    report = calibration.review()
    bucket = [b for b in report.buckets if b.bucket == "0.5-1.0"][0]
    assert bucket.n == 1
    assert abs(bucket.mean_fwd_5d - 2.5) < 1e-9
    assert abs(bucket.mean_fwd_20d - (-1.0)) < 1e-9
    assert report.rows_with_forward_returns == 1


def test_universe_gates_reject_illiquid_and_short_history():
    fresh_state(TMP_LOG)
    provider = DictProvider()
    provider.inject("THIN", bars_from_closes(compressed_closes(), volume=10))   # ~1k/day
    provider.inject("SHORT", bars_from_closes([100.0] * 30))
    provider.inject("CHEAP", bars_from_closes([1.0] * 300, spread=0.01))
    members = universe.build(provider, ["THIN", "SHORT", "CHEAP"])
    assert members == [], [m.symbol for m in members]


def test_excluded_symbols_never_reach_the_scan():
    fresh_state(TMP_LOG)
    provider = DictProvider()
    excluded = sorted(config.EXCLUDED_SYMBOLS)[0]
    provider.inject(excluded, bars_from_closes(compressed_closes()))
    assert universe.build(provider, [excluded]) == []


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    original_log = config.LOG_PATH
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(e).__name__}: {e}")
    config.LOG_PATH = original_log
    if os.path.exists(TMP_LOG):
        os.remove(TMP_LOG)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
