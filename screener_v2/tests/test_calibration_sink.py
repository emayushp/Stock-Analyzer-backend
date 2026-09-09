"""
Calibration storage tests, against BOTH sinks.

The database path is exercised for real via SQLite rather than mocked:
db.py builds the engine from DATABASE_URL with generic SQLAlchemy column
types, so pointing it at a temp SQLite file runs the same model, the same
create_all, the same session and the same queries Postgres will. A mock
would only prove the mock works.

Run: python3 screener_v2/tests/test_calibration_sink.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from screener_v2 import calibration, config  # noqa: E402
from screener_v2.schemas import Candidate, Catalysts, CompressionState, EntryPlan, StrengthState  # noqa: E402


def make_candidate(symbol="TEST", extension=0.75, score=120.0, timestamp="2026-01-02T00:00:00Z"):
    return Candidate(
        symbol=symbol, listing_used=symbol, generated_at=timestamp, score=score,
        compression=CompressionState(ok=True, bbw_pct_rank=8.0, atr_pct=1.2, adx_14=15.0,
                                     adx_slope_5=2.0, extension_atr=extension),
        strength=StrengthState(ok=True, rs_5=1.0, rs_20=3.0, rs_60=-2.0, rs_120=-5.0,
                               accumulating=True),
        catalysts=Catalysts(available=True, insider_cluster_buy=True, days_to_earnings=30),
        entry=EntryPlan(ok=True, entry_target=98.0, stop=94.0, r_pct=4.08, shares=125),
        perf_1d=0.5,
    )


def use_file_sink(tmpdir):
    os.environ.pop("DATABASE_URL", None)
    _reload_db()
    config.LOG_PATH = os.path.join(tmpdir, "log.jsonl")
    if os.path.exists(config.LOG_PATH):
        os.remove(config.LOG_PATH)


def use_db_sink(tmpdir):
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(tmpdir, 'calib.sqlite')}"
    _reload_db()
    config.LOG_PATH = os.path.join(tmpdir, "unused.jsonl")


def _reload_db():
    """db.py memoizes its engine, so the module is reloaded whenever the
    URL changes rather than reaching into its private globals."""
    import db as db_lib
    importlib.reload(db_lib)


# ---------------------------------------------------------------------------
def test_file_sink_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        use_file_sink(tmpdir)
        assert calibration.active_sink().startswith("jsonl:")
        assert calibration.log_candidates([make_candidate()]) == 1
        rows = calibration.read_rows()
        assert len(rows) == 1
        assert rows[0]["extension_atr"] == 0.75


def test_database_sink_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        use_db_sink(tmpdir)
        assert calibration.active_sink() == "postgres", calibration.active_sink()
        assert calibration.log_candidates([make_candidate()]) == 1
        rows = calibration.read_rows()
        assert len(rows) == 1, rows
        assert rows[0]["symbol"] == "TEST"
        assert rows[0]["extension_atr"] == 0.75
        # Nothing was written to the file sink.
        assert not os.path.exists(config.LOG_PATH)


def test_database_sink_survives_a_restart():
    """The whole point: the log outlives the process that wrote it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        use_db_sink(tmpdir)
        calibration.log_candidates([make_candidate(symbol="AAA")])
        _reload_db()  # simulates a redeploy: new process, same database
        rows = calibration.read_rows()
        assert [r["symbol"] for r in rows] == ["AAA"]


def test_database_sink_merges_emission_fill_and_forward():
    with tempfile.TemporaryDirectory() as tmpdir:
        use_db_sink(tmpdir)
        candidate = make_candidate(extension=0.75)
        calibration.log_candidates([candidate])
        calibration.record_fill(
            candidate.listing_used, candidate.generated_at,
            actual_fill_price=98.0, extension_atr_at_fill=0.6,
        )
        calibration.append_rows([{
            "kind": "forward",
            "id": calibration.row_id(candidate.listing_used, candidate.generated_at),
            "fwd_return_5d": 3.0, "fwd_return_10d": 5.0, "fwd_return_20d": 8.0,
        }])
        report = calibration.review()
        # extension_at_fill 0.6 buckets into 0.5-1.0, not the emission's 0.75
        bucket = [b for b in report.buckets if b.bucket == "0.5-1.0"][0]
        assert bucket.n == 1
        assert abs(bucket.mean_fwd_10d - 5.0) < 1e-9
        assert report.sink == "postgres"
        assert report.rows_with_forward_returns == 1


def test_falls_back_to_file_when_the_database_is_unreachable():
    """A configured-but-broken database must not lose the emission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["DATABASE_URL"] = "postgresql://nobody@127.0.0.1:1/doesnotexist"
        try:
            _reload_db()
        except Exception:
            pass
        config.LOG_PATH = os.path.join(tmpdir, "fallback.jsonl")
        written = calibration.log_candidates([make_candidate(symbol="BBB")])
        assert written == 1, "the row must land somewhere rather than vanish"
        rows = calibration.read_rows()
        assert [r["symbol"] for r in rows] == ["BBB"]
        assert os.path.exists(config.LOG_PATH)


def main() -> int:
    original_url = os.environ.get("DATABASE_URL")
    original_path = config.LOG_PATH
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(e).__name__}: {e}")
    if original_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = original_url
    config.LOG_PATH = original_path
    _reload_db()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
