"""
HTTP-surface tests: the feature flag actually gates what it claims to.

Run: python3 screener_v2/tests/test_router.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from screener_v2 import runtime  # noqa: E402
from screener_v2.router import router  # noqa: E402
from screener_v2.schemas import ScanResult  # noqa: E402

app = FastAPI()
app.include_router(router)
client = TestClient(app)

FLAG = "SCREENER_V2_ENABLED"


def flag_off():
    os.environ.pop(FLAG, None)


def flag_on():
    os.environ[FLAG] = "true"


def test_status_is_readable_with_the_flag_off():
    flag_off()
    response = client.get("/api/v2/screener/status")
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_scan_is_refused_with_the_flag_off():
    flag_off()
    response = client.get("/api/v2/screener")
    assert response.status_code == 503
    assert "SCREENER_V2_ENABLED" in response.json()["detail"]


def test_scan_runs_with_the_flag_on():
    flag_on()
    canned = ScanResult(universe_size=3, scanned=3, note="ok", generated_at="2026-01-01T00:00:00Z")
    with patch.object(runtime, "is_configured", return_value=True), \
         patch.object(runtime, "run_scan", return_value=canned) as scan:
        response = client.get("/api/v2/screener")
    assert response.status_code == 200
    assert response.json()["scanned"] == 3
    assert scan.called
    flag_off()


def test_scan_reports_an_unconfigured_universe_rather_than_crashing():
    flag_on()
    with patch.object(runtime, "is_configured", return_value=False):
        response = client.get("/api/v2/screener")
    assert response.status_code == 503
    assert "universe" in response.json()["detail"].lower()
    flag_off()


def test_calibration_is_readable_with_the_flag_off():
    # The whole point of the parallel run is reading this table BEFORE
    # flipping the flag, so it must not be gated behind it.
    flag_off()
    response = client.get("/api/v2/screener/calibration")
    assert response.status_code == 200
    body = response.json()
    assert [b["bucket"] for b in body["buckets"]] == ["<0", "0-0.5", "0.5-1.0", "1.0-1.5"]


def test_fill_recording_is_gated():
    flag_off()
    response = client.post(
        "/api/v2/screener/fill",
        params={"symbol": "TEST", "generated_at": "x", "actual_fill_price": 10.0},
    )
    assert response.status_code == 503


def test_jobs_are_gated():
    flag_off()
    assert client.post("/api/v2/screener/jobs/universe").status_code == 503
    assert client.post("/api/v2/screener/jobs/catalysts").status_code == 503


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
    flag_off()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
