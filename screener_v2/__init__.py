"""
Screener v2 — a pre-move screener built alongside the existing momentum
screener, behind the SCREENER_V2_ENABLED flag.

The v1 momentum screen conditions on completed moves (1-day performance,
an RSI band); every input to it is a realized return, which selects into
short-horizon mean reversion. This one conditions on pre-move state —
volatility compression, relative-strength inflection, accumulation — and
separates candidate selection from entry timing, so a good name found at a
bad price becomes a resting limit order rather than a market fill.

The edge being targeted is a better entry price and a cheaper stop, not
better forecasting.

Layer map (each module owns exactly one):
    universe.py     1  liquidity / size / history floors, weekly cache
    compression.py  2  stored energy + the absolute extension veto
    strength.py     3  which way, by RS inflection rather than RS maximum
    catalysts.py    4  nightly-cached only, never fetched inline
    entry.py        5  target price, stop, size, LIMIT draft
    pipeline.py        orchestration
    calibration.py  6  emission log + the extension bucket review table

Hard invariants, enforced by tools/check_screener_v2_invariants.py:
    1. No gate or ranking input is a recent return.
    2. The extension veto has no override path.
    3. Selection never places an order on the same bar.
    4. No AI verdict / sentiment / news classification gates or ranks.
    5. No catalyst network fetch at scan time.
"""
