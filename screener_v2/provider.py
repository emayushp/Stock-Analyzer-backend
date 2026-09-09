"""
The data layer, behind a thin interface so an alternative source can be
swapped in without touching any layer module (spec §5).

The default implementation is yfinance, matching the rest of the backend.
It is deliberately batch-first: a 350-name universe fetched one symbol at a
time is exactly the pattern that got this deployment rate-limited by Yahoo
for half an hour in production. `prefetch()` pulls the whole list in
chunked multi-symbol requests, each behind a hard wall-clock cap, and
`daily_bars()` then serves every layer from memory.

The timeout guard is NOT a `with ThreadPoolExecutor(...) as pool:` block:
that form blocks on exit until the submitted work finishes regardless of
whether .result() already timed out, which silently undoes the timeout.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional, Protocol

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger("stock-analyzer")


class PriceProvider(Protocol):
    def daily_bars(self, symbol: str, lookback: int) -> pd.DataFrame: ...


class YFinanceProvider:
    """Batch-prefetching yfinance provider.

    `daily_bars` returns an empty frame rather than raising for a symbol
    with no data — a single bad ticker must never fail a scan (spec §3).
    """

    def __init__(self) -> None:
        self._bars: Dict[str, pd.DataFrame] = {}

    # -- interface ---------------------------------------------------------
    def daily_bars(self, symbol: str, lookback: int = config.MIN_SESSIONS) -> pd.DataFrame:
        frame = self._bars.get(symbol.upper())
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame.tail(lookback) if lookback else frame

    # -- batch loading -----------------------------------------------------
    def prefetch(self, symbols: List[str], days: int = config.BARS_LOOKBACK_DAYS) -> int:
        """Load bars for every symbol into memory. Returns how many symbols
        came back with usable data."""
        unique = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        for start in range(0, len(unique), config.BATCH_CHUNK_SIZE):
            chunk = unique[start : start + config.BATCH_CHUNK_SIZE]
            self._load_chunk(chunk, days)
        return len([s for s in unique if not self.daily_bars(s, 0).empty])

    def _load_chunk(self, chunk: List[str], days: int) -> None:
        raw = self._download(chunk, days)
        if raw is None or raw.empty:
            return
        for symbol in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol not in raw.columns.get_level_values(0):
                        continue
                    frame = raw[symbol]
                else:
                    frame = raw
                frame = frame.dropna(how="all")
                if not frame.empty:
                    self._bars[symbol] = frame
            except Exception as e:
                logger.info(f"screener_v2: could not unpack bars for {symbol}: {e}")

    @staticmethod
    def _download(chunk: List[str], days: int) -> Optional[pd.DataFrame]:
        guard = ThreadPoolExecutor(max_workers=1)
        try:
            return guard.submit(
                yf.download,
                chunk,
                period=f"{days}d",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            ).result(timeout=config.BATCH_DOWNLOAD_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            logger.error(
                f"screener_v2: bars download timed out after "
                f"{config.BATCH_DOWNLOAD_TIMEOUT_SECONDS}s for {len(chunk)} symbols"
            )
            return None
        except Exception as e:
            logger.error(f"screener_v2: bars download failed for {len(chunk)} symbols: {e}")
            return None
        finally:
            guard.shutdown(wait=False)

    # -- helpers -----------------------------------------------------------
    def loaded_symbols(self) -> List[str]:
        return sorted(self._bars.keys())

    def inject(self, symbol: str, bars: pd.DataFrame) -> None:
        """Seed bars directly. Used by tests and by the calibration
        backfill, so neither has to reach the network."""
        self._bars[symbol.strip().upper()] = bars


class DictProvider:
    """A provider backed by an in-memory dict — the seam that lets every
    layer be tested against fixed frames with no network at all."""

    def __init__(self, bars: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        self._bars = {k.upper(): v for k, v in (bars or {}).items()}

    def daily_bars(self, symbol: str, lookback: int = config.MIN_SESSIONS) -> pd.DataFrame:
        frame = self._bars.get(symbol.upper())
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame.tail(lookback) if lookback else frame

    def prefetch(self, symbols: List[str], days: int = config.BARS_LOOKBACK_DAYS) -> int:
        return len([s for s in symbols if not self.daily_bars(s, 0).empty])

    def inject(self, symbol: str, bars: pd.DataFrame) -> None:
        self._bars[symbol.strip().upper()] = bars

    def loaded_symbols(self) -> List[str]:
        return sorted(self._bars.keys())
