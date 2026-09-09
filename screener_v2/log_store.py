"""
Storage for the emission log. Split out of calibration.py so that module
can stay about the analysis (buckets, forward returns, the review table)
rather than about where bytes land.

Two sinks, chosen at call time:

  postgres   whenever DATABASE_URL is configured and reachable
  jsonl      config.LOG_PATH otherwise, and as the fallback if a
             configured database write fails

That order is the point. Render's web-service filesystem is wiped on every
deploy, so a file log cannot survive the 30-session parallel run the flag
flip is gated on. Both sinks store identical row shapes and are merged by
the same code, so switching between them changes durability and nothing
else.

Rows are append-only in both sinks. An emission, a later fill, and a later
forward-return backfill are three separate appends sharing a row id;
nothing is ever updated in place, so a bug in the backfill cannot corrupt
the original emission record.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List

from . import config

logger = logging.getLogger("stock-analyzer")

_write_lock = threading.Lock()


def _db():
    """The app's optional database module, or None.

    Imported lazily and defensively: screener_v2 must stay importable on
    its own (the tests import it with no app around it), and a database
    that is configured but unreachable has to degrade to the file sink
    rather than taking the scan down with it."""
    try:
        import db as db_lib

        if db_lib.get_engine() is None:
            return None
        return db_lib
    except Exception as e:
        logger.info(f"screener_v2: database sink unavailable, using the file log: {e}")
        return None


def active_sink() -> str:
    return "postgres" if _db() is not None else f"jsonl:{config.LOG_PATH}"


def _append_jsonl(rows: List[Dict[str, Any]]) -> int:
    try:
        path = config.LOG_PATH
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
        return len(rows)
    except Exception as e:
        logger.error(f"screener_v2: could not append to the emission log: {e}")
        return 0


def _append_db(db_lib, rows: List[Dict[str, Any]]) -> int:
    """Write to Postgres. Returns 0 on failure so the caller can fall back
    to the file sink rather than losing the rows outright."""
    session = None
    try:
        session = db_lib.get_session()
        if session is None:
            return 0
        # json.loads(json.dumps(...)) normalizes anything the JSON column
        # can't take directly (dates, numpy scalars) the same way the file
        # sink's default=str does, so both sinks store identical shapes.
        for row in rows:
            payload = json.loads(json.dumps(row, default=str))
            session.add(db_lib.ScreenerV2Log(
                row_key=str(payload.get("id") or ""),
                symbol=str(payload.get("symbol") or payload.get("listing_used") or ""),
                kind=str(payload.get("kind") or "emission"),
                payload=payload,
            ))
        session.commit()
        return len(rows)
    except Exception as e:
        logger.error(f"screener_v2: database append failed, falling back to the file log: {e}")
        try:
            if session is not None:
                session.rollback()
        except Exception:
            pass
        return 0
    finally:
        try:
            if session is not None:
                session.close()
        except Exception:
            pass


def append(rows: List[Dict[str, Any]]) -> int:
    """Append rows to whichever sink is active. Returns how many landed."""
    if not rows:
        return 0
    db_lib = _db()
    if db_lib is not None:
        written = _append_db(db_lib, rows)
        if written:
            return written
    return _append_jsonl(rows)


def _read_jsonl() -> List[Dict[str, Any]]:
    path = config.LOG_PATH
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"screener_v2: could not read the emission log: {e}")
    return rows


def read_rows() -> List[Dict[str, Any]]:
    """Every row in the log, oldest first.

    A corrupt file line, or a database that has gone away since the last
    write, degrades to what can still be read rather than raising — the
    calibration report is diagnostic output and is more useful partially
    populated than not at all."""
    db_lib = _db()
    if db_lib is not None:
        session = None
        try:
            session = db_lib.get_session()
            if session is not None:
                records = (
                    session.query(db_lib.ScreenerV2Log)
                    .order_by(db_lib.ScreenerV2Log.id.asc())
                    .all()
                )
                return [r.payload for r in records if isinstance(r.payload, dict)]
        except Exception as e:
            logger.error(f"screener_v2: database read failed, falling back to the file log: {e}")
        finally:
            try:
                if session is not None:
                    session.close()
            except Exception:
                pass
    return _read_jsonl()
