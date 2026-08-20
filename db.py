"""
Optional persistence layer for cross-device sync. Entirely opt-in: the rest
of the app has zero dependency on a database, so if DATABASE_URL isn't set,
get_engine() returns None and every endpoint that needs it degrades to a
503 rather than crashing anything else — same graceful-degradation pattern
as _get_anthropic_client() in main.py.

Render's free web-service disk is ephemeral (wiped on every deploy), so
this deliberately points at an external managed Postgres (e.g. Neon) rather
than a local SQLite file.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("stock-analyzer")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserData(Base):
    """
    A generic per-user key-value store, keyed by the same names the client
    already uses for its own localStorage/AsyncStorage (ma_portfolio,
    ma_watchlist, etc.) — see SYNC_ALLOWED_KEYS in main.py. This mirrors the
    client's existing store.get/set(key, value) shape exactly, so syncing is
    "push this key's value" / "pull all keys," not a bespoke schema per
    feature.
    """
    __tablename__ = "user_data"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key = Column(String, nullable=False)
    value = Column(JSON, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_data_user_id_key"),)


_engine = None
_SessionLocal = None
_unavailable_logged = False


def get_engine():
    global _engine, _SessionLocal, _unavailable_logged
    if _engine is not None:
        return _engine
    url = os.environ.get("DATABASE_URL")
    if not url:
        if not _unavailable_logged:
            logger.info("DATABASE_URL not set — cross-device sync disabled, rest of the app is unaffected.")
            _unavailable_logged = True
        return None
    # Some providers (Render, Heroku-style) hand out postgres:// URLs;
    # SQLAlchemy's psycopg2 driver requires the postgresql:// scheme.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    _engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session() -> Optional[Session]:
    if get_engine() is None:
        return None
    return _SessionLocal()
