"""
Password hashing + JWT issuance/verification for the optional cross-device
sync feature (see db.py). Kept separate from main.py's request handlers so
the crypto logic is easy to audit in one place.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import jwt
from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days — a personal-use app, not a bank


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, password_hash)
    except Exception:
        return False


def create_token(user_id: int) -> Optional[str]:
    """None means JWT_SECRET isn't configured — caller should 503, not issue
    an unsigned or default-secret token."""
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        return None
    now = int(time.time())
    payload = {"sub": str(user_id), "iat": now, "exp": now + JWT_TTL_SECONDS}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except Exception:
        return None
