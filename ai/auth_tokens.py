"""JWT helpers for bearer auth."""

from __future__ import annotations

import os
import time
from typing import Any

import jwt

_ALG = "HS256"
_DEFAULT_DEV_SECRET = "dev-insecure-change-me"


def jwt_secret() -> str:
    return (os.environ.get("JWT_SECRET_KEY") or "").strip() or _DEFAULT_DEV_SECRET


def issue_access_token(*, user_id: int, ttl_sec: int = 60 * 60 * 24 * 14) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {"sub": str(user_id), "iat": now, "exp": now + ttl_sec}
    return jwt.encode(payload, jwt_secret(), algorithm=_ALG)


def decode_access_token(token: str) -> int | None:
    try:
        data = jwt.decode(token, jwt_secret(), algorithms=[_ALG])
        sub = data.get("sub")
        if isinstance(sub, str) and sub.isdigit():
            return int(sub)
        if isinstance(sub, int):
            return sub
    except jwt.PyJWTError:
        return None
    return None
