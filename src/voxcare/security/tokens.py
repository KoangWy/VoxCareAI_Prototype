"""``VoxCare-Session-Token``: an ephemeral HS256 JWT carrying ``{session_id, customer_id|null, exp}`` (guide §15).

Issued by ``POST /session`` (and ``POST /api/sessions``); verified on every REST
call and on the voice WebSocket handshake. ``exp`` never exceeds the session TTL
(1800 s). The secret is ``SESSION_TOKEN_SECRET``; without it a random per-process
secret is generated (tokens then die with the process) and a warning is logged.
The demo simulates verification by binding a fixture customer (OPD-04);
production needs an OTP challenge.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

import jwt

logger = logging.getLogger("voxcare.security")
_warned_random_secret = False

ALGORITHM = "HS256"
HEADER_NAME = "VoxCare-Session-Token"
_MIN_SECRET_BYTES = 32


_PROCESS_SECRET = secrets.token_urlsafe(48)


class InvalidToken(ValueError):
    """Missing, malformed, unsigned, expired or mismatched session token."""


@dataclass(frozen=True)
class SessionClaims:
    session_id: str
    customer_id: str | None
    exp: int
    probe: bool = False


class TokenService:
    def __init__(self, secret: str | None = None, *, ttl_s: int = 1800) -> None:
        configured = secret if secret is not None else os.getenv("SESSION_TOKEN_SECRET", "")
        if not configured:
            global _warned_random_secret
            configured = _PROCESS_SECRET
            if not _warned_random_secret:
                _warned_random_secret = True
                logger.warning("SESSION_TOKEN_SECRET is not set; using a random per-process secret")
        elif len(configured.encode("utf-8")) < _MIN_SECRET_BYTES:
            raise ValueError("SESSION_TOKEN_SECRET must be at least 32 bytes")
        self._secret = configured
        self.ttl_s = ttl_s

    def issue(self, session_id: str, customer_id: str | None, *, now: datetime, probe: bool = False) -> str:
        exp = int((now + timedelta(seconds=self.ttl_s)).timestamp())
        claims: dict[str, object] = {"session_id": session_id, "customer_id": customer_id, "exp": exp}
        if probe:
            claims["probe"] = True
        return jwt.encode(claims, self._secret, algorithm=ALGORITHM)

    def verify(self, token: str | None, *, now: datetime, session_id: str | None = None) -> SessionClaims:
        if not token:
            raise InvalidToken("missing token")
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                options={"require": ["exp", "session_id"], "verify_exp": False},
            )
        except jwt.PyJWTError as exc:
            raise InvalidToken(type(exc).__name__) from None
        if int(claims["exp"]) <= int(now.timestamp()):
            raise InvalidToken("expired")
        if session_id is not None and claims["session_id"] != session_id:
            raise InvalidToken("session mismatch")
        customer = claims.get("customer_id")
        return SessionClaims(
            session_id=str(claims["session_id"]),
            customer_id=str(customer) if customer else None,
            exp=int(claims["exp"]),
            probe=bool(claims.get("probe", False)),
        )


def bearer_token(authorization: str | None, header_token: str | None) -> str | None:
    """The token from ``VoxCare-Session-Token`` or ``Authorization: Bearer …``."""
    if header_token:
        return header_token.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


__all__ = ["ALGORITHM", "HEADER_NAME", "InvalidToken", "SessionClaims", "TokenService", "bearer_token"]
