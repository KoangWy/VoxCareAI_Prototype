"""The shared tool envelope (guide §6.1, design D§4.1).

Success responses always carry ``"outcome": "SUCCESS"``. Errors share one shape::

    {"outcome": "ERROR", "error_code": "<ErrorCode>", "http": 404, "message": "<human-readable>"}

Keys starting with ``_`` are internal (order hex id, audit hints); the tool
middleware strips them before a result reaches the dialogue layer or the model.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..enums import ERROR_HTTP_STATUS, ErrorCode

logger = logging.getLogger("voxcare.services")

DEFAULT_TOOL_TIMEOUT_S = 1.5  # CARRIER_TOOL_TIMEOUT_MS; reused for OMS, vector store, catalog, promotion (guide §3.2)


def success(**fields: Any) -> dict[str, Any]:
    return {"outcome": "SUCCESS", **fields}


def error(code: ErrorCode, message: str, **extra: Any) -> dict[str, Any]:
    return {"outcome": "ERROR", "error_code": str(code), "http": ERROR_HTTP_STATUS[code], "message": message, **extra}


def is_error(envelope: Mapping[str, Any], code: ErrorCode | None = None) -> bool:
    if envelope.get("outcome") != "ERROR":
        return False
    return code is None or envelope.get("error_code") == code


def public(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The envelope without internal ``_`` keys."""
    return {key: value for key, value in envelope.items() if not key.startswith("_")}


class ServiceUnavailable(RuntimeError):
    """A backing store answered with an error (maps to SERVICE_ERROR)."""


async def guarded(
    operation: Callable[[], Awaitable[dict[str, Any]]],
    *,
    timeout_s: float,
    system: str,
) -> dict[str, Any]:
    """Run one service read under its timeout: timeout → SERVICE_TIMEOUT 504, failure → SERVICE_ERROR 500."""
    try:
        return await asyncio.wait_for(operation(), timeout=timeout_s)
    except TimeoutError:
        logger.warning("%s timed out", system)
        return error(ErrorCode.SERVICE_TIMEOUT, f"The {system} did not respond within {int(timeout_s * 1000)} ms.")
    except Exception as exc:
        logger.warning("%s failed: %s", system, type(exc).__name__)
        return error(ErrorCode.SERVICE_ERROR, f"The {system} returned an error.")


__all__ = ["DEFAULT_TOOL_TIMEOUT_S", "ServiceUnavailable", "error", "guarded", "is_error", "public", "success"]
