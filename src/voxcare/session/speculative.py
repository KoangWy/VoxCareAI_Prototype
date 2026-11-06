"""Speculative reads (guide §10.4, design D§5.4) — deferred; the cache and its write guard are built, not wired.

Trigger: an ``ORD-\\d{4}`` match on an ASR partial may start a read-only
``lookup_order_details``. The result is cached under ``(session_id, turn_epoch,
query_hash)`` and injected only when the committed transcript carries the same
code; otherwise it is evicted and the real read is issued. **No speculative
writes, ever**, not even as a dry run: :func:`assert_speculative_allowed`
refuses every tool name but the one read.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..middleware.registry import FUTURE_WRITE_TOOLS

SPECULATIVE_TOOLS = frozenset({"lookup_order_details"})
_ORDER_CODE = re.compile(r"\bORD-\d{4}\b", re.I)


class SpeculativeWriteForbidden(ValueError):
    """A tool other than the read-only lookup was offered to the speculative path."""


def assert_speculative_allowed(tool_name: str) -> None:
    if tool_name in FUTURE_WRITE_TOOLS or tool_name not in SPECULATIVE_TOOLS:
        raise SpeculativeWriteForbidden(tool_name)


def _hash(code: str) -> str:
    return hashlib.sha256(code.upper().encode("ascii")).hexdigest()[:16]


@dataclass
class SpeculativeCache:
    entries: dict[tuple[str, int, str], Any] = field(default_factory=dict)

    @staticmethod
    def code_in(partial: str) -> str | None:
        match = _ORDER_CODE.search(partial)
        return match.group(0).upper() if match else None

    def put(self, session_id: str, turn_epoch: int, tool_name: str, order_code: str, result: Any) -> None:
        assert_speculative_allowed(tool_name)
        self.entries[(session_id, turn_epoch, _hash(order_code))] = result

    def take(self, session_id: str, turn_epoch: int, committed_transcript: str) -> Any | None:
        """The cached read when the committed transcript names the same code; every other entry is evicted."""
        code = self.code_in(committed_transcript)
        key = (session_id, turn_epoch, _hash(code)) if code else None
        hit = self.entries.pop(key, None) if key else None
        for stale in [k for k in self.entries if k[0] == session_id and k[1] <= turn_epoch]:
            del self.entries[stale]
        return hit


__all__ = ["SPECULATIVE_TOOLS", "SpeculativeCache", "SpeculativeWriteForbidden", "assert_speculative_allowed"]
