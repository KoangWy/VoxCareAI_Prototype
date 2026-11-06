"""Anti-scraping window (FR-SEC-01 proposal; guide §15, design D§9.1, D§9.2.3).

A sorted set per session (``throttle:{session_id}``) holds the distinct SKUs the
caller touched in the last 120 s. When a ``search_catalog`` call would bring
that count to 4 with a SKU not yet in the window, the call is refused with
OUT_OF_SCOPE 403 and audited as ``ANTI_SCRAPING_THROTTLE``. A bulk stock-count
request is refused the same way. Questions about models already in the window
stay allowed ("single-model questions stay allowed"). The 4 SKUs / 2 minutes
figure is the design's operationalisation of threat T3, not an approved limit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..storage.redis_store import RedisStore

THROTTLE_DISTINCT_SKUS = 4
BULK_STOCK_REQUEST = re.compile(
    r"\b(?:stock\s+(?:count|counts|level|levels|quantit\w*|numbers?)|how\s+many\s+(?:units?|pieces|are\s+(?:left|in\s+stock)|"
    r"do\s+you\s+have)|inventory|units?\s+(?:left|available|in\s+stock)|exact\s+stock|every\s+(?:sku|model)|all\s+(?:skus|"
    r"models|products)\s+(?:and|with)\s+(?:stock|quantit\w*))\b",
    re.IGNORECASE,
)


class CatalogThrottle:
    def __init__(self, store: RedisStore, *, distinct_skus: int = THROTTLE_DISTINCT_SKUS, enabled: bool = True) -> None:
        self.store = store
        self.distinct_skus = distinct_skus
        self.enabled = enabled

    @staticmethod
    def bulk_request(*texts: str | None) -> bool:
        return any(text and BULK_STOCK_REQUEST.search(text) for text in texts)

    async def admit(self, session_id: str, skus: Iterable[str]) -> bool:
        """Record ``skus`` unless they would push the window to the limit with a new SKU."""
        wanted = {sku for sku in skus if sku}
        if not self.enabled:
            await self.store.throttle_record(session_id, wanted)
            return True
        window = await self.store.throttle_window(session_id)
        new = wanted - window
        if new and len(window | wanted) >= self.distinct_skus:
            return False
        await self.store.throttle_record(session_id, wanted)
        return True

    async def record(self, session_id: str, skus: Iterable[str]) -> None:
        await self.store.throttle_record(session_id, {sku for sku in skus if sku})


__all__ = ["BULK_STOCK_REQUEST", "THROTTLE_DISTINCT_SKUS", "CatalogThrottle"]
