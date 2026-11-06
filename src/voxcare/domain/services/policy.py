"""PolicyService: ``retrieve_policy_clauses(query, category)`` (guide §7.3, design D§4.5, D§8.1).

1. Filter by topic (the category, unless ``ALL``).
2. Filter by applicability inferred from the query text (market, channel,
   product categories, brands, version): a ``KITCHEN`` installation question never
   gets ``POL-INST-001``, an LG warranty question never gets ``POL-WAR-001``.
3. Vector top_k = 3.
4. Highest similarity ≥ threshold → SUCCESS (the clauses at or above the
   threshold); otherwise ``INSUFFICIENT_EVIDENCE`` with ``http: 200``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...storage.vector_store import CLAUSE_VERSION, CORPUS_ID, PolicyVectorStore
from ..enums import ErrorCode
from ..rules import brands_in, product_categories_in
from .envelope import DEFAULT_TOOL_TIMEOUT_S, error, guarded, success

MISS_MESSAGE = "No fictional demo policy clause matched the inquiry above the grounding threshold."


class PolicyService:
    def __init__(
        self,
        store: PolicyVectorStore | None,
        *,
        threshold: float,
        top_k: int = 3,
        timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.timeout_s = timeout_s

    async def retrieve(self, query: str, category: str) -> dict[str, Any]:
        if self.store is None:
            return error(ErrorCode.SERVICE_ERROR, "The policy index is unavailable.")
        return await guarded(lambda: self._retrieve(query, category), timeout_s=self.timeout_s, system="policy system")

    async def _retrieve(self, query: str, category: str) -> dict[str, Any]:
        assert self.store is not None  # noqa: S101 - checked by retrieve()
        product_categories = product_categories_in(query)
        brands = brands_in(query)
        hits = await asyncio.to_thread(
            self.store.query,
            query,
            category=category,
            product_categories=product_categories,
            brands=brands,
            top_k=self.top_k,
        )
        applicability = {"product_categories": list(product_categories), "brands": list(brands)}
        best = max((hit.similarity_score for hit in hits), default=0.0)
        if not hits or best < self.threshold:
            return {
                **error(
                    ErrorCode.INSUFFICIENT_EVIDENCE,
                    MISS_MESSAGE,
                    highest_similarity_score=round(best, 4),
                    threshold_required=self.threshold,
                    corpus_id=CORPUS_ID,
                    is_fictional=True,
                ),
                "_applicability": applicability,
                "_candidates": [(hit.clause_id, hit.similarity_score) for hit in hits],
            }
        grounded = [hit for hit in hits if hit.similarity_score >= self.threshold]
        return {
            **success(
                corpus_id=CORPUS_ID,
                is_fictional=True,
                clause_version=CLAUSE_VERSION,
                clauses=[
                    {
                        "clause_id": hit.clause_id,
                        "clause_ref": hit.clause_ref,
                        "category": hit.category,
                        "title": hit.title,
                        "text": hit.text,
                        "similarity_score": hit.similarity_score,
                        "applicability_rules": dict(hit.applicability_rules),
                    }
                    for hit in grounded
                ],
            ),
            "_applicability": applicability,
        }


__all__ = ["MISS_MESSAGE", "PolicyService"]
