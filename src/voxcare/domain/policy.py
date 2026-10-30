"""Deterministic, read-only retrieval over the fictional VoxCom demo policy corpus.

Clause texts are transcribed verbatim from spec 42 §8.2 (9 clauses, SHIPPING 2,
INSTALLATION 2, WARRANTY 3, RETURNS 2); retrieval is deterministic lexical
overlap — a demo simplification of the spec §8.2 vector design (top_k=3,
threshold 0.70), not a semantic similarity score, and must not be compared
with the design threshold.

This module only returns authored evidence. It does not decide eligibility, make
business promises, query order data, or call a language model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

_ALLOWED_CATEGORIES = frozenset({"SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS"})
_ALLOWED_PRODUCT_CATEGORIES = frozenset({"TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN"})
_ALLOWED_FILTERS = frozenset({"category", "product_category", "brand", "market", "channel"})
_DEFAULT_MARKET = "VN"
_DEFAULT_CHANNEL = "VOICE_CSKH"
_MIN_QUERY_COVERAGE = 0.2
_TOPIC_TERMS = {
    "SHIPPING": frozenset({"ship", "delivery", "estimate", "carrier", "tracking", "freight", "fee", "quote", "route"}),
    "INSTALLATION": frozenset(
        {"install", "installation", "mount", "mounting", "setup", "technician", "pedestal", "drill"}
    ),
    "WARRANTY": frozenset({"warranty", "repair", "coverage", "manufacturer", "guarantee", "service"}),
    "RETURNS": frozenset({"return", "refund", "replace", "exchange", "defect", "restocking"}),
}
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "order",
        "please",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "with",
        "would",
        "voxcom",
    ]
)
_SYNONYMS = {
    "eta": "estimate",
    "estimated": "estimate",
    "date": "estimate",
    "delivery": "ship",
    "delivered": "ship",
    "shipment": "ship",
    "shipping": "ship",
    "cost": "fee",
    "price": "fee",
    "charge": "fee",
    "charges": "fee",
    "returning": "return",
    "returned": "return",
    "replacement": "replace",
    "replace": "replace",
    "coverage": "warranty",
    "covered": "warranty",
    "guarantee": "warranty",
    "appliance": "product",
    "model": "product",
}


@dataclass(frozen=True)
class PolicyFilters:
    """Allowlisted policy filters. `ALL` is allowed only as a query category."""

    category: str = "ALL"
    product_category: str | None = None
    brand: str | None = None
    market: str = _DEFAULT_MARKET
    channel: str = _DEFAULT_CHANNEL

    @classmethod
    def from_value(cls, value: PolicyFilters | Mapping[str, object] | None) -> PolicyFilters:
        if value is None:
            return cls()
        if isinstance(value, cls):
            value = asdict(value)
        if not isinstance(value, Mapping):
            raise ValueError("filters must be a PolicyFilters object or mapping")
        unknown = set(value) - _ALLOWED_FILTERS
        if unknown:
            raise ValueError(f"unsupported policy filter(s): {', '.join(sorted(map(str, unknown)))}")
        category = (_clean_filter(value.get("category"), default="ALL") or "ALL").upper()
        product_category = _clean_filter(value.get("product_category"))
        product_category = product_category.upper() if product_category else None
        brand = _clean_filter(value.get("brand"))
        market = (_clean_filter(value.get("market"), default=_DEFAULT_MARKET) or _DEFAULT_MARKET).upper()
        channel = (_clean_filter(value.get("channel"), default=_DEFAULT_CHANNEL) or _DEFAULT_CHANNEL).upper()
        if category not in _ALLOWED_CATEGORIES | {"ALL"}:
            raise ValueError("category must be SHIPPING, INSTALLATION, WARRANTY, RETURNS, or ALL")
        if product_category is not None and product_category not in _ALLOWED_PRODUCT_CATEGORIES:
            raise ValueError(
                "product_category must be TV_AV, REFRIGERATOR, WASHING_MACHINE, AIR_CONDITIONER, or KITCHEN"
            )
        return cls(category, product_category, brand, market, channel)


@dataclass(frozen=True)
class PolicyEvidence:
    policy_id: str
    clause_ref: str
    category: str
    title: str
    content_en: str
    applicability_rules: Mapping[str, Any]
    source_reference: str
    clause_version: str
    is_fictional: bool
    match_score: float


@dataclass(frozen=True)
class PolicyEvidenceResult:
    outcome: str
    corpus_id: str
    is_fictional: bool
    clause_version: str
    query: str
    filters: PolicyFilters
    evidence: Sequence[PolicyEvidence] = ()
    error_code: str | None = None
    missing_context: Sequence[str] = ()


def _clean_filter(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("filter values must be non-empty strings")
    return value.strip()


def _normalize(value: str) -> str:
    tokens = []
    for token in _TOKEN_RE.findall(value.lower()):
        if token in _STOP_WORDS:
            continue
        normalized = _SYNONYMS.get(token, token)
        if len(normalized) > 4 and normalized.endswith("s"):
            normalized = normalized[:-1]
        tokens.append(normalized)
    return " ".join(tokens)


def _tokens(value: str) -> set[str]:
    return set(_normalize(value).split())


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    raw = files("voxcare.domain").joinpath("data", "policies.json").read_text(encoding="utf-8")
    corpus = json.loads(raw)
    clauses = corpus.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        raise ValueError("policy corpus must contain a non-empty clauses list")
    seen_ids: set[str] = set()
    for clause in clauses:
        required = {
            "policy_id",
            "clause_ref",
            "category",
            "title",
            "content_en",
            "source_reference",
            "applicability_rules",
            "corpus_id",
            "clause_version",
            "is_fictional",
        }
        if not isinstance(clause, dict) or not required <= clause.keys():
            raise ValueError("policy clause is missing required contract fields")
        if clause["policy_id"] in seen_ids:
            raise ValueError(f"duplicate policy_id: {clause['policy_id']}")
        seen_ids.add(clause["policy_id"])
        if clause["category"] not in _ALLOWED_CATEGORIES:
            raise ValueError(f"invalid stored topic category: {clause['category']}")
        if clause["corpus_id"] != corpus.get("corpus_id") or clause["is_fictional"] is not True:
            raise ValueError(f"invalid fictional-corpus metadata on {clause['policy_id']}")
        if clause["clause_version"] != corpus.get("corpus_version"):
            raise ValueError(f"version mismatch on {clause['policy_id']}")
    return corpus, tuple(clauses)


def _applies(clause: Mapping[str, Any], filters: PolicyFilters) -> bool:
    rules = clause["applicability_rules"]
    if rules.get("market") != filters.market or rules.get("channel") != filters.channel:
        return False
    requested_category = filters.category.upper()
    if requested_category != "ALL" and clause["category"] != requested_category:
        return False
    requested_product = filters.product_category
    allowed_products = rules.get("product_categories", "ALL")
    if allowed_products != "ALL" and (
        requested_product is None or requested_product.upper() not in {str(item).upper() for item in allowed_products}
    ):
        return False
    requested_brand = filters.brand
    allowed_brands = rules.get("brands", "ALL")
    return allowed_brands == "ALL" or (
        requested_brand is not None and requested_brand.casefold() in {str(item).casefold() for item in allowed_brands}
    )


def _missing_context(clause: Mapping[str, Any], filters: PolicyFilters) -> tuple[str, ...]:
    rules = clause["applicability_rules"]
    missing: list[str] = []
    if rules.get("product_categories", "ALL") != "ALL" and filters.product_category is None:
        missing.append("product_category")
    if rules.get("brands", "ALL") != "ALL" and filters.brand is None:
        missing.append("brand")
    return tuple(missing)


def _topic_mismatch(query_tokens: set[str], category: str) -> bool:
    if category == "ALL":
        return False
    requested_terms = _TOPIC_TERMS[category]
    other_terms = set().union(*(_TOPIC_TERMS[name] for name in _ALLOWED_CATEGORIES if name != category))
    return bool(query_tokens & other_terms) and not bool(query_tokens & requested_terms)


def _score(query_tokens: set[str], clause: Mapping[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    evidence_tokens = _tokens(" ".join((clause["title"], clause["content_en"])))
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def retrieve_policy_clauses_sync(
    query: str,
    filters: PolicyFilters | Mapping[str, object] | None = None,
    *,
    limit: int = 3,
) -> PolicyEvidenceResult:
    """Retrieve applicable fictional clauses with deterministic lexical ranking.

    This deterministic overlap score is not a semantic similarity score and must
    not be compared with the design's vector-retrieval 0.70 threshold.
    """
    corpus, clauses = _load_corpus()
    try:
        parsed_filters = PolicyFilters.from_value(filters)
    except ValueError:
        return PolicyEvidenceResult(
            "INVALID_INPUT",
            corpus["corpus_id"],
            True,
            corpus["corpus_version"],
            query if isinstance(query, str) else "",
            PolicyFilters(),
            error_code="INVALID_FILTERS",
        )
    if not isinstance(query, str) or not query.strip():
        return PolicyEvidenceResult(
            "MISSING_INPUT",
            corpus["corpus_id"],
            True,
            corpus["corpus_version"],
            "" if not isinstance(query, str) else query,
            parsed_filters,
            error_code="QUERY_REQUIRED",
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return PolicyEvidenceResult(
            "INVALID_INPUT",
            corpus["corpus_id"],
            True,
            corpus["corpus_version"],
            query.strip(),
            parsed_filters,
            error_code="INVALID_LIMIT",
        )

    query_tokens = _tokens(query)
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    needs_context: list[tuple[float, Mapping[str, Any], tuple[str, ...]]] = []
    for clause in clauses:
        rules = clause["applicability_rules"]
        if rules.get("market") != parsed_filters.market or rules.get("channel") != parsed_filters.channel:
            continue
        if parsed_filters.category != "ALL" and clause["category"] != parsed_filters.category:
            continue
        score = _score(query_tokens, clause)
        if score < _MIN_QUERY_COVERAGE:
            continue
        missing = _missing_context(clause, parsed_filters)
        if missing:
            needs_context.append((score, clause, missing))
        elif _applies(clause, parsed_filters):
            candidates.append((score, clause))

    if _topic_mismatch(query_tokens, parsed_filters.category):
        return PolicyEvidenceResult(
            "INSUFFICIENT_EVIDENCE",
            corpus["corpus_id"],
            True,
            corpus["corpus_version"],
            query.strip(),
            parsed_filters,
            error_code="TOPIC_FILTER_MISMATCH",
        )
    if needs_context:
        needs_context.sort(key=lambda item: (-item[0], item[1]["policy_id"]))
        best_context_score = needs_context[0][0]
        best_evidence_score = max((score for score, _ in candidates), default=0.0)
        if best_context_score > best_evidence_score:
            required = sorted(
                {field for score, _, missing in needs_context if score == best_context_score for field in missing}
            )
            return PolicyEvidenceResult(
                "NEEDS_CONTEXT",
                corpus["corpus_id"],
                True,
                corpus["corpus_version"],
                query.strip(),
                parsed_filters,
                error_code="APPLICABILITY_CONTEXT_REQUIRED",
                missing_context=required,
            )
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["policy_id"]))
    if candidates:
        best_score = candidates[0][0]
        candidates = [pair for pair in candidates if pair[0] >= max(_MIN_QUERY_COVERAGE, best_score * 0.75)]
    evidence = tuple(
        PolicyEvidence(
            policy_id=clause["policy_id"],
            clause_ref=clause["clause_ref"],
            category=clause["category"],
            title=clause["title"],
            content_en=clause["content_en"],
            applicability_rules=clause["applicability_rules"],
            source_reference=clause["source_reference"],
            clause_version=clause["clause_version"],
            is_fictional=clause["is_fictional"],
            match_score=round(score, 4),
        )
        for score, clause in candidates[:limit]
    )
    return PolicyEvidenceResult(
        "SUCCESS" if evidence else "INSUFFICIENT_EVIDENCE",
        corpus["corpus_id"],
        True,
        corpus["corpus_version"],
        query.strip(),
        parsed_filters,
        evidence,
        None if evidence else "NO_APPLICABLE_EVIDENCE",
    )


async def retrieve_policy_clauses(
    query: str,
    filters: PolicyFilters | Mapping[str, object] | None = None,
) -> PolicyEvidenceResult:
    """Async session-seam adapter over the deterministic, read-only retriever."""
    return retrieve_policy_clauses_sync(query, filters)


class PolicyRetriever:
    """Small injectable adapter matching `SessionBoundary.policy_retriever`."""

    async def retrieve_policy_clauses(
        self,
        query: str,
        filters: PolicyFilters | Mapping[str, object] | None = None,
    ) -> PolicyEvidenceResult:
        return await retrieve_policy_clauses(query, filters)
