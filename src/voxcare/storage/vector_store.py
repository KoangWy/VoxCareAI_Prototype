"""Chroma collection ``voxcom_policy_v1`` (guide §4.3, §13; design D§3.3.13, D§8).

One document per clause (``id`` = clause id, document text = ``content_en``),
cosine space, ``similarity_score = 1 - cosine_distance``. Metadata carries
``policy_id, clause_ref, category, title``, the corpus fields, and the
applicability rules flattened into boolean keys (``pc_TV_AV``, ``brand_LG``,
``pc_ALL``, ``brand_ALL``) so a Chroma ``where`` filter can apply them.

Embeddings are computed by :mod:`voxcare.storage.embeddings` and passed to Chroma
explicitly; the embedded text is ``title + ". " + content_en`` (implementation
choice: the clause title is part of the clause header and improves top-1 on the
calibration set). The stored document stays ``content_en`` verbatim.

The collection is rebuilt when the corpus, the embedding model or this recipe
changes (a fingerprint in the collection metadata).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from .embeddings import Embedder

COLLECTION_NAME = "voxcom_policy_v1"
CORPUS_ID = "VOXCOM-DEMO-POLICY-V1"
CLAUSE_VERSION = "1.0-demo"
RECIPE_VERSION = "title+content_en/v1"
PRODUCT_CATEGORIES = ("TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN")
BRANDS = ("Samsung", "LG", "Panasonic", "Electrolux", "Toshiba")


def load_corpus() -> dict[str, Any]:
    return json.loads(files("voxcare.domain").joinpath("data/policies.json").read_text(encoding="utf-8"))


def embedding_text(clause: Mapping[str, Any]) -> str:
    return f"{clause['title']}. {clause['content_en']}"


@dataclass(frozen=True)
class PolicyHit:
    clause_id: str
    clause_ref: str
    category: str
    title: str
    text: str
    similarity_score: float
    applicability_rules: Mapping[str, Any]


def _flatten(clause: Mapping[str, Any]) -> dict[str, Any]:
    rules = clause["applicability_rules"]
    categories = rules["product_categories"]
    brands = rules["brands"]
    metadata: dict[str, Any] = {
        "policy_id": clause["policy_id"],
        "clause_ref": clause["clause_ref"],
        "category": clause["category"],
        "title": clause["title"],
        "market": rules["market"],
        "channel": rules["channel"],
        "version_active": rules["version_active"],
        "conditions": rules["conditions"],
        "source_reference": clause.get("source_reference", ""),
        "corpus_id": clause["corpus_id"],
        "clause_version": clause["clause_version"],
        "is_fictional": bool(clause["is_fictional"]),
        "pc_ALL": categories == "ALL",
        "brand_ALL": brands == "ALL",
    }
    for category in PRODUCT_CATEGORIES:
        metadata[f"pc_{category}"] = categories == "ALL" or category in categories
    for brand in BRANDS:
        metadata[f"brand_{brand}"] = brands == "ALL" or brand in brands
    return metadata


def where_filter(
    category: str, product_categories: Sequence[str], brands: Sequence[str], *, market: str, channel: str, version: str
) -> dict[str, Any]:
    """Topic filter (unless ``ALL``) plus applicability (guide §7.3 steps 1-2).

    A clause is excluded when the query names a product category or brand outside its applicability.
    A dimension the query does not name never excludes a clause.
    """
    conditions: list[dict[str, Any]] = [{"market": market}, {"channel": channel}, {"version_active": version}]
    if category != "ALL":
        conditions.append({"category": category})
    known_categories = [item for item in product_categories if item in PRODUCT_CATEGORIES]
    if known_categories:
        conditions.append({"$or": [{f"pc_{item}": True} for item in known_categories]})
    known_brands = [item for item in brands if item in BRANDS]
    if known_brands:
        conditions.append({"$or": [{f"brand_{item}": True} for item in known_brands]})
    for condition in conditions:
        if "$or" in condition and len(condition["$or"]) == 1:
            condition.update(condition.pop("$or")[0])
    return {"$and": conditions}


class PolicyVectorStore:
    def __init__(self, embedder: Embedder, *, path: str | None = None, corpus: Mapping[str, Any] | None = None) -> None:
        import chromadb
        from chromadb.config import Settings

        self.embedder = embedder
        self.corpus = dict(corpus or load_corpus())
        self.clauses = {clause["policy_id"]: clause for clause in self.corpus["clauses"]}
        settings = Settings(anonymized_telemetry=False, allow_reset=False)
        self.path = path
        if path is None:
            # Chroma's EphemeralClient shares one in-process system between instances; a private temporary
            # directory keeps each unpersisted index isolated (tests build several embeddings side by side).
            self._tempdir = tempfile.TemporaryDirectory(prefix="voxcare-chroma-")
            path = self._tempdir.name
        self.client = chromadb.PersistentClient(path=path, settings=settings)
        self.collection: Any = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.corpus, sort_keys=True, ensure_ascii=False) + self.embedder.name + RECIPE_VERSION
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def build(self) -> PolicyVectorStore:
        """Create (or reuse) the collection; rebuilt only when the fingerprint changes."""
        fingerprint = self.fingerprint
        existing = None
        try:
            existing = self.client.get_collection(COLLECTION_NAME, embedding_function=None)
        except Exception:  # chromadb raises NotFoundError (or ValueError in older releases)
            existing = None
        if (
            existing is not None
            and (existing.metadata or {}).get("fingerprint") == fingerprint
            and existing.count() == len(self.clauses)
        ):
            self.collection = existing
            return self
        if existing is not None:
            self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "fingerprint": fingerprint, "embedding": self.embedder.name},
        )
        clauses = list(self.clauses.values())
        self.collection.add(
            ids=[clause["policy_id"] for clause in clauses],
            documents=[clause["content_en"] for clause in clauses],
            embeddings=self.embedder.embed([embedding_text(clause) for clause in clauses]),
            metadatas=[_flatten(clause) for clause in clauses],
        )
        return self

    def count(self) -> int:
        return int(self.collection.count()) if self.collection is not None else 0

    def query(
        self,
        query: str,
        *,
        category: str,
        product_categories: Sequence[str] = (),
        brands: Sequence[str] = (),
        top_k: int = 3,
        market: str = "VN",
        channel: str = "VOICE_CSKH",
        version: str = CLAUSE_VERSION,
    ) -> list[PolicyHit]:
        if self.collection is None:
            raise RuntimeError("policy index is not built")
        where = where_filter(category, product_categories, brands, market=market, channel=channel, version=version)
        candidates = self.collection.get(where=where, include=[])["ids"]
        if not candidates:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(top_k, len(candidates)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for clause_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            clause = self.clauses[clause_id]
            hits.append(
                PolicyHit(
                    clause_id=clause_id,
                    clause_ref=metadata["clause_ref"],
                    category=metadata["category"],
                    title=metadata["title"],
                    text=document,
                    similarity_score=round(1.0 - float(distance), 4),
                    applicability_rules=clause["applicability_rules"],
                )
            )
        return hits


__all__ = [
    "CLAUSE_VERSION",
    "COLLECTION_NAME",
    "CORPUS_ID",
    "PolicyHit",
    "PolicyVectorStore",
    "embedding_text",
    "load_corpus",
    "where_filter",
]
