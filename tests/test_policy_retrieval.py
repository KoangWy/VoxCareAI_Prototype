"""Policy retrieval (guide §13.3): assert clause ids and threshold behaviour, never exact scores.

Runs with the offline hashing embedding; the same table runs with all-MiniLM-L6-v2
when its ONNX model is already in Chroma's cache (no download inside tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voxcare.domain.services.policy import PolicyService
from voxcare.storage.embeddings import CALIBRATED_THRESHOLDS, create_embedder
from voxcare.storage.vector_store import COLLECTION_NAME, PolicyVectorStore, embedding_text, load_corpus

MINILM_CACHED = (Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx").exists()
EMBEDDINGS = [
    "hashing",
    pytest.param("minilm", marks=pytest.mark.skipif(not MINILM_CACHED, reason="MiniLM model not cached")),
]
_STORES: dict[str, PolicyVectorStore] = {}


def _service(name: str) -> PolicyService:
    if name not in _STORES:
        corpus = load_corpus()
        _STORES[name] = PolicyVectorStore(create_embedder(name, [embedding_text(c) for c in corpus["clauses"]])).build()
    return PolicyService(_STORES[name], threshold=CALIBRATED_THRESHOLDS[name])


def test_corpus_has_exactly_nine_fictional_clauses_with_headers():
    corpus = load_corpus()
    clauses = corpus["clauses"]
    assert [c["policy_id"] for c in clauses] == [
        "POL-SHIP-001",
        "POL-SHIP-002",
        "POL-INST-001",
        "POL-INST-002",
        "POL-WAR-001",
        "POL-WAR-002",
        "POL-WAR-003",
        "POL-RET-001",
        "POL-RET-002",
    ]
    header = (
        "[Fictional VoxCom Demo Policy v1.0, Market: VN, Language: EN, Channel: Voice CSKH, "
        "Scope: Consumer Electronics]"
    )
    for clause in clauses:
        assert clause["header"] == header and clause["is_fictional"] is True
        assert clause["corpus_id"] == "VOXCOM-DEMO-POLICY-V1" and clause["clause_version"] == "1.0-demo"
        assert clause["category"] in {"SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS"}  # never ALL


@pytest.mark.parametrize("embedding", EMBEDDINGS)
async def test_retrieval_table(embedding):
    service = _service(embedding)
    both = await service.retrieve(
        "What is the warranty period for a Samsung refrigerator compressor and can I return it if it has a "
        "manufacturing defect?",
        "ALL",
    )
    cited = [c["clause_id"] for c in both["clauses"]]
    assert cited[0] == "POL-WAR-001" and "POL-RET-001" in cited
    trade_in = await service.retrieve("Do you accept trade-ins for 10-year-old microwave ovens with cash back?", "ALL")
    assert trade_in["error_code"] == "INSUFFICIENT_EVIDENCE" and trade_in["http"] == 200
    assert trade_in["highest_similarity_score"] < trade_in["threshold_required"]
    kitchen = await service.retrieve("Can you install my kitchen oven?", "INSTALLATION")
    assert kitchen["error_code"] == "INSUFFICIENT_EVIDENCE"  # POL-INST-001 excluded by applicability
    assert "POL-INST-001" not in str(kitchen)
    lg = await service.retrieve("What warranty does an LG washer have?", "WARRANTY")
    assert [c["clause_id"] for c in lg["clauses"]] == ["POL-WAR-002"]  # never POL-WAR-001
    warranty = await service.retrieve("How long is the warranty on Panasonic appliances?", "WARRANTY")
    assert "POL-RET-001" not in [c["clause_id"] for c in warranty["clauses"]]  # topic mismatch
    tv_install = await service.retrieve("Is installation free for my TV on a table stand?", "INSTALLATION")
    assert tv_install["outcome"] == "SUCCESS" and tv_install["clauses"][0]["clause_id"] == "POL-INST-001"


async def test_success_carries_the_mandatory_corpus_fields_and_scores_in_range():
    result = await _service("hashing").retrieve(
        "Can I get a 1-for-1 replacement for a defective TV within 30 days?", "RETURNS"
    )
    assert {"corpus_id", "is_fictional", "clause_version", "clauses"} <= set(result)
    for clause in result["clauses"]:
        assert set(clause) >= {"clause_id", "clause_ref", "category", "title", "text", "similarity_score"}
        assert 0.0 <= clause["similarity_score"] <= 1.0
        assert clause["similarity_score"] >= result.get("threshold_required", 0)


async def test_index_down_is_service_error():
    service = PolicyService(None, threshold=0.5)
    result = await service.retrieve("anything", "ALL")
    assert result["error_code"] == "SERVICE_ERROR" and result["http"] == 500


def test_collection_metadata_and_flattened_applicability():
    store = _service("hashing").store
    assert store.collection.name == COLLECTION_NAME
    assert store.collection.metadata["hnsw:space"] == "cosine"
    record = store.collection.get(ids=["POL-INST-001"], include=["metadatas", "documents"])
    metadata = record["metadatas"][0]
    assert metadata["category"] == "INSTALLATION" and metadata["pc_TV_AV"] is True and metadata["pc_KITCHEN"] is False
    assert metadata["brand_ALL"] is True and metadata["is_fictional"] is True
    assert record["documents"][0].startswith("Standard complimentary installation service covers")
