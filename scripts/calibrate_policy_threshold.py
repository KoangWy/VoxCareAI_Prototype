"""Calibrate the policy-retrieval threshold for an embedding model (G-10).

The design's 0.70 threshold and its demo scores (0.88, 0.52) belong to an
embedding model the design does not name. This script runs the real retrieval
path (topic filter, applicability filter, Chroma top-k) over the labeled set in
``evaluation/policy_calibration.json`` and reports, for each candidate threshold,
how many in-domain queries ground (best score at or above it, expected clause in
the results) and how many out-of-domain queries are refused. Scenario 4 (both
POL-WAR-001 and POL-RET-001 grounded) and Scenario 5 (refused) are hard checks.

    uv run python scripts/calibrate_policy_threshold.py --embedding minilm
    uv run python scripts/calibrate_policy_threshold.py --embedding hashing --json

Report the chosen value as "measured on N queries" (docs/POLICY_RETRIEVAL.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxcare.domain.rules import brands_in, product_categories_in  # noqa: E402
from voxcare.storage.embeddings import CALIBRATED_THRESHOLDS, DESIGN_THRESHOLD, create_embedder  # noqa: E402
from voxcare.storage.vector_store import PolicyVectorStore, embedding_text, load_corpus  # noqa: E402


def retrieve_all(store: PolicyVectorStore, cases: list[dict]) -> list[list]:
    """One retrieval per case through the production filters (embeddings are the slow part)."""
    return [
        store.query(
            case["query"],
            category=case["category"],
            product_categories=product_categories_in(case["query"]),
            brands=brands_in(case["query"]),
        )
        for case in cases
    ]


def evaluate(cached: list[list], cases: list[dict], threshold: float) -> dict:
    grounded = refused = top_ok = 0
    failures = []
    positives = [case for case in cases if case["expected"]]
    negatives = [case for case in cases if not case["expected"]]
    for case, hits in zip(cases, cached, strict=True):
        best = max((hit.similarity_score for hit in hits), default=0.0)
        cited = [hit.clause_id for hit in hits if hit.similarity_score >= threshold]
        if case["expected"]:
            ok = best >= threshold and all(expected in cited for expected in case["expected"][:1])
            top_ok += bool(hits) and hits[0].clause_id in case["expected"]
            grounded += ok
        else:
            ok = best < threshold
            refused += ok
        if not ok:
            failures.append(
                {"query": case["query"], "best": round(best, 4), "cited": cited, "expected": case["expected"]}
            )
    total = len(cases)
    return {
        "threshold": round(threshold, 4),
        "in_domain_grounded": f"{grounded}/{len(positives)}",
        "out_of_domain_refused": f"{refused}/{len(negatives)}",
        "accuracy": round((grounded + refused) / total, 4),
        "top1_correct": f"{top_ok}/{len(positives)}",
        "failures": failures,
    }


def scenario_checks(cached: list[list], threshold: float) -> dict:
    hits4, hits5 = cached[0], cached[1]
    cited4 = {hit.clause_id for hit in hits4 if hit.similarity_score >= threshold}
    best5 = max((hit.similarity_score for hit in hits5), default=0.0)
    return {
        "scenario_4_cites_POL-WAR-001_and_POL-RET-001": {"POL-WAR-001", "POL-RET-001"} <= cited4,
        "scenario_4_scores": {hit.clause_id: hit.similarity_score for hit in hits4},
        "scenario_5_refused": best5 < threshold,
        "scenario_5_best_score": best5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedding", choices=("minilm", "hashing"), default="minilm")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation" / "policy_calibration.json").read_text(encoding="utf-8"))["cases"]
    corpus = load_corpus()
    store = PolicyVectorStore(create_embedder(args.embedding, [embedding_text(c) for c in corpus["clauses"]])).build()
    cached = retrieve_all(store, cases)
    scores = [max((hit.similarity_score for hit in hits), default=0.0) for hits in cached]
    candidates = sorted({round(score, 4) for score in scores} | {round(score + 1e-4, 4) for score in scores})
    best = max(
        candidates,
        key=lambda thr: (evaluate(cached, cases, thr)["accuracy"], -abs(thr - CALIBRATED_THRESHOLDS[args.embedding])),
    )
    report = {
        "embedding": args.embedding,
        "queries": len(cases),
        "in_domain": sum(1 for case in cases if case["expected"]),
        "out_of_domain": sum(1 for case in cases if not case["expected"]),
        "best_accuracy_threshold": evaluate(cached, cases, best),
        "configured_default": evaluate(cached, cases, CALIBRATED_THRESHOLDS[args.embedding]),
        "design_threshold_0.70": evaluate(cached, cases, DESIGN_THRESHOLD),
        "scenario_checks_at_default": scenario_checks(cached, CALIBRATED_THRESHOLDS[args.embedding]),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"embedding={args.embedding}  measured on {len(cases)} queries "
            f"({report['in_domain']} in-domain, {report['out_of_domain']} out-of-domain)"
        )
        for label in ("best_accuracy_threshold", "configured_default", "design_threshold_0.70"):
            row = report[label]
            print(
                f"  {label:26s} thr={row['threshold']:<7} accuracy={row['accuracy']:<6} "
                f"grounded={row['in_domain_grounded']:<6} "
                f"refused={row['out_of_domain_refused']:<6} top1={row['top1_correct']}"
            )
        checks = report["scenario_checks_at_default"]
        print(
            f"  Scenario 4 cites WAR-001+RET-001: {checks['scenario_4_cites_POL-WAR-001_and_POL-RET-001']}  "
            f"Scenario 5 refused: {checks['scenario_5_refused']} (best {checks['scenario_5_best_score']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
