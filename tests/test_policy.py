import asyncio
import unittest

from voxcare.domain.policy import (
    PolicyFilters,
    retrieve_policy_clauses,
    retrieve_policy_clauses_sync,
)


class PolicyRetrievalTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_retrieval_returns_fictional_metadata_and_grounded_chunk(self):
        """Corpus clause is the spec §8.2 verbatim text; fictionality rides on metadata, not prose."""
        result = self.run_async(
            retrieve_policy_clauses(
                "What is the return window for a defective product?",
                {"category": "returns", "brand": "Samsung"},
            )
        )

        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual(result.corpus_id, "VOXCOM-DEMO-POLICY-V1")
        self.assertTrue(result.is_fictional)
        self.assertEqual(result.clause_version, "1.0-demo")
        self.assertEqual([row.policy_id for row in result.evidence], ["POL-RET-001"])
        self.assertEqual(result.evidence[0].category, "RETURNS")

    def test_topic_and_brand_filters_do_not_cross_apply_warranty_terms(self):
        result = retrieve_policy_clauses_sync(
            "LG washer warranty duration",
            PolicyFilters(category="WARRANTY", product_category="WASHING_MACHINE", brand="LG"),
        )

        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual([row.policy_id for row in result.evidence], ["POL-WAR-002"])
        self.assertTrue(all(row.category == "WARRANTY" for row in result.evidence))
        self.assertNotIn("POL-WAR-001", {row.policy_id for row in result.evidence})

    def test_generic_warranty_query_does_not_claim_a_brand_specific_clause(self):
        result = retrieve_policy_clauses_sync("warranty period for a washing machine", {"category": "WARRANTY"})

        self.assertEqual(result.outcome, "NEEDS_CONTEXT")
        self.assertEqual(result.evidence, ())
        self.assertIn("brand", result.missing_context)

    def test_explicit_topic_conflict_returns_no_evidence(self):
        result = retrieve_policy_clauses_sync(
            "What is the return and refund window?",
            {"category": "WARRANTY"},
        )

        self.assertEqual(result.outcome, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.error_code, "TOPIC_FILTER_MISMATCH")
        self.assertEqual(result.evidence, ())

    def test_product_applicability_excludes_unknown_category(self):
        result = retrieve_policy_clauses_sync(
            "bulky appliance stair carry threshold",
            {"category": "SHIPPING", "product_category": "KITCHEN", "brand": "Samsung"},
        )

        self.assertEqual(result.outcome, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.error_code, "NO_APPLICABLE_EVIDENCE")

    def test_unknown_topic_does_not_retrieve_unrelated_corpus_content(self):
        result = retrieve_policy_clauses_sync(
            "Do you accept trade-ins for a ten-year-old microwave?",
            {"category": "ALL"},
        )

        self.assertEqual(result.outcome, "INSUFFICIENT_EVIDENCE")

    def test_delivery_eta_uses_carrier_evidence_and_quotes_distance_terms(self):
        result = retrieve_policy_clauses_sync(
            "delivery ETA estimate",
            {"category": "SHIPPING"},
        )

        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual(result.evidence[0].policy_id, "POL-SHIP-001")
        self.assertIn("500,000 VND", result.evidence[0].content_en)

    def test_missing_query_and_invalid_filters_are_typed(self):
        missing = retrieve_policy_clauses_sync("  ")
        invalid = retrieve_policy_clauses_sync("Samsung warranty", {"category": "ALL", "order_id": "ORD-1001"})

        self.assertEqual((missing.outcome, missing.error_code), ("MISSING_INPUT", "QUERY_REQUIRED"))
        self.assertEqual((invalid.outcome, invalid.error_code), ("INVALID_INPUT", "INVALID_FILTERS"))

    def test_installation_topic_retrieves_installation_clauses(self):
        result = retrieve_policy_clauses_sync(
            "standard installation for a Samsung television pedestal",
            {"category": "INSTALLATION", "product_category": "TV_AV", "brand": "Samsung"},
        )

        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual([row.policy_id for row in result.evidence], ["POL-INST-001"])
        self.assertTrue(all(row.category == "INSTALLATION" for row in result.evidence))
        self.assertIn("pedestal", result.evidence[0].content_en)

    def test_kitchen_appliance_installation_reports_insufficient_evidence(self):
        result = retrieve_policy_clauses_sync(
            "installation for a kitchen appliance",
            {"category": "INSTALLATION", "product_category": "KITCHEN", "brand": "Samsung"},
        )

        self.assertEqual(result.outcome, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.evidence, ())

    def test_corpus_carries_nine_clauses_with_installation_pair(self):
        from voxcare.domain.policy import _load_corpus

        _corpus, clauses = _load_corpus()
        ids = sorted(clause["policy_id"] for clause in clauses)
        self.assertEqual(
            ids,
            [
                "POL-INST-001",
                "POL-INST-002",
                "POL-RET-001",
                "POL-RET-002",
                "POL-SHIP-001",
                "POL-SHIP-002",
                "POL-WAR-001",
                "POL-WAR-002",
                "POL-WAR-003",
            ],
        )
        self.assertTrue(all(clause["is_fictional"] for clause in clauses))

    def test_all_is_query_only_and_never_a_stored_category(self):
        result = retrieve_policy_clauses_sync(
            "defective return window",
            {"category": "ALL"},
            limit=10,
        )

        self.assertEqual(result.outcome, "SUCCESS")
        self.assertNotIn("ALL", {row.category for row in result.evidence})
        self.assertLessEqual(
            {row.category for row in result.evidence}, {"SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS"}
        )


if __name__ == "__main__":
    unittest.main()
