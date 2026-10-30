import asyncio
import unittest

from voxcare.contracts import PlannedToolCall
from voxcare.domain.seed import SeedOMS
from voxcare.session import TOOLS, SessionBoundary


class ScriptedDialogue:
    def __init__(self, *plans):
        self.plans = list(plans)

    async def plan_turn(self, *, user_text, history, available_tools):
        return self.plans.pop(0)


def run(coro):
    return asyncio.run(coro)


class CatalogTrioTests(unittest.TestCase):
    def setUp(self):
        self.oms = SeedOMS()

    def turn(self, plan):
        service = SessionBoundary(dialogue=ScriptedDialogue(plan), oms=self.oms)
        session = run(service.create_session())
        return service, run(service.handle_turn(session.session_id, "catalog question", source="manual"))

    def test_registry_lists_the_full_demo_tool_set(self):
        names = {tool["name"] for tool in TOOLS}
        self.assertEqual(
            names,
            {
                "lookup_order_details",
                "track_shipment_status",
                "retrieve_policy_clauses",
                "search_catalog",
                "compare_products",
                "apply_voucher",
                "escalate_to_human",
            },
        )
        for tool in TOOLS:
            self.assertEqual(tool["parameters"]["type"], "object")
            self.assertFalse(tool["parameters"]["additionalProperties"])
            self.assertEqual(set(tool["parameters"]["required"]), set(tool["parameters"]["properties"]))

    def test_future_only_writes_stay_absent(self):
        names = {tool["name"] for tool in TOOLS}
        for write_tool in ("cancel_order", "reschedule_delivery", "change_delivery_address", "create_rma_request"):
            self.assertNotIn(write_tool, names)

    def test_unknown_tool_is_rejected(self):
        _service, result = self.turn(PlannedToolCall("cancel_order", {"order_code": "ORD-1001"}))
        self.assertEqual(result.outcome, "TOOL_ERROR")
        self.assertEqual(result.tool_outcomes[0].error_code, "TOOL_NOT_ALLOWLISTED")

    def test_search_catalog_happy_path(self):
        oms = SeedOMS()
        service = SessionBoundary(
            dialogue=ScriptedDialogue(
                PlannedToolCall(
                    "search_catalog",
                    {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": "55"},
                )
            ),
            oms=oms,
        )
        snapshot = oms.business_snapshot()
        session = run(service.create_session())
        result = run(service.handle_turn(session.session_id, "Compare the Samsung and the LG 55-inch TV."))

        self.assertEqual(result.outcome, "ANSWERED")
        self.assertEqual(result.tool_outcomes[0].status, "SUCCESS")
        skus = {item["sku"] for item in result.tool_outcomes[0].record["matches"]}
        self.assertEqual(skus, {"TV-SAM-55", "TV-LG-55"})
        self.assertEqual(oms.business_snapshot(), snapshot)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_search_catalog_empty_result_is_not_found(self):
        _service, result = self.turn(
            PlannedToolCall(
                "search_catalog",
                {"category": "KITCHEN", "brand": None, "max_price_vnd": None, "keyword": "55 inch"},
            )
        )
        self.assertEqual(result.tool_outcomes[0].status, "NOT_FOUND")
        self.assertIn("could not find", result.assistant_text.lower())

    def test_compare_products_happy_path(self):
        _service, result = self.turn(PlannedToolCall("compare_products", {"skus": ["TV-SAM-55", "TV-LG-55"]}))
        self.assertEqual(result.outcome, "ANSWERED")
        self.assertEqual(result.tool_outcomes[0].status, "SUCCESS")
        record = result.tool_outcomes[0].record
        self.assertEqual(record["category"], "TV_AV")
        prices = {row["sku"]: row["price_vnd"] for row in record["comparison"]}
        self.assertEqual(prices, {"TV-SAM-55": 11_490_000, "TV-LG-55": 18_990_000})

    def test_compare_products_rejects_single_sku(self):
        _service, result = self.turn(PlannedToolCall("compare_products", {"skus": ["TV-SAM-55"]}))
        self.assertEqual(result.tool_outcomes[0].status, "INVALID_INPUT")
        self.assertIn("two or three", result.assistant_text.lower())

    def test_compare_products_rejects_four_skus(self):
        _service, result = self.turn(
            PlannedToolCall("compare_products", {"skus": ["TV-SAM-55", "TV-LG-55", "TV-SAM-55", "TV-LG-55"]})
        )
        self.assertEqual(result.tool_outcomes[0].status, "INVALID_INPUT")

    def test_compare_products_rejects_mixed_category(self):
        _service, result = self.turn(PlannedToolCall("compare_products", {"skus": ["TV-SAM-55", "WM-PAN-09"]}))
        self.assertEqual(result.tool_outcomes[0].status, "INVALID_INPUT")

    def test_compare_products_unknown_sku_is_not_found(self):
        _service, result = self.turn(PlannedToolCall("compare_products", {"skus": ["TV-SAM-55", "TV-XXXX-55"]}))
        self.assertEqual(result.tool_outcomes[0].status, "NOT_FOUND")
        self.assertIn("TV-XXXX-55", result.assistant_text)

    def test_apply_voucher_happy_path(self):
        oms = SeedOMS()
        service = SessionBoundary(
            dialogue=ScriptedDialogue(
                PlannedToolCall(
                    "apply_voucher",
                    {"voucher_code": "VOXCOM_500K", "order_value_vnd": 11_490_000, "category": "TV_AV"},
                )
            ),
            oms=oms,
        )
        snapshot = oms.business_snapshot()
        session = run(service.create_session())
        result = run(service.handle_turn(session.session_id, "Apply VOXCOM_500K to the Samsung."))

        self.assertEqual(result.outcome, "ANSWERED")
        record = result.tool_outcomes[0].record
        self.assertEqual(result.tool_outcomes[0].status, "SUCCESS")
        self.assertTrue(record["valid"])
        self.assertEqual(record["discount_vnd"], 500_000)
        self.assertEqual(record["capped_at_vnd"], 500_000)
        self.assertEqual(record["quoted_price_vnd"], 10_990_000)
        self.assertEqual(oms.business_snapshot(), snapshot)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_apply_voucher_wrong_category_and_low_value_is_invalid(self):
        _service, result = self.turn(
            PlannedToolCall(
                "apply_voucher",
                {"voucher_code": "VOXCOM_500K", "order_value_vnd": 8_990_000, "category": "WASHING_MACHINE"},
            )
        )
        self.assertEqual(result.tool_outcomes[0].status, "INVALID_INPUT")
        self.assertFalse(result.tool_outcomes[0].record["valid"])
        self.assertEqual(result.tool_outcomes[0].record["discount_vnd"], 0)

    def test_apply_voucher_unknown_code_is_invalid(self):
        _service, result = self.turn(
            PlannedToolCall(
                "apply_voucher",
                {"voucher_code": "VOXCOM_TET_500K", "order_value_vnd": 11_490_000, "category": "TV_AV"},
            )
        )
        self.assertEqual(result.tool_outcomes[0].status, "INVALID_INPUT")
        self.assertFalse(result.tool_outcomes[0].record["valid"])

    def test_apply_voucher_does_not_mutate_business_snapshot(self):
        oms = SeedOMS()
        before = oms.business_snapshot()
        outcome = oms.apply_voucher(
            session_customer_id="CUST-001",
            voucher_code="VOXCOM_500K",
            order_value_vnd=11_490_000,
            category="TV_AV",
        )
        self.assertEqual(outcome.status, "SUCCESS")
        self.assertEqual(oms.business_snapshot(), before)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)


if __name__ == "__main__":
    unittest.main()
