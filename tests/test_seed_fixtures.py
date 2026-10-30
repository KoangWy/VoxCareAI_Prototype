import unittest

from voxcare.domain.seed import SeedOMS

# Fixture table transcribed from spec 42 §13 (canonical fixtures) +
# CANONICAL_DECISIONS D8 + FIXTURE_ADDENDUM. Spec section cited per row
# family in the assertion messages. One order_code = one lifecycle state.
EXPECTED_ORDERS = {
    "ORD-1001": {
        "customer_id": "CUST-001",
        "status": "confirmed",
        "version": 1,
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",
        "total_amount_vnd": 11_490_000,
        "requested_delivery_time": None,
        "item_sku": "TV-SAM-55",
    },
    "ORD-1002": {
        "customer_id": "CUST-001",
        "status": "shipped",
        "version": 3,
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",
        "total_amount_vnd": 18_990_000,
        "requested_delivery_time": "2026-10-16 09:00-12:00",
        "item_sku": "TV-LG-55",
    },
    "ORD-1003": {
        "customer_id": "CUST-001",
        "status": "delivered",
        "version": 4,
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",
        "total_amount_vnd": 8_990_000,
        "delivered_on": "2026-10-02",
        "item_sku": "WM-PAN-09",
    },
    "ORD-2002": {
        "customer_id": "CUST-002",
        "status": "processing",
        "version": 1,
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",
        "item_sku": "TV-SAM-55",
    },
}

EXPECTED_SHIPMENTS = {
    "ORD-1002": {
        "shipment_id": "SHIP-1002",
        "tracking_code": "GHN-VN-882910",
        "carrier": "GHN",
        "status": "in_transit",
        "estimated_delivery_date": "2026-10-15",
    },
}


class SeedFixtureTests(unittest.TestCase):
    def setUp(self):
        self.oms = SeedOMS()

    def test_orders_match_spec_fixture_table(self):
        self.assertEqual(set(self.oms.orders), set(EXPECTED_ORDERS))
        for code, expected in EXPECTED_ORDERS.items():
            with self.subTest(order=code):
                order = self.oms.orders[code]
                for field, value in expected.items():
                    if field == "item_sku":
                        self.assertEqual(order["items"][0]["sku"], value)
                    else:
                        self.assertEqual(order[field], value, f"spec 42 §13/D8: {code}.{field}")

    def test_ord1001_has_no_shipment_row(self):
        self.assertNotIn("ORD-1001", self.oms.shipments)

    def test_shipments_match_spec_fixture_table(self):
        for code, expected in EXPECTED_SHIPMENTS.items():
            with self.subTest(order=code):
                shipment = self.oms.shipments[code]
                for field, value in expected.items():
                    self.assertEqual(shipment[field], value, f"spec 42 §13/D8: {code}.{field}")

    def test_no_invented_skus_survive(self):
        skus = set()
        for order in self.oms.orders.values():
            skus.update(item["sku"] for item in order["items"])
        skus.update(self.oms.catalog)
        self.assertLessEqual(skus, {"TV-SAM-55", "TV-LG-55", "WM-PAN-09"})

    def test_spec_item_ids_are_present(self):
        items = {item["item_id"]: item["sku"] for order in self.oms.orders.values() for item in order["items"]}
        self.assertEqual(
            items,
            {
                "ITEM-1001-1": "TV-SAM-55",
                "ITEM-1002-1": "TV-LG-55",
                "ITEM-1003-1": "WM-PAN-09",
                "ITEM-2002-1": "TV-SAM-55",
            },
        )


if __name__ == "__main__":
    unittest.main()
