from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..contracts import ToolOutcome

# Demo anchor date (CANONICAL_DECISIONS D8): today = 2026-10-14
# (Asia/Ho_Chi_Minh). Relative phrasing ("tomorrow" = 2026-10-15) and the
# VOXCOM_500K validity window are evaluated against this fixed date so the
# read-only fixtures stay deterministic. Spec: 42 §13 + D8 + FIXTURE_ADDENDUM.
_ANCHOR_TODAY = "2026-10-14"

# ProductCategoryEnum members (spec 42 §3.5). Stored on catalog rows and
# accepted by search_catalog / apply_voucher; never includes the retrieval-only
# sentinel ALL.
_PRODUCT_CATEGORIES = ("TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN")

# Fixture SKUs transcribed from spec 42 §13 + D8 + §9.1. No other SKU may
# appear in seed data: PHONE-SAM-S24 / HEADPHONES-DEMO were prototype
# inventions with no spec provenance and are removed by this module.
_SPEC_SKUS = frozenset({"TV-SAM-55", "TV-LG-55", "WM-PAN-09"})


class SeedOMS:
    """Deterministic, read-only OMS fixtures from the accepted VoxCare design.

    Values follow CANONICAL_DECISIONS D8 and FIXTURE_ADDENDUM (one order_code =
    one lifecycle state; ORD-1001 confirmed CARD/PAID with no Shipment row;
    ORD-1002 shipped v3 + SHIP-1002 ETA 2026-10-15; ORD-1003 delivered CARD/PAID
    v4; ORD-2002 processing under CUST-002). The catalog trio
    (search/compare/voucher) reads the same fixture rows; nothing here mutates.
    """

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {
            "ORD-1001": {
                "order_code": "ORD-1001",
                "customer_id": "CUST-001",
                "status": "confirmed",
                "version": 1,
                "payment_method": "CREDIT_CARD",
                "payment_status": "PAID",
                "total_amount_vnd": 11_490_000,
                "requested_delivery_time": None,
                "items": [
                    {
                        "item_id": "ITEM-1001-1",
                        "sku": "TV-SAM-55",
                        "product_name": "Samsung 55-inch 4K TV",
                        "quantity": 1,
                        "unit_price_vnd": 11_490_000,
                    }
                ],
            },
            "ORD-1002": {
                "order_code": "ORD-1002",
                "customer_id": "CUST-001",
                "status": "shipped",
                "version": 3,
                "payment_method": "CREDIT_CARD",
                "payment_status": "PAID",
                "total_amount_vnd": 18_990_000,
                "requested_delivery_time": "2026-10-16 09:00-12:00",
                "items": [
                    {
                        "item_id": "ITEM-1002-1",
                        "sku": "TV-LG-55",
                        "product_name": "LG 55-inch 4K TV",
                        "quantity": 1,
                        "unit_price_vnd": 18_990_000,
                    }
                ],
            },
            "ORD-1003": {
                "order_code": "ORD-1003",
                "customer_id": "CUST-001",
                "status": "delivered",
                "version": 4,
                "payment_method": "CREDIT_CARD",
                "payment_status": "PAID",
                "total_amount_vnd": 8_990_000,
                "requested_delivery_time": None,
                "delivered_on": "2026-10-02",
                "items": [
                    {
                        "item_id": "ITEM-1003-1",
                        "sku": "WM-PAN-09",
                        "product_name": "Panasonic Inverter 9 kg washer",
                        "quantity": 1,
                        "unit_price_vnd": 8_990_000,
                    }
                ],
            },
            "ORD-2002": {
                "order_code": "ORD-2002",
                "customer_id": "CUST-002",
                "status": "processing",
                "version": 1,
                "payment_method": "CREDIT_CARD",
                "payment_status": "PAID",
                "total_amount_vnd": 11_490_000,
                "requested_delivery_time": None,
                "items": [
                    {
                        "item_id": "ITEM-2002-1",
                        "sku": "TV-SAM-55",
                        "product_name": "Samsung 55-inch 4K TV",
                        "quantity": 1,
                        "unit_price_vnd": 11_490_000,
                    }
                ],
            },
        }
        self.shipments: dict[str, dict[str, Any]] = {
            "ORD-1002": {
                "shipment_id": "SHIP-1002",
                "order_code": "ORD-1002",
                "tracking_code": "GHN-VN-882910",
                "carrier": "GHN",
                "status": "in_transit",
                "estimated_delivery_date": "2026-10-15",
                "driver_name": "Trần Quốc Dũng",
                "driver_phone_masked": "+8490***45",
                # Events transcribed from the §4.3 contract example (UTC). They
                # are the same two scans FIXTURE_ADDENDUM D8 lists in local
                # time (08:20 / 19:05 +07:00 on 2026-10-13).
                "events": [
                    {
                        "event_id": "EVT-1002-1",
                        "event_time": "2026-10-13T01:20:00Z",
                        "event_type": "ORIGIN_PICKUP",
                        "location": "VoxCom Warehouse D7, HCMC",
                        "description": "Carrier collected the package from the fulfilment centre",
                    },
                    {
                        "event_id": "EVT-1002-2",
                        "event_time": "2026-10-13T12:05:00Z",
                        "event_type": "HUB_DEPARTURE",
                        "location": "Binh Tan Sorting Hub, HCMC",
                        "description": "Package forwarded to the local delivery station",
                    },
                ],
            },
            # FIXTURE_ADDENDUM A3 rows: delivered shipments carry no ETA.
            "ORD-1003": {
                "shipment_id": "SHIP-1003",
                "order_code": "ORD-1003",
                "tracking_code": "GHTK-VN-771204",
                "carrier": "GHTK",
                "status": "delivered",
                "estimated_delivery_date": None,
                "events": [],
            },
            "ORD-2002": {
                "shipment_id": "SHIP-2002",
                "order_code": "ORD-2002",
                "tracking_code": "VTP-VN-330517",
                "carrier": "ViettelPost",
                "status": "delivered",
                "estimated_delivery_date": None,
                "events": [],
            },
        }
        # Small fixture catalog consistent with spec 42 §9.1/§9.2.2 and §13:
        # TV-SAM-55 at 11,490,000 VND and TV-LG-55 at 18,990,000 VND (both
        # TV_AV), WM-PAN-09 at 8,990,000 VND (WASHING_MACHINE). Stock is a
        # coarse availability flag only; quantities are never stored or
        # disclosed (spec §9.1).
        self.catalog: dict[str, dict[str, Any]] = {
            "TV-SAM-55": {
                "sku": "TV-SAM-55",
                "name": "Samsung 55-inch 4K TV",
                "brand": "Samsung",
                "category": "TV_AV",
                "price_vnd": 11_490_000,
                "in_stock": True,
            },
            "TV-LG-55": {
                "sku": "TV-LG-55",
                "name": "LG 55-inch 4K TV",
                "brand": "LG",
                "category": "TV_AV",
                "price_vnd": 18_990_000,
                "in_stock": True,
            },
            "WM-PAN-09": {
                "sku": "WM-PAN-09",
                "name": "Panasonic Inverter 9 kg washer",
                "brand": "Panasonic",
                "category": "WASHING_MACHINE",
                "price_vnd": 8_990_000,
                "in_stock": True,
            },
        }
        # Fictional demo promotion (spec 42 §9.1 + D8). Quote-only evaluation;
        # never mutates an order or promotion.
        self.promotions: dict[str, dict[str, Any]] = {
            "VOXCOM_500K": {
                "voucher_code": "VOXCOM_500K",
                "campaign": "Autumn 2026",
                "category": "TV_AV",
                "min_order_value_vnd": 10_000_000,
                "discount_cap_vnd": 500_000,
                "valid_from": "2026-10-01",
                "valid_to": "2026-10-31",
                "is_fictional": True,
            }
        }
        self.initial_snapshot = self.business_snapshot()

    def business_snapshot(self) -> tuple[tuple[str, str], ...]:
        """A compact immutable projection used to verify business records stay read-only."""
        rows = []
        for code, order in sorted(self.orders.items()):
            rows.append((f"order:{code}", repr(order)))
        for code, shipment in sorted(self.shipments.items()):
            rows.append((f"shipment:{code}", repr(shipment)))
        for sku, product in sorted(self.catalog.items()):
            rows.append((f"catalog:{sku}", repr(product)))
        for code, promotion in sorted(self.promotions.items()):
            rows.append((f"promotion:{code}", repr(promotion)))
        return tuple(rows)

    def lookup_order_details(self, *, session_customer_id: str, order_code: str) -> ToolOutcome:
        if not order_code:
            return ToolOutcome("lookup_order_details", "MISSING_INPUT")
        order = self.orders.get(order_code)
        if order is None:
            return ToolOutcome("lookup_order_details", "NOT_FOUND")
        if order["customer_id"] != session_customer_id:
            return ToolOutcome("lookup_order_details", "OUT_OF_SCOPE")
        public_order = {key: value for key, value in order.items() if key != "customer_id"}
        return ToolOutcome("lookup_order_details", "SUCCESS", record=deepcopy(public_order))

    def track_shipment_status(self, *, session_customer_id: str, order_code: str) -> ToolOutcome:
        if not order_code:
            return ToolOutcome("track_shipment_status", "MISSING_INPUT")
        order = self.orders.get(order_code)
        if order is None:
            return ToolOutcome("track_shipment_status", "NOT_FOUND")
        if order["customer_id"] != session_customer_id:
            return ToolOutcome("track_shipment_status", "OUT_OF_SCOPE")
        shipment = self.shipments.get(order_code)
        return ToolOutcome(
            "track_shipment_status",
            "SUCCESS",
            record={
                "order_code": order_code,
                "shipment": deepcopy(shipment) if shipment else None,
                "events": deepcopy(shipment["events"]) if shipment else [],
            },
        )

    def search_catalog(
        self,
        *,
        session_customer_id: str,
        category: str,
        brand: str | None = None,
        max_price_vnd: int | None = None,
        keyword: str | None = None,
    ) -> ToolOutcome:
        """Read-only catalog search (spec 42 §9.2.1). Empty results are NOT_FOUND, never invented."""
        # Public catalog: no customer-owned resource; session must still be active (gated by caller).
        _ = session_customer_id
        if category not in _PRODUCT_CATEGORIES:
            return ToolOutcome("search_catalog", "INVALID_INPUT", error_code="INVALID_CATEGORY")
        if max_price_vnd is not None and (
            not isinstance(max_price_vnd, int) or isinstance(max_price_vnd, bool) or max_price_vnd < 0
        ):
            return ToolOutcome("search_catalog", "INVALID_INPUT", error_code="INVALID_MAX_PRICE")
        lowered_keyword = keyword.lower() if isinstance(keyword, str) and keyword else None
        matches = []
        for sku in sorted(self.catalog):
            product = self.catalog[sku]
            if product["category"] != category:
                continue
            if brand is not None and product["brand"].casefold() != brand.casefold():
                continue
            if max_price_vnd is not None and product["price_vnd"] > max_price_vnd:
                continue
            if lowered_keyword is not None and (
                lowered_keyword not in sku.lower() and lowered_keyword not in product["name"].lower()
            ):
                continue
            matches.append(
                {
                    "sku": sku,
                    "name": product["name"],
                    "brand": product["brand"],
                    "price_vnd": product["price_vnd"],
                    "availability": "in_stock" if product["in_stock"] else "out_of_stock",
                }
            )
        record: dict[str, Any] = {
            "category": category,
            "brand": brand,
            "max_price_vnd": max_price_vnd,
            "keyword": keyword,
            "match_count": len(matches),
            "matches": matches,
        }
        if not matches:
            return ToolOutcome("search_catalog", "NOT_FOUND", record=record, error_code="NO_MATCHING_MODEL")
        if all(match["availability"] == "out_of_stock" for match in matches):
            record["out_of_stock_skus"] = [match["sku"] for match in matches]
            return ToolOutcome("search_catalog", "STATE_CONFLICT", record=record, error_code="OUT_OF_STOCK")
        return ToolOutcome("search_catalog", "SUCCESS", record=deepcopy(record))

    def compare_products(self, *, session_customer_id: str, skus: list[str]) -> ToolOutcome:
        """Read-only spec comparison (spec 42 §4.6): 2-3 SKUs, same category, recorded attributes only."""
        _ = session_customer_id  # public catalog: scope gated by caller session, as in search_catalog.
        normalized = [sku.strip().upper() for sku in skus]
        if not 2 <= len(normalized) <= 3:
            return ToolOutcome(
                "compare_products",
                "INVALID_INPUT",
                record={"skus": normalized},
                error_code="ARITY_OUT_OF_RANGE",
            )
        for sku in normalized:
            if sku not in self.catalog:
                return ToolOutcome(
                    "compare_products",
                    "NOT_FOUND",
                    record={"skus": normalized, "missing_sku": sku},
                    error_code="UNKNOWN_SKU",
                )
        categories = {self.catalog[sku]["category"] for sku in normalized}
        if len(categories) > 1:
            return ToolOutcome(
                "compare_products",
                "INVALID_INPUT",
                record={"skus": normalized, "categories": sorted(categories)},
                error_code="MIXED_CATEGORY",
            )
        comparison = [
            {
                "sku": sku,
                "name": self.catalog[sku]["name"],
                "brand": self.catalog[sku]["brand"],
                "price_vnd": self.catalog[sku]["price_vnd"],
                "availability": "in_stock" if self.catalog[sku]["in_stock"] else "out_of_stock",
            }
            for sku in normalized
        ]
        return ToolOutcome(
            "compare_products",
            "SUCCESS",
            record={"skus": normalized, "category": next(iter(categories)), "comparison": comparison},
        )

    def apply_voucher(
        self,
        *,
        session_customer_id: str,
        voucher_code: str,
        order_value_vnd: int,
        category: str,
    ) -> ToolOutcome:
        """Read-only voucher evaluation (spec 42 §4.6). Returns a quote; never mutates an order."""
        _ = session_customer_id  # public promotion read: scope gated by caller session, as in search_catalog.
        code = voucher_code.strip().upper()
        promotion = self.promotions.get(code)
        if promotion is None:
            return ToolOutcome(
                "apply_voucher",
                "INVALID_INPUT",
                record={
                    "voucher_code": code,
                    "valid": False,
                    "discount_vnd": 0,
                    "capped_at_vnd": 0,
                    "reason": "UNKNOWN_CODE",
                },
                error_code="UNKNOWN_VOUCHER",
            )
        if category not in _PRODUCT_CATEGORIES:
            return ToolOutcome(
                "apply_voucher",
                "INVALID_INPUT",
                record={
                    "voucher_code": code,
                    "valid": False,
                    "discount_vnd": 0,
                    "capped_at_vnd": promotion["discount_cap_vnd"],
                    "reason": "INVALID_CATEGORY",
                },
                error_code="INVALID_CATEGORY",
            )
        if not isinstance(order_value_vnd, int) or isinstance(order_value_vnd, bool) or order_value_vnd < 0:
            return ToolOutcome(
                "apply_voucher",
                "INVALID_INPUT",
                record={
                    "voucher_code": code,
                    "valid": False,
                    "discount_vnd": 0,
                    "capped_at_vnd": promotion["discount_cap_vnd"],
                    "reason": "INVALID_ORDER_VALUE",
                },
                error_code="INVALID_ORDER_VALUE",
            )
        if category != promotion["category"]:
            reason = "CATEGORY_MISMATCH"
        elif order_value_vnd < promotion["min_order_value_vnd"]:
            reason = "BELOW_MINIMUM"
        elif not promotion["valid_from"] <= _ANCHOR_TODAY <= promotion["valid_to"]:
            reason = "EXPIRED"
        else:
            reason = None
        if reason is not None:
            return ToolOutcome(
                "apply_voucher",
                "INVALID_INPUT",
                record={
                    "voucher_code": code,
                    "category": category,
                    "order_value_vnd": order_value_vnd,
                    "valid": False,
                    "discount_vnd": 0,
                    "capped_at_vnd": promotion["discount_cap_vnd"],
                    "reason": reason,
                },
                error_code="VOUCHER_NOT_APPLICABLE",
            )
        discount = min(promotion["discount_cap_vnd"], order_value_vnd)
        return ToolOutcome(
            "apply_voucher",
            "SUCCESS",
            record={
                "voucher_code": code,
                "category": category,
                "order_value_vnd": order_value_vnd,
                "valid": True,
                "discount_vnd": discount,
                "capped_at_vnd": promotion["discount_cap_vnd"],
                "quoted_price_vnd": order_value_vnd - discount,
            },
        )
