"""Seed fixtures (P1 build guide §17, design D§13 and D§4.2-4.3): one row set, two stores.

The same rows populate PostgreSQL (``apply_seed``) and the in-memory repository,
so both backends serve identical data. Values marked ``# †`` are fixture filler:
chosen only so the rows are complete and the foreign keys resolve.

Policy clauses are not here: they live in ``domain/data/policies.json`` (verbatim
from D§8.2, checked by ``scripts/sync_policy_corpus.py``) and are indexed in Chroma.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

from ..domain.clock import DEMO_TIMEZONE


def order_id_for(order_code: str) -> str:
    """32-char lowercase hex primary key, derived from the code so fixture rows are stable. †"""
    return hashlib.md5(order_code.encode("ascii"), usedforsecurity=False).hexdigest()


def _local(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=DEMO_TIMEZONE)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


CUST_001_ADDRESS = "123 Nguyen Trai, Ward 2, District 5, Ho Chi Minh City"

CUSTOMERS: list[dict[str, Any]] = [
    {
        "customer_id": "CUST-001",
        "full_name": "Lê Minh Anh",
        "phone_number": "+84900000001",  # † placeholder
        "email": None,
        "default_address": CUST_001_ADDRESS,
        "created_at": _utc(2026, 1, 5, 2),  # †
    },
    {
        "customer_id": "CUST-002",
        "full_name": "Phạm Thu Hà",
        "phone_number": "+84900000002",  # †
        "email": None,
        "default_address": "45 Le Loi, Ben Nghe Ward, District 1, Ho Chi Minh City",  # †
        "created_at": _utc(2026, 2, 9, 3),  # †
    },
]

PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": "PROD-TV-SAM-01",
        "product_name": "Samsung 55-inch 4K TV",
        "category": "TV_AV",
        "brand": "Samsung",
        "base_price": 11_490_000,
        "warranty_months": 24,  # †
        "specifications_json": {  # † recorded attributes
            "screen_size_in": 55,
            "resolution": "4K UHD",
            "panel_type": "Crystal UHD LED",
            "refresh_rate_hz": 60,
            "hdr": "HDR10+",
            "smart_platform": "Tizen",
        },
    },
    {
        "product_id": "PROD-TV-LG-01",  # †
        "product_name": "LG 55-inch 4K TV",
        "category": "TV_AV",
        "brand": "LG",
        "base_price": 18_990_000,
        "warranty_months": 24,  # †
        # † recorded attributes; refresh_rate_hz is deliberately missing ("That figure is not recorded for TV-LG-55.")
        "specifications_json": {
            "screen_size_in": 55,
            "resolution": "4K UHD",
            "panel_type": "OLED",
            "hdr": "Dolby Vision",
            "smart_platform": "webOS",
        },
    },
    {
        "product_id": "PROD-WM-PAN-01",  # †
        "product_name": "Panasonic Inverter 9 kg washer",
        "category": "WASHING_MACHINE",
        "brand": "Panasonic",
        "base_price": 8_990_000,  # † (equals the ORD-1003 total)
        "warranty_months": 24,  # †
        "specifications_json": {"capacity_kg": 9, "inverter_motor": True, "max_spin_rpm": 1200},  # †
    },
]

VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "VAR-SAM-55-BLK",
        "product_id": "PROD-TV-SAM-01",
        "sku": "TV-SAM-55",
        "color": "Black",  # †
        "storage_capacity": None,
        "screen_size": "55 inch",
        "price": 11_490_000,
        "stock_quantity": 24,  # † INTERNAL ONLY
    },
    {
        "variant_id": "VAR-LG-55-BLK",  # †
        "product_id": "PROD-TV-LG-01",
        "sku": "TV-LG-55",
        "color": "Black",  # †
        "storage_capacity": None,
        "screen_size": "55 inch",
        "price": 18_990_000,
        "stock_quantity": 7,  # † INTERNAL ONLY
    },
    {
        "variant_id": "VAR-PAN-09-WHT",  # †
        "product_id": "PROD-WM-PAN-01",
        "sku": "WM-PAN-09",
        "color": "White",  # †
        "storage_capacity": "9 kg",
        "screen_size": None,
        "price": 8_990_000,
        "stock_quantity": 15,  # † INTERNAL ONLY
    },
]

ORDERS: list[dict[str, Any]] = [
    {
        "order_id": order_id_for("ORD-1001"),
        "order_code": "ORD-1001",
        "customer_id": "CUST-001",
        "status": "confirmed",
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",
        "total_amount": 11_490_000,
        "shipping_address": CUST_001_ADDRESS,  # †
        "requested_delivery_time": None,
        "requested_delivery_window_end": None,
        "estimated_delivery_date": None,
        "version": 1,
        "created_at": _utc(2026, 10, 12, 3),  # †
        "updated_at": _utc(2026, 10, 12, 3),  # †
    },
    {
        "order_id": order_id_for("ORD-1002"),
        "order_code": "ORD-1002",
        "customer_id": "CUST-001",
        "status": "shipped",
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",  # †
        "total_amount": 18_990_000,
        "shipping_address": CUST_001_ADDRESS,
        "requested_delivery_time": _local(2026, 10, 16, 9),
        "requested_delivery_window_end": _local(2026, 10, 16, 12),
        "estimated_delivery_date": date(2026, 10, 15),
        "version": 3,
        "created_at": _utc(2026, 10, 10, 4),  # †
        "updated_at": _utc(2026, 10, 13, 1, 20),  # †
    },
    {
        "order_id": order_id_for("ORD-1003"),
        "order_code": "ORD-1003",
        "customer_id": "CUST-001",
        "status": "delivered",
        "payment_method": "CREDIT_CARD",
        "payment_status": "PAID",  # †
        "total_amount": 8_990_000,
        "shipping_address": CUST_001_ADDRESS,  # †
        "requested_delivery_time": None,
        "requested_delivery_window_end": None,
        "estimated_delivery_date": None,
        "version": 4,
        "created_at": _utc(2026, 9, 28, 5),  # †
        "updated_at": _utc(2026, 10, 2, 7, 30),  # †
    },
    {
        "order_id": order_id_for("ORD-2002"),
        "order_code": "ORD-2002",
        "customer_id": "CUST-002",
        "status": "processing",
        "payment_method": "COD",  # †
        "payment_status": "UNPAID",  # †
        "total_amount": 11_490_000,  # †
        "shipping_address": "45 Le Loi, Ben Nghe Ward, District 1, Ho Chi Minh City",  # †
        "requested_delivery_time": None,
        "requested_delivery_window_end": None,
        "estimated_delivery_date": None,
        "version": 1,  # †
        "created_at": _utc(2026, 10, 13, 6),  # †
        "updated_at": _utc(2026, 10, 13, 6),  # †
    },
]


def _item(item_id: str, order_code: str, variant_id: str, name: str, sku: str, price: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "order_id": order_id_for(order_code),
        "variant_id": variant_id,
        "product_name": name,
        "sku": sku,
        "quantity": 1,
        "unit_price": price,
        "total_price": price,
    }


ORDER_ITEMS: list[dict[str, Any]] = [
    _item("ITEM-1001-1", "ORD-1001", "VAR-SAM-55-BLK", "Samsung 55-inch 4K TV", "TV-SAM-55", 11_490_000),  # †
    _item("ITEM-1002-1", "ORD-1002", "VAR-LG-55-BLK", "LG 55-inch 4K TV", "TV-LG-55", 18_990_000),
    _item("ITEM-1003-1", "ORD-1003", "VAR-PAN-09-WHT", "Panasonic Inverter 9 kg washer", "WM-PAN-09", 8_990_000),
    _item("ITEM-2002-1", "ORD-2002", "VAR-SAM-55-BLK", "Samsung 55-inch 4K TV", "TV-SAM-55", 11_490_000),  # †
]

SHIPMENTS: list[dict[str, Any]] = [
    {
        "shipment_id": "SHIP-1002",
        "order_id": order_id_for("ORD-1002"),
        "tracking_code": "GHN-VN-882910",
        "carrier_name": "GHN",
        "status": "in_transit",
        "driver_name": "Trần Quốc Dũng",
        "driver_phone_masked": "+8490***45",
        "origin_hub": "VoxCom Warehouse D7, HCMC",  # †
        "destination_address": CUST_001_ADDRESS,
        "current_location": "Binh Tan Sorting Hub, HCMC",  # †
        "last_update_time": _utc(2026, 10, 13, 12, 5),
    },
    {
        # † ORD-1003 is delivered on 2026-10-02 (D§13); the DELIVERED event carries that date.
        "shipment_id": "SHIP-1003",
        "order_id": order_id_for("ORD-1003"),
        "tracking_code": "GHTK-VN-771204",
        "carrier_name": "GHTK",
        "status": "delivered",
        "driver_name": "Nguyễn Văn Bình",
        "driver_phone_masked": "+8491***07",
        "origin_hub": "VoxCom Warehouse D7, HCMC",
        "destination_address": CUST_001_ADDRESS,
        "current_location": "Delivered",
        "last_update_time": _utc(2026, 10, 2, 7, 30),
    },
]

SHIPMENT_EVENTS: list[dict[str, Any]] = [
    {
        "event_id": "EVT-1002-1",
        "shipment_id": "SHIP-1002",
        "event_time": _utc(2026, 10, 13, 1, 20),
        "event_type": "ORIGIN_PICKUP",
        "location": "VoxCom Warehouse D7, HCMC",
        "description": "Carrier collected the package from the fulfilment centre",
    },
    {
        "event_id": "EVT-1002-2",
        "shipment_id": "SHIP-1002",
        "event_time": _utc(2026, 10, 13, 12, 5),
        "event_type": "HUB_DEPARTURE",
        "location": "Binh Tan Sorting Hub, HCMC",
        "description": "Package forwarded to the local delivery station",
    },
    {
        "event_id": "EVT-1003-1",  # †
        "shipment_id": "SHIP-1003",
        "event_time": _utc(2026, 10, 1, 2),
        "event_type": "ORIGIN_PICKUP",
        "location": "VoxCom Warehouse D7, HCMC",
        "description": "Carrier collected the package from the fulfilment centre",
    },
    {
        "event_id": "EVT-1003-2",  # †
        "shipment_id": "SHIP-1003",
        "event_time": _utc(2026, 10, 2, 7, 30),
        "event_type": "DELIVERED",
        "location": "District 5, HCMC",
        "description": "Delivered to the customer",
    },
]

INSTALLATIONS: list[dict[str, Any]] = [
    {
        "installation_id": "INST-3001",  # †
        "order_id": order_id_for("ORD-1002"),
        "scheduled_date": date(2026, 10, 16),
        "time_slot": "13:00-16:00",
        "technician_name": None,
        "status": "Pending",
        "notes": "TV pedestal assembly",
    },
    {
        "installation_id": "INST-3002",
        "order_id": order_id_for("ORD-1003"),
        "scheduled_date": date(2026, 10, 15),
        "time_slot": "14:00-17:00",
        "technician_name": "Đỗ Minh Khoa",  # †
        "status": "Assigned",  # †
        "notes": "Washer service visit",  # †
    },
]

WARRANTY_RMAS: list[dict[str, Any]] = [
    {
        # Future-only continuation fixture (Scenario 8): no Phase 1 tool reads or creates it.
        "rma_id": "RMA-2026-8821",
        "order_item_id": "ITEM-1003-1",
        "customer_id": "CUST-001",
        "installation_id": "INST-3002",
        "diagnosis_class": "MOTOR_DRIVE_FAULT",
        "symptom_code": "SPIN_CYCLE_ERROR_E4",
        "rma_status": "requested",
        "requested_at": _utc(2026, 10, 14, 2, 50),  # †
        "inspection_date": None,
    },
]

PROMOTIONS: list[dict[str, Any]] = [
    {
        "promotion_id": "PROMO-2026-AUTUMN",  # †
        "voucher_code": "VOXCOM_500K",
        "campaign_name": "Autumn 2026",
        "category": "TV_AV",
        "min_order_value": 10_000_000,
        "discount_cap": 500_000,
        "valid_from": date(2026, 10, 1),
        "valid_to": date(2026, 10, 31),
        "is_fictional": True,
    },
    {
        # Retired (D§9.1): the row exists so a lookup finds it; domain/rules.py treats the code as not valid.
        "promotion_id": "PROMO-2026-TET",  # †
        "voucher_code": "VOXCOM_TET_500K",
        "campaign_name": "Tet 2026",  # †
        "category": "TV_AV",
        "min_order_value": 10_000_000,
        "discount_cap": 500_000,
        "valid_from": date(2026, 1, 15),  # †
        "valid_to": date(2026, 2, 28),  # †
        "is_fictional": True,
    },
]

# Fixture sessions (guide §17). SES-VOX-8839205 is bound to CUST-001 as filler (G-6).
FIXTURE_SESSIONS: dict[str, str | None] = {
    "SES-VOX-8839201": "CUST-001",
    "SES-VOX-8839202": "CUST-001",
    "SES-VOX-8839203": "CUST-001",
    "SES-VOX-8839205": "CUST-001",
}

# Write fixture: ACT-8821 cancels ORD-1001 at expected_version 1 with key K1 (guide §11.5).
K1 = "550e8400-e29b-41d4-a716-446655440007"
ACT_8821 = {"action_id": "ACT-8821", "action_type": "Cancel", "order_code": "ORD-1001", "expected_version": 1}

TABLES: tuple[tuple[str, list[dict[str, Any]]], ...] = (
    ("customer", CUSTOMERS),
    ("product", PRODUCTS),
    ("product_variant", VARIANTS),
    ("orders", ORDERS),
    ("order_item", ORDER_ITEMS),
    ("shipment", SHIPMENTS),
    ("shipment_event", SHIPMENT_EVENTS),
    ("installation", INSTALLATIONS),
    ("warranty_rma", WARRANTY_RMAS),
    ("promotion", PROMOTIONS),
)

__all__ = [
    "ACT_8821",
    "CUSTOMERS",
    "CUST_001_ADDRESS",
    "FIXTURE_SESSIONS",
    "INSTALLATIONS",
    "K1",
    "ORDERS",
    "ORDER_ITEMS",
    "PRODUCTS",
    "PROMOTIONS",
    "SHIPMENTS",
    "SHIPMENT_EVENTS",
    "TABLES",
    "VARIANTS",
    "WARRANTY_RMAS",
    "order_id_for",
]
