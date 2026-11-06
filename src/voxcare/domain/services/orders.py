"""OrderService, ShipmentService and the purchase-history read (guide §7.1, §7.2, G-2, G-3).

The ownership check runs inside each service, after loading by code and before
building any response: a row that belongs to someone else never leaves the
service (guide §7.1). A denial is flagged with ``_scope_violation`` so the
middleware appends ``AUTH_SCOPE_VIOLATION`` with the details concealed.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from ...storage.db import Repository, delivered_on
from ..clock import DEMO_TIMEZONE
from ..enums import ErrorCode
from ..rules import scope_allows
from .envelope import DEFAULT_TOOL_TIMEOUT_S, error, guarded, success

ORDER_CODE = re.compile(r"^ORD-\d{4}$")
_ORDER_CODE_VARIANT = re.compile(r"^ORD[\s_.-]*(\d{4})$")

ORDER_STATUS_LABELS = {  # implementation choice ("Shipped - In Transit" is the §7.1 example)
    "pending_confirmation": "Pending Confirmation",
    "confirmed": "Confirmed",
    "processing": "Processing",
    "ready_for_pickup": "Ready for Pickup",
    "shipped": "Shipped - In Transit",
    "out_for_delivery": "Out for Delivery",
    "delivered": "Delivered",
    "delivery_failed": "Delivery Failed",
    "return_requested": "Return Requested",
    "returned": "Returned",
    "cancelled": "Cancelled",
}
SHIPMENT_STATUS_LABELS = {  # implementation choice ("In Transit to Destination Hub" is the §7.2 example)
    "label_created": "Label Created",
    "in_transit": "In Transit to Destination Hub",
    "out_for_delivery": "Out for Delivery",
    "delivered": "Delivered",
    "delivery_failed": "Delivery Attempt Failed",
    "returned": "Returned to Sender",
}
SHIPMENT_FIELDS = (
    "shipment_id",
    "order_code",
    "tracking_code",
    "carrier_name",
    "shipment_status",
    "status_label",
    "estimated_delivery_date",
    "driver_name",
    "driver_phone_masked",
)


def normalize_order_code(raw: str) -> str:
    compact = raw.strip().upper()
    match = _ORDER_CODE_VARIANT.fullmatch(compact)
    return f"ORD-{match.group(1)}" if match else compact


def delivery_window(start: datetime | None, end: datetime | None) -> str | None:
    """G-1: ``YYYY-MM-DD HH:MM-HH:MM`` in Asia/Ho_Chi_Minh."""
    if start is None:
        return None
    local_start = start.astimezone(DEMO_TIMEZONE)
    text = local_start.strftime("%Y-%m-%d %H:%M")
    if end is not None:
        text += "-" + end.astimezone(DEMO_TIMEZONE).strftime("%H:%M")
    return text


def _iso(value: date | None) -> str | None:
    return value.isoformat() if isinstance(value, date) else value


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _denied(order: dict[str, Any], message: str) -> dict[str, Any]:
    envelope = error(ErrorCode.OUT_OF_SCOPE, message)
    envelope["_scope_violation"] = True
    envelope["_order_id"] = order["order_id"]
    envelope["_order_code"] = order["order_code"]
    return envelope


class OrderService:
    def __init__(self, repository: Repository, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        self.repository = repository
        self.timeout_s = timeout_s

    async def lookup_order(self, order_code: str, session_customer_id: str) -> dict[str, Any]:
        return await guarded(
            lambda: self._lookup(order_code, session_customer_id), timeout_s=self.timeout_s, system="order system"
        )

    async def _lookup(self, raw_code: str, session_customer_id: str) -> dict[str, Any]:
        code = normalize_order_code(raw_code)
        order = await self.repository.get_order_by_code(code) if ORDER_CODE.fullmatch(code) else None
        if order is None:
            return {**error(ErrorCode.NOT_FOUND, "No order matches that code."), "_order_code": code}
        if not scope_allows(order["customer_id"], session_customer_id):
            return _denied(order, "The order is not on the verified account.")
        items = await self.repository.get_order_items(order["order_id"])
        return {
            **success(
                order={
                    "order_code": order["order_code"],
                    "order_status": order["status"],
                    "status_label": ORDER_STATUS_LABELS.get(order["status"], order["status"]),
                    "version": order["version"],
                    "total_amount_vnd": int(order["total_amount"]),
                    "payment_method": order["payment_method"],
                    "payment_status": order["payment_status"],
                    "delivery_address": order["shipping_address"],
                    "requested_delivery_time": delivery_window(
                        order["requested_delivery_time"], order["requested_delivery_window_end"]
                    ),
                    "items": [
                        {
                            "item_id": item["item_id"],
                            "sku": item["sku"],
                            "product_name": item["product_name"],
                            "quantity": item["quantity"],
                            "unit_price_vnd": int(item["unit_price"]),
                        }
                        for item in items
                    ],
                }
            ),
            "_order_id": order["order_id"],
            "_order_code": order["order_code"],
        }


class ShipmentService:
    def __init__(self, repository: Repository, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        self.repository = repository
        self.timeout_s = timeout_s

    async def track_shipment(
        self, order_code: str | None, tracking_code: str | None, session_customer_id: str
    ) -> dict[str, Any]:
        return await guarded(
            lambda: self._track(order_code, tracking_code, session_customer_id),
            timeout_s=self.timeout_s,
            system="courier network",
        )

    async def _track(
        self, order_code: str | None, tracking_code: str | None, session_customer_id: str
    ) -> dict[str, Any]:
        shipment: dict[str, Any] | None
        if order_code:
            code = normalize_order_code(order_code)
            order = await self.repository.get_order_by_code(code) if ORDER_CODE.fullmatch(code) else None
            if order is None:
                return {**error(ErrorCode.NOT_FOUND, "No order matches that code."), "_order_code": code}
            if not scope_allows(order["customer_id"], session_customer_id):
                return _denied(order, "The shipment is not on the verified account.")
            shipment = await self.repository.get_shipment_by_order(order["order_id"])
        else:
            tracking = (tracking_code or "").strip().upper()
            shipment = await self.repository.get_shipment_by_tracking(tracking)
            if shipment is None:
                return {
                    **error(ErrorCode.NOT_FOUND, "No active shipment matches that tracking code."),
                    "_tracking_code": tracking,
                }
            order = await self.repository.get_order_by_id(shipment["order_id"])
            if order is None:
                return error(ErrorCode.SERVICE_ERROR, "The shipment row has no order.")
            if not scope_allows(order["customer_id"], session_customer_id):
                return _denied(order, "The shipment is not on the verified account.")
        eta = _iso(order["estimated_delivery_date"])
        if shipment is None:
            # G-2: an order with no manifest yet is not an error; every shipment field but the code is null.
            body = {field: None for field in SHIPMENT_FIELDS}
            body["order_code"] = order["order_code"]
            body["estimated_delivery_date"] = None
            return {
                **success(shipment=body, events=[]),
                "_order_id": order["order_id"],
                "_order_code": order["order_code"],
            }
        events = await self.repository.get_shipment_events(shipment["shipment_id"])
        status = shipment["status"]
        return {
            **success(
                shipment={
                    "shipment_id": shipment["shipment_id"],
                    "order_code": order["order_code"],
                    "tracking_code": shipment["tracking_code"],
                    "carrier_name": shipment["carrier_name"],
                    "shipment_status": status,
                    "status_label": SHIPMENT_STATUS_LABELS.get(status, status),
                    "estimated_delivery_date": None if status in {"delivered", "returned"} else eta,
                    "driver_name": shipment["driver_name"],
                    "driver_phone_masked": shipment["driver_phone_masked"],
                },
                events=[
                    {
                        "event_id": event["event_id"],
                        "event_time": _utc(event["event_time"]),
                        "event_type": event["event_type"],
                        "location": event["location"],
                        "description": event["description"],
                    }
                    for event in events
                ],
            ),
            "_order_id": order["order_id"],
            "_order_code": order["order_code"],
        }


class CustomerOrdersService:
    """``list_customer_orders`` (G-3 decision): the verified caller's own orders, never another customer's."""

    def __init__(self, repository: Repository, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        self.repository = repository
        self.timeout_s = timeout_s

    async def list_orders(self, session_customer_id: str) -> dict[str, Any]:
        return await guarded(lambda: self._list(session_customer_id), timeout_s=self.timeout_s, system="order system")

    async def _list(self, session_customer_id: str) -> dict[str, Any]:
        rows = await self.repository.list_orders_by_customer(session_customer_id)
        orders = []
        for order in rows:
            if not scope_allows(order["customer_id"], session_customer_id):
                continue  # defence in depth: the query is already scoped
            delivered = None
            if order["status"] in {"delivered", "return_requested", "returned"}:
                shipment = await self.repository.get_shipment_by_order(order["order_id"])
                if shipment is not None:
                    delivered = delivered_on(await self.repository.get_shipment_events(shipment["shipment_id"]))
            items = await self.repository.get_order_items(order["order_id"])
            orders.append(
                {
                    "order_code": order["order_code"],
                    "order_status": order["status"],
                    "status_label": ORDER_STATUS_LABELS.get(order["status"], order["status"]),
                    "total_amount_vnd": int(order["total_amount"]),
                    "delivered_on": _iso(delivered),
                    "item_names": [item["product_name"] for item in items],
                }
            )
        return success(orders=orders)


__all__ = [
    "ORDER_CODE",
    "ORDER_STATUS_LABELS",
    "SHIPMENT_STATUS_LABELS",
    "CustomerOrdersService",
    "OrderService",
    "ShipmentService",
    "delivery_window",
    "normalize_order_code",
]
