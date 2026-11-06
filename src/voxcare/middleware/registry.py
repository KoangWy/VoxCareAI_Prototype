"""The Phase 1 model-callable tool registry (guide §1.1, §7; design D§4, D§7.2, D§9.2; G-3 decision).

Eight tools: the design's seven read/record tools plus the scope-guarded read
``list_customer_orders`` chosen for G-3 (a design change, recorded in
docs/SPEC_CONFORMANCE.md). The future-only guarded writes (``cancel_order``,
``reschedule_delivery``, ``change_delivery_address``, ``request_reversal``,
``create_rma_request``) are **absent**: the model can never propose them.

Schemas are sent with ``strict: true`` and ``additionalProperties: false``.
OpenAI strict mode needs every property in ``required``, so optional fields are
typed ``[T, "null"]`` and the validator treats ``null`` as absent (guide §7).
No schema carries ``session_customer_id``: the middleware injects it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PRODUCT_CATEGORIES = ("TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN")
POLICY_CATEGORIES = ("SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS", "ALL")
TRIGGER_REASONS = (
    "CUSTOMER_REQUESTED_ESCALATION",
    "CLARIFICATION_LIMIT_EXCEEDED",
    "LOW_CONFIDENCE_THRESHOLD",
    "COMPLEX_EXCEPTION",
)
PRIORITIES = ("NORMAL", "HIGH", "URGENT")


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "lookup_order_details",
        "description": (
            "Read status, items, payment and the requested delivery window of one order on the verified caller's "
            "account, by order code (ORD- plus four digits). Returns no shipment or carrier data."
        ),
        "parameters": _object({"order_code": {"type": "string", "minLength": 1}}),
    },
    {
        "name": "track_shipment_status",
        "description": (
            "Read shipment status, carrier, tracking events and the carrier ETA for an owned order, by order code "
            "or by the tracking code from the confirmation. Pass null for a code the caller did not give."
        ),
        "parameters": _object(
            {"order_code": {"type": ["string", "null"]}, "tracking_code": {"type": ["string", "null"]}}
        ),
    },
    {
        "name": "list_customer_orders",
        "description": (
            "List the verified caller's own orders (purchase history): code, status and delivery date. "
            "Use when the caller asks about earlier purchases or which orders they have."
        ),
        "parameters": _object({}),
    },
    {
        "name": "retrieve_policy_clauses",
        "description": (
            "Search the fictional VoxCom demo policy corpus (delivery, installation, warranty, returns) for clauses "
            "that ground an answer."
        ),
        "parameters": _object(
            {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language question regarding customer rights, fees, warranty terms, or return windows."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": list(POLICY_CATEGORIES),
                    "description": "Category filter; ALL means no category filter.",
                },
            }
        ),
    },
    {
        "name": "search_catalog",
        "description": "Query electronics catalog by category, brand, maximum budget, and model keywords",
        "parameters": _object(
            {
                "category": {"type": "string", "enum": list(PRODUCT_CATEGORIES)},
                "brand": {"type": ["string", "null"]},
                "max_price_vnd": {"type": ["integer", "null"]},
                "keyword": {"type": ["string", "null"]},
            }
        ),
    },
    {
        "name": "compare_products",
        "description": (
            "Compare two or three catalog models of the same category on recorded attributes and price. "
            "Never invent an unrecorded specification."
        ),
        "parameters": _object({"skus": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}}),
    },
    {
        "name": "apply_voucher",
        "description": (
            "Quote a fictional demo voucher against an order value and product category. A quote only: it never "
            "applies the discount to an order."
        ),
        "parameters": _object(
            {
                "voucher_code": {"type": "string"},
                "order_value_vnd": {"type": "integer"},
                "category": {"type": "string", "enum": list(PRODUCT_CATEGORIES)},
            }
        ),
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Record an official customer request for human agent escalation and package call context when automated "
            "troubleshooting is inconclusive or caller demands a representative."
        ),
        "parameters": _object(
            {
                "trigger_reason": {
                    "type": "string",
                    "enum": list(TRIGGER_REASONS),
                    "description": "Root cause triggering human escalation.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(PRIORITIES),
                    "description": (
                        "Priority recorded with the request. Phase 1 has no live queue and no agent pool, so this "
                        "value is carried into the ticket for Phase 2 routing."
                    ),
                },
            }
        ),
    },
)

PHASE1_REGISTRY = frozenset(tool["name"] for tool in TOOLS)
# Future-only guarded writes: never exposed to the LLM in Phase 1 (guide §1.1).
FUTURE_WRITE_TOOLS = frozenset(
    {"cancel_order", "reschedule_delivery", "change_delivery_address", "request_reversal", "create_rma_request"}
)
# Guide §6.2: the design names the first three; the rest are implementation choices (G-9).
REQUIRES_VERIFIED_CUSTOMER = frozenset(
    {"lookup_order_details", "track_shipment_status", "search_catalog", "escalate_to_human", "list_customer_orders"}
)
ORDER_TOOLS = frozenset({"lookup_order_details", "track_shipment_status", "list_customer_orders"})

__all__ = [
    "FUTURE_WRITE_TOOLS",
    "ORDER_TOOLS",
    "PHASE1_REGISTRY",
    "POLICY_CATEGORIES",
    "PRIORITIES",
    "PRODUCT_CATEGORIES",
    "REQUIRES_VERIFIED_CUSTOMER",
    "TOOLS",
    "TRIGGER_REASONS",
]
