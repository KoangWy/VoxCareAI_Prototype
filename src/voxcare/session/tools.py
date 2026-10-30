"""The model-visible read-only tool registry and its argument contracts.

Seven tools are exposed to the conversation model. Every one is read-only:
order/shipment reads are scoped to the session customer server-side, the
catalog trio and the policy retriever read fictional fixtures, and
``escalate_to_human`` only *records* a support request. There are no write
tools; an unknown tool name is rejected before dispatch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

MAX_HANDOFF_REASON_CHARS = 200
PRODUCT_CATEGORIES = ("TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN")
POLICY_CATEGORIES = ("SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS", "ALL")

ORDER_CODE = re.compile(r"^ORD-[0-9]{4}$")
# Spoken or typed variants of an order code: "ord 1002", "ORD1002", "ord_1002".
_ORDER_CODE_VARIANT = re.compile(r"^ORD[\s_.-]*([0-9]{4})$")

ORDER_TOOLS = frozenset({"lookup_order_details", "track_shipment_status"})
CATALOG_TOOLS = frozenset({"search_catalog", "compare_products", "apply_voucher"})

TOOLS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "lookup_order_details",
        "description": "Read the caller's order status and items using its VoxCom order code (format ORD-1234).",
        "parameters": {
            "type": "object",
            "properties": {"order_code": {"type": "string"}},
            "required": ["order_code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "track_shipment_status",
        "description": "Read shipment, carrier status, tracking events, and carrier ETA for an order code (ORD-1234).",
        "parameters": {
            "type": "object",
            "properties": {"order_code": {"type": "string"}},
            "required": ["order_code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "retrieve_policy_clauses",
        "description": (
            "Read applicable fictional VoxCom customer-support policy clauses "
            "for shipping, installation, returns, or warranty. Never infer order facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "category": {"type": "string", "enum": list(POLICY_CATEGORIES)},
                "product_category": {
                    "type": ["string", "null"],
                    "enum": [*PRODUCT_CATEGORIES, None],
                },
                "brand": {"type": ["string", "null"]},
            },
            "required": ["query", "category", "product_category", "brand"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record the caller's request for human support. This does not transfer or contact an agent.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": MAX_HANDOFF_REASON_CHARS}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_catalog",
        "description": (
            "Read-only electronics catalog search by category, brand, max budget, and keyword. "
            "Empty results are reported, never invented."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(PRODUCT_CATEGORIES)},
                "brand": {"type": ["string", "null"]},
                "max_price_vnd": {"type": ["integer", "null"]},
                "keyword": {"type": ["string", "null"]},
            },
            "required": ["category", "brand", "max_price_vnd", "keyword"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_products",
        "description": (
            "Read-only comparison of two or three catalog models of the same category "
            "on recorded attributes only. Never invent an unrecorded specification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skus": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
            },
            "required": ["skus"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_voucher",
        "description": (
            "Read-only evaluation of a fictional demo voucher against an order value and category. "
            "Returns the discount quote without changing any order or promotion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "voucher_code": {"type": "string", "minLength": 1},
                "order_value_vnd": {"type": "integer"},
                "category": {"type": "string", "enum": list(PRODUCT_CATEGORIES)},
            },
            "required": ["voucher_code", "order_value_vnd", "category"],
            "additionalProperties": False,
        },
    },
)

TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)


class InvalidToolArguments(ValueError):
    """The model sent arguments outside the tool's contract."""


def normalize_order_code(raw: str) -> str:
    """Canonical ``ORD-####`` form for typed or spoken variants; unknown shapes pass through upper-cased."""
    compact = raw.strip().upper()
    match = _ORDER_CODE_VARIANT.fullmatch(compact)
    return f"ORD-{match.group(1)}" if match else compact


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_str(arguments: Mapping[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is not None and not isinstance(value, str):
        raise InvalidToolArguments(key)
    return value


def validate_arguments(name: str, arguments: object) -> dict[str, Any]:
    """Check one call against its tool contract; raises :exc:`InvalidToolArguments`.

    Optional (nullable) fields may be omitted. Unknown keys are rejected.
    """
    if not isinstance(arguments, Mapping):
        raise InvalidToolArguments("arguments")
    keys = set(arguments)
    if name in ORDER_TOOLS:
        if keys != {"order_code"} or not isinstance(arguments["order_code"], str):
            raise InvalidToolArguments("order_code")
        return {"order_code": arguments["order_code"]}
    if name == "retrieve_policy_clauses":
        if keys - {"query", "category", "product_category", "brand"} or not {"query", "category"} <= keys:
            raise InvalidToolArguments("keys")
        query, category = arguments["query"], arguments["category"]
        if not isinstance(query, str) or not query.strip() or not isinstance(category, str):
            raise InvalidToolArguments("query")
        filters = {"category": category}
        for key in ("product_category", "brand"):
            if key in arguments:
                filters[key] = _optional_str(arguments, key)
        return {"query": query, **filters}
    if name == "escalate_to_human":
        reason = arguments.get("reason")
        if keys != {"reason"} or not isinstance(reason, str) or not reason.strip():
            raise InvalidToolArguments("reason")
        if len(reason) > MAX_HANDOFF_REASON_CHARS:
            raise InvalidToolArguments("reason")
        return {"reason": reason}
    if name == "search_catalog":
        if keys - {"category", "brand", "max_price_vnd", "keyword"} or not isinstance(arguments.get("category"), str):
            raise InvalidToolArguments("category")
        max_price = arguments.get("max_price_vnd")
        if max_price is not None and not _is_int(max_price):
            raise InvalidToolArguments("max_price_vnd")
        return {
            "category": arguments["category"],
            "brand": _optional_str(arguments, "brand"),
            "max_price_vnd": max_price,
            "keyword": _optional_str(arguments, "keyword"),
        }
    if name == "compare_products":
        skus = arguments.get("skus")
        if keys != {"skus"} or not isinstance(skus, list) or not all(isinstance(item, str) for item in skus):
            raise InvalidToolArguments("skus")
        return {"skus": list(skus)}
    if name == "apply_voucher":
        if keys != {"voucher_code", "order_value_vnd", "category"}:
            raise InvalidToolArguments("keys")
        code, value, category = arguments["voucher_code"], arguments["order_value_vnd"], arguments["category"]
        if not isinstance(code, str) or not code.strip() or not _is_int(value) or not isinstance(category, str):
            raise InvalidToolArguments("voucher")
        return {"voucher_code": code, "order_value_vnd": value, "category": category}
    raise InvalidToolArguments("tool")


def safe_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Audit-safe projection of tool arguments: identifiers kept, free text reduced to a length."""
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "query" and isinstance(value, str):
            safe["query_char_count"] = min(len(value), 500)
        elif key == "keyword" and isinstance(value, str):
            safe["keyword_char_count"] = min(len(value), 100)
        elif key in {"order_code", "category", "product_category", "brand", "voucher_code"} and isinstance(value, str):
            safe[key] = value[:100]
        elif key == "skus" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            safe["skus"] = [item[:30] for item in value[:3]]
        elif key in {"order_value_vnd", "max_price_vnd"} and (value is None or _is_int(value)):
            safe[key] = value
    return safe


def audit_outcome(status: str) -> str:
    """Map a tool status onto the audit ``outcome_status`` vocabulary."""
    return {
        "SUCCESS": "SUCCESS",
        "OUT_OF_SCOPE": "DENIED",
        "INSUFFICIENT_EVIDENCE": "SUCCESS",
        "NEEDS_CONTEXT": "SUCCESS",
        "HANDOFF_REQUEST_RECORDED": "SUCCESS",
    }.get(status, "FAILED")


__all__ = [
    "CATALOG_TOOLS",
    "MAX_HANDOFF_REASON_CHARS",
    "ORDER_CODE",
    "ORDER_TOOLS",
    "POLICY_CATEGORIES",
    "PRODUCT_CATEGORIES",
    "TOOLS",
    "TOOL_NAMES",
    "InvalidToolArguments",
    "audit_outcome",
    "normalize_order_code",
    "safe_arguments",
    "validate_arguments",
]
