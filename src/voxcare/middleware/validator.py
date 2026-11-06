"""Pre-invocation validator (guide §6.3, design D§4.4): JSON-schema checks plus the identifier rule.

Runs before any I/O. A rejected call is **not invoked**: no PostgreSQL read, no
carrier call. ``null`` counts as absent for the nullable optional fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..domain.enums import ErrorCode
from ..domain.services.envelope import error
from .registry import POLICY_CATEGORIES, PRIORITIES, PRODUCT_CATEGORIES, TOOLS, TRIGGER_REASONS

_SCHEMAS = {tool["name"]: tool["parameters"] for tool in TOOLS}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _reject(reason: str) -> dict[str, Any]:
    return {**error(ErrorCode.INVALID_INPUT, f"Invalid arguments: {reason}."), "_reason": reason}


def _shape(name: str, arguments: Mapping[str, Any]) -> str | None:
    """Generic JSON-schema checks: unknown keys, types, enums, array bounds."""
    schema = _SCHEMAS[name]
    properties: Mapping[str, Any] = schema["properties"]
    unknown = set(arguments) - set(properties)
    if unknown:
        return f"unknown argument {sorted(unknown)[0]}"
    for key, spec in properties.items():
        value = arguments.get(key)
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if value is None:
            continue  # required/null handling is tool-specific below
        if "string" in types and isinstance(value, str):
            if "enum" in spec and value not in spec["enum"]:
                return f"{key} outside enum"
            continue
        if "integer" in types and _is_int(value):
            continue
        if "array" in types and isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                return f"{key} items must be strings"
            continue
        return f"{key} has the wrong type"
    return None


def check(name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """``None`` when the call may run; otherwise an INVALID_INPUT 400 envelope (with an internal ``_reason``)."""
    if not isinstance(arguments, Mapping):
        return _reject("arguments must be an object")
    problem = _shape(name, arguments)
    if problem:
        return _reject(problem)
    if name == "lookup_order_details":
        if _blank(arguments.get("order_code")):
            return {**_reject("order_code missing"), "_missing_slot": "order_code"}
    elif name == "track_shipment_status":
        if _blank(arguments.get("order_code")) and _blank(arguments.get("tracking_code")):
            return {**_reject("order_code and tracking_code missing"), "_missing_slot": "order_code"}
    elif name == "retrieve_policy_clauses":
        if _blank(arguments.get("query")):
            return _reject("query empty")
        if arguments.get("category") not in POLICY_CATEGORIES:
            return _reject("category outside PolicyCategoryEnum")
    elif name == "search_catalog":
        if arguments.get("category") not in PRODUCT_CATEGORIES:
            return _reject("category outside ProductCategoryEnum")
        price = arguments.get("max_price_vnd")
        if price is not None and (not _is_int(price) or price < 0):
            return _reject("max_price_vnd must be a non-negative integer")
    elif name == "compare_products":
        skus = arguments.get("skus")
        if not isinstance(skus, list) or not 2 <= len(skus) <= 3 or any(_blank(sku) for sku in skus):
            return _reject("skus must hold 2-3 codes")
        if len({sku.strip().upper() for sku in skus}) != len(skus):
            return _reject("skus must be distinct")
    elif name == "apply_voucher":
        if _blank(arguments.get("voucher_code")):
            return {**_reject("voucher_code empty"), "valid": False, "discount_vnd": 0, "capped_at_vnd": 0}
        value = arguments.get("order_value_vnd")
        if not _is_int(value) or value < 0:
            return {
                **_reject("order_value_vnd must be a non-negative integer"),
                "valid": False,
                "discount_vnd": 0,
                "capped_at_vnd": 0,
            }
        if arguments.get("category") not in PRODUCT_CATEGORIES:
            return {
                **_reject("category outside ProductCategoryEnum"),
                "valid": False,
                "discount_vnd": 0,
                "capped_at_vnd": 0,
            }
    elif name == "escalate_to_human":
        if arguments.get("trigger_reason") not in TRIGGER_REASONS:
            return _reject("trigger_reason outside TriggerReason")
        if arguments.get("priority") not in PRIORITIES:
            return _reject("priority outside Priority")
    elif name == "list_customer_orders":
        pass
    else:
        return _reject("unknown tool")
    return None


__all__ = ["check"]
