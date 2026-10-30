"""Deterministic grounded answers: every spoken fact comes from a tool record.

The conversation model only *plans* tool calls; the assistant text the caller
hears is rendered here from the tool outcomes, so the model cannot invent an
order status, ETA, price, policy term, or discount.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import ToolOutcome
from .tools import CATALOG_TOOLS

HANDOFF_TEXT = "Your request for human support was recorded. This demo does not make a live transfer."
CLARIFY_TEXT = (
    "I can help with order status and delivery if you share the order code, like ORD-1002. "
    "I can also search and compare products, check a demo voucher, or explain VoxCom shipping, "
    "installation, warranty and return policies. I can't change an order. What would you like to check?"
)
CLARIFY_LIMIT_TEXT = (
    "I still couldn't tell what to look up. Please share an order code or a product question, "
    "or request human support. You can also start a new demo session."
)
_RETRY_TEXT = "I couldn't retrieve that information right now. Please try again."

_CATEGORY_LABELS = {
    "TV_AV": "TV and audio",
    "REFRIGERATOR": "refrigerator",
    "WASHING_MACHINE": "washing machine",
    "AIR_CONDITIONER": "air conditioner",
    "KITCHEN": "kitchen",
}
_VOUCHER_REASONS = {
    "UNKNOWN_CODE": "I couldn't find voucher {code} in this demo.",
    "CATEGORY_MISMATCH": "Voucher {code} does not apply to {category} products.",
    "BELOW_MINIMUM": "Voucher {code} needs a higher order value than {value} VND.",
    "EXPIRED": "Voucher {code} is not valid on the demo date.",
    "INVALID_CATEGORY": "I need a valid product category to check voucher {code}.",
    "INVALID_ORDER_VALUE": "I need a valid order value to check voucher {code}.",
}


def render_results(tools: Sequence[ToolOutcome]) -> tuple[str, str]:
    """Compose one answer from every executed tool result, in emission order.

    A recorded handoff wins outright; otherwise every fragment is joined so a
    multi-call turn answers each lookup. The outcome is ANSWERED when every
    call answered, else the first non-answered outcome in emission order.
    """
    if any(tool.status == "HANDOFF_REQUEST_RECORDED" for tool in tools):
        return HANDOFF_TEXT, "HANDOFF_REQUEST_RECORDED"
    rendered = [render_result(tool) for tool in tools]
    outcome = next((outcome for _, outcome in rendered if outcome != "ANSWERED"), "ANSWERED")
    return " ".join(text for text, _ in rendered), outcome


def render_result(tool: ToolOutcome) -> tuple[str, str]:
    if tool.status == "HANDOFF_REQUEST_RECORDED":
        return HANDOFF_TEXT, "HANDOFF_REQUEST_RECORDED"
    if tool.error_code == "TOOL_NOT_ALLOWLISTED":
        return (
            "I can't perform that action. This demo only supports order, delivery, catalog, voucher "
            "and policy lookups.",
            "TOOL_ERROR",
        )
    if tool.tool_name == "retrieve_policy_clauses":
        return _render_policy(tool)
    if tool.tool_name in CATALOG_TOOLS:
        return _render_catalog(tool)
    return _render_order(tool)


def _render_policy(tool: ToolOutcome) -> tuple[str, str]:
    record = tool.record or {}
    if tool.error_code == "INVALID_TOOL_ARGUMENTS":
        return "I couldn't safely process that policy question. Please rephrase it.", "TOOL_ERROR"
    if tool.status in {"SERVICE_ERROR", "DEPENDENCY_UNAVAILABLE", "SERVICE_TIMEOUT", "TOOL_ERROR"}:
        outcome = "DEPENDENCY_UNAVAILABLE" if tool.status == "DEPENDENCY_UNAVAILABLE" else "TOOL_ERROR"
        return (
            "Policy lookup is unavailable right now, so I can't verify the applicable terms. You can request support.",
            outcome,
        )
    if tool.status == "SUCCESS" and tool.evidence:
        clauses = " ".join(str(item.get("content_en", "")) for item in tool.evidence)
        ids = ", ".join(str(item.get("policy_id", "")) for item in tool.evidence)
        return f"According to fictional VoxCom demo policy ({ids}): {clauses}", "ANSWERED"
    if tool.status == "NEEDS_CONTEXT":
        missing = " and ".join(str(item).replace("_", " ") for item in record.get("missing_context", ()))
        return (
            "I need more information to identify the applicable fictional policy, "
            f"including the {missing or 'product and brand'}.",
            "INSUFFICIENT_EVIDENCE",
        )
    return (
        "I don't have applicable policy evidence for that question, so I can't confirm the term. "
        "You can request support.",
        "INSUFFICIENT_EVIDENCE",
    )


def _render_order(tool: ToolOutcome) -> tuple[str, str]:
    if tool.error_code == "INVALID_TOOL_ARGUMENTS":
        return "I couldn't safely process that lookup. Please try again with your order code.", "TOOL_ERROR"
    if tool.status == "MISSING_INPUT":
        return "I can look up order status and delivery. Please share the order code, like ORD-1002.", "NEEDS_ORDER_ID"
    if tool.status == "NOT_FOUND":
        return "I couldn't find an order with that code. Please check it and try again.", "NOT_FOUND"
    if tool.status == "OUT_OF_SCOPE":
        return "I can only share details for orders in this simulated customer session.", "OUT_OF_SCOPE"
    if tool.status != "SUCCESS":
        return _RETRY_TEXT, "TOOL_ERROR"
    record = tool.record or {}
    code = str(record.get("order_code", ""))
    if tool.tool_name == "lookup_order_details":
        status = str(record.get("status", "")).replace("_", " ")
        items = record.get("items") or []
        names = ", ".join(
            str(item.get("product_name", item.get("sku", ""))) for item in items if isinstance(item, dict)
        )
        if names:
            return f"Order {code} is {status}. Item(s): {names}.", "ANSWERED"
        return f"Order {code} is {status}.", "ANSWERED"
    shipment = record.get("shipment")
    if shipment is None:
        return (
            f"I found order {code}, but no shipment record or carrier delivery estimate is available yet.",
            "ANSWERED",
        )
    carrier = shipment.get("carrier")
    status = str(shipment.get("status") or "").replace("_", " ")
    eta = shipment.get("estimated_delivery_date")
    first = f"Order {code} is {status}" if status else f"Order {code} shipment status is unavailable"
    if carrier:
        first += f" with {carrier}"
    second = (
        f"The carrier currently estimates delivery on {eta}" if eta else "No carrier delivery estimate is available"
    )
    return f"{first}. {second}.", "ANSWERED"


def _product_phrase(row: dict) -> str:
    return f"{row.get('name') or row.get('sku')} ({row.get('sku')}) at {row.get('price_vnd'):,} VND"


def _render_catalog(tool: ToolOutcome) -> tuple[str, str]:
    record = tool.record or {}
    if tool.error_code == "INVALID_TOOL_ARGUMENTS":
        return "I couldn't safely process that catalog request. Please try again.", "TOOL_ERROR"
    if tool.status == "SUCCESS":
        if tool.tool_name == "search_catalog":
            matches = record.get("matches", [])
            listed = "; ".join(_product_phrase(item) for item in matches)
            noun = "model" if len(matches) == 1 else "models"
            return f"I found {len(matches)} {noun} in our catalog: {listed}.", "ANSWERED"
        if tool.tool_name == "compare_products":
            rows = record.get("comparison", [])
            category = _CATEGORY_LABELS.get(str(record.get("category")), str(record.get("category")))
            listed = "; ".join(_product_phrase(row) for row in rows)
            return f"Comparing {len(rows)} {category} models on recorded attributes: {listed}.", "ANSWERED"
        return (
            f"Voucher {record.get('voucher_code')} applies: {record.get('discount_vnd', 0):,} VND off, "
            f"for a quoted price of {record.get('quoted_price_vnd'):,} VND. "
            "This is a quote only; no order was changed.",
            "ANSWERED",
        )
    if tool.status == "NOT_FOUND":
        missing = record.get("missing_sku")
        if missing:
            return f"I cannot find {missing} in our catalog.", "NOT_FOUND"
        return "I could not find a matching model in our catalog.", "NOT_FOUND"
    if tool.status == "INVALID_INPUT":
        if tool.tool_name == "compare_products":
            return (
                "I can compare two or three models from the same category. Which models would you like to compare?",
                "INVALID_INPUT",
            )
        if tool.tool_name == "apply_voucher":
            template = _VOUCHER_REASONS.get(str(record.get("reason")), "Voucher {code} is not applicable in this demo.")
            category = _CATEGORY_LABELS.get(str(record.get("category")), "those")
            value = record.get("order_value_vnd")
            return (
                template.format(
                    code=record.get("voucher_code") or "that",
                    category=category,
                    value=f"{value:,}" if isinstance(value, int) else "that",
                ),
                "INVALID_INPUT",
            )
        return "That catalog request is not valid in this demo. Please name a product category.", "INVALID_INPUT"
    if tool.status == "STATE_CONFLICT":
        skus = ", ".join(record.get("out_of_stock_skus", ())) or "the requested model"
        return f"{skus} is out of stock today.", "STATE_CONFLICT"
    return _RETRY_TEXT, "TOOL_ERROR"


__all__ = ["CLARIFY_LIMIT_TEXT", "CLARIFY_TEXT", "HANDOFF_TEXT", "render_result", "render_results"]
