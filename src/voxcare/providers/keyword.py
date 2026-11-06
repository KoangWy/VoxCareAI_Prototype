"""Offline keyword planner: a deterministic stand-in for the conversation model.

Selected with ``VOXCARE_CONVERSATION_PROVIDER=keyword``.

Not an LLM and not part of the design's provider set: it exists so the demo and
the offline voice harness run with no model credentials or network. It maps an
utterance to the same tool calls a model is instructed to make (see
``SYSTEM_PROMPT``); the application still speaks only from tool results.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..contracts import PlannedToolCall
from ..domain.rules import brands_in, product_categories_in

_ORDER = re.compile(r"\bORD-(\d{4})\b", re.I)
_TRACKING = re.compile(r"\b(?:GHN|GHTK|VTP)-VN-\d{4,}\b", re.I)
_SKU = re.compile(r"\b(?:TV|WM|RF|AC|KT)-[A-Z]{2,4}-\d{2,3}\b", re.I)
_VOUCHER = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_PRICE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(million|m|triệu|tr)\b", re.I)
_HISTORY = re.compile(
    r"\b(earlier|previous|past|history|bought|buy\s+before|ordered\s+before|purchases?|my orders|other orders)\b", re.I
)
_DELIVERY = re.compile(r"\b(where|track|tracking|deliver\w*|arriv\w*|package|parcel|shipment|eta|status)\b", re.I)
_HUMAN = re.compile(r"\b(human|person|agent|representative|supervisor|someone)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference)\b", re.I)
_PRODUCT = re.compile(
    r"\b(tv|tvs|television|fridge|refrigerator|washer|washing machine|air ?con\w*|microwave|oven|"
    r"model|models|buy|price|cheap\w*|under)\b",
    re.I,
)
_POLICY_TOPICS = (
    ("RETURNS", re.compile(r"\b(return|refund|exchange|replace\w*|defect\w*)\b", re.I)),
    ("WARRANTY", re.compile(r"\b(warrant\w*|guarantee|cover\w*|compressor)\b", re.I)),
    ("INSTALLATION", re.compile(r"\b(install\w*|mount\w*|set ?up|drill\w*|bracket)\b", re.I)),
    ("SHIPPING", re.compile(r"\b(delivery fee|shipping|freight|surcharge|stairs?|cod|cash on delivery)\b", re.I)),
)
_POLICY_HINT = re.compile(
    r"\b(policy|policies|can i|do you|is there|how long|how much|free|fee|accept|trade-?in)\b", re.I
)
_BUYING = re.compile(
    r"\b(buy|price|prices|cost of|cheap\w*|under \d|budget|models?|show me|looking for|in stock)\b", re.I
)
_LABEL_SKUS = {
    ("Samsung", "TV_AV"): "TV-SAM-55",
    ("LG", "TV_AV"): "TV-LG-55",
    ("Panasonic", "WASHING_MACHINE"): "WM-PAN-09",
}
_SKU_PRICES = {"TV-SAM-55": 11_490_000, "TV-LG-55": 18_990_000, "WM-PAN-09": 8_990_000}


def _category(text: str) -> str | None:
    found = product_categories_in(text)
    return found[0] if found else None


class KeywordPlanner:
    label = "Offline keyword planner"
    model = "keyword-planner"

    async def plan_turn(
        self, *, user_text: str, history: Sequence[Mapping[str, str]], available_tools: Sequence[Mapping[str, object]]
    ) -> list[PlannedToolCall]:
        text = user_text
        calls: list[PlannedToolCall] = []
        code = _ORDER.search(text)
        if code:
            order_code = f"ORD-{code.group(1)}"
            calls.append(PlannedToolCall("lookup_order_details", {"order_code": order_code}, "k1"))
            if _DELIVERY.search(text) or len(text.split()) <= 4:
                calls.append(
                    PlannedToolCall("track_shipment_status", {"order_code": order_code, "tracking_code": None}, "k2")
                )
            if _HISTORY.search(text):
                calls.append(PlannedToolCall("list_customer_orders", {}, "k3"))
            return calls
        tracking = _TRACKING.search(text)
        if tracking:
            return [
                PlannedToolCall(
                    "track_shipment_status", {"order_code": None, "tracking_code": tracking.group(0).upper()}, "k1"
                )
            ]
        if _HISTORY.search(text):
            return [PlannedToolCall("list_customer_orders", {}, "k1")]
        if _HUMAN.search(text):
            return [
                PlannedToolCall(
                    "escalate_to_human", {"trigger_reason": "CUSTOMER_REQUESTED_ESCALATION", "priority": "NORMAL"}, "k1"
                )
            ]
        if re.search(r"\b(package|parcel|my order|delivery status)\b", text, re.I):
            return [PlannedToolCall("lookup_order_details", {"order_code": ""}, "k1")]
        category = _category(text)
        brands = brands_in(text)
        voucher = _VOUCHER.search(text)
        if _COMPARE.search(text) or voucher:
            category = category or "TV_AV"
            skus = [sku.upper() for sku in _SKU.findall(text)] or [
                _LABEL_SKUS[(brand, category)] for brand in brands if (brand, category) in _LABEL_SKUS
            ]
            if _COMPARE.search(text):
                keyword = "55 inch" if "55" in text else None
                calls.append(
                    PlannedToolCall(
                        "search_catalog",
                        {"category": category, "brand": None, "max_price_vnd": None, "keyword": keyword},
                        "k1",
                    )
                )
                if len(skus) >= 2:
                    calls.append(PlannedToolCall("compare_products", {"skus": skus[:3]}, "k2"))
            if voucher:
                value = _SKU_PRICES.get(skus[0], 0) if skus else 0
                price = _PRICE.search(text)
                if price:
                    value = int(float(price.group(1).replace(",", ".")) * 1_000_000)
                calls.append(
                    PlannedToolCall(
                        "apply_voucher",
                        {"voucher_code": voucher.group(0), "order_value_vnd": value, "category": category},
                        "k3",
                    )
                )
            return calls[:3]
        policy = next((topic for topic, pattern in _POLICY_TOPICS if pattern.search(text)), None)
        if policy or (_POLICY_HINT.search(text) and not _BUYING.search(text)):
            topics = [topic for topic, pattern in _POLICY_TOPICS if pattern.search(text)]
            return [
                PlannedToolCall(
                    "retrieve_policy_clauses",
                    {"query": text, "category": topics[0] if len(topics) == 1 else "ALL"},
                    "k1",
                )
            ]
        if category and _PRODUCT.search(text):
            price = _PRICE.search(text)
            max_price = int(float(price.group(1).replace(",", ".")) * 1_000_000) if price else None
            return [
                PlannedToolCall(
                    "search_catalog",
                    {
                        "category": category,
                        "brand": brands[0] if brands else None,
                        "max_price_vnd": max_price,
                        "keyword": None,
                    },
                    "k1",
                )
            ]
        return []


__all__ = ["KeywordPlanner"]
