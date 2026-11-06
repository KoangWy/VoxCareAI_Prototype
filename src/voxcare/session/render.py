"""Grounded answers: one spoken line per tool result, taken verbatim from :mod:`voxcare.domain.copy`.

The model only *plans* tool calls; every sentence the caller hears is rendered
here from a tool envelope (guide §8.2 rule 7: "When a tool result has a spoken
line, use it verbatim"). Two variants come out of one render:

- ``text``: the transcript/screen version (dates as the tools return them, clause text in full);
- ``voice``: what TTS says (:func:`speech.speakable` rewrites, policy shortened to one
  sentence per cited clause plus a pointer to the screen).

Output guards (guide §8.2) run on both: the ETA guard and the write guard.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..domain import copy
from ..domain.rules import mentions_date
from ..domain.services.catalog import CATEGORY_LABELS
from ..middleware.tool_middleware import ToolResult
from .speech import first_sentence, speakable

_CONTRAST_WORDS = re.compile(r"\b(?:earlier|other|also|else|previous|rest)\b", re.I)
_HISTORY_WORDS = re.compile(r"\b(?:earlier|previous|past|history|before|bought|purchases?|ordered\s+before)\b", re.I)
_ATTRIBUTE_WORDS = {
    "refresh_rate_hz": re.compile(r"\brefresh(?:\s+rate)?|\bhz\b|hertz", re.I),
    "hdr": re.compile(r"\bhdr\b|dolby\s+vision", re.I),
    "panel_type": re.compile(r"\bpanel\b|oled|qled|\bled\b", re.I),
    "resolution": re.compile(r"\bresolution\b|\b4k\b|\b8k\b", re.I),
    "smart_platform": re.compile(r"\bsmart\b|operating\s+system|\bos\b|tizen|webos", re.I),
    "capacity_kg": re.compile(r"\bcapacity\b|\bkg\b|kilo", re.I),
    "max_spin_rpm": re.compile(r"\bspin\s+speed\b|\brpm\b", re.I),
}
_QUESTION_LEAD = re.compile(
    r"^\s*(?:hi|hello|hey|so|okay|ok|um|uh|please|and|also|well)?[\s,]*"
    r"(?:do|does|can|could|will|would|is|are|what(?:'s| is)?|how|tell\s+me(?:\s+about)?|"
    r"i(?:\s+want|'d\s+like|\s+would\s+like)\s+to\s+know(?:\s+about)?)\s+"
    # every optional word ends at a word boundary: "is installation" must not lose the "i" of "installation"
    r"(?:(?:you|voxcom|there|i|we|your)\b\s*)?(?:(?:have|offer|accept|support|take|do|get|allow|a|an|any|the)\b\s*)?",
    re.I,
)
_TOPIC_TAIL = re.compile(r"[?.!\s]+$")


def inquiry_topic(text: str) -> str:
    """The ``[inquiry topic]`` slot of the out-of-domain admission line: the question without its lead words."""
    topic = _QUESTION_LEAD.sub("", text.strip(), count=1)
    topic = _TOPIC_TAIL.sub("", topic).strip()
    topic = re.sub(
        r"\b(?:my|me|i|i'm|we|our)\b", lambda m: {"my": "your", "our": "your"}.get(m[0].lower(), ""), topic, flags=re.I
    )
    topic = re.sub(r"\s{2,}", " ", topic).strip(" ,")
    return topic or "that topic"


@dataclass
class Rendered:
    text: str
    voice: str
    outcome: str
    topic: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    needs_verification: bool = False
    missing_slot: str | None = None


_OUTCOME_PRIORITY = (
    "NEEDS_VERIFICATION",
    "OUT_OF_SCOPE",
    "THROTTLED",
    "SERVICE_TIMEOUT",
    "TOOL_ERROR",
    "NEEDS_ORDER_ID",
    "INVALID_INPUT",
    "NOT_FOUND",
    "STATE_CONFLICT",
    "INSUFFICIENT_EVIDENCE",
    "HANDOFF_REQUEST_RECORDED",
    "ANSWERED",
)


def _outcome_for(result: ToolResult) -> str:
    code = result.error_code
    if result.ok:
        return "HANDOFF_REQUEST_RECORDED" if result.name == "escalate_to_human" else "ANSWERED"
    if code == "UNAUTHENTICATED_SESSION":
        return "NEEDS_VERIFICATION"
    if code == "OUT_OF_SCOPE":
        return "THROTTLED" if result.internal.get("_throttled") else "OUT_OF_SCOPE"
    if code == "SERVICE_TIMEOUT":
        return "SERVICE_TIMEOUT"
    if code == "SERVICE_ERROR":
        return "TOOL_ERROR"
    if code == "INVALID_INPUT":
        return "NEEDS_ORDER_ID" if result.internal.get("_missing_slot") else "INVALID_INPUT"
    return str(code or "ANSWERED")


def _policy_voice(clauses: Sequence[Mapping[str, Any]]) -> str:
    """One sentence per cited clause (top_k = 3), each with its clause reference."""
    parts = []
    for index, clause in enumerate(clauses[:3]):
        name = copy.POLICY_CATEGORY_NAMES.get(clause["category"], "Policy")
        lead = "According to VoxCom" if index == 0 else "Under"
        parts.append(f"{lead} {name} {clause['clause_ref']}, {first_sentence(clause['text'])}")
    extra = len(clauses) - len(parts)
    if extra:
        parts.append(
            "There is 1 more applicable clause." if extra == 1 else f"There are {extra} more applicable clauses."
        )
    return " ".join(parts)


def _policy_text(clauses: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for index, clause in enumerate(clauses):
        name = copy.POLICY_CATEGORY_NAMES.get(clause["category"], "Policy")
        lead = "According to VoxCom" if index == 0 else "Under"
        parts.append(f"{lead} {name} {clause['clause_ref']}: {clause['text']}")
    return " ".join(parts)


def _price_labels(results: Sequence[ToolResult]) -> dict[int, str]:
    """Order value → brand ("the Samsung") from catalog results in the same turn."""
    labels: dict[int, str] = {}
    for result in results:
        if not result.ok:
            continue
        rows = result.envelope.get("products") or result.envelope.get("results") or []
        for row in rows:
            labels.setdefault(int(row["price_vnd"]), row["brand"])
    return labels


def render_results(results: Sequence[ToolResult], *, user_text: str = "", active_order: str | None = None) -> Rendered:
    texts: list[str] = []
    voices: list[str] = []
    evidence: list[dict[str, Any]] = []
    outcomes: list[str] = []
    topic: str | None = None
    needs_verification = False
    missing_slot: str | None = None
    names = [result.name for result in results]
    tracked_orders = {
        result.envelope["shipment"]["order_code"]
        for result in results
        if result.name == "track_shipment_status" and result.ok
    }
    mentioned = set(tracked_orders)
    labels = _price_labels(results)

    def add(text: str, voice: str | None = None) -> None:
        if text and text not in texts:
            texts.append(text)
            voices.append(speakable(voice if voice is not None else text))

    for result in results:
        env, internal = result.envelope, result.internal
        outcome = _outcome_for(result)
        outcomes.append(outcome)
        code = result.error_code
        name = result.name
        if code == "UNAUTHENTICATED_SESSION":
            needs_verification = True
            continue  # the dialogue layer speaks the verification prompt
        if name == "lookup_order_details":
            order_code = internal.get("_order_code") or result.arguments.get("order_code", "")
            if result.ok:
                order = env["order"]
                mentioned.add(order["order_code"])
                topic = f"your order {order['order_code']}"
                if order["order_code"] not in tracked_orders:
                    items = [item["product_name"] for item in order["items"]]
                    add(
                        copy.order_status_line(
                            order["order_code"], order["order_status"], items, order["total_amount_vnd"]
                        )
                    )
            elif code == "INVALID_INPUT":
                missing_slot = "order_code"
                add(copy.ORDER_INVALID_INPUT)
            elif code == "OUT_OF_SCOPE":
                add(copy.order_out_of_scope_full(order_code))
            elif code == "NOT_FOUND":
                add(copy.order_not_found(order_code))
            elif code == "SERVICE_TIMEOUT":
                add(copy.ORDER_TIMEOUT)
            else:
                add(copy.ORDER_SERVICE_ERROR)
        elif name == "track_shipment_status":
            if result.ok:
                shipment = env["shipment"]
                topic = f"your order {shipment['order_code']}"
                if shipment["shipment_id"] is None or (
                    shipment["estimated_delivery_date"] is None
                    and shipment["shipment_status"] not in {"delivered", "returned"}
                ):
                    line = copy.SHIPMENT_NULL_ETA
                else:
                    line = copy.shipment_status_line(
                        shipment["order_code"],
                        shipment["shipment_status"],
                        shipment["carrier_name"],
                        shipment["estimated_delivery_date"],
                    )
                    if shipment["estimated_delivery_date"] is None and mentions_date(line):
                        line = copy.SHIPMENT_NULL_ETA  # ETA guard (guide §8.2)
                add(line)
            elif code == "INVALID_INPUT":
                missing_slot = "order_code"
                add(copy.SHIPMENT_INVALID_INPUT)
            elif code == "OUT_OF_SCOPE":
                denied_lookup = any(
                    r.name == "lookup_order_details" and r.error_code == "OUT_OF_SCOPE" for r in results
                )
                if not denied_lookup:  # one refusal per order: the lookup line already covers it
                    add(copy.SHIPMENT_OUT_OF_SCOPE)
            elif code == "NOT_FOUND":
                if result.arguments.get("order_code"):
                    add(copy.order_not_found(internal.get("_order_code") or result.arguments["order_code"]))
                else:
                    add(
                        copy.shipment_not_found(
                            internal.get("_tracking_code") or result.arguments.get("tracking_code", "")
                        )
                    )
            elif code == "SERVICE_TIMEOUT":
                add(copy.SHIPMENT_TIMEOUT)
            else:
                add(copy.SHIPMENT_SERVICE_ERROR)
        elif name == "list_customer_orders":
            if result.ok:
                history_mode = bool(_HISTORY_WORDS.search(user_text)) or bool(mentioned)
                entries = []
                for order in env["orders"]:
                    if order["order_code"] in mentioned:
                        continue
                    if history_mode and order["order_status"] not in {"delivered", "returned", "return_requested"}:
                        continue
                    delivered = date.fromisoformat(order["delivered_on"]) if order["delivered_on"] else None
                    entries.append((order["order_code"], order["order_status"], delivered))
                # "also" when this reply already named an order, or the caller contrasts with the one discussed
                # ("What about my earlier purchases?"); not for a plain "What did I buy before?" (log SES-ba23ffdc)
                also = bool(mentioned) or (bool(active_order) and bool(_CONTRAST_WORDS.search(user_text)))
                add(copy.purchase_history_line(entries, also=also) if entries else copy.NO_OTHER_ORDERS)
                topic = topic or "your purchase history"
            elif code == "SERVICE_TIMEOUT":
                add(copy.ORDER_TIMEOUT)
            else:
                add(copy.ORDER_SERVICE_ERROR)
        elif name == "retrieve_policy_clauses":
            topic = "your policy question"
            if result.ok:
                clauses = env["clauses"]
                text = _policy_text(clauses)
                voice = _policy_voice(clauses)
                if any(clause["clause_id"] == "POL-RET-001" for clause in clauses):
                    text += " " + copy.POLICY_TECHNICIAN_CONDITION
                    voice += " " + copy.POLICY_TECHNICIAN_CONDITION
                add(text, voice + " " + copy.POLICY_SCREEN_POINTER)
                for clause in clauses:
                    evidence.append(
                        {
                            "policy_id": clause["clause_id"],
                            "clause_ref": clause["clause_ref"],
                            "category": clause["category"],
                            "title": clause["title"],
                            "content_en": clause["text"],
                            "similarity_score": clause["similarity_score"],
                            "applicability_rules": clause.get("applicability_rules"),
                            "corpus_id": env["corpus_id"],
                            "is_fictional": env["is_fictional"],
                            "clause_version": env["clause_version"],
                        }
                    )
            elif code == "INSUFFICIENT_EVIDENCE":
                add(copy.policy_insufficient_evidence(inquiry_topic(user_text or result.arguments.get("query", ""))))
            elif code == "INVALID_INPUT":
                add(copy.POLICY_INVALID_INPUT)
            elif code == "SERVICE_TIMEOUT":
                add(copy.POLICY_TIMEOUT)
            else:
                add(copy.POLICY_SERVICE_ERROR)
        elif name == "search_catalog":
            topic = "your product search"
            if result.ok:
                if "compare_products" in names:
                    continue  # the comparison line carries the prices (Scenario 7)
                rows = env["results"]
                short = internal.get("_labels", {})
                add(
                    copy.catalog_results_line(
                        [(row["product_name"], row["sku"], row["price_vnd"], row["availability"]) for row in rows]
                    ),
                    copy.catalog_results_line(
                        [
                            (
                                short.get(row["sku"], row["product_name"]),
                                row["sku"],
                                row["price_vnd"],
                                row["availability"],
                            )
                            for row in rows
                        ]
                    ),
                )
            elif code == "OUT_OF_SCOPE":
                add(f"{copy.CATALOG_THROTTLED} {copy.CATALOG_THROTTLE_POINTER}")
            elif code == "NOT_FOUND":
                add(copy.catalog_not_found(internal.get("_missing") or result.arguments.get("keyword") or "that model"))
            elif code == "STATE_CONFLICT":
                add(copy.out_of_stock(internal.get("_sku", "That model")))
            elif code == "INVALID_INPUT":
                add(copy.CATALOG_INVALID_INPUT)
            elif code == "SERVICE_TIMEOUT":
                add(copy.CATALOG_TIMEOUT)
            else:
                add(copy.CATALOG_SERVICE_ERROR)
        elif name == "compare_products":
            topic = "the product comparison"
            if result.ok:
                short = internal.get("_labels", {})
                line = copy.comparison_line(
                    [(short.get(row["sku"], row["sku"]), row["price_vnd"]) for row in env["products"]]
                )
                missing = []
                for sku, attributes in (env.get("not_recorded") or {}).items():
                    if any(
                        _ATTRIBUTE_WORDS.get(attr) and _ATTRIBUTE_WORDS[attr].search(user_text) for attr in attributes
                    ):
                        missing.append(copy.attribute_not_recorded(sku))
                add(" ".join([line, *missing]))
            elif code == "INVALID_INPUT":
                add(copy.COMPARE_INVALID_INPUT)
            elif code == "NOT_FOUND":
                add(copy.catalog_not_found(internal.get("_missing") or "that model"))
            elif code == "SERVICE_TIMEOUT":
                add(copy.COMPARE_TIMEOUT)
            else:
                add(copy.COMPARE_SERVICE_ERROR)
        elif name == "apply_voucher":
            topic = topic or "the voucher"
            voucher = internal.get("_code") or str(result.arguments.get("voucher_code", "")).upper()
            if result.ok:
                value = int(result.arguments["order_value_vnd"])
                brand = labels.get(value)
                add(
                    copy.voucher_quote_line(
                        voucher, brand, internal.get("_quoted_price_vnd", value - env["discount_vnd"])
                    )
                )
            elif code == "INVALID_INPUT":
                reason = internal.get("_reason")
                promotion = internal.get("_promotion")
                if reason in {"CATEGORY_MISMATCH", "BELOW_MINIMUM", "NOT_IN_VALIDITY"} and promotion:
                    if voucher == "VOXCOM_500K":
                        add(copy.VOUCHER_500K_NOT_APPLICABLE)
                    else:
                        add(
                            copy.voucher_not_applicable(
                                voucher,
                                CATEGORY_LABELS.get(promotion["category"], promotion["category"]),
                                promotion["min_order_value"],
                            )
                        )
                elif reason == "RETIRED":
                    add(copy.VOUCHER_RETIRED)
                else:
                    add(copy.VOUCHER_INVALID_INPUT)
            elif code == "SERVICE_TIMEOUT":
                add(copy.VOUCHER_TIMEOUT)
            else:
                add(copy.VOUCHER_SERVICE_ERROR)
        elif name == "escalate_to_human":
            if result.ok:
                add(copy.handoff_recorded(env["ticket_number"]))
            elif code == "SERVICE_TIMEOUT":
                add(copy.HANDOFF_TIMEOUT)
            else:
                add(copy.HANDOFF_SERVICE_ERROR)
        else:  # not allowlisted (a future-only write or an invented name): nothing ran
            add("I could not complete that request. Nothing was changed.")

    outcome = next((item for item in _OUTCOME_PRIORITY if item in outcomes), "ANSWERED")
    return Rendered(
        text=" ".join(texts),
        voice=" ".join(voices),
        outcome=outcome,
        topic=topic,
        evidence=evidence,
        needs_verification=needs_verification,
        missing_slot=missing_slot,
    )


_WRITE_CLAIM = re.compile(
    r"\b(?:is|has\s+been|was|were)\s+(?:now\s+)?(?:cancelled|canceled|rescheduled|moved|reinstated|refunded|updated)\b"
    r"|\bis\s+now\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\bactive\s+again\b",
    re.I,
)


def write_guard(text: str, *, execution_confirmed: bool) -> str:
    """A success claim for a write is allowed only after an explicit execution result (guide §8.2)."""
    if execution_confirmed or not _WRITE_CLAIM.search(text):
        return text
    return copy.WRITE_STILL_PROCESSING_INTERIM


def eta_guard(text: str, *, null_eta: bool) -> str:
    """With a null ETA, any date or relative-date word replaces the text with the null-ETA line (guide §8.2)."""
    if null_eta and mentions_date(text):
        return copy.SHIPMENT_NULL_ETA
    return text


__all__ = ["Rendered", "eta_guard", "inquiry_topic", "render_results", "write_guard"]
