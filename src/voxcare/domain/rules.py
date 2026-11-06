"""Pure business rules: scope, order FSM, write lifecycle, voucher math, eligibility, reversal.

No provider, storage or framework imports: NFR-07 requires zero changed lines
here when an ASR/LLM/TTS provider is swapped. Every function is deterministic
and takes the current date or instant as an argument (never the wall clock).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .enums import OrderStatus as O
from .enums import PendingActionStatus as P
from .enums import ProductCategory

# --- scope (FR-AUTH-02 proposal; guide §15) -------------------------------------------------


def scope_allows(resource_customer_id: str | None, session_customer_id: str | None) -> bool:
    """A customer-owned row may be read only by the verified session owner."""
    return bool(session_customer_id) and resource_customer_id == session_customer_id


# --- order FSM (D§6.5, guide §11.6) --------------------------------------------------------

ORDER_TRANSITIONS: dict[O, frozenset[O]] = {
    O.PENDING_CONFIRMATION: frozenset({O.CONFIRMED, O.CANCELLED}),
    O.CONFIRMED: frozenset({O.PROCESSING, O.CANCELLED}),
    O.PROCESSING: frozenset({O.READY_FOR_PICKUP, O.CANCELLED}),
    O.READY_FOR_PICKUP: frozenset({O.SHIPPED}),
    O.SHIPPED: frozenset({O.OUT_FOR_DELIVERY}),
    O.OUT_FOR_DELIVERY: frozenset({O.DELIVERED, O.DELIVERY_FAILED}),
    O.DELIVERY_FAILED: frozenset({O.OUT_FOR_DELIVERY, O.RETURNED}),
    O.DELIVERED: frozenset({O.RETURN_REQUESTED}),
    O.RETURN_REQUESTED: frozenset({O.RETURNED}),
    O.RETURNED: frozenset(),
    O.CANCELLED: frozenset(),
}
FORBIDDEN_TRANSITIONS = frozenset({(O.SHIPPED, O.CANCELLED), (O.OUT_FOR_DELIVERY, O.CANCELLED)})
TERMINAL_ORDER_STATUSES = frozenset({O.CANCELLED, O.RETURNED})
MAX_DELIVERY_REATTEMPTS = 3


class ForbiddenTransition(ValueError):
    """A transition the design forbids outright (``shipped``/``out_for_delivery`` → ``cancelled``)."""


def can_transition(from_status: str, to_status: str, *, reattempts: int = 0) -> bool:
    """``True`` for a valid order transition, ``False`` otherwise; raises on the two forbidden ones.

    ``reattempts`` counts earlier ``delivery_failed → out_for_delivery`` moves (max 3).
    """
    src, dst = O(from_status), O(to_status)
    if (src, dst) in FORBIDDEN_TRANSITIONS:
        raise ForbiddenTransition(f"{src} -> {dst} is forbidden")
    if dst not in ORDER_TRANSITIONS[src]:
        return False
    return not (src is O.DELIVERY_FAILED and dst is O.OUT_FOR_DELIVERY and reattempts >= MAX_DELIVERY_REATTEMPTS)


def is_dispatched(status: str) -> bool:
    """Shipped or later on the delivery path: a cancel gets the OPD-03 line and no state change."""
    return O(status) in {O.SHIPPED, O.OUT_FOR_DELIVERY}


def cancellable(status: str) -> bool:
    try:
        return can_transition(status, O.CANCELLED)
    except ForbiddenTransition:
        return False


# --- guarded-write lifecycle (D§6.1, guide §11.2) -----------------------------------------

ACTION_TRANSITIONS: dict[P, frozenset[P]] = {
    P.PENDING: frozenset({P.SUBMITTED, P.CANCELLED, P.FAILED}),
    P.SUBMITTED: frozenset({P.EXECUTING, P.CANCELLED, P.FAILED}),
    P.EXECUTING: frozenset({P.COMPLETED, P.CANCELLED, P.FAILED}),
    P.COMPLETED: frozenset(),
    P.CANCELLED: frozenset(),
    P.FAILED: frozenset(),
}
TERMINAL_ACTION_STATUSES = frozenset({P.COMPLETED, P.CANCELLED, P.FAILED})
MIN_CONFIRMATION_STEPS, MAX_CONFIRMATION_STEPS = 1, 2


def can_advance_action(from_status: str, to_status: str) -> bool:
    return P(to_status) in ACTION_TRANSITIONS[P(from_status)]


def confirmation_steps_valid(steps: int) -> bool:
    """Confirmation takes 1-2 steps, minimum 1; there is no third step (D§6, Section/12-YeuCau.tex:146)."""
    return MIN_CONFIRMATION_STEPS <= steps <= MAX_CONFIRMATION_STEPS


def preview_hash(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical previewed terms (implementation choice); 64 hex chars."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- voucher (D§4.6, D§9.2.2, guide §7.6, G-8) ---------------------------------------------

RETIRED_VOUCHER_CODES = frozenset({"VOXCOM_TET_500K"})
VOUCHER_CODE_SHAPE = re.compile(r"^[A-Z0-9_]{3,32}$")


@dataclass(frozen=True)
class VoucherQuote:
    valid: bool
    discount_vnd: int
    capped_at_vnd: int
    reason: str | None = None  # UNKNOWN_CODE | RETIRED | CATEGORY_MISMATCH | BELOW_MINIMUM | NOT_IN_VALIDITY

    @property
    def outcome(self) -> str:
        return "SUCCESS" if self.valid else "INVALID_INPUT"


def evaluate_voucher(
    promotion: Mapping[str, Any] | None, code: str, order_value_vnd: int, category: str, today: date
) -> VoucherQuote:
    """Valid iff the code exists, is not retired, the category matches, the value meets the minimum
    and ``today`` is inside ``[valid_from, valid_to]``. The discount is a fixed amount equal to the cap (G-8)."""
    if code in RETIRED_VOUCHER_CODES:
        return VoucherQuote(False, 0, 0, "RETIRED")
    if promotion is None:
        return VoucherQuote(False, 0, 0, "UNKNOWN_CODE")
    cap = int(promotion["discount_cap"])
    if category != promotion["category"]:
        return VoucherQuote(False, 0, cap, "CATEGORY_MISMATCH")
    if order_value_vnd < int(promotion["min_order_value"]):
        return VoucherQuote(False, 0, cap, "BELOW_MINIMUM")
    if not _as_date(promotion["valid_from"]) <= today <= _as_date(promotion["valid_to"]):
        return VoucherQuote(False, 0, cap, "NOT_IN_VALIDITY")
    discount_amount = cap  # G-8: the design defines only the cap; a fixed-amount voucher equal to the cap is assumed
    return VoucherQuote(True, min(cap, discount_amount, order_value_vnd), cap)


# --- return window and after-sales diagnosis (D§8.2.4, D§9.3, guide §14.1) -----------------

RETURN_WINDOW_DAYS = 30
DIAGNOSIS_MAX_TURNS = 3


def within_return_window(delivered_on: date, today: date, *, days: int = RETURN_WINDOW_DAYS) -> bool:
    """POL-RET-001: 30 calendar days from the validated delivery date. Eligibility stays conditional on
    technician certification; this only answers the window half."""
    return delivered_on <= today <= delivered_on + timedelta(days=days)


SYMPTOM_FAULT_CLASSES = {"SPIN_CYCLE_ERROR_E4": "MOTOR_DRIVE_FAULT"}
_SYMPTOM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "SPIN_CYCLE_ERROR_E4",
        re.compile(r"\b(?:error\s*(?:code\s*)?e\s*-?\s*4|e4\b|stops?\s+(?:during|in|on)\s+(?:the\s+)?spin)", re.I),
    ),
)


def detect_symptom(text: str) -> str | None:
    for code, pattern in _SYMPTOM_PATTERNS:
        if pattern.search(text):
            return code
    return None


def classify_fault(symptom_code: str) -> str | None:
    """Turn 3: the fault class, or ``None`` (escalate immediately)."""
    return SYMPTOM_FAULT_CLASSES.get(symptom_code)


@dataclass(frozen=True)
class RmaCheck:
    ok: bool
    error: str | None = None  # INVALID_INPUT | AUTH_SCOPE_VIOLATION


def rma_invariant(
    item_order_id: str | None, order_id: str, order_customer_id: str, session_customer_id: str
) -> RmaCheck:
    """The item must belong to the order and the order to the session customer (guide §4.1)."""
    if item_order_id != order_id:
        return RmaCheck(False, "INVALID_INPUT")
    if not scope_allows(order_customer_id, session_customer_id):
        return RmaCheck(False, "AUTH_SCOPE_VIOLATION")
    return RmaCheck(True)


# --- reversal eligibility (D§6.3, guide §11.4); default is deny -----------------------------


@dataclass(frozen=True)
class ReversalDecision:
    allowed: bool
    reason: str


def reinstate_cancel_allowed(
    *,
    order_status: str | None,
    shipment_status: str | None,
    inventory_reserved: bool | None,
    refund_dispatched: bool | None,
    version_current: bool | None,
    oms_validates: bool | None,
) -> ReversalDecision:
    if order_status != O.CANCELLED:
        return ReversalDecision(False, "ORDER_NOT_CANCELLED")
    if shipment_status not in (None, "label_created"):
        return ReversalDecision(False, "SHIPMENT_PROGRESSED")
    if refund_dispatched is not False:
        return ReversalDecision(False, "REFUND_DISPATCHED")
    if inventory_reserved is not True:
        return ReversalDecision(False, "INVENTORY_RELEASED")
    if version_current is not True:
        return ReversalDecision(False, "STALE_VERSION")
    if oms_validates is not True:
        return ReversalDecision(False, "OMS_REJECTED")
    return ReversalDecision(True, "ELIGIBLE")


def reverse_address_change_allowed(
    *, order_status: str | None, shipment_status: str | None, oms_confirms: bool | None
) -> ReversalDecision:
    if order_status not in (O.PENDING_CONFIRMATION, O.CONFIRMED, O.PROCESSING):
        return ReversalDecision(False, "ORDER_PROGRESSED")
    if shipment_status not in (None, "label_created"):
        return ReversalDecision(False, "SHIPMENT_PROGRESSED")
    if oms_confirms is not True:
        return ReversalDecision(False, "OMS_NOT_CONFIRMED")
    return ReversalDecision(True, "ELIGIBLE")


def reverse_reschedule_allowed(
    *,
    now: datetime | None,
    target_slot_start: datetime | None,
    existing_window_start: datetime | None,
    cutoff: datetime | None,
    order_status: str | None,
    shipment_status: str | None,
    carrier_result: str | None,
) -> ReversalDecision:
    """Deterministic reschedule test (guide §11.4): CONFIRMED before the cutoff → eligible; PENDING / REFUSED → deny."""
    if None in (now, target_slot_start, existing_window_start, cutoff):
        return ReversalDecision(False, "MISSING_INPUT")
    if target_slot_start <= now:
        return ReversalDecision(False, "TARGET_IN_PAST")
    if existing_window_start <= now:
        return ReversalDecision(False, "WINDOW_BEGUN")
    if now >= cutoff:
        return ReversalDecision(False, "PAST_CUTOFF")
    in_transit = order_status == O.SHIPPED or shipment_status == "in_transit"
    if in_transit and carrier_result != "CONFIRMED":
        return ReversalDecision(False, f"CARRIER_{carrier_result or 'MISSING'}")
    return ReversalDecision(True, "ELIGIBLE")


# --- ETA / date words (D§4.3 null-ETA invariant; output guard §8.2) ------------------------

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_WORDS = re.compile(
    rf"\b(?:today|tonight|tomorrow|yesterday|soon|shortly|next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
    rf"friday|saturday|sunday)|this\s+(?:week|weekend)|in\s+(?:a|one|two|three|four|five|\d+)\s+(?:days?|weeks?|hours?)|"
    rf"monday|tuesday|wednesday|thursday|friday|saturday|sunday|{_MONTHS})\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|"
    rf"\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b",
    re.IGNORECASE,
)


def mentions_date(text: str) -> bool:
    return DATE_WORDS.search(text) is not None


# --- product categories named in free text (applicability inference, guide §7.3 step 2) ----

PRODUCT_CATEGORY_TERMS: dict[ProductCategory, re.Pattern[str]] = {
    ProductCategory.TV_AV: re.compile(r"\b(?:tvs?|televisions?|displays?|soundbars?|speakers?|audio)\b", re.I),
    ProductCategory.REFRIGERATOR: re.compile(r"\b(?:refrigerators?|fridges?|freezers?)\b", re.I),
    ProductCategory.WASHING_MACHINE: re.compile(r"\b(?:washing\s+machines?|washers?|dryers?|laundry)\b", re.I),
    ProductCategory.AIR_CONDITIONER: re.compile(r"\b(?:air[\s-]?conditioners?|air[\s-]?con|a/?c|aircon)\b", re.I),
    ProductCategory.KITCHEN: re.compile(
        r"\b(?:kitchen|microwaves?|ovens?|cookers?|hobs?|stoves?|dishwashers?|blenders?|kettles?|rice\s+cookers?)\b",
        re.I,
    ),
}
KNOWN_BRANDS = ("Samsung", "LG", "Panasonic", "Electrolux", "Toshiba")
_BRAND_TERMS = {brand: re.compile(rf"\b{brand}\b", re.I) for brand in KNOWN_BRANDS}


def product_categories_in(text: str) -> tuple[str, ...]:
    return tuple(category.value for category, pattern in PRODUCT_CATEGORY_TERMS.items() if pattern.search(text))


def brands_in(text: str) -> tuple[str, ...]:
    return tuple(brand for brand, pattern in _BRAND_TERMS.items() if pattern.search(text))


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


__all__ = [
    "ACTION_TRANSITIONS",
    "DATE_WORDS",
    "DIAGNOSIS_MAX_TURNS",
    "FORBIDDEN_TRANSITIONS",
    "KNOWN_BRANDS",
    "MAX_CONFIRMATION_STEPS",
    "MAX_DELIVERY_REATTEMPTS",
    "MIN_CONFIRMATION_STEPS",
    "ORDER_TRANSITIONS",
    "PRODUCT_CATEGORY_TERMS",
    "RETIRED_VOUCHER_CODES",
    "RETURN_WINDOW_DAYS",
    "SYMPTOM_FAULT_CLASSES",
    "TERMINAL_ACTION_STATUSES",
    "TERMINAL_ORDER_STATUSES",
    "VOUCHER_CODE_SHAPE",
    "ForbiddenTransition",
    "ReversalDecision",
    "RmaCheck",
    "VoucherQuote",
    "brands_in",
    "can_advance_action",
    "can_transition",
    "cancellable",
    "classify_fault",
    "confirmation_steps_valid",
    "detect_symptom",
    "evaluate_voucher",
    "is_dispatched",
    "mentions_date",
    "preview_hash",
    "product_categories_in",
    "reinstate_cancel_allowed",
    "reverse_address_change_allowed",
    "reverse_reschedule_allowed",
    "rma_invariant",
    "scope_allows",
    "within_return_window",
]
