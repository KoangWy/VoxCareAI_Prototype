"""Every spoken line, verbatim (P1 build guide §7, §11, §12, §13.4, §14; design D§4, D§6, D§7.2, D§8.4, D§13).

Lines marked ``# implementation choice`` are not given by the design; everything
else is copied character for character. A ``[CODE]``/``[inquiry topic]`` slot in
the design is a function parameter here. The renderer and the dialogue layer
import from this module only, so a spoken line has exactly one definition.
"""

from __future__ import annotations

from datetime import date

# --- lookup_order_details (D§4.2, guide §7.1) --------------------------------------------
ORDER_INVALID_INPUT = "Could you please tell me your order code?"
ORDER_UNAUTHENTICATED = "I could not verify this call session. Please start the call again so I can check your account."
ORDER_OUT_OF_SCOPE = "I can only share details for orders on your verified account."
ORDER_TIMEOUT = "The order system is not responding. I can connect you with an agent."
ORDER_SERVICE_ERROR = (
    "I could not complete that check. Nothing was changed. I can connect you with a customer service representative."
)


def order_not_found(order_code: str) -> str:
    return f"I could not find order {order_code}. Please check your order confirmation."


def order_out_of_scope_full(order_code: str) -> str:
    """Scenario 3: the full version of the OUT_OF_SCOPE line (G-5)."""
    return (
        "For data privacy and security, I can only share details for orders on your verified account. "
        f"Order {order_code} is not associated with your phone number."
    )


# --- verification sub-dialogue (FR-AUTH-01 proposal; Scenario 2, G-5) ---------------------
VERIFY_ACCOUNT_PROMPT = (
    "To access your order information, I need to verify your registered account. Please provide the phone "
    "number used during checkout, or check the verification SMS sent to you."
)
VERIFY_SUCCESS = (  # implementation choice (simulated verification, OPD-04)
    "Thank you. Your account is verified for this call."
)
VERIFY_FAILED = (  # implementation choice
    "I could not match that phone number to a VoxCom account. Please check the number used during checkout."
)

# --- track_shipment_status (D§4.3, guide §7.2) --------------------------------------------
SHIPMENT_NULL_ETA = (
    "The carrier has not provided an estimated delivery date yet. You will get an SMS once it is scheduled."
)
SHIPMENT_INVALID_INPUT = "Could you please tell me your order code, or the tracking code from your confirmation?"
SHIPMENT_OUT_OF_SCOPE = (
    "For privacy and security, I can only provide shipment updates for orders placed under your account."
)
SHIPMENT_TIMEOUT = "The courier network is not responding. I can connect you with an agent."
SHIPMENT_SERVICE_ERROR = "I could not reach the courier tracking network. I can connect you with an agent."


def shipment_not_found(tracking_code: str) -> str:
    return f"I checked tracking code {tracking_code} and could not find any active shipment for it."


_SHIPMENT_PHRASES = {  # implementation choice: status wording for the Scenario 1 sentence
    "label_created": "has a shipping label with",
    "in_transit": "is in transit with",
    "out_for_delivery": "is out for delivery with",
    "delivered": "was delivered by",
    "delivery_failed": "had a failed delivery attempt with",
    "returned": "was returned to VoxCom by",
}


def shipment_status_line(order_code: str, shipment_status: str, carrier: str, eta: str | None) -> str:
    """Scenario 1: "Order ORD-1002 is in transit with GHN. Estimated delivery date: 2026-10-15."."""
    phrase = _SHIPMENT_PHRASES.get(shipment_status, "is with")
    line = f"Order {order_code} {phrase} {carrier}."
    if eta and shipment_status not in {"delivered", "returned"}:
        line += f" Estimated delivery date: {eta}."
    return line


def shipment_eta_example(order_code: str, carrier: str, eta: date) -> str:
    """The guide §7.2 wording for a non-null ETA (kept for the live template check)."""
    return f"Your order {order_code} has shipped with {carrier} and is estimated to arrive on {month_day(eta)}."


# --- order status (lookup alone) and purchase history (G-3) -------------------------------
_ORDER_STATUS_PHRASES = {  # implementation choice
    "pending_confirmation": "is waiting for confirmation",
    "confirmed": "is confirmed",
    "processing": "is being processed",
    "ready_for_pickup": "is ready for pickup",
    "shipped": "has shipped",
    "out_for_delivery": "is out for delivery",
    "delivered": "was delivered",
    "delivery_failed": "had a failed delivery attempt",
    "return_requested": "has a return requested",
    "returned": "was returned",
    "cancelled": "is cancelled",
}


def order_status_line(order_code: str, status: str, items: list[str], total_vnd: int) -> str:
    """lookup_order_details alone. implementation choice."""
    phrase = _ORDER_STATUS_PHRASES.get(status, f"has status {status}")
    line = f"Order {order_code} {phrase}."
    if items:
        line += f" Items: {', '.join(items)}."
    return line + f" Total: {vnd(total_vnd)}."


def purchase_history_line(entries: list[tuple[str, str, date | None]], *, also: bool) -> str:
    """Scenario 1: "Your account also shows ORD-1003, delivered on October 2."."""
    parts = []
    for code, status, on in entries:
        if status == "delivered" and on is not None:
            parts.append(f"{code}, delivered on {month_day(on)}")
        else:
            parts.append(f"{code}, which {_ORDER_STATUS_PHRASES.get(status, status)}")
    lead = "Your account also shows" if also else "Your account shows"
    return f"{lead} {_join(parts)}."


NO_OTHER_ORDERS = "I do not see any other orders on your account."  # implementation choice

# --- retrieve_policy_clauses (D§4.5, guide §7.3) ------------------------------------------
POLICY_INVALID_INPUT = "Could you rephrase what you would like to know about our policies?"
POLICY_TIMEOUT = "The policy system is not responding. I can connect you with an agent."
POLICY_SERVICE_ERROR = "I could not complete that check. I can connect you with a customer service representative."


def policy_insufficient_evidence(inquiry_topic: str) -> str:
    return (
        f"VoxCom does not currently have a published policy regarding {inquiry_topic}. "
        "Would you like me to connect you with our customer support desk for further assistance?"
    )


POLICY_CATEGORY_NAMES = {  # implementation choice: "According to VoxCom Warranty Policy Clause 3.1" (Scenario 4)
    "SHIPPING": "Shipping Policy",
    "INSTALLATION": "Installation Policy",
    "WARRANTY": "Warranty Policy",
    "RETURNS": "Return Policy",
}
# Rule 5 (guide §8.2): return/exchange eligibility is always conditional on technician certification.
POLICY_TECHNICIAN_CONDITION = (  # implementation choice, wording after Scenario 4
    "This applies only if a VoxCom or authorised brand technician certifies the defect, "
    "and that certification has not been done yet."
)
POLICY_NOT_A_DECISION = (  # implementation choice: rule 4, the terms are stated, never applied as an approval
    "These are the fictional demo terms, not a decision about your order."
)
POLICY_SCREEN_POINTER = "The full clause text is shown in the transcript."  # implementation choice (voice only)

# --- search_catalog / compare_products (D§9.2.3, D§4.6, guide §7.4-7.5) --------------------
CATALOG_INVALID_INPUT = (
    "Which category should I search — TV and audio, refrigeration, laundry, air conditioning, or kitchen?"
)
CATALOG_UNAUTHENTICATED = (
    "Before I look anything up, I need to verify your account. Could you confirm the phone number on the order?"
)
CATALOG_THROTTLED = "Bulk stock counts are not available by phone."
CATALOG_THROTTLE_POINTER = "You can browse every model on the VoxCom web storefront."  # implementation choice
CATALOG_TIMEOUT = "I could not reach the catalog system just now. Let me try once more."
CATALOG_SERVICE_ERROR = (
    "I cannot reach the catalog system right now. I can pass this to a customer service representative."
)
COMPARE_INVALID_INPUT = "I can compare two or three models from the same category — which second model would you like?"
COMPARE_TIMEOUT = "The catalog is not responding. I can connect you with an agent."
COMPARE_SERVICE_ERROR = (
    "I could not complete that comparison. Nothing was changed. I can connect you with a customer service "
    "representative."
)


def catalog_not_found(sku: str) -> str:
    return f"I cannot find {sku} in our catalog."


def attribute_not_recorded(sku: str) -> str:
    return f"That figure is not recorded for {sku}."


def out_of_stock(sku: str) -> str:
    return f"{sku} is out of stock today."


def catalog_results_line(results: list[tuple[str, str, int, str]]) -> str:
    """search_catalog success: (short name, sku, price, availability). implementation choice."""
    parts = [
        f"{name} ({sku}) at {vnd(price)}" + ("" if availability == "in_stock" else ", out of stock today")
        for name, sku, price, availability in results
    ]
    noun = "model" if len(parts) == 1 else "models"
    return f"I found {len(parts)} {noun}: {_join(parts)}."


def comparison_line(prices: list[tuple[str, int]]) -> str:
    """Scenario 7: "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND."."""
    return " ".join(f"{label}: {vnd(price)}." for label, price in prices)


# --- apply_voucher (D§4.6, D§9.2.2, guide §7.6) -------------------------------------------
VOUCHER_INVALID_INPUT = "Could you read me the voucher code again?"
VOUCHER_TIMEOUT = "The promotion system is not responding. I can connect you with an agent."
VOUCHER_SERVICE_ERROR = (
    "I could not check that voucher. Nothing was changed. I can connect you with a customer service representative."
)
VOUCHER_500K_NOT_APPLICABLE = "VOXCOM_500K covers TV and audio only, and needs an order of 10,000,000 VND or more."
VOUCHER_RETIRED = "That voucher code is no longer valid."  # implementation choice (VOXCOM_TET_500K)


def voucher_not_applicable(code: str, category_label: str, minimum_vnd: int) -> str:
    """The design line, generalised to any promotion row (the VOXCOM_500K instance is verbatim)."""
    return f"{code} covers {category_label} only, and needs an order of {vnd(minimum_vnd)} or more."


def voucher_quote_line(code: str, label: str | None, quoted_vnd: int) -> str:
    """Scenario 7: "With VOXCOM_500K the Samsung is 10,990,000 VND." (a quote, never applied)."""
    target = f"the {label}" if label else "the order"
    return f"With {code} {target} is {vnd(quoted_vnd)}."


# --- escalate_to_human (D§7.2, guide §12) -------------------------------------------------
def handoff_recorded(ticket_number: str) -> str:
    return (
        "I have recorded your request to speak with a customer service representative under reference ticket "
        f"{ticket_number}, along with the full details and summary of our conversation so you won't need to "
        "repeat yourself. A support specialist will review your file and contact you via phone or SMS during "
        "business hours. Is there anything else I can assist you with right now?"
    )


HANDOFF_TIMEOUT = "The support system is not responding. Please try again in a moment."  # implementation choice
HANDOFF_SERVICE_ERROR = (  # implementation choice
    "I could not record your request just now. Nothing was changed. Please try again in a moment."
)
OFFER_HANDOFF = "Would you like me to record a request for a customer service representative?"  # implementation choice

# --- after-sales diagnosis (D§9.3, guide §14.1, Scenario 8) --------------------------------
DIAGNOSIS_Q1 = "Does it drain before it spins?"
DIAGNOSIS_Q2 = "Is the drum stiff to turn? One more question before I decide if a technician is needed."
DIAGNOSIS_CONDITIONAL = "Inside the 30-day window, but eligible only if a VoxCom technician confirms the fault."
DIAGNOSIS_OUTSIDE_WINDOW = (  # implementation choice
    "Your delivery date is outside the 30-day replacement window, so a technician will assess warranty repair."
)
DIAGNOSIS_OPTIONS = (  # implementation choice: AD-05 option set (UC-14, UC-15, UC-16)
    "I can record a request for an exchange, a return and refund, or a repair under warranty. Which would you like?"
)
DIAGNOSIS_RECORDED = "I recorded your request. No RMA ticket has been created yet."
DIAGNOSIS_DECLINED = (
    "All right, I have not recorded any request. Is there anything else I can help with?"  # impl. choice
)
DIAGNOSIS_FINAL = (
    "Your claim is inside the 30-day window, but it is eligible only if a VoxCom technician confirms the fault. "
    "I have recorded your request — no RMA ticket has been created yet."
)
DIAGNOSIS_UNKNOWN_SYMPTOM = (  # implementation choice
    "I cannot diagnose that symptom by phone. Would you like me to record a request for a technician to review it?"
)
DIAGNOSIS_NEEDS_ORDER = (  # implementation choice
    "I could not find a delivered washing machine on your verified account. Could you tell me the order code?"
)
RMA_STATUS_HANDOFF = (  # implementation choice (UC-17 has no read tool in Phase 1)
    "I cannot read repair ticket status in this demo. I have recorded a request so a representative can check it."
)

# --- open policy decisions (D§8.4, guide §13.4): the only permitted answer on the topic ----
OPD_01_RETURN_FREIGHT = (
    "For voluntary returns, return shipping conditions are evaluated by store support based on item condition "
    "upon inspection."
)
OPD_02_REFUND_TURNAROUND = (
    "Once cancellation is processed, your refund will be returned via your original payment method in accordance "
    "with your card issuer or bank processing schedule."
)
OPD_03_CANCEL_SHIPPED = (
    "Your order has already been dispatched to the carrier. I have recorded your cancellation request; please "
    "politely decline receipt when the courier calls."
)
OPD_04_NOTE = (  # design note, not a spoken line
    "The prototype uses the simulated session_customer_id scope. Production needs an OTP challenge."
)
OPD_05_COSMETIC = (
    "I will note this cosmetic report for our warranty team. A specialist will review photos of the packaging to "
    "confirm replacement eligibility."
)
OPD_06_BEYOND_30KM = (
    "Deliveries beyond 30 km are fulfilled via our third-party logistics partners and require specialized regional "
    "freight scheduling."
)

# --- guarded writes (D§6, guide §11, Scenarios 6, 11-16) -----------------------------------


def cancel_preview(order_code: str, total_vnd: int) -> str:  # implementation choice ("reads back order, fee, action")
    return (
        f"I can submit a cancellation for {order_code}, total {vnd(total_vnd)}, with no cancellation fee. "
        "Do you confirm?"
    )


def reschedule_preview(order_code: str, target: date) -> str:
    return f"I can move {order_code} to {weekday_month_day(target)}. Do you confirm?"


def reschedule_unchanged(order_code: str, target: date, window: str | None) -> str:  # implementation choice
    slot = f", between {window.replace('-', ' and ')}" if window else ""
    return (
        f"{order_code} is already scheduled for delivery on {weekday_month_day(target)}{slot}. Nothing needs to change."
    )


def reschedule_done(order_code: str, target: date) -> str:
    return f"The delivery window for {order_code} is now {weekday_month_day(target)}."


def address_preview(order_code: str) -> str:  # implementation choice (address is masked PII)
    return f"I can submit the new delivery address for {order_code}. Do you confirm?"


def cancel_done(order_code: str, total_vnd: int) -> str:
    """Scenario 6, only after the explicit execution result."""
    return f"{order_code} is cancelled. VoxCom will refund {vnd(total_vnd)} to your card."


REINSTATE_PREVIEW = "I can submit the reinstatement. Do you confirm?"


def reinstated(order_code: str) -> str:
    return f"Order {order_code} is active again."


REVERSAL_REFUSED_REFUND_DISPATCHED = (
    "The refund batch was already dispatched, so this cancellation cannot be undone. I can ask a human colleague "
    "to help."
)
REVERSAL_REFUSED_GENERIC = (  # implementation choice
    "That change cannot be reversed by phone. I can ask a human colleague to help."
)


def write_recorded(order_code: str) -> str:
    """Phase 1 demo: the request is recorded for VoxCom, never claimed as executed. implementation choice."""
    return (
        f"I have recorded your request for {order_code}. VoxCom will process it and confirm the result; "
        "nothing has changed yet."
    )


WRITE_STILL_PROCESSING_INTERIM = "Your request is being processed. I will confirm only after VoxCom replies."  # choice


def pending_dropped(order_code: str) -> str:  # implementation choice: the caller moved on without confirming
    return f"I have not submitted the change to {order_code}."


NEEDS_ADDRESS = "What is the new delivery address?"  # implementation choice


def withdrawn_pending(order_code: str) -> str:
    """Case A: pending, zero OMS interaction."""
    return f"Cancelled. No changes were made to order {order_code}."


def withdrawn_submitted(order_code: str) -> str:
    """Case B: submitted, intercepted before execution."""
    return f"Stopped before execution. No changes were made to {order_code}."


WITHDRAW_ASKING_OMS = "I am asking the fulfilment system to stop the request."


def withdrawn_executing(order_code: str) -> str:
    """Case C, abort accepted. The Scenario 12 wording, which matches Case A (G-14)."""
    return f"The request was stopped. No changes were made to order {order_code}."


WITHDRAW_TOO_LATE = (
    "It was already committed by the fulfilment centre. Undoing it needs a reinstatement (reversal) request "
    "— I can submit one if you want."
)
REPLAY_STILL_PROCESSING = "The change is still being processed."
REPLAY_ALREADY_APPLIED = "That change was already applied."
REPLAY_OUTCOME_UNKNOWN = "I could not confirm the result yet. I am checking with the fulfilment system."


def replay_conflict(order_code: str) -> str:
    return f"That change can no longer be applied to {order_code}."


WRITE_NOT_CANCELLABLE = (  # implementation choice: FSM refuses the transition (not the shipped case, which is OPD-03)
    "That order can no longer be cancelled by phone. I can ask a human colleague to help."
)
WRITE_PREVIEW_EXPIRED = (  # implementation choice: pending -> failed after the 300 s TTL
    "That request expired before it was confirmed, so nothing was changed. Would you like to start again?"
)
WRITE_DECLINED_KEEP = "Okay, I will leave it as it is. Nothing was changed."  # implementation choice

# --- dialogue control (implementation choice) ---------------------------------------------
# --- capability and repair intake (implementation choices: no design line covers these) -----------------
VOUCHER_LIST_UNAVAILABLE = (
    "I cannot list every current promotion in this demo. I can check whether a specific voucher code applies to a "
    "product; for example, ask me whether VOXCOM_500K applies to the Samsung 55-inch TV."
)


POLICY_TOPICS = (
    "VoxCom's demo policies cover delivery fees, installation, warranty, and returns or exchanges. "
    "Which one would you like to know about?"
)


_SYMPTOM_EXAMPLES = {  # implementation choice: examples that fit the product (log SES-e388fcd3, turn 1)
    "washing machine": "does it show an error code, or stop during a cycle?",
    "TV": "is there no picture, no sound, or does the screen flicker?",
    "refrigerator": "is it not cooling, leaking, or making a noise?",
    "air conditioner": "is it not cooling, leaking, or showing an error code?",
}


def repair_symptom_question(product: str) -> str:
    example = _SYMPTOM_EXAMPLES.get(product, "does it show an error code, or stop working?")
    return f"I'm sorry to hear that. What is happening with your {product}? For example, {example}"


CLARIFY = (
    "I can help with order status, delivery tracking, VoxCom policies, product questions, vouchers, "
    "or a request for a customer service representative. What would you like to do?"
)
CLARIFY_LIMIT = (
    "I am still not sure how to help with that. I can record a request for a customer service representative "
    "if you like."
)
PROVIDER_UNAVAILABLE = "I am having trouble right now. Please try again in a moment."


def resume_question(topic: str) -> str:
    """UC-23: after an interruption, ask whether to resume the paused topic."""
    return f"Would you like to go back to {topic}?"


RESUME_DECLINED = "Okay, I have set that aside. Nothing on your orders was changed."
RESUME_NOTHING = "There is nothing to go back to."


# --- formatting helpers -------------------------------------------------------------------
def vnd(amount: int) -> str:
    return f"{int(amount):,} VND"


def month_day(day: date) -> str:
    return f"{day.strftime('%B')} {day.day}"


def weekday_month_day(day: date) -> str:
    return f"{day.strftime('%A')}, {month_day(day)}"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


__all__ = [name for name in dir() if not name.startswith("_") and name not in {"annotations", "date"}]
