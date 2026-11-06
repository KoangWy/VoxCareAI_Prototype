"""Deterministic intent detectors for the sub-dialogues the model never drives.

The model plans *read* tool calls. Four kinds of turn are decided by rules
instead, because the design fixes their wording or forbids model control:

- confirmations and withdrawals of a previewed write (writes are never exposed to the LLM);
- the open-policy-decision topics, whose fallback line is the **only** permitted answer (D§8.4);
- the bounded after-sales diagnosis (D§9.3), whose questions are verbatim;
- explicit requests for a person, and the UC-23 resume question.

All patterns run on the masked transcript (implementation choice).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from ..domain.rules import product_categories_in

ORDER_CODE_IN_TEXT = re.compile(r"\bORD-\d{4}\b", re.I)
TRACKING_CODE_IN_TEXT = re.compile(r"\b(?:GHN|GHTK|VTP)-VN-\d{4,}\b", re.I)

YES = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|correct|confirm(?:ed)?|i\s+confirm|please\s+do|go\s+ahead|do\s+it|ok(?:ay)?|"
    r"that'?s\s+right|right|absolutely|please)\b",
    re.I,
)
NO = re.compile(
    r"^\s*(?:no|nope|nah|don'?t|do\s+not|stop|wait|never\s*mind|not\s+now|leave\s+it|cancel\s+(?:that|it)|"
    r"keep\s+it)\b",
    re.I,
)
WITHDRAW = re.compile(
    r"\b(?:wait|stop|hold\s+on|don'?t\s+(?:cancel|change|move|do)|do\s+not\s+(?:cancel|change|move|do)|"
    r"never\s*mind|withdraw|call\s+(?:it|that)\s+off|keep\s+(?:it|my\s+order))\b",
    re.I,
)
HUMAN = re.compile(
    r"\b(?:speak|talk|connect|put\s+me\s+through|transfer\s+me|let\s+me\s+speak)\b.{0,40}\b(?:human|person|agent|"
    r"representative|supervisor|manager|someone|somebody|operator|staff)\b|\b(?:human|real\s+person|live\s+agent|"
    r"representative|supervisor)\s*(?:please|now|right\s+now)?\s*[!.?]*\s*$",
    re.I,
)
RMA_STATUS = re.compile(
    r"\b(?:status|update|progress|where\s+is)\b.{0,40}\b(?:rma|repair|warranty\s+claim|service\s+(?:ticket|request)|"
    r"technician\s+visit|replacement\s+request)\b",
    re.I,
)

CANCEL = re.compile(r"\bcancel(?:l?ing|l?ed|lation)?\b", re.I)
REINSTATE = re.compile(r"\b(?:reinstate|re-?activate|restore|un-?cancel|undo\s+(?:the\s+)?cancel\w*)\b", re.I)
RESCHEDULE = re.compile(
    r"\b(?:reschedule|re-schedule|move|push|postpone|change)\b.{0,40}\b(?:deliver\w*|slot|window|date|day|to\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next))\b|\breschedule\b",
    re.I,
)
ADDRESS_CHANGE = re.compile(r"\b(?:change|update|correct)\b.{0,30}\b(?:delivery\s+)?address\b", re.I)

OPD_REFUND_TURNAROUND = re.compile(
    r"\b(?:when|how\s+long|how\s+many\s+days|how\s+soon)\b.{0,50}\brefund|\brefund\b.{0,40}\b(?:take|arrive|come\s+back|"
    r"processed|reach|land|turnaround)\b",
    re.I,
)
OPD_VOLUNTARY_RETURN = re.compile(
    r"\b(?:change(?:d)?\s+(?:of\s+|my\s+)mind|don'?t\s+(?:like|want|need)\s+it|no\s+longer\s+(?:want|need)|voluntar\w+|"
    r"not\s+defective)\b",
    re.I,
)
OPD_COSMETIC = re.compile(
    r"\b(?:scratch(?:ed|es)?|dent(?:ed|s)?|cosmetic|scuff(?:ed)?|(?:box|packaging|carton)(?:\s+\w+){0,2}?\s+"
    r"(?:damaged|torn|crushed|broken|dented)|damaged\s+(?:box|packaging|carton))\b",
    re.I,
)
OPD_BEYOND_30KM = re.compile(
    r"\b(?:3[1-9]|[4-9]\d|\d{3,})\s*(?:km|kms|kilomet\w+)\b|\b(?:beyond|more\s+than|over|outside)\s+30\s*(?:km|kilomet\w+)",
    re.I,
)
RETURN_WORDS = re.compile(r"\b(?:return|send\s+(?:it\s+)?back|refund)\w*\b", re.I)

DIAG_EXCHANGE = re.compile(r"\b(?:exchange|replace\w*|swap|1[- ]for[- ]1|one[- ]for[- ]one|new\s+one)\b", re.I)
DIAG_RETURN = re.compile(r"\b(?:return|refund|money\s+back)\b", re.I)
DIAG_REPAIR = re.compile(r"\b(?:repair|fix|warranty|technician)\b", re.I)
WASHER_CONTEXT = re.compile(r"\b(?:washer|washing\s+machine|laundry|drum|spin)\b", re.I)

# Repair intake: a fault or repair request about a product, without a recognised symptom yet.
REPAIR_REQUEST = re.compile(
    r"\b(?:repair\w*|fix(?:ed|ing)?|broken|breaks?\s+down|not\s+working|(?:does\s*n[o']?t|won'?t)\s+(?:work|start|"
    r"turn\s+on|spin|drain)|stopped\s+working|faulty|malfunction\w*)\b",
    re.I,
)
# Policy vocabulary: used only when the model proposed no tool (a fallback, never ahead of the model).
POLICY_TOPIC = {
    "WARRANTY": re.compile(r"\b(?:warrant\w*|guarantee\w*)\b", re.I),
    "RETURNS": re.compile(r"\b(?:returns?|returning|refunds?|exchanges?|replacements?|send\s+(?:it\s+)?back)\b", re.I),
    "SHIPPING": re.compile(
        r"\b(?:shipping|freight)\b|\b(?:charges?|pay|costs?|fees?|free)\b.{0,25}\bdeliver\w*|"
        r"\bdeliver\w*\b.{0,25}\b(?:fees?|costs?|charges?|free|policy|polic(?:y|ies))\b",
        re.I,
    ),
    "INSTALLATION": re.compile(r"\b(?:install\w*|mounting|set\s*up)\b", re.I),
}
POLICY_WORD = re.compile(r"\b(?:polic(?:y|ies)|rules?|terms|conditions)\b", re.I)
VOUCHER_LIST = re.compile(
    r"\b(?:vouchers?|promo(?:tion)?s?|discounts?|coupons?|deals?|offers?)\b.{0,40}\b(?:available|current\w*|have|"
    r"any|list|now|running)\b|\b(?:what|which|any)\b.{0,25}\b(?:vouchers?|promo(?:tion)?s?|discounts?|coupons?)\b",
    re.I,
)
VOUCHER_CODE_IN_TEXT = re.compile(r"\b[A-Z][A-Z0-9]+_[A-Z0-9_]+\b")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip


def policy_category(text: str) -> str | None:
    """The policy category a question names (ALL when it names several), or ``None``."""
    topics = [name for name, pattern in POLICY_TOPIC.items() if pattern.search(text)]
    if not topics:
        return "ALL" if POLICY_WORD.search(text) else None
    return topics[0] if len(topics) == 1 else "ALL"


_VAGUE_FILLER = re.compile(
    r"\b(?:what|which|are|is|the|your|you|voxcom'?s?|me|tell|about|all|can|could|please|i|want|to|know|would|like|"
    r"do|have|any|show|list|give|explain|polic(?:y|ies)|rules?|terms|conditions|and|of|store|shop)\b|[?.!,']",
    re.I,
)


def vague_policy_question(text: str) -> bool:
    """'Tell me your policies': a policy word and nothing that names a topic, product or brand."""
    if not POLICY_WORD.search(text) or any(pattern.search(text) for pattern in POLICY_TOPIC.values()):
        return False
    if product_categories_in(text):
        return False
    return not _VAGUE_FILLER.sub(" ", text).strip()


def repair_request(text: str) -> bool:
    """A repair/fault request about a product that is not a policy question (those go to the policy tool)."""
    if not REPAIR_REQUEST.search(text) or POLICY_WORD.search(text) or POLICY_TOPIC["WARRANTY"].search(text):
        return False
    return bool(product_categories_in(text))


def order_code_in(text: str) -> str | None:
    match = ORDER_CODE_IN_TEXT.search(text)
    return match.group(0).upper() if match else None


def opd_topic(text: str) -> str | None:
    """The OPD row a question falls under, if any (D§8.4). OPD-03 is decided in the write flow."""
    if OPD_REFUND_TURNAROUND.search(text):
        return "OPD-02"
    if OPD_VOLUNTARY_RETURN.search(text) and RETURN_WORDS.search(text):
        return "OPD-01"
    if OPD_COSMETIC.search(text):
        return "OPD-05"
    if OPD_BEYOND_30KM.search(text):
        return "OPD-06"
    return None


def write_intent(text: str) -> str | None:
    """Cancel | Reinstate | Reschedule | AddressChange, for a request naming (or implying) an order."""
    if REINSTATE.search(text):
        return "Reversal"
    if ADDRESS_CHANGE.search(text):
        return "AddressChange"
    if RESCHEDULE.search(text) and not CANCEL.search(text):
        return "Reschedule"
    if CANCEL.search(text) and not re.search(r"\bcancel\w*\s+(?:policy|fee|fees|rules?)\b", text, re.I):
        return "Cancel"
    return None


def parse_target_date(text: str, today: date) -> date | None:
    """ "next Monday" / "Monday" / "tomorrow" / "October 19" / "19 October" / "2026-10-19" against the anchor."""
    lowered = text.lower()
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if iso:
        return date(int(iso[1]), int(iso[2]), int(iso[3]))
    if re.search(r"\btomorrow\b", lowered):
        return today + timedelta(days=1)
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", lowered):
            delta = (index - today.weekday()) % 7 or 7
            return today + timedelta(days=delta)
    for index, name in enumerate(_MONTHS, start=1):
        match = re.search(
            rf"\b{name}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{name}\b", lowered
        )
        if match:
            day = int(match.group(1) or match.group(2))
            candidate = date(today.year, index, day)
            return candidate if candidate >= today else date(today.year + 1, index, day)
    match = re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", lowered)
    if match:
        day = int(match.group(1))
        candidate = date(today.year, today.month, day)
        if candidate <= today:
            month = today.month % 12 + 1
            candidate = date(today.year + (today.month == 12), month, day)
        return candidate
    return None


_ADDRESS_IS = re.compile(
    r"\b(?:new\s+)?(?:delivery\s+)?address\s+(?:is|will\s+be|should\s+be)\s+(.+?)(?:,?\s+(?:please|and|so|could|can)\b|"
    r"[.?!]*$)",
    re.I,
)
ADDRESS_SHAPE = re.compile(
    r"\d.*\b(?:st(?:reet)?|road|rd|ward|district|phuong|quan|lane|alley|hem|duong)\b|\d+\s+\w+.*,", re.I
)


def new_address_in(text: str) -> str | None:
    """The new address in "...address to X" or "my (new) address is X, please...", else ``None``."""
    match = re.search(r"\baddress\b.{0,20}?\bto\s+(.+?)[.?!]*$", text, re.I) or _ADDRESS_IS.search(text)
    return match.group(1).strip(" ,") if match else None


def diagnosis_choice(text: str) -> str | None:
    """UC-14 exchange, UC-15 return and refund, UC-16 repair under warranty."""
    if DIAG_EXCHANGE.search(text):
        return "EXCHANGE"
    if DIAG_RETURN.search(text):
        return "RETURN_REFUND"
    if DIAG_REPAIR.search(text):
        return "REPAIR"
    return None


__all__ = [
    "ADDRESS_CHANGE",
    "CANCEL",
    "HUMAN",
    "NO",
    "POLICY_TOPIC",
    "REPAIR_REQUEST",
    "RMA_STATUS",
    "VOUCHER_CODE_IN_TEXT",
    "VOUCHER_LIST",
    "WASHER_CONTEXT",
    "WITHDRAW",
    "YES",
    "diagnosis_choice",
    "new_address_in",
    "opd_topic",
    "order_code_in",
    "parse_target_date",
    "policy_category",
    "repair_request",
    "vague_policy_question",
    "write_intent",
]
