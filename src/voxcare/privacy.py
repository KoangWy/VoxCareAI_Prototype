"""Small deterministic redaction boundary for text sent to external turn services.

Pattern-based masking with explicit limits — NOT a general PII detector:

- Catches label-anchored identifiers (email/phone/card/introductions/addresses)
  plus a closed set of unanchored shapes: dotted/spaced/dashed card numbers
  (Luhn-checked, 13-19 digits, to limit false positives), US-style DOB dates,
  SSN-like numbers, number+street free-form addresses, and +country phone
  formats. Short digit runs (order totals, years, clause numbers) are ignored.
- NEVER masks VoxCare reference codes: order codes (ORD-####), shipment ids
  (SHIP-####), clause ids (POL-XXX-###), SKUs (TV-/WM-/RF-…), voucher codes,
  tracking codes, or session ids — these are business identifiers the voice
  flow must keep verbatim.
- Non-Latin scripts, passport numbers, and free-form names without an
  introduction label are out of scope and pass through unchanged.

Masking is applied at LLM/Jev/TTS/handoff edges; stored history, live
transcripts, broadcasts and audit payloads keep their own boundaries (see
session._safe_arguments). Never treat a [REDACTED_*] token as evidence that
no raw text persists elsewhere.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<![A-Z0-9Plus])(?:\+?\d[\d ().-]{6,}\d)(?![A-Z0-9])",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_US_DATE = re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(19|20)\d{2}\b")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_NAME_LABEL = re.compile(
    r"\b(my name is|name is|full name is|I'm|I am|this is)\s+"
    r"([A-ZÀ-Þ][A-Za-zÀ-ÿ'-]*(?:\s+[A-ZÀ-Þ][A-Za-zÀ-ÿ'-]*){0,2})"
)
_ADDRESS_LABEL = re.compile(r"\b(my address is|address is|I live at|deliver (?:it )?to)\s+([^,.!?;]+)", re.IGNORECASE)
_STREET_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-zÀ-ÿ0-9'.-]+\s+){0,3}"
    r"(?:street|st|avenue|ave|road|rd|lane|ln|boulevard|blvd|drive|dr|district|ward|phuong|quan)\b"
    r"[^,.!?;]*",
    re.IGNORECASE,
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ .-]*){13,19}(?!\d)")
_REFERENCE_CODES = re.compile(
    r"\b(?:ORD|SHIP|SES-VOX|ACT|RMA|EVT|ITEM|TICK-CS|CHK-POL|POL-[A-Z]+|TV|WM|RF)-[A-Z0-9-]+\b"
    r"|\bGHN-VN-\d+\b|\bGHTK-VN-\d+\b|\bVTP-VN-\d+\b|\bVOXCOM_[A-Z0-9]+\b"
)


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum gate so plain long numbers are not treated as card numbers."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = ord(char) - 48
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _mask_cards(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        core = raw.rstrip(" .-")
        tail = raw[len(core) :]
        digits = re.sub(r"\D", "", core)
        if not 13 <= len(digits) <= 19:
            return raw
        if not core[0].isdigit() or not core[-1].isdigit():
            return raw
        if _REFERENCE_CODES.search(core):
            return raw
        if not _luhn_ok(digits):
            return raw
        return "[REDACTED_NUMBER]" + tail

    return _CARD_CANDIDATE.sub(replace, text)


def _mask_phones(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        if _ISO_DATE.fullmatch(raw):
            return raw
        digits = re.sub(r"\D", "", raw)
        if raw.lstrip().startswith("+"):
            if not 8 <= len(digits) <= 15:
                return raw
        elif not 9 <= len(digits) <= 12:
            return raw
        return "[REDACTED_PHONE]"

    return _PHONE.sub(replace, text)


# Private-use sentinels: no masking pattern matches them, so a protected code
# survives every pass and is restored verbatim afterwards.
_SENTINEL_OPEN, _SENTINEL_CLOSE = "\ue000", "\ue001"
_SENTINEL = re.compile(f"{_SENTINEL_OPEN}([a-z]+){_SENTINEL_CLOSE}")


def _sentinel(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(97 + remainder) + letters
    return f"{_SENTINEL_OPEN}{letters}{_SENTINEL_CLOSE}"


def mask_pii(text: str) -> str:
    """Mask recognizable direct identifiers while preserving surrounding intent text."""
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        token = _sentinel(len(protected))
        protected[token] = match.group(0)
        return token

    masked = _REFERENCE_CODES.sub(protect, str(text))
    masked = _NAME_LABEL.sub(lambda match: f"{match.group(1)} [REDACTED]", masked)
    masked = _ADDRESS_LABEL.sub(lambda match: f"{match.group(1)} [REDACTED_ADDRESS]", masked)
    masked = _EMAIL.sub("[REDACTED_EMAIL]", masked)
    masked = _mask_cards(masked)
    masked = _US_DATE.sub("[REDACTED_DOB]", masked)
    masked = _SSN.sub("[REDACTED_SSN]", masked)
    masked = _STREET_ADDRESS.sub("[REDACTED_ADDRESS]", masked)
    masked = _mask_phones(masked)
    return _SENTINEL.sub(lambda match: protected.get(match.group(0), ""), masked)
