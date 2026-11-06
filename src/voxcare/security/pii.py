"""PII masking (guide §15, NFR-05): phone numbers, addresses and names, before anything is persisted or logged.

Approach (implementation choice, guide §15):

- **Phones:** Vietnamese (``0xxxxxxxxx``, ``+84…``) and other E.164 numbers keep the
  ``+8490***45`` style: country code and two leading digits, ``***``, the last two digits.
- **Addresses:** a pattern list (number + street + Ward/District, label-anchored
  phrases) plus every address string known from the customer and shipment rows.
- **Names:** a dictionary built from the customer and driver rows, matched with and
  without Vietnamese diacritics (ASR output has none), plus label-anchored introductions.

The generic masker of :mod:`voxcare.privacy` (emails, card numbers, SSN-like and
date-of-birth shapes) runs first. Business identifiers (ORD-####, SKUs, tracking
codes, voucher codes, ticket ids) are never masked. Masking is an engineering
simulation of safe access boundaries, not statutory compliance with Decree
356/2025/NĐ-CP (D§8.3).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from .. import privacy

_PHONE = re.compile(r"(?<![\w*+])(\+?\d[\d .()-]{7,16}\d)(?![\w*])")
_VN_ADDRESS = re.compile(
    r"\b\d{1,5}[A-Za-z]?(?:/\d{1,4})?\s+(?:[A-ZÀ-Ỹa-zà-ỹ][\wÀ-ỹ'.-]*\s*){1,4},\s*"
    r"(?:Ward|Phường|Phuong|P\.)\s*[\wÀ-ỹ ]{1,20}"
    r"(?:,\s*(?:District|Quận|Quan|Q\.)\s*[\wÀ-ỹ ]{1,20})?"
    r"(?:,\s*(?:Ho Chi Minh City|HCMC|Hà Nội|Ha Noi|Hanoi|Da Nang|Đà Nẵng|"
    r"[A-ZÀ-Ỹ][\wÀ-ỹ]+(?:\s[A-ZÀ-Ỹ][\wÀ-ỹ]+){0,3}))?",
)
_MASK_NAME = "[REDACTED_NAME]"
_MASK_ADDRESS = "[REDACTED_ADDRESS]"
# Already-masked phone shapes (+8490***45) must pass through untouched.
_MASKED_PHONE = re.compile(r"\+?\d{2,6}\*{3}\d{2}")


def _fold(text: str) -> str:
    """Lower-case and strip diacritics (đ → d) for dictionary matching."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").lower()


def mask_phone_number(raw: str) -> str:
    """``+84900000001`` → ``+8490***01``; ``0901234567`` → ``090***67``."""
    digits = re.sub(r"\D", "", raw)
    if raw.strip().startswith("+"):
        country = "84" if digits.startswith("84") else digits[: max(1, len(digits) - 10)]
        rest = digits[len(country) :]
        return f"+{country}{rest[:2]}***{rest[-2:]}"
    return f"{digits[:3]}***{digits[-2:]}"


def _looks_like_phone(raw: str) -> bool:
    """E.164 or Vietnamese shapes only; other long digit runs fall to the generic masker afterwards."""
    stripped = raw.strip()
    digits = re.sub(r"\D", "", stripped)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return False  # ISO date
    if stripped.startswith("+"):
        return 8 <= len(digits) <= 15
    if digits.startswith("0"):
        return len(digits) in (10, 11)  # Vietnamese national format
    return digits.startswith("84") and len(digits) == 11


def normalize_phone(raw: str) -> str | None:
    """E.164 form of a Vietnamese or international number heard in a transcript, or ``None``."""
    digits = re.sub(r"\D", "", raw)
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return f"+{digits}"
    if digits.startswith("84") and len(digits) == 11:
        return f"+{digits}"
    if digits.startswith("0") and len(digits) == 10:
        return f"+84{digits[1:]}"
    return None


_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "plus": "+",
}  # fmt: skip


def find_phone_number(text: str) -> str | None:
    """The first phone number in a typed or spoken utterance, in E.164 form (verification sub-dialogue)."""
    for match in _PHONE.finditer(text):
        phone = normalize_phone(match.group(1))
        if phone:
            return phone
    tokens = re.findall(r"[a-z]+|\d+|\+", text.lower())
    spoken = "".join(
        _SPOKEN_DIGITS.get(token, token if token.isdigit() or token == "+" else " ")  # noqa: S105 (a "+" sign)
        for token in tokens
    )
    for run in re.findall(r"\+?\d{9,15}", spoken):
        phone = normalize_phone(run)
        if phone:
            return phone
    return None


class PiiMasker:
    """Dictionary- and pattern-based masker; ``learn`` adds names/addresses from the stores."""

    def __init__(self, names: Iterable[str] = (), addresses: Iterable[str] = ()) -> None:
        self._names: set[str] = set()
        self._addresses: set[str] = set()
        self._name_pattern: re.Pattern[str] | None = None
        self.learn(names=names, addresses=addresses)

    def learn(self, *, names: Iterable[str] = (), addresses: Iterable[str] = ()) -> None:
        self._names.update(name.strip() for name in names if name and name.strip())
        self._addresses.update(address.strip() for address in addresses if address and address.strip())
        variants: set[str] = set()
        for name in self._names:
            variants.add(re.escape(_fold(name)))
        self._name_pattern = (
            re.compile(r"\b(?:" + "|".join(sorted(variants, key=len, reverse=True)) + r")\b") if variants else None
        )

    def mask(self, text: Any) -> str:
        value = str(text)
        if not value:
            return value
        for address in sorted(self._addresses, key=len, reverse=True):
            value = re.sub(re.escape(address), _MASK_ADDRESS, value, flags=re.IGNORECASE)
        value = _VN_ADDRESS.sub(_MASK_ADDRESS, value)
        value = self._mask_names(value)
        value = self._mask_phones(value)
        return privacy.mask_pii(value)

    def _mask_names(self, value: str) -> str:
        if self._name_pattern is None:
            return value
        folded = _fold(value)
        if len(folded) != len(value):  # NFD folding changed the length; fall back to exact names
            for name in sorted(self._names, key=len, reverse=True):
                value = re.sub(re.escape(name), _MASK_NAME, value, flags=re.IGNORECASE)
            return value
        spans = [match.span() for match in self._name_pattern.finditer(folded)]
        for start, end in reversed(spans):
            value = value[:start] + _MASK_NAME + value[end:]
        return value

    @staticmethod
    def _mask_phones(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            if _MASKED_PHONE.fullmatch(raw) or not _looks_like_phone(raw):
                return raw
            return mask_phone_number(raw)

        return _PHONE.sub(replace, value)

    def mask_payload(self, payload: Any) -> Any:
        """Recursively mask every string in a JSON-like payload (keys are kept)."""
        if isinstance(payload, str):
            return self.mask(payload)
        if isinstance(payload, Mapping):
            return {key: self.mask_payload(item) for key, item in payload.items()}
        if isinstance(payload, list | tuple):
            return [self.mask_payload(item) for item in payload]
        return payload


masker = PiiMasker()
_bootstrapped = False


def _bootstrap() -> None:
    """Learn the fixture names and addresses once, so text masked before the runtime starts is covered too."""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    from ..storage import seed  # local import: the seed module is plain data

    masker.learn(
        names=[row["full_name"] for row in seed.CUSTOMERS] + [row["driver_name"] for row in seed.SHIPMENTS],
        addresses=[row["default_address"] for row in seed.CUSTOMERS] + [row["shipping_address"] for row in seed.ORDERS],
    )


def mask(text: Any) -> str:
    _bootstrap()
    return masker.mask(text)


def mask_payload(payload: Any) -> Any:
    _bootstrap()
    return masker.mask_payload(payload)


_CREDENTIAL = re.compile(r"(?i)(token=|bearer\s+|voxcare-session-token[\"':\s]+)[A-Za-z0-9._~+/=-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_API_KEY = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")


def mask_credentials(text: str) -> str:
    """Session tokens and API keys never reach a log line."""
    text = _CREDENTIAL.sub(lambda m: f"{m.group(1)}[REDACTED_TOKEN]", text)
    text = _JWT.sub("[REDACTED_TOKEN]", text)
    return _API_KEY.sub("[REDACTED_KEY]", text)


def _scrub(text: str) -> str:
    return mask(mask_credentials(text))


# Loggers whose formatters unpack ``record.args`` themselves (uvicorn's AccessFormatter expects exactly
# client, method, path, HTTP version and status): their string arguments are masked in place.
_POSITIONAL_ARGS_LOGGERS = ("uvicorn.access",)


class MaskingLogFilter(logging.Filter):
    """Masks every log record's message (PII and credentials) before any handler formats it.

    Usually the message is rendered, masked and stored with no arguments left. Two uvicorn
    conventions are kept intact: the access logger's positional arguments (masked one by one)
    and the ``color_message`` extra that the coloured console formatter renders instead of
    ``msg`` (rendered with the same arguments, then masked).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        try:
            message = record.getMessage()
        except Exception:  # a malformed record is logged as-is rather than dropped
            return True
        color = record.__dict__.get("color_message")
        if isinstance(color, str):
            try:
                color = color % args if args else color
            except Exception:  # fall back to the plain rendering
                color = message
            record.color_message = _scrub(color)
        if record.name in _POSITIONAL_ARGS_LOGGERS and isinstance(args, tuple):
            record.args = tuple(_scrub(arg) if isinstance(arg, str) else arg for arg in args)
            return True
        record.msg = _scrub(message)
        record.args = ()
        return True


__all__ = [
    "MaskingLogFilter",
    "PiiMasker",
    "find_phone_number",
    "mask",
    "mask_credentials",
    "mask_payload",
    "mask_phone_number",
    "masker",
    "normalize_phone",
]
