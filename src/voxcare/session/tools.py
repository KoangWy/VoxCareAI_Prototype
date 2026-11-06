"""Spoken order-code normalisation for ASR transcripts, plus the tool registry re-export.

The model-visible registry, the validator and the middleware live in
:mod:`voxcare.middleware`; this module keeps the ASR-side helper that turns
"o r d one zero zero two" into ``ORD-1002`` before a transcript is planned.
"""

from __future__ import annotations

import re

from ..domain.services.orders import ORDER_CODE, normalize_order_code
from ..middleware.registry import PHASE1_REGISTRY as TOOL_NAMES
from ..middleware.registry import TOOLS

_UNIT_WORDS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}  # fmt: skip
_TEEN_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS_WORDS = {"twenty": 2, "thirty": 3, "forty": 4, "fifty": 5, "sixty": 6, "seventy": 7, "eighty": 8, "ninety": 9}
# "o r d", "ord", and the small-model mishearings "or d" / "or rd" / "o rd" / "oh are dee";
# or the word "order" (optionally "number"/"code") directly followed by four spoken digits.
_SPOKEN_CODE_PREFIX = re.compile(
    r"\b(?:(?P<order>order(?:\s+(?:number|code))?)|o\s*r\s*d|or\s+r?d|o\s+rd|(?:o|oh)\s+(?:r|are|ar)\s+(?:d|dee))(?![a-z])",
    re.IGNORECASE,
)
_SPOKEN_TOKEN = re.compile(r"[\s,.-]*([a-z]+|[0-9]+)", re.IGNORECASE)


def _is_number_token(token: str) -> bool:
    low = token.lower()
    return token.isdigit() or low in _UNIT_WORDS or low in _TEEN_WORDS or low in _TENS_WORDS


def _spoken_digits(text: str, start: int) -> tuple[str, int] | None:
    """Exactly four spoken digits from ``start`` ("one zero zero two", "ten oh two", "forty five sixty seven")."""
    digits, pos = "", start
    while len(digits) < 4:
        match = _SPOKEN_TOKEN.match(text, pos)
        if match is None:
            break
        token = match.group(1).lower()
        if token.isdigit():
            digits += token
        elif token in _UNIT_WORDS:
            digits += str(_UNIT_WORDS[token])
        elif token in _TEEN_WORDS:
            digits += str(_TEEN_WORDS[token])
        elif token in _TENS_WORDS:
            unit = _SPOKEN_TOKEN.match(text, match.end())
            unit_word = unit.group(1).lower() if unit else ""
            if unit_word in _UNIT_WORDS and _UNIT_WORDS[unit_word] > 0 and unit_word not in {"o", "oh"}:
                digits += f"{_TENS_WORDS[token]}{_UNIT_WORDS[unit_word]}"
                match = unit
            else:
                digits += f"{_TENS_WORDS[token]}0"
        else:
            break
        pos = match.end()
    if len(digits) != 4:
        return None
    following = _SPOKEN_TOKEN.match(text, pos)
    if following is not None and _is_number_token(following.group(1)):
        return None  # a longer number is not an order code
    return digits, pos


def normalize_spoken_order_codes(text: str) -> tuple[str, tuple[str, ...]]:
    """Rewrite spoken or misheard order codes in an ASR transcript to ``ORD-####``.

    Deterministic and conservative: exactly four digits must follow the prefix,
    so "order two TVs" or a five-digit number is left alone. Returns the new
    text and the codes it produced.
    """
    parts: list[str] = []
    codes: list[str] = []
    pos = 0
    for prefix in _SPOKEN_CODE_PREFIX.finditer(text):
        if prefix.start() < pos:
            continue
        spoken = _spoken_digits(text, prefix.end())
        if spoken is None:
            continue
        digits, end = spoken
        code = f"ORD-{digits}"
        parts.append(text[pos : prefix.start()])
        parts.append(f"{prefix.group('order')} {code}" if prefix.group("order") else code)
        codes.append(code)
        pos = end
    if not codes:
        return text, ()
    parts.append(text[pos:])
    return "".join(parts), tuple(codes)


# Domain terms the small offline ASR model mishears or spells out (implementation choice, voice turns only).
# Each rewrite is narrow: a known VoxCom term, in the shapes Vosk produces for it.
_VOUCHER_SPOKEN = re.compile(
    r"\b(?:vox|box|fox|vocs|vax)\s*-?\s*com(?P<tet>\s+(?:tet|tat|tets|ted))?(?:\s+underscore)?\s+"
    r"(?:five\s+hundred|500)\s*(?:k|kay|okay|thousand)\b",
    re.IGNORECASE,
)
_ERROR_CODE_SPOKEN = re.compile(
    r"\b(?P<lead>error(?:\s+code)?|code|shows?|showing|displays?|displaying)\s+(?:an?\s+)?(?:e|ee|he)\s*(?:four|for|4)\b",
    re.IGNORECASE,
)
_INCH_SPOKEN = re.compile(
    r"\b(?P<tens>forty|fifty|sixty|seventy|eighty)[\s-]+(?P<unit>[a-z]+)[\s-]+inch(?:es)?\b", re.IGNORECASE
)
_TERM_REWRITES = (
    (re.compile(r"\b(?:samson|sam\s+sung|sam\s+song|samsong)\b", re.IGNORECASE), "Samsung"),
    (re.compile(r"\b(?:l\s+g|el\s+gee?|elle\s+gee)\b", re.IGNORECASE), "LG"),
)


def normalize_spoken_terms(text: str) -> tuple[str, tuple[str, ...]]:
    """Rewrite misheard VoxCom terms in an ASR transcript; returns the text and "heard → term" notes."""
    notes: list[str] = []

    def note(heard: str, term: str) -> str:
        notes.append(f"“{heard}” → {term}")
        return term

    text = _VOUCHER_SPOKEN.sub(lambda m: note(m.group(0), "VOXCOM_TET_500K" if m.group("tet") else "VOXCOM_500K"), text)
    text = _ERROR_CODE_SPOKEN.sub(
        lambda m: f"{m.group('lead')} {note(m.group(0)[len(m.group('lead')) :].strip(), 'E4')}", text
    )

    def inches(match: re.Match[str]) -> str:
        unit = _UNIT_WORDS.get(match.group("unit").lower())
        if unit is None or match.group("unit").lower() in {"o", "oh"}:
            return match.group(0)
        return note(match.group(0), f"{_TENS_WORDS[match.group('tens').lower()]}{unit}-inch")

    text = _INCH_SPOKEN.sub(inches, text)
    for pattern, term in _TERM_REWRITES:
        text = pattern.sub(lambda m, term=term: note(m.group(0), term), text)
    return text, tuple(notes)


__all__ = [
    "ORDER_CODE",
    "TOOLS",
    "TOOL_NAMES",
    "normalize_order_code",
    "normalize_spoken_order_codes",
    "normalize_spoken_terms",
]
