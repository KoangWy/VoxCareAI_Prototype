"""Voice rendering helpers: rewrite a grounded answer so text-to-speech reads it well.

The screen keeps the full grounded text. What the caller *hears* is derived by
deterministic rewrites only, so no fact is added: ``2026-10-15`` becomes
"October 15, 2026", ``VND`` becomes "dong", ``ORD-1002`` is spelled out, and
clause ids and SKU parentheticals are dropped. :mod:`.render` builds the voice
variant of every answer with :func:`speakable`; policy answers are shortened
there to one sentence per cited clause plus a pointer to the transcript.
"""

from __future__ import annotations

import calendar
import re

_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")  # VN-market clauses write day/month/year
_POLICY_IDS = re.compile(r"\s*\((?:POL-[A-Z]+-\d{3})(?:,\s*POL-[A-Z]+-\d{3})*\)")
_SKU_PAREN = re.compile(r"\s*\((?:[A-Z]{2}-[A-Z0-9]+(?:-[A-Z0-9]+)*)\)")
_ORDER_CODE = re.compile(r"\bORD-(\d{4})\b")
_TICKET = re.compile(r"\bTICK-CS-(\d{4})\b")
_VND_PER = re.compile(r"\bVND/([a-z]+)\b")
_VND = re.compile(r"\bVND\b")
_UNDERSCORE_CODE = re.compile(r"\b([A-Z][A-Z0-9]*)_([A-Z0-9]+)\b")


def _date(year: str, month: str, day: str) -> str | None:
    m, d = int(month), int(day)
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{calendar.month_name[m]} {d}, {int(year)}"


def _digits(code: str) -> str:
    return " ".join(_DIGIT_WORDS[int(char)] for char in code)


def speakable(text: str) -> str:
    """Rewrite codes, dates and currency into forms TTS pronounces; meaning is unchanged."""
    spoken = _POLICY_IDS.sub("", text)
    spoken = _SKU_PAREN.sub("", spoken)
    spoken = _ISO_DATE.sub(lambda m: _date(m[1], m[2], m[3]) or m[0], spoken)
    spoken = _DMY_DATE.sub(lambda m: _date(m[3], m[2], m[1]) or m[0], spoken)
    spoken = _ORDER_CODE.sub(lambda m: "O R D " + _digits(m[1]), spoken)
    spoken = _TICKET.sub(lambda m: "T I C K, C S, " + _digits(m[1]), spoken)
    spoken = _VND_PER.sub(r"dong per \1", spoken)
    spoken = _VND.sub("dong", spoken)
    spoken = _UNDERSCORE_CODE.sub(r"\1 \2", spoken)
    spoken = spoken.replace("—", ",")
    return re.sub(r"\s{2,}", " ", spoken).strip()


def first_sentence(text: str) -> str:
    return _SENTENCE_END.split(text.strip(), maxsplit=1)[0]


def spoken_reply(result) -> str:  # type: ignore[no-untyped-def]
    """What TTS says for a turn: the rendered voice variant, else the speakable transcript text."""
    spoken = getattr(result, "spoken_text", None)
    if spoken:
        return spoken
    return speakable(getattr(result, "assistant_text", "") or "")


__all__ = ["first_sentence", "speakable", "spoken_reply"]
