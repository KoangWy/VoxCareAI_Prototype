"""Regex log scanner for NFR-05: zero plain-text PII (and zero credentials) in logs.

    uv run python scripts/pii_log_scan.py server.log [more.log ...]
    some-command 2>&1 | uv run python scripts/pii_log_scan.py -

Exits 1 and prints each finding (file, line number, kind) when a phone number,
email address, fixture customer name or address, session token or API key
appears unmasked. The patterns mirror ``voxcare.security.pii``; masked shapes
such as ``+8490***45`` and ``[REDACTED_*]`` are allowed.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxcare.storage import seed  # noqa: E402

PATTERNS = {
    "phone": re.compile(
        r"(?<![\w*])(?:\+84|0)(?:[\s.-]?\d){9,10}(?![\w*])|(?<![\w*])\+\d(?:[\s().-]?\d){7,14}(?![\w*])"
    ),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),
}


def _fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").lower()


NAMES = {_fold(row["full_name"]) for row in seed.CUSTOMERS} | {_fold(row["driver_name"]) for row in seed.SHIPMENTS}
ADDRESSES = {_fold(row["default_address"]) for row in seed.CUSTOMERS}


def scan_lines(lines: Iterable[str]) -> Iterator[tuple[int, str, str]]:
    for number, line in enumerate(lines, start=1):
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                yield number, kind, match.group(0)
        folded = _fold(line)
        for name in NAMES:
            if name in folded:
                yield number, "name", name
        for address in ADDRESSES:
            if address in folded or address.split(",")[0] in folded:
                yield number, "address", address


def main(argv: list[str]) -> int:
    sources = argv or ["-"]
    findings = 0
    for source in sources:
        lines = sys.stdin if source == "-" else Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
        for number, kind, value in scan_lines(lines):
            findings += 1
            print(f"{source}:{number}: unmasked {kind}: {value[:6]}…")
    print(f"{findings} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
