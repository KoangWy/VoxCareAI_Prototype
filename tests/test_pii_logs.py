"""NFR-05: zero plain-text PII in logs, audit payloads, stored history and handoff packets (guide §15)."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
from pathlib import Path

from support import P, ScriptedPlanner, boundary_for, make_runtime

from voxcare.observability.logging import JsonFormatter
from voxcare.security import pii
from voxcare.security.pii import MaskingLogFilter

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pii_log_scan", ROOT / "scripts" / "pii_log_scan.py")
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)

PII_TEXT = (
    "Hi, I am Lê Minh Anh, my phone is 0901 234 567 or +84 900 000 001, email anh@example.com, "
    "deliver to 123 Nguyen Trai, Ward 2, District 5, Ho Chi Minh City please. Where is ORD-1002?"
)


def test_masker_masks_phones_names_addresses_and_keeps_business_codes():
    masked = pii.mask(PII_TEXT)
    for raw in ("0901 234 567", "+84 900 000 001", "anh@example.com", "Nguyen Trai", "Lê Minh Anh"):
        assert raw not in masked
    assert "+8490***01" in masked and "090***67" in masked
    assert "ORD-1002" in masked  # business identifiers are never masked
    assert pii.mask("driver Tran Quoc Dung called") == "driver [REDACTED_NAME] called"  # no diacritics (ASR)
    assert pii.mask("+8490***45") == "+8490***45"  # stored-masked values pass through
    assert pii.mask("key 550e8400-e29b-41d4-a716-446655440007") == "key 550e8400-e29b-41d4-a716-446655440007"


def test_phone_number_extraction_for_verification():
    assert pii.find_phone_number("It is 0900 000 001") == "+84900000001"
    assert pii.find_phone_number("plus eight four nine zero zero zero zero zero zero zero one") == "+84900000001"
    assert pii.find_phone_number("my order is ORD-1002") is None


def test_logs_are_masked_and_the_scanner_finds_nothing():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(MaskingLogFilter())
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("voxcare.test.pii")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("caller said: %s", PII_TEXT)
        logger.warning("ws path /ws/voice?token=eyJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiU0VTIn0.c2lnbmF0dXJlMTIzNDU")
        logger.info("key sk-proj-abcdefghijklmnopqrstuvwxyz")
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert [json.loads(line)["level"] for line in output.splitlines()] == ["INFO", "WARNING", "INFO"]
    assert list(scanner.scan_lines(output.splitlines())) == []
    assert list(scanner.scan_lines([PII_TEXT]))  # and the scanner does catch the unmasked text


async def test_a_full_conversation_persists_no_pii():
    runtime = await make_runtime("memory")
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-1002"})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    await boundary.handle_turn(session.session_id, PII_TEXT)
    await boundary.handle_turn(session.session_id, "Let me speak to a human please, call me on 0901 234 567")
    stored = json.dumps(
        {
            "history": await boundary.get_conversation_history(session.session_id),
            "audit": await boundary.session_audit(session.session_id),
            "trace": await boundary.session_trace(session.session_id),
            "handoffs": await runtime.store.list_handoffs(session.session_id),
            "model_saw": planner.seen,
        },
        ensure_ascii=False,
        default=str,
    )
    assert list(scanner.scan_lines(stored.splitlines())) == []
    for raw in ("0901 234 567", "anh@example.com", "Nguyen Trai", "Lê Minh Anh"):
        assert raw not in stored


def _uvicorn_record(name, msg, args, **extra):
    record = logging.LogRecord(name, logging.INFO, __file__, 1, msg, args, None)
    record.__dict__.update(extra)
    return record


def test_uvicorn_console_lines_keep_their_arguments():
    from uvicorn.logging import AccessFormatter, DefaultFormatter

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiU0VTIn0.c2lnbmF0dXJlMTIzNDU"
    startup = _uvicorn_record(
        "uvicorn.error",
        "Started server process [%d]",
        (4242,),
        color_message="Started server process [\x1b[36m%d\x1b[0m]",
    )
    running = _uvicorn_record(
        "uvicorn.error",
        "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
        ("http", "127.0.0.1", 8000),
        color_message="Uvicorn running on \x1b[1m%s://%s:%d\x1b[0m (Press CTRL+C to quit)",
    )
    access = _uvicorn_record(
        "uvicorn.access", '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5000", "GET", f"/ws/voice?token={jwt}", "1.1", 101)
    )
    flt = MaskingLogFilter()
    assert flt.filter(startup) and flt.filter(running) and flt.filter(access)
    for use_colors in (True, False):
        console = DefaultFormatter("%(levelprefix)s %(message)s", use_colors=use_colors)
        assert "4242" in console.format(startup) and "%d" not in console.format(startup)
        assert "127.0.0.1:8000" in console.format(running).replace("\x1b[1m", "").replace("\x1b[0m", "")
        line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=use_colors).format(
            access
        )
        assert "GET /ws/voice?token=[REDACTED_TOKEN] HTTP/1.1" in line.replace("\x1b[1m", "").replace("\x1b[0m", "")
        assert jwt not in line
