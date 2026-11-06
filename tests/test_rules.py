"""Pure business rules (domain/rules.py): voucher fixtures, return window, reversal eligibility, hashes, guards."""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from voxcare.domain import rules
from voxcare.storage.seed import PROMOTIONS

TODAY = date(2026, 10, 14)
PROMO = next(row for row in PROMOTIONS if row["voucher_code"] == "VOXCOM_500K")
TET = next(row for row in PROMOTIONS if row["voucher_code"] == "VOXCOM_TET_500K")


def test_voucher_fixture_evaluations():
    quote = rules.evaluate_voucher(PROMO, "VOXCOM_500K", 11_490_000, "TV_AV", TODAY)
    assert (quote.valid, quote.discount_vnd, quote.capped_at_vnd, quote.outcome) == (True, 500_000, 500_000, "SUCCESS")
    washer = rules.evaluate_voucher(PROMO, "VOXCOM_500K", 8_990_000, "WASHING_MACHINE", TODAY)
    assert (washer.valid, washer.discount_vnd, washer.outcome, washer.reason) == (
        False,
        0,
        "INVALID_INPUT",
        "CATEGORY_MISMATCH",
    )


@pytest.mark.parametrize(
    ("value", "category", "today", "reason"),
    [
        (9_999_999, "TV_AV", TODAY, "BELOW_MINIMUM"),
        (12_000_000, "KITCHEN", TODAY, "CATEGORY_MISMATCH"),
        (12_000_000, "TV_AV", date(2026, 11, 1), "NOT_IN_VALIDITY"),
        (12_000_000, "TV_AV", date(2026, 9, 30), "NOT_IN_VALIDITY"),
    ],
)
def test_voucher_conditions(value, category, today, reason):
    quote = rules.evaluate_voucher(PROMO, "VOXCOM_500K", value, category, today)
    assert not quote.valid and quote.discount_vnd == 0 and quote.reason == reason


def test_voucher_boundaries_and_retired_code():
    assert rules.evaluate_voucher(PROMO, "VOXCOM_500K", 10_000_000, "TV_AV", date(2026, 10, 31)).valid
    assert rules.evaluate_voucher(PROMO, "VOXCOM_500K", 10_000_000, "TV_AV", date(2026, 10, 1)).valid
    retired = rules.evaluate_voucher(TET, "VOXCOM_TET_500K", 20_000_000, "TV_AV", date(2026, 2, 1))
    assert not retired.valid and retired.reason == "RETIRED"  # retired even inside its old validity window
    assert rules.evaluate_voucher(None, "NOPE_123", 20_000_000, "TV_AV", TODAY).reason == "UNKNOWN_CODE"


def test_return_window_is_thirty_calendar_days():
    delivered = date(2026, 10, 2)
    assert rules.within_return_window(delivered, TODAY)
    assert rules.within_return_window(delivered, date(2026, 11, 1))
    assert not rules.within_return_window(delivered, date(2026, 11, 2))


def test_diagnosis_symptom_and_fault():
    assert rules.detect_symptom("My washer stops during spin cycle and displays error E4.") == "SPIN_CYCLE_ERROR_E4"
    assert rules.detect_symptom("it shows error code e-4") == "SPIN_CYCLE_ERROR_E4"
    assert rules.detect_symptom("the screen flickers") is None
    assert rules.classify_fault("SPIN_CYCLE_ERROR_E4") == "MOTOR_DRIVE_FAULT"
    assert rules.classify_fault("UNKNOWN") is None


def test_rma_invariant():
    assert rules.rma_invariant("o1", "o1", "CUST-001", "CUST-001").ok
    assert rules.rma_invariant("o2", "o1", "CUST-001", "CUST-001").error == "INVALID_INPUT"
    assert rules.rma_invariant("o1", "o1", "CUST-002", "CUST-001").error == "AUTH_SCOPE_VIOLATION"


def test_reinstate_cancel_eligibility_defaults_to_deny():
    ok = dict(
        order_status="cancelled",
        shipment_status=None,
        inventory_reserved=True,
        refund_dispatched=False,
        version_current=True,
        oms_validates=True,
    )
    assert rules.reinstate_cancel_allowed(**ok).allowed
    assert rules.reinstate_cancel_allowed(**{**ok, "refund_dispatched": True}).reason == "REFUND_DISPATCHED"
    assert rules.reinstate_cancel_allowed(**{**ok, "inventory_reserved": False}).reason == "INVENTORY_RELEASED"
    assert rules.reinstate_cancel_allowed(**{**ok, "shipment_status": "in_transit"}).reason == "SHIPMENT_PROGRESSED"
    assert rules.reinstate_cancel_allowed(**{**ok, "order_status": "confirmed"}).reason == "ORDER_NOT_CANCELLED"
    assert not rules.reinstate_cancel_allowed(**{**ok, "refund_dispatched": None}).allowed  # missing input → deny


def test_deterministic_reschedule_reversal():
    now = datetime(2026, 10, 16, 7, 30, tzinfo=UTC)
    base = dict(
        now=now,
        target_slot_start=datetime(2026, 10, 17, 9, tzinfo=UTC),
        existing_window_start=datetime(2026, 10, 16, 9, tzinfo=UTC),
        cutoff=datetime(2026, 10, 16, 8, tzinfo=UTC),
        order_status="shipped",
        shipment_status="in_transit",
    )
    assert rules.reverse_reschedule_allowed(**base, carrier_result="CONFIRMED").allowed
    assert not rules.reverse_reschedule_allowed(**base, carrier_result="PENDING").allowed
    assert not rules.reverse_reschedule_allowed(**base, carrier_result="REFUSED").allowed
    late = {**base, "now": datetime(2026, 10, 16, 8, 0, tzinfo=UTC)}
    assert rules.reverse_reschedule_allowed(**late, carrier_result="CONFIRMED").reason == "PAST_CUTOFF"
    assert (
        rules.reverse_reschedule_allowed(**{**base, "cutoff": None}, carrier_result="CONFIRMED").reason
        == "MISSING_INPUT"
    )


def test_address_change_reversal():
    assert rules.reverse_address_change_allowed(
        order_status="confirmed", shipment_status=None, oms_confirms=True
    ).allowed
    assert not rules.reverse_address_change_allowed(
        order_status="shipped", shipment_status="in_transit", oms_confirms=True
    ).allowed


def test_preview_hash_and_confirmation_steps():
    first = rules.preview_hash({"a": 1, "b": [2, 3]})
    assert first == rules.preview_hash({"b": [2, 3], "a": 1}) and len(first) == 64
    assert first != rules.preview_hash({"a": 2, "b": [2, 3]})
    assert [rules.confirmation_steps_valid(n) for n in (0, 1, 2, 3)] == [False, True, True, False]


@pytest.mark.parametrize(
    "text",
    ["arrives tomorrow", "soon", "next week", "in two days", "on October 15", "2026-10-15", "by Monday", "15/10"],
)
def test_date_words_are_detected(text):
    assert rules.mentions_date(text)


def test_null_eta_line_has_no_date_word():
    from voxcare.domain.copy import SHIPMENT_NULL_ETA

    assert not rules.mentions_date(SHIPMENT_NULL_ETA)


def test_applicability_terms():
    assert rules.product_categories_in("Can you install my kitchen oven?") == ("KITCHEN",)
    assert rules.product_categories_in("Samsung refrigerator compressor") == ("REFRIGERATOR",)
    assert rules.brands_in("What warranty does an LG washer have?") == ("LG",)


def test_rules_module_imports_no_provider_or_storage_code():
    """NFR-07: zero lines of domain/rules.py change on a provider swap; it imports nothing outside the domain."""
    tree = ast.parse(Path(rules.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {
        "__future__",
        "hashlib",
        "json",
        "re",
        "collections.abc",
        "dataclasses",
        "datetime",
        "typing",
        "enums",
    }
