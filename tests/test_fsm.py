"""Every valid and forbidden order transition (guide §11.6, design D§6.5), plus the write-lifecycle machine."""

from __future__ import annotations

import itertools

import pytest

from voxcare.domain.enums import OrderStatus, PendingActionStatus
from voxcare.domain.rules import (
    ACTION_TRANSITIONS,
    MAX_DELIVERY_REATTEMPTS,
    ForbiddenTransition,
    can_advance_action,
    can_transition,
    cancellable,
    is_dispatched,
)

VALID = {
    ("pending_confirmation", "confirmed"), ("pending_confirmation", "cancelled"),
    ("confirmed", "processing"), ("confirmed", "cancelled"),
    ("processing", "ready_for_pickup"), ("processing", "cancelled"),
    ("ready_for_pickup", "shipped"),
    ("shipped", "out_for_delivery"),
    ("out_for_delivery", "delivered"), ("out_for_delivery", "delivery_failed"),
    ("delivery_failed", "out_for_delivery"), ("delivery_failed", "returned"),
    ("delivered", "return_requested"),
    ("return_requested", "returned"),
}  # fmt: skip
FORBIDDEN = {("shipped", "cancelled"), ("out_for_delivery", "cancelled")}
STATUSES = [status.value for status in OrderStatus]


@pytest.mark.parametrize(("source", "target"), sorted(VALID))
def test_valid_transition(source, target):
    assert can_transition(source, target) is True


@pytest.mark.parametrize(("source", "target"), sorted(FORBIDDEN))
def test_forbidden_transition_raises(source, target):
    with pytest.raises(ForbiddenTransition):
        can_transition(source, target)


def test_every_other_pair_is_invalid():
    for source, target in itertools.product(STATUSES, STATUSES):
        if (source, target) in VALID or (source, target) in FORBIDDEN:
            continue
        assert can_transition(source, target) is False, (source, target)


@pytest.mark.parametrize("terminal", ["cancelled", "returned"])
def test_terminal_states_have_no_exit(terminal):
    assert not any(can_transition(terminal, target) for target in STATUSES)


def test_redelivery_is_capped_at_three_reattempts():
    for attempts in range(MAX_DELIVERY_REATTEMPTS):
        assert can_transition("delivery_failed", "out_for_delivery", reattempts=attempts)
    assert not can_transition("delivery_failed", "out_for_delivery", reattempts=MAX_DELIVERY_REATTEMPTS)
    assert can_transition("delivery_failed", "returned", reattempts=MAX_DELIVERY_REATTEMPTS)


def test_cancel_helpers():
    assert cancellable("confirmed") and cancellable("processing") and cancellable("pending_confirmation")
    assert not cancellable("shipped") and not cancellable("delivered")
    assert is_dispatched("shipped") and is_dispatched("out_for_delivery") and not is_dispatched("confirmed")


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError):
        can_transition("Shipped", "delivered")  # the capitalised form is not a status value (D§3.5)


def test_write_lifecycle_machine():
    P = PendingActionStatus
    expected = {
        (P.PENDING, P.SUBMITTED), (P.PENDING, P.CANCELLED), (P.PENDING, P.FAILED),
        (P.SUBMITTED, P.EXECUTING), (P.SUBMITTED, P.CANCELLED), (P.SUBMITTED, P.FAILED),
        (P.EXECUTING, P.COMPLETED), (P.EXECUTING, P.CANCELLED), (P.EXECUTING, P.FAILED),
    }  # fmt: skip
    for source, target in itertools.product(P, P):
        assert can_advance_action(source, target) is ((source, target) in expected), (source, target)
    assert all(not ACTION_TRANSITIONS[state] for state in (P.COMPLETED, P.CANCELLED, P.FAILED))
