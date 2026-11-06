"""Dialogue manager behaviours outside the 17 scenarios: OPD lines, offers, clarification, flags, guards."""

from __future__ import annotations

import pytest
from support import P, ScriptedPlanner, boundary_for, make_runtime

from voxcare.domain import copy
from voxcare.session.render import eta_guard, inquiry_topic, write_guard


async def _boundary(planner=None, **overrides):
    runtime = await make_runtime("memory", **overrides)
    return boundary_for(runtime, planner or ScriptedPlanner()), runtime


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("Who pays the return shipping if I changed my mind about the TV?", copy.OPD_01_RETURN_FREIGHT),
        ("How long will my refund take to arrive?", copy.OPD_02_REFUND_TURNAROUND),
        ("The box was damaged and the TV has a small scratch", copy.OPD_05_COSMETIC),
        ("Can you deliver to an address 45 km away?", copy.OPD_06_BEYOND_30KM),
    ],
)
async def test_open_policy_decisions_get_only_their_fallback_line(text, line):
    planner = ScriptedPlanner([[P("retrieve_policy_clauses", {"query": text, "category": "ALL"})]])
    boundary, _ = await _boundary(planner)
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, text)
    assert result.assistant_text == line and result.outcome == "OPD_FALLBACK"
    assert planner.seen == []  # the model is not consulted on an undecided topic


async def test_cancel_on_a_shipped_order_gets_opd_03_and_records_the_request():
    boundary, runtime = await _boundary()
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, "Please cancel ORD-1002")
    assert result.assistant_text == copy.OPD_03_CANCEL_SHIPPED
    order = await runtime.repository.get_order_by_code("ORD-1002")
    assert (order["status"], order["version"]) == ("shipped", 3)  # no state change
    assert [a["outcome_code"] for a in await runtime.repository.list_agent_activity(session.session_id)] == ["QUEUED"]


async def test_handoff_offer_after_a_policy_miss_is_accepted_with_yes():
    query = "Do you accept trade-ins for 10-year-old microwave ovens with cash back?"
    planner = ScriptedPlanner([[P("retrieve_policy_clauses", {"query": query, "category": "ALL"})]])
    boundary, runtime = await _boundary(planner)
    session = await boundary.create_session()
    await boundary.handle_turn(session.session_id, query)
    accepted = await boundary.handle_turn(session.session_id, "Yes please")
    assert accepted.outcome == "HANDOFF_REQUEST_RECORDED"
    row = (await runtime.repository.list_audit(session.session_id))[-1]
    assert row["request_payload_json"]["trigger_reason"] == "LOW_CONFIDENCE_THRESHOLD"


async def test_clarification_limit_offers_a_handoff():
    boundary, runtime = await _boundary()
    session = await boundary.create_session()
    replies = [await boundary.handle_turn(session.session_id, "hmm") for _ in range(3)]
    assert [r.assistant_text for r in replies] == [copy.CLARIFY, copy.CLARIFY, copy.CLARIFY_LIMIT]
    accepted = await boundary.handle_turn(session.session_id, "yes")
    row = (await runtime.repository.list_audit(session.session_id))[-1]
    assert accepted.outcome == "HANDOFF_REQUEST_RECORDED"
    assert row["request_payload_json"]["trigger_reason"] == "CLARIFICATION_LIMIT_EXCEEDED"


async def test_service_timeout_speaks_the_failure_line_and_offers_an_agent():
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-1001"})]])
    boundary, runtime = await _boundary(planner)
    runtime.middleware.orders.timeout_s = 0.001
    runtime.repository.delay_s = 0.05
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, "Check ORD-1001")
    assert result.assistant_text == copy.ORDER_TIMEOUT and result.outcome == "SERVICE_TIMEOUT"


async def test_flags_off_use_the_matrix_lines():
    planner = ScriptedPlanner(
        [
            [P("lookup_order_details", {"order_code": "ORD-1001"})],
            [P("lookup_order_details", {"order_code": "ORD-2002"})],
        ]
    )
    boundary, _ = await _boundary(planner, fr_auth_01=False, fr_auth_02=False)
    unverified = await boundary.create_session(verified=False)
    first = await boundary.handle_turn(unverified.session_id, "Check ORD-1001")
    assert first.assistant_text == copy.ORDER_UNAUTHENTICATED
    verified = await boundary.create_session()
    second = await boundary.handle_turn(verified.session_id, "Check ORD-2002")
    assert second.assistant_text == copy.ORDER_OUT_OF_SCOPE


async def test_verification_with_an_unknown_phone_number():
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-1001"})]])
    boundary, _ = await _boundary(planner)
    session = await boundary.create_session(verified=False)
    await boundary.handle_turn(session.session_id, "Check ORD-1001")
    first = await boundary.handle_turn(session.session_id, "0911 222 333")
    second = await boundary.handle_turn(session.session_id, "0911 222 334")
    assert first.assistant_text == copy.VERIFY_FAILED
    assert second.assistant_text == f"{copy.VERIFY_FAILED} {copy.OFFER_HANDOFF}"
    assert (await boundary.get_session(session.session_id)).customer_id is None


async def test_write_preview_expiry_and_decline():
    boundary, runtime = await _boundary()
    session = await boundary.create_session()
    preview = await boundary.handle_turn(session.session_id, "I want to cancel ORD-1001")
    assert preview.outcome == "CONFIRMATION_REQUESTED"
    declined = await boundary.handle_turn(session.session_id, "No, leave it")
    assert declined.assistant_text == copy.withdrawn_pending("ORD-1001")
    again = await boundary.handle_turn(session.session_id, "Cancel ORD-1001")
    record = await runtime.store.get_session(session.session_id)
    await runtime.store.client.delete(f"pending_action:{record['pending_action_id']}")  # the 300 s TTL elapsed
    expired = await boundary.handle_turn(session.session_id, "Yes")
    assert again.outcome == "CONFIRMATION_REQUESTED" and expired.assistant_text == copy.WRITE_PREVIEW_EXPIRED


async def test_unverified_caller_cannot_preview_a_write():
    boundary, _ = await _boundary()
    session = await boundary.create_session(verified=False)
    result = await boundary.handle_turn(session.session_id, "Cancel ORD-1001")
    assert result.assistant_text == copy.VERIFY_ACCOUNT_PROMPT


async def test_rma_status_question_records_a_handoff():
    boundary, runtime = await _boundary()
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, "What is the status of my repair ticket?")
    assert result.assistant_text == copy.RMA_STATUS_HANDOFF
    assert await runtime.repository.list_agent_activity(session.session_id)


async def test_diagnosis_meta_question_does_not_consume_the_budget():
    boundary, _ = await _boundary()
    session = await boundary.create_session()
    await boundary.handle_turn(session.session_id, "My washing machine shows error E4")
    meta = await boundary.handle_turn(session.session_id, "Sorry, what do you mean?")
    assert meta.assistant_text == copy.DIAGNOSIS_Q1


def test_output_guards():
    assert eta_guard("It should arrive tomorrow.", null_eta=True) == copy.SHIPMENT_NULL_ETA
    assert eta_guard("It should arrive tomorrow.", null_eta=False) == "It should arrive tomorrow."
    assert write_guard("ORD-1001 is cancelled.", execution_confirmed=False) == copy.WRITE_STILL_PROCESSING_INTERIM
    assert write_guard("ORD-1001 is cancelled.", execution_confirmed=True) == "ORD-1001 is cancelled."
    assert write_guard(copy.RESUME_DECLINED, execution_confirmed=False) == copy.RESUME_DECLINED


def test_inquiry_topic_slot():
    assert inquiry_topic("Do you accept trade-ins for 10-year-old microwave ovens with cash back?") == (
        "trade-ins for 10-year-old microwave ovens with cash back"
    )
    assert inquiry_topic("Do you sell gift cards?") == "sell gift cards"
