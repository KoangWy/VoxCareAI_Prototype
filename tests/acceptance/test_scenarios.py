"""The 17 acceptance scenarios of the P1 build guide §18 (design D§13), at the dialogue level.

Level (a): text in, the LLM replaced by a deterministic scripted adapter; each test
asserts the tool calls, audit rows and state changes, plus the spoken line verbatim.
Scenarios 6 and 11-16 drive the guarded-write library against the fake VoxCom OMS
(the writes are future-only and never exposed to the model). Every test runs on the
in-memory stores and, when ``VOXCARE_TEST_DATABASE_URL`` is set, on PostgreSQL too.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from support import BACKENDS, P, ScriptedPlanner, actions, boundary_for, drop_runtime, make_runtime

from voxcare.domain import copy
from voxcare.domain.services.handoff import HandoffContext
from voxcare.storage.seed import K1, order_id_for
from voxcare.turns import JevJudgment, TurnController, TurnInput

S1_LINE = (
    "Order ORD-1002 is in transit with GHN. Estimated delivery date: 2026-10-15. "
    "Your account also shows ORD-1003, delivered on October 2."
)
S2_LINE = (
    "To access your order information, I need to verify your registered account. Please provide the phone number "
    "used during checkout, or check the verification SMS sent to you."
)
S3_LINE = (
    "For data privacy and security, I can only share details for orders on your verified account. "
    "Order ORD-2002 is not associated with your phone number."
)
S7_LINE = "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND. With VOXCOM_500K the Samsung is 10,990,000 VND."
S8_LINE = (
    "Your claim is inside the 30-day window, but it is eligible only if a VoxCom technician confirms the fault. "
    "I have recorded your request — no RMA ticket has been created yet."
)
S9_LINE = (
    "I have recorded your request to speak with a customer service representative under reference ticket "
    "TICK-CS-8821, along with the full details and summary of our conversation so you won't need to repeat "
    "yourself. A support specialist will review your file and contact you via phone or SMS during business hours. "
    "Is there anything else I can assist you with right now?"
)
S17_LINE = "The carrier has not provided an estimated delivery date yet. You will get an SMS once it is scheduled."
SHIPMENT_FIELDS = {"tracking_code", "carrier_name", "estimated_delivery_date", "driver_name", "driver_phone_masked"}


@pytest.fixture(params=BACKENDS)
async def runtime(request):
    rt = await make_runtime(request.param)
    yield rt
    await drop_runtime(rt)


@pytest.fixture
async def fake_oms_runtime(request):
    rt = await make_runtime("memory", write_mode="fake_oms")
    yield rt
    await drop_runtime(rt)


async def audit_of(boundary, session_id):
    return await boundary.runtime.repository.list_audit(session_id)


# --------------------------------------------------------------------------------------------- T01


async def test_scenario_01_verified_lookup_track_and_history(runtime):
    planner = ScriptedPlanner(
        [
            [
                P("lookup_order_details", {"order_code": "ORD-1002"}),
                P("track_shipment_status", {"order_code": "ORD-1002", "tracking_code": None}),
                P("list_customer_orders", {}),
            ]
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    result = await boundary.handle_turn(
        session.session_id, "Can you check the delivery status of my order ORD-1002? And my earlier purchases?"
    )
    assert result.assistant_text == S1_LINE
    lookup = next(item for item in result.tool_outcomes if item.tool_name == "lookup_order_details")
    assert lookup.status == "SUCCESS"
    assert not SHIPMENT_FIELDS & set(lookup.record["order"])  # the lookup contract returns { order } only
    assert lookup.record["order"]["requested_delivery_time"] == "2026-10-16 09:00-12:00"
    history = next(item for item in result.tool_outcomes if item.tool_name == "list_customer_orders")
    assert {order["order_code"] for order in history.record["orders"]} == {"ORD-1001", "ORD-1002", "ORD-1003"}
    rows = await audit_of(boundary, session.session_id)
    assert actions(rows) == ["TOOL_INVOCATION"] * 3
    assert [row["request_payload_json"]["tool_name"] for row in rows] == [
        "lookup_order_details",
        "track_shipment_status",
        "list_customer_orders",
    ]
    assert all(row["outcome_status"] == "SUCCESS" and row["mutation_idempotency_key"] is None for row in rows)


async def test_scenario_01_two_turn_history_branch(runtime):
    planner = ScriptedPlanner(
        [
            [
                P("lookup_order_details", {"order_code": "ORD-1002"}),
                P("track_shipment_status", {"order_code": "ORD-1002", "tracking_code": None}),
            ],
            [P("list_customer_orders", {})],
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    first = await boundary.handle_turn(session.session_id, "Can you check the delivery status of my order ORD-1002?")
    second = await boundary.handle_turn(session.session_id, "What about my earlier purchases?")
    assert f"{first.assistant_text} {second.assistant_text}" == S1_LINE


async def test_scenario_02_unauthenticated_before_any_orders_read(runtime):
    planner = ScriptedPlanner([[P("track_shipment_status", {"order_code": "ORD-1001", "tracking_code": None})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=False)
    runtime.repository.read_log.clear()
    result = await boundary.handle_turn(session.session_id, "Track order ORD-1001.")
    assert result.assistant_text == S2_LINE
    assert result.outcome == "NEEDS_VERIFICATION"
    assert not {"orders", "shipment", "shipment_event", "order_item"} & set(runtime.repository.read_log)  # DB spy
    track = result.tool_outcomes[0]
    assert track.status == "UNAUTHENTICATED_SESSION" and track.record["http"] == 401
    rows = await audit_of(boundary, session.session_id)
    assert [(r["action_name"], r["action_status"], r["outcome_status"]) for r in rows] == [
        ("TOOL_INVOCATION", "Denied", "DENIED")
    ]


async def test_scenario_02_verification_sub_dialogue_binds_customer(runtime):
    planner = ScriptedPlanner(
        [
            [P("track_shipment_status", {"order_code": "ORD-1001", "tracking_code": None})],
            [
                P("lookup_order_details", {"order_code": "ORD-1001"}),
                P("track_shipment_status", {"order_code": "ORD-1001", "tracking_code": None}),
            ],
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=False)
    await boundary.handle_turn(session.session_id, "Track order ORD-1001.")
    result = await boundary.handle_turn(session.session_id, "It is 0900 000 001")
    assert result.binding_changed and result.customer_id == "CUST-001"
    assert result.assistant_text == f"{copy.VERIFY_SUCCESS} {S17_LINE}"
    assert "0900 000 001" not in str(await boundary.get_conversation_history(session.session_id))


async def test_scenario_03_cross_customer_order_is_refused_and_audited(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-2002"})]]))
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    result = await boundary.handle_turn(session.session_id, "Please check order ORD-2002.")
    assert result.assistant_text == S3_LINE
    envelope = result.tool_outcomes[0].record
    assert envelope == {
        "outcome": "ERROR",
        "error_code": "OUT_OF_SCOPE",
        "http": 403,
        "message": "The order is not on the verified account.",
    }
    rows = await audit_of(boundary, session.session_id)
    violation = next(row for row in rows if row["action_name"] == "AUTH_SCOPE_VIOLATION")
    assert (violation["actor"], violation["action_status"], violation["outcome_status"]) == (
        "System",
        "Denied",
        "DENIED",
    )
    assert violation["response_payload_json"] == {"error_code": "AUTH_SCOPE_VIOLATION", "details_concealed": True}
    assert violation["order_id"] == order_id_for("ORD-2002")  # G-4: the hex FK, the code in the request
    assert violation["request_payload_json"]["order_code"] == "ORD-2002"
    leaked = str(result.tool_outcomes) + result.assistant_text + str(rows)
    for secret in ("processing", "CUST-002", "Phạm", "Le Loi", "COD", "UNPAID"):
        assert secret not in leaked


async def test_scenario_17_validator_blocks_null_id_then_null_eta(runtime):
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": ""})], []])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    runtime.repository.read_log.clear()
    first = await boundary.handle_turn(session.session_id, "Where is my package?")
    assert first.assistant_text == "Could you please tell me your order code?"
    assert first.outcome == "NEEDS_ORDER_ID"
    assert first.tool_outcomes[0].record["error_code"] == "INVALID_INPUT"
    assert runtime.repository.read_log == []  # the tool was not invoked
    assert await audit_of(boundary, session.session_id) == []
    second = await boundary.handle_turn(session.session_id, "ORD-1001")
    assert second.assistant_text == S17_LINE
    track = next(item for item in second.tool_outcomes if item.tool_name == "track_shipment_status")
    assert track.record["shipment"]["estimated_delivery_date"] is None
    assert track.record["events"] == []
    assert not re.search(r"\b(tomorrow|today|soon|next week|in two days|October|Monday)\b", second.assistant_text, re.I)


# --------------------------------------------------------------------------------------------- T02


async def test_scenario_04_grounded_policy_answer_cites_both_clauses(runtime):
    query = (
        "What is the warranty period for a Samsung refrigerator compressor and can I return it if it has a "
        "manufacturing defect?"
    )
    boundary = boundary_for(
        runtime, ScriptedPlanner([[P("retrieve_policy_clauses", {"query": query, "category": "ALL"})]])
    )
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, query)
    envelope = result.tool_outcomes[0].record
    assert envelope["outcome"] == "SUCCESS"
    assert (envelope["corpus_id"], envelope["is_fictional"], envelope["clause_version"]) == (
        "VOXCOM-DEMO-POLICY-V1",
        True,
        "1.0-demo",
    )
    cited = [clause["clause_id"] for clause in envelope["clauses"]]
    assert cited[0] == "POL-WAR-001" and "POL-RET-001" in cited
    assert "POL-WAR-002" not in cited and "POL-WAR-003" not in cited  # applicability: brand Samsung
    assert "Warranty Policy Clause 3.1" in result.assistant_text and "Return Policy Clause 4.1" in result.assistant_text
    assert copy.POLICY_TECHNICIAN_CONDITION in result.assistant_text  # conditional wording
    assert "you are eligible" not in result.assistant_text.lower()
    assert result.spoken_text and "Clause 3.1" in result.spoken_text and "Clause 4.1" in result.spoken_text


async def test_scenario_05_insufficient_evidence_is_http_200_below_threshold(runtime):
    query = "Do you accept trade-ins for 10-year-old microwave ovens with cash back?"
    boundary = boundary_for(
        runtime, ScriptedPlanner([[P("retrieve_policy_clauses", {"query": query, "category": "ALL"})]])
    )
    session = await boundary.create_session()
    result = await boundary.handle_turn(session.session_id, query)
    envelope = result.tool_outcomes[0].record
    assert envelope["error_code"] == "INSUFFICIENT_EVIDENCE" and envelope["http"] == 200
    assert envelope["highest_similarity_score"] < envelope["threshold_required"] == runtime.threshold
    assert envelope["corpus_id"] == "VOXCOM-DEMO-POLICY-V1" and envelope["is_fictional"] is True
    text = result.assistant_text
    assert text.startswith("VoxCom does not currently have a published policy regarding trade-ins")
    assert text.endswith("Would you like me to connect you with our customer support desk for further assistance?")
    assert not re.search(r"\d[\d,]*\s*VND|\bdays?\b|\bfee\b|\brefund\b", text)
    assert result.evidence == ()


# --------------------------------------------------------------------------------------------- T05


async def test_scenario_07_compare_and_voucher(runtime):
    planner = ScriptedPlanner(
        [
            [
                P("search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": "55 inch"}),
                P("compare_products", {"skus": ["TV-SAM-55", "TV-LG-55"]}),
                P("apply_voucher", {"voucher_code": "VOXCOM_500K", "order_value_vnd": 11490000, "category": "TV_AV"}),
            ]
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839205")
    result = await boundary.handle_turn(
        session.session_id, "Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies."
    )
    assert result.assistant_text == S7_LINE
    rows = await audit_of(boundary, session.session_id)
    assert [row["request_payload_json"]["tool_name"] for row in rows] == [
        "search_catalog",
        "compare_products",
        "apply_voucher",
    ]
    voucher = result.tool_outcomes[2].record
    assert voucher == {
        "outcome": "SUCCESS",
        "valid": True,
        "discount_vnd": 500000,
        "capped_at_vnd": 500000,
        "is_fictional": True,
    }
    compare = result.tool_outcomes[1].record
    lg = next(item for item in compare["products"] if item["sku"] == "TV-LG-55")
    assert "refresh_rate_hz" not in lg["attributes"]  # never invent an unrecorded attribute
    assert compare["not_recorded"] == {"TV-LG-55": ["refresh_rate_hz"]}
    assert "stock_quantity" not in str(result.tool_outcomes)  # availability state only, never counts
    assert {row["availability"] for row in result.tool_outcomes[0].record["results"]} == {"in_stock"}


# --------------------------------------------------------------------------------------------- T06


async def test_scenario_08_bounded_diagnosis_records_request_without_rma(runtime):
    boundary = boundary_for(runtime)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839203")
    sid = session.session_id
    turn1 = await boundary.handle_turn(sid, "My washer stops during spin cycle and displays error E4.")
    turn2 = await boundary.handle_turn(sid, "Yes, it drains first.")
    turn3 = await boundary.handle_turn(sid, "Yes, the drum is stiff.")
    final = await boundary.handle_turn(sid, "I would like the 1-for-1 exchange.")
    assert turn1.assistant_text == copy.DIAGNOSIS_Q1
    assert turn2.assistant_text == copy.DIAGNOSIS_Q2
    assert turn3.assistant_text.startswith(copy.DIAGNOSIS_CONDITIONAL)  # ≤ 3 diagnostic turns
    assert final.assistant_text == S8_LINE
    assert "RMA-" not in final.assistant_text and "TICK-CS" not in final.assistant_text
    rows = await audit_of(boundary, sid)
    policy = next(row for row in rows if row["request_payload_json"].get("tool_name") == "retrieve_policy_clauses")
    assert policy["request_payload_json"]["category"] == "RETURNS"
    assert policy["response_payload_json"]["clauses"][0][0] == "POL-RET-001"
    handoff = next(row for row in rows if row["action_name"] == "HANDOFF_REQUEST_RECORDED")
    assert handoff["request_payload_json"]["trigger_reason"] == "COMPLEX_EXCEPTION"
    assert handoff["request_payload_json"]["priority"] == "NORMAL"
    assert not any("create_rma_request" in str(row) for row in rows)
    packet = (await runtime.store.list_handoffs(sid))[0]
    history = packet["context_payload_json"]["symptom_history"]
    assert history[0]["symptom_code"] == "SPIN_CYCLE_ERROR_E4"
    assert history[-1]["diagnosis_class"] == "MOTOR_DRIVE_FAULT"


async def test_scenario_09_handoff_records_request_activity_and_packet(runtime):
    boundary = boundary_for(runtime)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    result = await boundary.handle_turn(session.session_id, "Let me speak to a human supervisor right now!")
    assert result.assistant_text == S9_LINE
    assert not re.search(r"\b(transferr|connecting you|connected you|call you back at)\w*", result.assistant_text, re.I)
    envelope = result.tool_outcomes[0].record
    assert envelope == {
        "outcome": "SUCCESS",
        "transfer_status": "HANDOFF_REQUEST_RECORDED",
        "handoff_id": "HND-20261014-0942-8821",
        "ticket_number": "TICK-CS-8821",
        "summary_recorded": True,
        "business_hours_notice": "Customer support operating hours: 08:00 - 21:00 Monday through Sunday.",
    }
    rows = await audit_of(boundary, session.session_id)
    assert [(r["action_name"], r["actor"], r["action_status"], r["outcome_status"]) for r in rows] == [
        ("HANDOFF_REQUEST_RECORDED", "Bot", "Success", "SUCCESS")
    ]
    assert rows[0]["response_payload_json"]["agent_activity_state"] == "queued"
    assert rows[0]["response_payload_json"]["agent_id"] is None
    activity = await runtime.repository.list_agent_activity(session.session_id)
    assert [(a["activity_code"], a["outcome_code"]) for a in activity] == [("HANDOFF_REQUEST_RECORDED", "QUEUED")]
    packet = await runtime.store.get_handoff("HND-20261014-0942-8821")
    assert set(packet["context_payload_json"]) == {
        "ai_summary",
        "verbatim_transcript",
        "symptom_history",
        "recommendation",
    }
    assert packet["ticket_number"] == "TICK-CS-8821"


# --------------------------------------------------------------------------------------------- T03


class _ScriptedJev:
    def __init__(self, judgments):
        self.judgments = list(judgments)

    async def __call__(self, _):
        item = self.judgments.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _turn_input(now, *, transcript="I want to change, um,", epoch=7, silence_ms=400):
    return TurnInput(
        session_id="SES-VOX-8839201",
        turn_epoch=epoch,
        transcript=transcript,
        silence_ms=silence_ms,
        asr_final=True,
        stable_word_ratio=1.0,
        local_vad_quiet=True,
        now_ms=now,
    )


async def test_scenario_10_hold_then_commit_at_epoch_7():
    controller = TurnController(
        judge=_ScriptedJev(
            [
                JevJudgment(turn_complete=0.12, holding_floor=0.88),
                JevJudgment(turn_complete=0.95, holding_floor=0.02),
                JevJudgment(turn_complete=0.95, holding_floor=0.02),
            ]
        )
    )
    hold = await controller.evaluate(_turn_input(0))
    assert hold.state == "HOLD" and hold.reason == "caller_holding_floor"  # never interrupts the hesitation
    # the caller finishes: authentic speech (local VAD onset) resets the non-commit count in the same episode
    resumed = await controller.on_genuine_speech(
        TurnInput(
            session_id="SES-VOX-8839201",
            turn_epoch=7,
            transcript="",
            source="local_vad",
            local_vad_quiet=False,
            now_ms=500,
        )
    )
    assert resumed.state == "HOLD" and resumed.jeval_noncommit_count == 0
    complete = "I want to change, um, my delivery address."
    first = await controller.evaluate(_turn_input(700, transcript=complete, silence_ms=300))
    # implementation choice: a high score is confirmed after a short extra quiet before COMMIT (docs/TURN_CONTROL.md)
    assert first.state == "HOLD" and first.reason == "confirming"
    commit = await controller.evaluate(_turn_input(1100, transcript=complete, silence_ms=700))
    assert commit.state == "COMMIT" and commit.turn_epoch == 7


async def test_scenario_10_degraded_after_two_noncommits_or_deadline():
    controller = TurnController(judge=_ScriptedJev([JevJudgment(turn_complete=0.5, holding_floor=0.3)] * 2))
    first = await controller.evaluate(_turn_input(0))
    second = await controller.evaluate(_turn_input(100))
    assert (first.state, second.state) == ("HOLD", "DEGRADED")
    event = second.audit_event
    assert (event["action_name"], event["outcome_status"], event["mutation_idempotency_key"]) == (
        "TURN_COMPLETION_DEGRADED",
        "PENDING",
        None,
    )
    assert event["response_payload_json"]["turn_state"] == "DEGRADED/NEEDS_CLARIFICATION"
    timeout = TurnController(judge=_ScriptedJev([TimeoutError()]))
    held = await timeout.evaluate(_turn_input(0))
    degraded = await timeout.on_hold_deadline("SES-VOX-8839201", 7, held.hold_episode_id, now_ms=held.hold_deadline_ms)
    assert held.state == "HOLD" and degraded.state == "DEGRADED"  # a timeout alone never commits


async def test_scenario_10_barge_in_advances_epoch_resume_question_and_decline_never_undoes(runtime):
    planner = ScriptedPlanner(
        [
            [
                P("lookup_order_details", {"order_code": "ORD-1002"}),
                P("track_shipment_status", {"order_code": "ORD-1002", "tracking_code": None}),
            ],
            [P("retrieve_policy_clauses", {"query": "What warranty does an LG washer have?", "category": "WARRANTY"})],
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839201")
    sid = session.session_id
    first = await boundary.handle_turn(sid, "Where is ORD-1002?")
    assert first.turn_epoch == 1
    epoch = await boundary.note_barge_in(sid, interrupted_epoch=first.turn_epoch)  # stale epoch-1 work is dropped
    assert epoch == 2
    before = await runtime.repository.get_order_by_code("ORD-1002")
    second = await boundary.handle_turn(sid, "What warranty does an LG washer have?")
    assert second.turn_epoch == 3
    assert second.assistant_text.endswith("Would you like to go back to your order ORD-1002?")
    declined = await boundary.handle_turn(sid, "No thanks.")
    assert declined.assistant_text == copy.RESUME_DECLINED
    after = await runtime.repository.get_order_by_code("ORD-1002")
    assert after["version"] == before["version"] and after["status"] == before["status"]
    assert not any(row["action_name"] not in {"TOOL_INVOCATION"} for row in await audit_of(boundary, sid))


# --------------------------------------------------------------------------------------------- T04 (guarded writes)


async def _cancel_fixture(runtime, session_id="SES-VOX-8839202"):
    preview = await runtime.writes.preview(
        session_id=session_id,
        customer_id="CUST-001",
        action_type="Cancel",
        order_code="ORD-1001",
        action_id="ACT-8821",
        idempotency_key=K1,
    )
    return preview


async def test_scenario_06_guarded_cancellation(fake_oms_runtime):
    rt = fake_oms_runtime
    preview = await _cancel_fixture(rt)
    assert preview.kind == "PREVIEW"
    assert (
        preview.spoken
        == "I can submit a cancellation for ORD-1001, total 11,490,000 VND, with no cancellation fee. Do you confirm?"
    )
    assert (await rt.repository.get_order_by_code("ORD-1001"))["version"] == 1  # zero mutations
    action = preview.action
    assert (action["status"], action["expected_version"], action["idempotency_key"]) == ("pending", 1, K1)
    assert len(action["preview_hash"]) == 64
    confirm = await rt.writes.confirm("ACT-8821")
    assert confirm.kind == "SUBMITTED" and "cancelled" not in confirm.spoken  # no success line before the result
    done = await rt.writes.execute("ACT-8821")
    assert done.spoken == "ORD-1001 is cancelled. VoxCom will refund 11,490,000 VND to your card."
    order = await rt.repository.get_order_by_code("ORD-1001")
    assert (order["status"], order["version"], order["payment_status"]) == ("cancelled", 2, "REFUND_PENDING")
    rows = await rt.repository.list_audit("SES-VOX-8839202")
    assert actions(rows) == ["EXECUTION_PENDING", "ORDER_CANCELLED"]
    assert {row["mutation_idempotency_key"] for row in rows} == {K1}
    assert len({row["audit_id"] for row in rows}) == 2
    assert (await rt.store.get_pending_action("ACT-8821"))["status"] == "completed"


async def test_scenario_06_dialogue_level_with_fake_oms(fake_oms_runtime):
    boundary = boundary_for(fake_oms_runtime)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839202")
    preview = await boundary.handle_turn(session.session_id, "I want to cancel order ORD-1001.")
    assert preview.outcome == "CONFIRMATION_REQUESTED"
    done = await boundary.handle_turn(session.session_id, "Yes, please cancel it.")
    assert done.assistant_text == "ORD-1001 is cancelled. VoxCom will refund 11,490,000 VND to your card."


async def test_scenario_11_reschedule_preview_record_only_and_execution(runtime):
    boundary = boundary_for(runtime)
    session = await boundary.create_session(fixture_session_id="SES-VOX-8839202")
    preview = await boundary.handle_turn(session.session_id, "Please move the delivery of ORD-1001 to next Monday.")
    assert preview.assistant_text == "I can move ORD-1001 to Monday, October 19. Do you confirm?"
    assert (await runtime.repository.get_order_by_code("ORD-1001"))["version"] == 1
    recorded = await boundary.handle_turn(session.session_id, "Yes.")
    assert recorded.outcome == "WRITE_REQUEST_RECORDED"
    assert "is now" not in recorded.assistant_text and "nothing has changed yet" in recorded.assistant_text
    assert actions(await audit_of(boundary, session.session_id)) == ["EXECUTION_PENDING"]
    assert (await runtime.repository.get_order_by_code("ORD-1001"))["requested_delivery_time"] is None


async def test_scenario_11_reschedule_occ_execution(fake_oms_runtime):
    rt = fake_oms_runtime
    preview = await rt.writes.preview(
        session_id="SES-VOX-8839202",
        customer_id="CUST-001",
        action_type="Reschedule",
        order_code="ORD-1001",
        target_date=date(2026, 10, 19),
        steps_required=2,
    )
    first = await rt.writes.confirm(preview.action["action_id"])
    assert first.kind == "CONFIRM_AGAIN"  # a second step is allowed (1-2 steps)
    await rt.writes.confirm(preview.action["action_id"])
    done = await rt.writes.execute(preview.action["action_id"])
    assert done.spoken == "The delivery window for ORD-1001 is now Monday, October 19."
    order = await rt.repository.get_order_by_code("ORD-1001")
    assert order["version"] == 2
    assert order["requested_delivery_time"].isoformat().startswith("2026-10-19T09:00")
    with pytest.raises(ValueError):  # there is no third confirmation step
        await rt.writes.preview(
            session_id="S", customer_id="CUST-001", action_type="Cancel", order_code="ORD-1001", steps_required=3
        )


async def test_scenario_12_withdrawal_paths(fake_oms_runtime):
    rt = fake_oms_runtime
    a = await _cancel_fixture(rt)
    case_a = await rt.writes.withdraw(a.action["action_id"])
    assert case_a.spoken == "Cancelled. No changes were made to order ORD-1001."
    assert rt.oms.table == {}  # zero OMS interaction
    b = await rt.writes.preview(
        session_id="SES-VOX-8839202", customer_id="CUST-001", action_type="Cancel", order_code="ORD-1001"
    )
    await rt.writes.confirm(b.action["action_id"])
    case_b = await rt.writes.withdraw(b.action["action_id"])
    assert case_b.spoken == "Stopped before execution. No changes were made to ORD-1001."
    c = await rt.writes.preview(
        session_id="SES-VOX-8839202", customer_id="CUST-001", action_type="Cancel", order_code="ORD-1001"
    )
    await rt.writes.confirm(c.action["action_id"])
    rt.oms.hold = True
    in_flight = await rt.writes.execute(c.action["action_id"])
    assert in_flight.kind == "IN_PROGRESS"
    case_c = await rt.writes.withdraw(c.action["action_id"])
    assert case_c.spoken == "The request was stopped. No changes were made to order ORD-1001."
    assert (await rt.repository.get_order_by_code("ORD-1001"))["version"] == 1
    names = actions(await rt.repository.list_audit("SES-VOX-8839202"))
    assert names == [
        "ACTION_ABORTED_PRE_SUBMIT",
        "EXECUTION_PENDING",
        "ACTION_ABORTED_POST_SUBMIT",
        "EXECUTION_PENDING",
        "WITHDRAWAL_CONFIRMED",
    ]
    assert copy.WITHDRAW_ASKING_OMS == "I am asking the fulfilment system to stop the request."


async def test_scenario_13_withdrawal_after_commit_routes_to_reversal(fake_oms_runtime):
    rt = fake_oms_runtime
    action = (await _cancel_fixture(rt)).action
    await rt.writes.confirm(action["action_id"])
    rt.oms.hold = True
    await rt.writes.execute(action["action_id"])
    rt.oms.abort_policy = "too_late"
    late = await rt.writes.withdraw(action["action_id"])
    assert late.spoken == (
        "It was already committed by the fulfilment centre. Undoing it needs a reinstatement "
        "(reversal) request — I can submit one if you want."
    )
    assert (await rt.store.get_pending_action("ACT-8821"))["status"] == "completed"
    assert (await rt.repository.get_order_by_code("ORD-1001"))["version"] == 2
    assert actions(await rt.repository.list_audit("SES-VOX-8839202"))[-1] == "ORDER_CANCELLED"


async def _completed_cancel(rt):
    action = (await _cancel_fixture(rt)).action
    await rt.writes.confirm(action["action_id"])
    await rt.writes.execute(action["action_id"])
    return await rt.store.get_pending_action("ACT-8821")


async def test_scenario_14_reinstatement_with_new_key(fake_oms_runtime):
    rt = fake_oms_runtime
    original = await _completed_cancel(rt)
    preview = await rt.writes.preview(
        session_id="SES-VOX-8839202",
        customer_id="CUST-001",
        action_type="Reversal",
        order_code="ORD-1001",
        reversal_of=original,
    )
    assert preview.spoken == "I can submit the reinstatement. Do you confirm?"
    reversal = preview.action
    assert reversal["idempotency_key"] != K1
    assert reversal["action_payload_json"]["reversal_of_action_id"] == "ACT-8821"
    assert reversal["expected_version"] == 2
    await rt.writes.confirm(reversal["action_id"])
    done = await rt.writes.execute(reversal["action_id"])
    assert done.spoken == "Order ORD-1001 is active again."
    order = await rt.repository.get_order_by_code("ORD-1001")
    assert (order["status"], order["version"]) == ("confirmed", 3)
    rows = await rt.repository.list_audit("SES-VOX-8839202")
    requested = next(row for row in rows if row["action_name"] == "REVERSAL_REQUESTED")
    assert requested["request_payload_json"]["reversal_of_action_id"] == "ACT-8821"
    assert requested["mutation_idempotency_key"] == reversal["idempotency_key"]


async def test_scenario_15_reinstatement_refused_after_refund_dispatched(fake_oms_runtime):
    rt = fake_oms_runtime
    original = await _completed_cancel(rt)
    rt.oms.refund_dispatched["ORD-1001"] = True
    refused = await rt.writes.preview(
        session_id="SES-VOX-8839202",
        customer_id="CUST-001",
        action_type="Reversal",
        order_code="ORD-1001",
        reversal_of=original,
    )
    assert refused.spoken == (
        "The refund batch was already dispatched, so this cancellation cannot be undone. "
        "I can ask a human colleague to help."
    )
    assert refused.action is None
    order = await rt.repository.get_order_by_code("ORD-1001")
    assert (order["status"], order["version"]) == ("cancelled", 2)  # no mutation
    rows = await rt.repository.list_audit("SES-VOX-8839202")
    denial = rows[-1]
    assert (denial["action_name"], denial["action_status"], denial["outcome_status"]) == (
        "REVERSAL_REFUSED",
        "Denied",
        "DENIED",
    )
    assert denial["request_payload_json"]["reversal_of_action_id"] == "ACT-8821"
    result = await rt.middleware.handle(
        _ctx("SES-VOX-8839202", order="ORD-1001"),
        "escalate_to_human",
        {"trigger_reason": "COMPLEX_EXCEPTION", "priority": "NORMAL"},
    )
    assert result.ok and result.envelope["ticket_number"].startswith("TICK-CS-")


def _ctx(session_id, *, order=None):
    from voxcare.middleware.tool_middleware import ToolContext

    return ToolContext(session_id=session_id, customer_id="CUST-001", active_order_code=order, fixture_session=True)


async def test_scenario_16_replay_k1_in_four_states(fake_oms_runtime):
    rt = fake_oms_runtime
    action = (await _cancel_fixture(rt)).action
    await rt.writes.confirm(action["action_id"])
    rt.oms.hold = True
    await rt.writes.execute(action["action_id"])
    executing = await rt.writes.replay("ACT-8821")
    assert executing.spoken == "The change is still being processed."
    await rt.oms.commit_held(K1)
    rt.oms.hold = False
    replay = await rt.writes.replay("ACT-8821")
    assert replay.spoken == "That change was already applied."
    assert rt.oms.executions == [K1]  # never two executions for one key
    assert {row["mutation_idempotency_key"] for row in await rt.repository.list_audit("SES-VOX-8839202")} == {K1}


async def test_scenario_16_permanent_conflict_and_lost_responses():
    conflict_rt = await make_runtime("memory", write_mode="fake_oms")
    action = (await _cancel_fixture(conflict_rt)).action
    await conflict_rt.writes.confirm(action["action_id"])
    conflict_rt.oms.hold = True
    await conflict_rt.writes.execute(action["action_id"])
    await conflict_rt.repository.update_order_occ(
        "ORD-1001", 1, {"status": "processing"}, conflict_rt.clock.now()
    )  # someone else moved v1 → v2
    conflict_rt.oms.hold = False
    conflict_rt.oms.table.clear()
    failed = await conflict_rt.writes.replay("ACT-8821")
    assert failed.spoken == "That change can no longer be applied to ORD-1001."
    row = (await conflict_rt.repository.list_audit("SES-VOX-8839202"))[-1]
    assert (row["action_name"], row["action_status"], row["outcome_status"]) == ("TOOL_INVOCATION", "Failed", "FAILED")
    assert row["request_payload_json"]["tool_name"] == "cancel_order"
    assert row["response_payload_json"]["oms_status"] == "FAILED_NO_COMMIT"

    lost_rt = await make_runtime("memory", write_mode="fake_oms")
    action = (await _cancel_fixture(lost_rt)).action
    await lost_rt.writes.confirm(action["action_id"])
    lost_rt.oms.lose_responses = 2  # the original and the one same-key retry
    unknown = await lost_rt.writes.execute(action["action_id"])
    assert unknown.spoken == "I could not confirm the result yet. I am checking with the fulfilment system."
    assert (await lost_rt.store.get_pending_action("ACT-8821"))["status"] == "executing"
    row = (await lost_rt.repository.list_audit("SES-VOX-8839202"))[-1]
    assert (row["action_name"], row["outcome_status"], row["mutation_idempotency_key"]) == (
        "MUTATION_OUTCOME_UNKNOWN",
        "UNKNOWN",
        K1,
    )
    assert lost_rt.oms.executions == [K1]  # the retry reused K1 and did not execute twice
    resolved = await lost_rt.writes.reconcile("ACT-8821")
    assert resolved.kind == "COMPLETED"
    assert (await lost_rt.store.get_pending_action("ACT-8821"))["status"] == "completed"


async def test_handoff_context_dataclass_defaults():
    context = HandoffContext(session_id="S", customer_id="CUST-001")
    assert context.active_order_code is None and list(context.transcript) == []
