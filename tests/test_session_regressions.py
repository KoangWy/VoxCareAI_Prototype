"""Regressions from a recorded demo session (activity log of 2026-11-04): each turn as the model planned it."""

from __future__ import annotations

import pytest
from support import BACKENDS, P, ScriptedPlanner, boundary_for, drop_runtime, make_runtime


@pytest.fixture(params=BACKENDS)
async def runtime(request):
    rt = await make_runtime(request.param)
    yield rt
    await drop_runtime(rt)


def _codes(result):
    return [step["code"] for step in result.activity["steps"] if step["kind"] == "route"]


def _tools(result):
    return [step["name"] for step in result.activity["steps"] if step["kind"] == "tool"]


async def test_slot_fill_and_verification_keep_the_original_where_is_my_order_intent(runtime):
    # the model answered turn 1 with an empty code; it is not asked again on turns 2 and 3
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": ""})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=False)
    sid = session.session_id

    first = await boundary.handle_turn(sid, "where is my order")
    assert first.assistant_text == "Could you please tell me your order code?"

    second = await boundary.handle_turn(sid, "ORD-1002")
    assert _codes(second) == ["SLOT_FILL"]
    assert _tools(second) == ["lookup_order_details", "track_shipment_status"]  # tracking, as asked
    assert second.outcome == "NEEDS_VERIFICATION"

    third = await boundary.handle_turn(sid, "It is 0900 000 001")
    assert _codes(third) == ["VERIFICATION", "RESUMED_REQUEST"]
    assert _tools(third) == ["lookup_order_details", "track_shipment_status"]
    assert third.assistant_text.startswith("Thank you. Your account is verified for this call.")
    assert "Estimated delivery date: 2026-10-15" in third.assistant_text
    assert len(planner.seen) == 1  # the model was consulted once, on the first turn only


async def test_rescheduling_to_the_day_already_booked_changes_nothing(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "i want to change the delivery of ORD-1002 to friday")

    assert result.outcome == "UNCHANGED"
    assert result.assistant_text == (
        "ORD-1002 is already scheduled for delivery on Friday, October 16, between 09:00 and 12:00. "
        "Nothing needs to change."
    )
    write = next(step for step in result.activity["steps"] if step["kind"] == "write")
    assert write["result"] == "UNCHANGED"
    record = await runtime.store.get_session(session.session_id)
    assert not record.get("pending_action_id")  # no preview, nothing to confirm

    moved = await boundary.handle_turn(session.session_id, "then move it to Monday")
    assert moved.outcome == "CONFIRMATION_REQUESTED"  # a different day is still a normal preview
    assert moved.assistant_text == "I can move ORD-1002 to Monday, October 19. Do you confirm?"


async def test_washer_repair_request_asks_for_the_symptom_then_diagnoses(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id

    ask = await boundary.handle_turn(sid, "i want to some repairs for my washing machine")
    assert ask.outcome == "REPAIR_QUESTION" and _codes(ask) == ["REPAIR_INTAKE"]
    assert "What is happening with your washing machine?" in ask.assistant_text

    diagnosis = await boundary.handle_turn(sid, "it shows error E4")
    assert diagnosis.outcome == "DIAGNOSIS_QUESTION" and _codes(diagnosis) == ["DIAGNOSIS_START"]


async def test_unknown_fault_offers_a_technician_request_and_yes_records_it(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id

    await boundary.handle_turn(sid, "my TV is broken")
    offer = await boundary.handle_turn(sid, "it makes a buzzing noise")
    assert offer.outcome == "REPAIR_UNKNOWN_SYMPTOM" and _codes(offer) == ["REPAIR_SYMPTOM"]
    assert offer.assistant_text.startswith("I cannot diagnose that symptom by phone.")

    accepted = await boundary.handle_turn(sid, "yes please")
    assert accepted.outcome == "HANDOFF_REQUEST_RECORDED"
    assert _codes(accepted) == ["HANDOFF_OFFER_ACCEPTED"]


async def test_policy_question_without_a_tool_still_searches_the_policy_corpus(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner([[]]))  # the model proposed no tool, as in the log
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "i want to know about refund policies")

    assert _codes(result) == ["POLICY_FALLBACK"]
    tool = next(step for step in result.activity["steps"] if step["kind"] == "tool")
    assert tool["name"] == "retrieve_policy_clauses" and tool["arguments"]["category"] == "RETURNS"
    assert result.outcome in {"ANSWERED", "INSUFFICIENT_EVIDENCE"}
    assert result.outcome != "CLARIFICATION"


async def test_voucher_list_request_gets_a_capability_answer_and_no_clarification_count(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner([[], [], []]))
    session = await boundary.create_session(verified=True)
    sid = session.session_id

    result = await boundary.handle_turn(sid, "which vouchers are currently available")
    assert result.outcome == "CAPABILITY_ANSWER" and _codes(result) == ["VOUCHER_LIST"]
    assert "VOXCOM_500K" in result.assistant_text

    # two unclear turns, one handled turn, one unclear turn: the limit needs three unclear turns in a row
    await boundary.handle_turn(sid, "hmm")
    await boundary.handle_turn(sid, "blah")
    await boundary.handle_turn(sid, "When will I get my refund?")  # OPD-02, a handled turn
    last = await boundary.handle_turn(sid, "erm")
    assert last.outcome == "CLARIFICATION"


@pytest.mark.parametrize(
    ("question", "topic"),
    [
        ("is installation free for my washing machine", "installation free for your washing machine"),
        (
            "Do you accept trade-ins for 10-year-old microwave ovens with cash back?",
            "trade-ins for 10-year-old microwave ovens with cash back",
        ),
        ("Can I get same-day delivery by drone?", "same-day delivery by drone"),
        ("i want to know about refund policies", "refund policies"),
    ],
)
def test_out_of_domain_topic_keeps_whole_words(question, topic):
    from voxcare.session.render import inquiry_topic

    assert inquiry_topic(question) == topic


async def test_vague_policy_question_lists_the_policy_topics(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "tell me your policies")
    assert result.outcome == "CAPABILITY_ANSWER" and _codes(result) == ["POLICY_TOPICS"]
    assert "delivery fees, installation, warranty, and returns or exchanges" in result.assistant_text


async def test_delivery_charge_question_falls_back_to_shipping_policy(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner([[]]))
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "do you charge for delivery")
    tool = next(step for step in result.activity["steps"] if step["kind"] == "tool")
    assert tool["arguments"]["category"] == "SHIPPING" and _codes(result) == ["POLICY_FALLBACK"]


async def test_damaged_box_wording_reaches_the_opd_05_line(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "The box arrived damaged")
    assert result.outcome == "OPD_FALLBACK"
    assert result.assistant_text.startswith("I will note this cosmetic report for our warranty team.")


async def test_keyword_planner_reads_purchase_history_questions():
    from voxcare.providers.keyword import KeywordPlanner

    calls = await KeywordPlanner().plan_turn(user_text="What did I buy before?", history=(), available_tools=())
    assert [call.name for call in calls] == ["list_customer_orders"]


async def test_an_unrelated_yes_never_confirms_an_earlier_preview(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id

    preview = await boundary.handle_turn(sid, "Move the delivery of ORD-1001 to Tuesday")
    assert preview.outcome == "CONFIRMATION_REQUESTED"
    moved_on = await boundary.handle_turn(sid, "My washer stops during spin cycle and displays error E4.")
    assert moved_on.assistant_text.startswith("I have not submitted the change to ORD-1001.")
    assert _codes(moved_on)[0] == "WRITE_DROPPED"
    audits = [step["action_name"] for step in moved_on.activity["steps"] if step["kind"] == "audit"]
    assert "ACTION_ABORTED_PRE_SUBMIT" in audits

    answer = await boundary.handle_turn(sid, "Yes")  # answers the diagnosis question, not the old preview
    assert answer.outcome == "DIAGNOSIS_QUESTION"
    names = [row["action_name"] for row in await runtime.repository.list_audit(sid)]
    assert "EXECUTION_PENDING" not in names


async def test_a_new_preview_of_the_same_change_replaces_the_old_one_quietly(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    await boundary.handle_turn(session.session_id, "Move the delivery of ORD-1001 to Tuesday")
    again = await boundary.handle_turn(session.session_id, "Actually move the delivery of ORD-1001 to Wednesday")
    assert again.assistant_text == "I can move ORD-1001 to Wednesday, October 21. Do you confirm?"


async def test_address_change_takes_the_address_before_the_request_or_asks_for_it(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id

    first = await boundary.handle_turn(
        sid,
        "My address is 123 Nguyen Trai, Ward 2, District 5, Ho Chi Minh City, please change the delivery "
        "address of ORD-1001",
    )
    assert first.assistant_text == "I can submit the new delivery address for ORD-1001. Do you confirm?"
    await boundary.handle_turn(sid, "no")

    ask = await boundary.handle_turn(sid, "I need to change the delivery address of ORD-1001")
    assert ask.outcome == "NEEDS_ADDRESS" and ask.assistant_text == "What is the new delivery address?"
    given = await boundary.handle_turn(sid, "45 Le Loi Street, Ward 3, District 1")
    assert given.outcome == "CONFIRMATION_REQUESTED" and _codes(given)[0] == "WRITE_ADDRESS"
    assert "Le Loi" not in str(given.activity)  # stored and logged masked


async def test_diagnosis_turns_log_answers_fault_class_window_and_choice(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id
    await boundary.handle_turn(sid, "My washer stops during spin cycle and displays error E4")
    first = await boundary.handle_turn(sid, "yes")
    second = await boundary.handle_turn(sid, "yes")
    final = await boundary.handle_turn(sid, "i want a refund")

    def states(result):
        return [step["text"] for step in result.activity["steps"] if step["kind"] == "state"]

    assert states(first) == ["Answer recorded for the technician: “yes” to “Does it drain before it spins?”"]
    assert any(text.startswith("Fault class MOTOR_DRIVE_FAULT from SPIN_CYCLE_ERROR_E4") for text in states(second))
    assert any("ORD-1003 delivered 2026-10-02: inside the 30-day return window" in text for text in states(second))
    assert any(text.startswith("Caller chose a return and refund") for text in states(final))


async def test_order_read_with_an_empty_code_takes_the_code_the_caller_said(runtime):
    # activity log SES-8103c38c, turn 2: lookup_order_details() with no arguments
    planner = ScriptedPlanner([[P("lookup_order_details", {})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "Is order ORD-1002 delivered yet?")

    assert _codes(result) == ["MODEL_PLAN"]
    assert _tools(result) == ["lookup_order_details", "track_shipment_status"]  # tracking paired for delivery
    inputs = [step["text"] for step in result.activity["steps"] if step["kind"] == "input"]
    assert any("order_code ORD-1002 for lookup_order_details" in text for text in inputs)
    assert "Estimated delivery date: 2026-10-15" in result.assistant_text


async def test_scenario_1_with_an_empty_lookup_runs_the_keyword_plan(runtime):
    planner = ScriptedPlanner([[P("lookup_order_details", {}), P("list_customer_orders", {})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    text = "Can you check the delivery status of my order ORD-1002 and my earlier purchases?"
    result = await boundary.handle_turn(session.session_id, text)
    assert result.outcome != "NEEDS_ORDER_CODE"
    assert "ORD-1002 is in transit with GHN" in result.assistant_text


async def test_compare_with_no_skus_is_replaced_by_the_keyword_plan(runtime):
    # same log, turn 4: compare_products() with no arguments
    planner = ScriptedPlanner([[P("compare_products", {})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(
        session.session_id, "Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies"
    )
    assert _codes(result) == ["KEYWORD_FALLBACK"]
    assert _tools(result) == ["search_catalog", "compare_products", "apply_voucher"]
    assert result.assistant_text == (
        "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND. With VOXCOM_500K the Samsung is 10,990,000 VND."
    )


async def test_no_tool_turn_tries_the_keyword_plan_and_logs_the_model_text(runtime):
    from voxcare.contracts import PlannedCalls

    empty = PlannedCalls()
    empty.note = "Sure! Let me check that for you."
    boundary = boundary_for(runtime, ScriptedPlanner([empty]))
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "Where is order ORD-1002?")
    assert _codes(result) == ["KEYWORD_FALLBACK"]
    notes = [step["text"] for step in result.activity["steps"] if step["kind"] == "note"]
    assert notes and "Let me check that for you." in notes[0]
    assert "Let me check" not in result.assistant_text  # never spoken


async def test_a_named_voucher_the_model_skipped_is_still_checked(runtime):
    # activity log SES-36e8a973, turn 2: compare_products only, the voucher was never checked
    planner = ScriptedPlanner([[P("compare_products", {"skus": ["TV-SAM-55", "TV-LG-55"]})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(
        session.session_id, "Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies"
    )
    assert _codes(result) == ["MODEL_PLAN"]
    assert _tools(result) == ["compare_products", "apply_voucher"]
    assert result.assistant_text == (
        "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND. With VOXCOM_500K the Samsung is 10,990,000 VND."
    )


async def test_a_superseded_turn_never_adds_its_audit_rows_to_the_next_turn(runtime):
    # log SES-345e37a0, turns 16 and 17: the late turn's tool finished after the next turn had started, and
    # its audit row appeared in both entries
    import asyncio

    boundary = boundary_for(runtime, ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-1002"})]] * 2))
    session = await boundary.create_session(verified=True)
    sid = session.session_id
    middleware = runtime.dialogue.middleware
    original = middleware.handle
    gate, entered = asyncio.Event(), asyncio.Event()
    calls = 0

    async def slow_first(context, name, arguments):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await gate.wait()
        return await original(context, name, arguments)

    middleware.handle = slow_first
    try:
        first = asyncio.create_task(boundary.handle_turn(sid, "show order ORD-1002 please"))
        await entered.wait()
        second = asyncio.create_task(boundary.handle_turn(sid, "show order ORD-1002 now"))
        await asyncio.sleep(0.05)
        gate.set()
        old, new = await first, await second
    finally:
        middleware.handle = original

    def audit_ids(result):
        return [step["audit_id"] for step in (result.activity or {}).get("steps", []) if step["kind"] == "audit"]

    assert old.outcome == "STALE_DISCARDED"
    assert len(audit_ids(new)) == 1
    assert not set(audit_ids(new)) & set(audit_ids(old))


async def test_asking_again_for_the_same_change_says_the_preview_was_replaced(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    await boundary.handle_turn(session.session_id, "Move the delivery of ORD-1001 to Tuesday")
    again = await boundary.handle_turn(session.session_id, "Move the delivery of ORD-1001 to Tuesday")
    assert _codes(again)[0] == "WRITE_REPLACED"
    assert again.assistant_text == "I can move ORD-1001 to Tuesday, October 20. Do you confirm?"


async def test_a_new_request_ends_the_diagnosis_instead_of_being_recorded_as_an_answer(runtime):
    # logs SES-a70870f2 (turn 7) and SES-7c2df43a (turn 31)
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id
    await boundary.handle_turn(sid, "My washer stops during spin cycle and displays error E4.")
    tv = await boundary.handle_turn(sid, "my TV is broken")
    assert _codes(tv) == ["DIAGNOSIS_LEFT", "REPAIR_INTAKE"]
    assert tv.outcome == "REPAIR_QUESTION"

    await boundary.handle_turn(sid, "My washer stops during spin cycle and displays error E4.")
    await boundary.handle_turn(sid, "yes")
    address = await boundary.handle_turn(sid, "I need to change the delivery address of ORD-1001")
    assert _codes(address)[0] == "DIAGNOSIS_LEFT"
    assert address.assistant_text == "What is the new delivery address?"


async def test_the_options_question_can_be_declined_and_is_not_asked_forever(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    sid = session.session_id
    for line in ("My washer stops during spin cycle and displays error E4.", "yes", "no"):
        await boundary.handle_turn(sid, line)
    declined = await boundary.handle_turn(sid, "no")
    assert declined.outcome == "DIAGNOSIS_DECLINED"
    record = await runtime.store.get_session(sid)
    assert not record.get("diagnosis")

    for line in ("My washer stops during spin cycle and displays error E4.", "yes", "no"):
        await boundary.handle_turn(sid, line)
    again = await boundary.handle_turn(sid, "hmm")
    assert again.outcome == "DIAGNOSIS_OPTIONS"
    limit = await boundary.handle_turn(sid, "erm")
    assert limit.outcome == "CLARIFICATION_LIMIT"


async def test_the_model_cannot_add_an_order_the_caller_did_not_name(runtime):
    # log SES-ba23ffdc, turn 3: ORD-2002 asked, ORD-1002 (an earlier order) tracked too
    planner = ScriptedPlanner(
        [
            [P("lookup_order_details", {"order_code": "ORD-1002"})],
            [
                P("lookup_order_details", {"order_code": "ORD-2002"}),
                P("track_shipment_status", {"order_code": "ORD-1002", "tracking_code": None}),
            ],
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    await boundary.handle_turn(session.session_id, "Where is order ORD-1002?")
    result = await boundary.handle_turn(session.session_id, "Please check order ORD-2002")
    assert "ORD-1002" not in result.assistant_text
    tracked = [s for s in result.activity["steps"] if s["kind"] == "tool" and s["name"] == "track_shipment_status"]
    assert all(step["arguments"]["order_code"] == "ORD-2002" for step in tracked)


async def test_a_policy_question_drops_order_reads_nobody_asked_for(runtime):
    # log SES-7c2df43a, turn 2: lookup + track of the active order added to a warranty question
    planner = ScriptedPlanner(
        [
            [
                P("retrieve_policy_clauses", {"query": "compressor warranty and defect returns", "category": "ALL"}),
                P("lookup_order_details", {"order_code": "ORD-1002"}),
                P("track_shipment_status", {"order_code": "ORD-1002", "tracking_code": None}),
            ]
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(
        session.session_id,
        "What is the warranty period for a Samsung refrigerator compressor and can I return it if it has a "
        "manufacturing defect?",
    )
    assert _tools(result) == ["retrieve_policy_clauses"]
    assert "ORD-1002" not in result.assistant_text


async def test_compare_request_without_a_comparison_runs_the_keyword_plan(runtime):
    # log SES-7c2df43a, turn 13: two searches and no compare_products
    planner = ScriptedPlanner(
        [
            [
                P("search_catalog", {"category": "TV_AV", "brand": "Samsung", "keyword": "55"}),
                P("search_catalog", {"category": "TV_AV", "brand": "LG", "keyword": "55"}),
            ]
        ]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(
        session.session_id, "Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies."
    )
    assert _codes(result) == ["KEYWORD_FALLBACK"]
    assert result.assistant_text == (
        "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND. With VOXCOM_500K the Samsung is 10,990,000 VND."
    )


async def test_voucher_list_is_a_fixed_rule_and_never_requotes_an_old_code(runtime):
    planner = ScriptedPlanner([[P("apply_voucher", {"voucher_code": "VOXCOM_500K", "order_value_vnd": 11490000})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, "which vouchers are currently available")
    assert _codes(result) == ["VOUCHER_LIST"] and result.outcome == "CAPABILITY_ANSWER"
    assert planner.seen == []


async def test_purchase_history_alone_does_not_say_also(runtime):
    planner = ScriptedPlanner(
        [[P("lookup_order_details", {"order_code": "ORD-1002"})], [P("list_customer_orders", {})]]
    )
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    await boundary.handle_turn(session.session_id, "Where is order ORD-1002?")
    result = await boundary.handle_turn(session.session_id, "What did I buy before?")
    assert result.assistant_text.startswith("Your account shows ORD-1003")


def test_repair_question_examples_fit_the_product():
    from voxcare.domain.copy import repair_symptom_question

    assert "cycle" not in repair_symptom_question("TV")
    assert repair_symptom_question("washing machine").endswith("does it show an error code, or stop during a cycle?")


async def test_voice_turn_logs_the_corrected_terms(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(
        session.session_id, "does box com tet five hundred k still work on the samson tv", source="asr_final"
    )
    inputs = [step["text"] for step in result.activity["steps"] if step["kind"] == "input"]
    assert any("VOXCOM_TET_500K" in text and "Samsung" in text for text in inputs)
    assert "VOXCOM_TET_500K" in result.user_text
