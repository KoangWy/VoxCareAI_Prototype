"""Session activity log: every turn's route, tool checks, audit rows, state changes and reply, server-side."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from support import BACKENDS, P, ScriptedPlanner, boundary_for, drop_runtime, make_runtime

import voxcare.app as api
from voxcare.session.activity import to_markdown

SCENARIO_1 = "Can you check the delivery status of my order ORD-1002 and my earlier purchases?"


@pytest.fixture(params=BACKENDS)
async def runtime(request):
    rt = await make_runtime(request.param)
    yield rt
    await drop_runtime(rt)


def _steps(entry, kind):
    return [step for step in entry["steps"] if step["kind"] == kind]


def _checks(step):
    return {check["check"]: check["result"] for check in step["checks"]}


async def test_turn_entry_lists_route_checks_audit_rows_state_and_reply(runtime):
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
    session = await boundary.create_session(verified=True)
    result = await boundary.handle_turn(session.session_id, SCENARIO_1)

    entry = result.activity
    assert entry["kind"] == "turn" and entry["turn"] == 1 and entry["source"] == "typed"
    assert entry["caller_text"] == SCENARIO_1 and entry["summary"] == "Answered"
    assert [step["code"] for step in _steps(entry, "route")] == ["MODEL_PLAN"]
    tools = _steps(entry, "tool")
    assert [step["name"] for step in tools] == ["lookup_order_details", "track_shipment_status", "list_customer_orders"]
    assert _checks(tools[0]) == {
        "allowlist": "pass",
        "arguments": "pass",
        "verified_session": "pass",
        "service": "pass",
        "customer_scope": "pass",
    }
    assert tools[1]["result"] == "in_transit, estimated delivery 2026-10-15, 2 carrier event(s)"
    stored = {str(row["audit_id"]) for row in await runtime.repository.list_audit(session.session_id)}
    logged = {step["audit_id"] for step in _steps(entry, "audit")}
    assert logged and logged <= stored  # the same rows as audit_log
    assert [step["text"] for step in _steps(entry, "state")] == ["Active order: ORD-1002"]
    assert _steps(entry, "reply")[0]["text"].startswith("Order ORD-1002 is in transit with GHN.")

    log = await boundary.session_activity(session.session_id)
    assert [item["kind"] for item in log["entries"]] == ["session", "turn"]


async def test_cross_customer_refusal_shows_the_failed_scope_check_and_violation_audit(runtime):
    planner = ScriptedPlanner([[P("lookup_order_details", {"order_code": "ORD-2002"})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    entry = (await boundary.handle_turn(session.session_id, "Please check order ORD-2002.")).activity

    tool = _steps(entry, "tool")[0]
    assert _checks(tool)["customer_scope"] == "fail"
    assert tool["result"] == "refused: belongs to another customer; no details returned"
    assert "AUTH_SCOPE_VIOLATION" in [step["action_name"] for step in _steps(entry, "audit")]
    assert entry["summary"] == "Refused: not on the verified account"


async def test_verification_is_logged_with_masked_phone_and_never_the_raw_number(runtime):
    lookup = [
        P("lookup_order_details", {"order_code": "ORD-1001"}),
        P("track_shipment_status", {"order_code": "ORD-1001", "tracking_code": None}),
    ]
    planner = ScriptedPlanner([lookup, lookup])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=False)
    first = (await boundary.handle_turn(session.session_id, "Track order ORD-1001")).activity
    second = (await boundary.handle_turn(session.session_id, "It is 0900 000 001")).activity

    assert {_checks(step)["verified_session"] for step in _steps(first, "tool")} == {"fail"}
    assert "Waiting for the phone number used at checkout" in [step["text"] for step in _steps(first, "state")]
    route = _steps(second, "route")
    assert route[0]["code"] == "VERIFICATION" and "+8490***01 matches CUST-001" in route[0]["text"]
    assert "Session bound to CUST-001 (simulated verification, OPD-04)" in [s["text"] for s in _steps(second, "state")]
    dumped = json.dumps((await boundary.session_activity(session.session_id))["entries"])
    assert "0900 000 001" not in dumped and "0900000001" not in dumped and "+84900000001" not in dumped


async def test_guarded_write_preview_and_record_only_confirmation(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    preview = (await boundary.handle_turn(session.session_id, "Move the delivery of ORD-1001 to next Monday.")).activity
    confirm = (await boundary.handle_turn(session.session_id, "Yes")).activity

    assert _steps(preview, "route")[0]["code"] == "WRITE_REQUEST"
    assert _steps(preview, "write")[0]["result"] == "PREVIEW"
    assert "nothing written" in _steps(preview, "write")[0]["text"]
    assert _steps(confirm, "route")[0]["code"] == "WRITE_CONFIRMATION"
    write = _steps(confirm, "write")[0]
    assert write["result"] == "SUBMITTED" and "ORD-1001 was not changed" in write["text"]
    assert "EXECUTION_PENDING" in [step["action_name"] for step in _steps(confirm, "audit")]
    assert confirm["summary"] == "Change request recorded, not executed"


async def test_support_button_and_barge_in_are_logged(runtime):
    boundary = boundary_for(runtime, ScriptedPlanner())
    session = await boundary.create_session(verified=True)
    handoff = await boundary.record_handoff(session.session_id, reason="Caller requested human support")
    await boundary.note_barge_in(session.session_id, interrupted_epoch=3)

    assert handoff["activity"]["kind"] == "support"
    tool = _steps(handoff["activity"], "tool")[0]
    assert tool["name"] == "escalate_to_human" and "no live transfer" in tool["result"]
    assert [step["action_name"] for step in _steps(handoff["activity"], "audit")] == ["HANDOFF_REQUEST_RECORDED"]
    entries = (await boundary.session_activity(session.session_id))["entries"]
    assert entries[-1]["summary"] == "Caller interrupted the reply (barge-in)"
    assert "Turn epoch advanced to 4" in entries[-1]["steps"][0]["text"]


async def test_the_log_carries_no_model_names_or_durations(runtime):
    planner = ScriptedPlanner([[P("retrieve_policy_clauses", {"query": "return window for a TV", "category": "ALL"})]])
    boundary = boundary_for(runtime, planner)
    session = await boundary.create_session(verified=True)
    await boundary.handle_turn(session.session_id, "What is the return window for my TV?")
    log = await boundary.session_activity(session.session_id)

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert not [key for key in keys(log) if "model" in key or "latency" in key or "duration" in key or "_ms" in key]
    text = json.dumps(log).lower()
    assert "claude" not in text and "gpt" not in text and "haiku" not in text
    markdown = to_markdown(log["session_id"], log["customer_id"], log["entries"])
    assert markdown.startswith(f"# VoxCare session activity: {session.session_id}")
    assert "## Turn 1 (typed)" in markdown and "  - checks: allowlist ✓" in markdown


def test_activity_endpoint_needs_the_token_and_exports_markdown(monkeypatch):
    monkeypatch.setenv("VOXCARE_CONVERSATION_PROVIDER", "keyword")
    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        headers = {"VoxCare-Session-Token": session["token"]}
        sid = session["session_id"]
        turn = client.post(f"/api/sessions/{sid}/turns", headers=headers, json={"text": "Where is order ORD-1002?"})
        denied = client.get(f"/api/sessions/{sid}/activity")
        log = client.get(f"/api/sessions/{sid}/activity", headers=headers)
        markdown = client.get(f"/api/sessions/{sid}/activity", params={"format": "markdown"}, headers=headers)
        invalid = client.get(f"/api/sessions/{sid}/activity", params={"format": "pdf"}, headers=headers)

    assert turn.json()["activity"]["kind"] == "turn"
    assert denied.status_code == 401
    assert [entry["kind"] for entry in log.json()["entries"]] == ["session", "turn"]
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "attachment" in markdown.headers["content-disposition"]
    assert "## Turn 1 (typed)" in markdown.text
    assert invalid.status_code == 400
