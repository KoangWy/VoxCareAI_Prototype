"""The audit writer against the D§3.5 golden row *format* (guide §16; G-11: values are format-only)."""

from __future__ import annotations

import uuid

import pytest
from support import BACKENDS, drop_runtime, make_runtime

from voxcare.observability.audit import AUDIT_FIELDS, audit_json
from voxcare.storage.seed import K1, order_id_for

GOLDEN = [  # D§3.5, audit_ids …440001-…440006 and …440008
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440001",
        "session_id": "SES-VOX-8839201",
        "order_id": "ORD-1002",
        "mutation_idempotency_key": None,
        "actor": "Bot",
        "action_name": "TOOL_INVOCATION",
        "action_status": "Success",
        "outcome_status": "SUCCESS",
        "request_payload_json": {"tool_name": "track_shipment_status", "order_code": "ORD-1002"},
        "response_payload_json": {"shipment_status": "in_transit", "estimated_delivery_date": None},
        "timestamp": "2026-10-14T10:00:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440002",
        "session_id": "SES-VOX-8839201",
        "order_id": "ORD-2002",
        "mutation_idempotency_key": None,
        "actor": "System",
        "action_name": "AUTH_SCOPE_VIOLATION",
        "action_status": "Denied",
        "outcome_status": "DENIED",
        "request_payload_json": {
            "tool_name": "lookup_order_details",
            "order_code": "ORD-2002",
            "session_customer_id": "CUST-001",
        },
        "response_payload_json": {"error_code": "AUTH_SCOPE_VIOLATION", "details_concealed": True},
        "timestamp": "2026-10-14T10:01:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440003",
        "session_id": "SES-VOX-8839201",
        "order_id": "ORD-1001",
        "mutation_idempotency_key": K1,
        "actor": "System",
        "action_name": "EXECUTION_PENDING",
        "action_status": "Success",
        "outcome_status": "PENDING",
        "request_payload_json": {"action_id": "ACT-8821", "tool_name": "cancel_order", "idempotency_key": K1},
        "response_payload_json": {"pending_action_status": "executing"},
        "timestamp": "2026-10-14T10:02:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440004",
        "session_id": "SES-VOX-8839201",
        "order_id": "ORD-1001",
        "mutation_idempotency_key": K1,
        "actor": "System",
        "action_name": "MUTATION_OUTCOME_UNKNOWN",
        "action_status": "Success",
        "outcome_status": "UNKNOWN",
        "request_payload_json": {"action_id": "ACT-8821", "tool_name": "cancel_order", "idempotency_key": K1},
        "response_payload_json": {
            "mutation_outcome": "UNKNOWN",
            "reason": "original_and_same_key_retry_responses_lost",
        },
        "timestamp": "2026-10-14T10:03:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440005",
        "session_id": "SES-VOX-8839201",
        "order_id": "ORD-1001",
        "mutation_idempotency_key": None,
        "actor": "Bot",
        "action_name": "HANDOFF_REQUEST_RECORDED",
        "action_status": "Success",
        "outcome_status": "SUCCESS",
        "request_payload_json": {"tool_name": "escalate_to_human", "trigger_reason": "CUSTOMER_REQUESTED_ESCALATION"},
        "response_payload_json": {
            "transfer_status": "HANDOFF_REQUEST_RECORDED",
            "agent_activity_state": "queued",
            "agent_id": None,
        },
        "timestamp": "2026-10-14T10:04:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440006",
        "session_id": "SES-VOX-8839201",
        "order_id": None,
        "mutation_idempotency_key": None,
        "actor": "System",
        "action_name": "LLM_PROVIDER_FAILOVER",
        "action_status": "Success",
        "outcome_status": "SUCCESS",
        "request_payload_json": {"primary_model": "gpt-6-luna", "consecutive_failures": 2},
        "response_payload_json": {"active_model": "claude-haiku-5-5", "session_scope": "remainder_of_session"},
        "timestamp": "2026-10-14T10:05:00Z",
    },
    {
        "audit_id": "550e8400-e29b-41d4-a716-446655440008",
        "session_id": "SES-VOX-8839201",
        "order_id": None,
        "mutation_idempotency_key": None,
        "actor": "System",
        "action_name": "TURN_COMPLETION_DEGRADED",
        "action_status": "Success",
        "outcome_status": "PENDING",
        "request_payload_json": {
            "reason": "jeval_noncommit_limit",
            "hold_episode_id": "HOLD-EP-07-01",
            "decision_request_id": "DR-7-01",
            "jeval_noncommit_count": 2,
            "hold_elapsed_ms": 620,
            "turn_epoch": 7,
        },
        "response_payload_json": {
            "turn_state": "DEGRADED/NEEDS_CLARIFICATION",
            "transcript_retained": True,
            "microphone_live": True,
            "prompt_when": "local_vad_quiet",
        },
        "timestamp": "2026-10-14T10:06:00Z",
    },
]


@pytest.fixture(params=BACKENDS)
async def runtime(request):
    rt = await make_runtime(request.param)
    yield rt
    await drop_runtime(rt)


def _shape(row):
    return {
        key: type(value).__name__
        for key, value in row.items()
        if key not in {"request_payload_json", "response_payload_json"}
    }


@pytest.mark.parametrize("golden", GOLDEN, ids=[row["action_name"] for row in GOLDEN])
async def test_writer_reproduces_golden_row_format(runtime, golden):
    order_hex = order_id_for(golden["order_id"]) if golden["order_id"] else None
    row = await runtime.audit.write(
        session_id=golden["session_id"],
        action_name=golden["action_name"],
        actor=golden["actor"],
        action_status=golden["action_status"],
        outcome_status=golden["outcome_status"],
        order_id=order_hex,
        mutation_idempotency_key=golden["mutation_idempotency_key"],
        request=golden["request_payload_json"],
        response=golden["response_payload_json"],
    )
    stored = audit_json(
        next(
            r
            for r in await runtime.repository.list_audit(golden["session_id"])
            if str(r["audit_id"]) == row["audit_id"]
        )
    )
    assert list(stored) == list(AUDIT_FIELDS) == list(golden)
    assert uuid.UUID(stored["audit_id"]).version == 4 and stored["audit_id"] != golden["mutation_idempotency_key"]
    assert stored["timestamp"].endswith("Z") and len(stored["timestamp"]) == len(golden["timestamp"])
    assert stored["request_payload_json"] == golden["request_payload_json"]
    assert stored["response_payload_json"] == golden["response_payload_json"]
    for key in ("session_id", "mutation_idempotency_key", "actor", "action_name", "action_status", "outcome_status"):
        assert stored[key] == golden[key]
    assert stored["order_id"] == order_hex  # G-4: the FK holds the 32-char hex id


async def test_resolution_gets_a_new_audit_id_and_keeps_the_key(runtime):
    first = await runtime.audit.write(
        session_id="S",
        action_name="EXECUTION_PENDING",
        actor="System",
        action_status="Success",
        outcome_status="PENDING",
        mutation_idempotency_key=K1,
    )
    second = await runtime.audit.write(
        session_id="S",
        action_name="ORDER_CANCELLED",
        actor="System",
        action_status="Success",
        outcome_status="SUCCESS",
        mutation_idempotency_key=K1,
    )
    assert first["audit_id"] != second["audit_id"]
    assert first["mutation_idempotency_key"] == second["mutation_idempotency_key"] == K1


async def test_payloads_are_masked_before_persisting(runtime):
    await runtime.audit.write(
        session_id="S-PII",
        action_name="TOOL_INVOCATION",
        actor="Bot",
        action_status="Success",
        outcome_status="SUCCESS",
        request={
            "tool_name": "x",
            "note": "call me on 0901234567 at 123 Nguyen Trai, Ward 2, District 5, Ho Chi Minh City, I am Lê Minh Anh",
        },
    )
    stored = str((await runtime.repository.list_audit("S-PII"))[0]["request_payload_json"])
    assert "0901234567" not in stored and "Nguyen Trai" not in stored and "Lê Minh Anh" not in stored
    assert "090***67" in stored


@pytest.mark.parametrize("bad", [{"actor": "Robot"}, {"action_name": "ORDER_DELETED"}, {"outcome_status": "OK"}])
async def test_vocabulary_outside_the_enums_is_rejected(runtime, bad):
    values = {
        "session_id": "S",
        "action_name": "TOOL_INVOCATION",
        "actor": "Bot",
        "action_status": "Success",
        "outcome_status": "SUCCESS",
        **bad,
    }
    with pytest.raises(ValueError):
        await runtime.audit.write(**values)
