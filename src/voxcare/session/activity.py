"""Session activity log: what happened in each turn, step by step, for the operator view and export.

Built on the server from what the turn actually did, never inferred in the browser:

- the caller's words as stored (PII-masked);
- the branch that handled the turn (a deterministic rule, or the conversation model proposing tool calls);
- each tool call with its arguments (masked), every middleware stage it passed or failed, and its result;
- every audit row the turn wrote (the same ``audit_id`` as in ``audit_log``);
- guarded-write state, session state that changed, and the reply.

Deliberately absent (product decision): model names and durations. Timings live in the OTel spans.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..security import pii

MAX_ENTRIES = 200
_ARG_CHARS = 80

ROUTE_TEXT: dict[str, str] = {
    "RESUME_ANSWER": "Caller answered the question about resuming the interrupted topic",
    "VERIFICATION": "Caller gave a phone number for the verification step (FR-AUTH-01)",
    "HANDOFF_OFFER_ACCEPTED": "Caller accepted the offer of a human follow-up",
    "WRITE_CONFIRMATION": "Caller confirmed the pending change",
    "WRITE_WITHDRAWAL": "Caller withdrew the change",
    "DIAGNOSIS_STEP": "Bounded diagnosis continues",
    "HUMAN_REQUEST": "Fixed rule: the caller asked for a human",
    "RMA_STATUS": "Fixed rule: repair-ticket status question (no read tool in Phase 1)",
    "OPD_FALLBACK": "Fixed rule: open policy decision, the fallback line is the only permitted answer",
    "WRITE_DATE": "Caller gave the new delivery date",
    "WRITE_REQUEST": "Fixed rule: change request detected (guarded write, preview first)",
    "DIAGNOSIS_START": "Fixed rule: product symptom detected, bounded diagnosis (at most 3 turns)",
    "MODEL_PLAN": "No fixed rule matched: the conversation model proposed tool calls",
    "SLOT_FILL": "Caller supplied the order code that was asked for: the order reads run again",
    "CLARIFY": "The conversation model proposed no tool: the caller is asked to rephrase",
    "POLICY_FALLBACK": "The conversation model proposed no tool, but the question names a policy topic: "
    "the policy clauses are searched",
    "CAPABILITY": "The conversation model proposed no tool: the caller is told what the assistant can do instead",
    "REPAIR_INTAKE": "Fixed rule: repair request without a recognised symptom, the caller is asked what is wrong",
    "REPAIR_SYMPTOM": "The described fault is not one the phone diagnosis covers: a technician request is offered",
    "RESUMED_REQUEST": "The calls refused before verification run again, unchanged",
    "POLICY_TOPICS": "Fixed rule: a policy question that names no topic, the caller is told which topics exist",
    "WRITE_DROPPED": "The caller moved on without confirming: the pending change is withdrawn, nothing was sent",
    "WRITE_ADDRESS": "Caller gave the new delivery address",
    "WRITE_REPLACED": "The caller asked again for the same kind of change: the earlier preview is withdrawn and "
    "replaced by the new one",
    "KEYWORD_FALLBACK": "The conversation model's plan was missing or unusable: the offline keyword planner's plan "
    "runs instead",
    "DIAGNOSIS_LEFT": "The caller made a new request instead of answering the diagnosis question: the diagnosis ends",
    "VOUCHER_LIST": "Fixed rule: a request to list promotions (no tool lists them), the caller is told what can be "
    "checked instead",
}

CHECK_LABELS: dict[str, str] = {
    "allowlist": "allowlist",
    "untrusted_input": "untrusted input",
    "arguments": "arguments",
    "verified_session": "verified session",
    "throttle": "anti-scraping throttle",
    "service": "service",
    "customer_scope": "customer scope",
}

OUTCOME_LABELS: dict[str, str] = {
    "ANSWERED": "Answered",
    "OUT_OF_SCOPE": "Refused: not on the verified account",
    "UNAUTHENTICATED_SESSION": "Refused: caller not verified",
    "NEEDS_VERIFICATION": "Asked the caller to verify",
    "VERIFIED": "Caller verified",
    "VERIFICATION_FAILED": "Verification failed",
    "NEEDS_ORDER_ID": "Asked for the order code",
    "NEEDS_DATE": "Asked for the new date",
    "INSUFFICIENT_EVIDENCE": "No policy clause applies",
    "OPD_FALLBACK": "Fallback line (open policy decision)",
    "HANDOFF_REQUEST_RECORDED": "Human follow-up recorded",
    "CONFIRMATION_REQUESTED": "Asked to confirm a change",
    "WRITE_REQUEST_RECORDED": "Change request recorded, not executed",
    "WRITE_COMPLETED": "Change executed (fake OMS)",
    "WITHDRAWN": "Change withdrawn",
    "REFUSED": "Change refused",
    "DIAGNOSIS_QUESTION": "Diagnosis question",
    "DIAGNOSIS_OPTIONS": "Diagnosis result and options",
    "CLARIFICATION": "Asked the caller to rephrase",
    "CLARIFICATION_LIMIT_REACHED": "Clarification limit reached",
    "RESUMED": "Resumed the interrupted topic",
    "RESUME_DECLINED": "Caller did not resume",
    "THROTTLED": "Refused: bulk catalog request",
    "DEPENDENCY_UNAVAILABLE": "Conversation model unavailable",
    "STALE_DISCARDED": "Discarded: superseded by a newer turn or an interruption",
    "CAPABILITY_ANSWER": "Explained what it can do instead",
    "REPAIR_QUESTION": "Asked what is wrong with the product",
    "REPAIR_UNKNOWN_SYMPTOM": "Offered a technician request",
    "UNCHANGED": "Already booked for that day: nothing to change",
    "NEEDS_ADDRESS": "Asked for the new address",
}

_STATE_TEXT: dict[str, Any] = {
    "customer_id": lambda old, new: f"Session bound to {new} (simulated verification, OPD-04)" if new else None,
    "active_order_id": lambda old, new: f"Active order: {new}" if new else "Active order cleared",
    "slot_state": lambda old, new: {
        "S_AUTH_AWAITING_PHONE": "Waiting for the phone number used at checkout",
        "S_ORDER_AWAITING_IDENTIFIER": "Waiting for the order code",
    }.get(str(new), "Slot cleared" if old else None),
    "pending_action_id": lambda old, new: f"Waiting for the caller to confirm {new}" if new else None,
    "handoff_offered": lambda old, new: f"Offered a human follow-up ({new})" if new else None,
    "resume_pending": lambda old, new: "Asked whether to resume the interrupted topic" if new else None,
    "diagnosis": lambda old, new: _diagnosis_changes(old or {}, new or {}),
    "symptom_history": lambda old, new: _diagnosis_outcome(new or []) if new and new != old else None,
}
STATE_KEYS = tuple(_STATE_TEXT)


def state_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    # deep copies: the diagnosis dict is updated in place during the turn
    return {key: copy.deepcopy(record.get(key)) for key in STATE_KEYS}


def _diagnosis_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """What the bounded diagnosis learned this turn: recorded answers, then the fault class and return window."""
    texts = []
    for item in list(new.get("answers", []))[len(old.get("answers", [])) :]:
        answer, question = item.get("answer"), item.get("question")
        texts.append(f"Answer recorded for the technician: \u201c{answer}\u201d to \u201c{question}\u201d")
    if new.get("diagnosis_class") and not old.get("diagnosis_class"):
        window = "inside" if new.get("inside_window") else "outside"
        delivered = new.get("delivered_on") or "unknown date"
        texts.append(
            f"Fault class {new['diagnosis_class']} from {new.get('symptom_code')} (fixed by the diagnosis rules); "
            f"{new.get('order_code') or 'the washer'} delivered {delivered}: {window} the 30-day return window"
        )
    return texts


def _diagnosis_outcome(history: list[Mapping[str, Any]]) -> str | None:
    choice = next((item.get("choice") for item in history if item.get("choice")), None)
    if not choice:
        return None
    label = _CHOICE_LABELS.get(str(choice), str(choice).replace("_", " ").lower())
    return (
        f"Caller chose {label}; symptom, answers and fault class go to the technician with the handoff (no RMA ticket)"
    )


_CHOICE_LABELS = {
    "EXCHANGE": "an exchange",
    "RETURN_REFUND": "a return and refund",
    "REPAIR": "a repair under warranty",
}


def outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(outcome, outcome.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------------------------- entries


def session_entry(*, customer_id: str | None, verification_mode: str) -> dict[str, Any]:
    if customer_id:
        text = f"Session started for a verified caller ({customer_id}, simulated, OPD-04)"
    else:
        text = "Session started for an unverified caller: order data needs the verification step first"
    return {"kind": "session", "summary": text, "steps": []}


def event_entry(summary: str, *steps: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": "event", "summary": summary, "steps": list(steps)}


def turn_entry(
    *,
    turn: int,
    source: str,
    caller_text: str,
    outcome: str,
    reply_text: str | None,
    route: Sequence[Mapping[str, Any]] = (),
    results: Sequence[Any] = (),
    write: Any = None,
    audit_rows: Iterable[Mapping[str, Any]] = (),
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    normalized_codes: Sequence[str] = (),
    normalized_terms: Sequence[str] = (),
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if normalized_codes:
        steps.append(step("input", f"Spoken order code normalised to {', '.join(normalized_codes)}"))
    if normalized_terms:
        steps.append(step("input", f"Misheard terms corrected: {', '.join(normalized_terms)}"))
    for item in route:
        steps.append(route_step(item))
        for fix in item.get("completed") or ():
            steps.append(step("input", f"Plan completed from the caller's words: {fix}"))
        if item.get("model_note"):
            note = pii.mask(" ".join(str(item["model_note"]).split()))
            note = note if len(note) <= 240 else note[:239] + "…"
            steps.append(step("note", f"Model text, not spoken (replies come from tool results only): {note}"))
    for result in results:
        steps.append(tool_step(result))
    if write is not None:
        steps.append(write_step(write))
    steps.extend(audit_step(row) for row in audit_rows)
    if before is not None and after is not None:
        steps.extend(state_steps(before, after))
    if reply_text:
        steps.append(step("reply", pii.mask(reply_text), outcome=outcome))
    return {
        "kind": "turn",
        "turn": turn,
        "source": "voice" if source == "asr_final" else "typed",
        "caller_text": caller_text,
        "outcome": outcome,
        "summary": outcome_label(outcome),
        "steps": steps,
    }


def support_entry(result: Any, audit_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = "Support button: human follow-up recorded" if result.ok else "Support button: request not recorded"
    steps = [tool_step(result), *(audit_step(row) for row in audit_rows)]
    return {"kind": "support", "summary": summary, "steps": steps}


# ---------------------------------------------------------------------------------------------- steps


def step(kind: str, text: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "text": text, **{key: value for key, value in fields.items() if value is not None}}


def route_step(item: Mapping[str, Any]) -> dict[str, Any]:
    code = str(item.get("code"))
    text = ROUTE_TEXT.get(code, code)
    detail = {key: value for key, value in item.items() if key != "code"}
    extra = ""
    if code == "VERIFICATION":
        if detail.get("matched"):
            extra = f": {detail.get('phone')} matches {detail.get('customer_id')}"
            if detail.get("resumed_request"):
                extra += "; the earlier request is answered now"
        else:
            extra = f": {detail.get('phone')} matches no customer (attempt {detail.get('attempt')})"
    elif code == "OPD_FALLBACK":
        extra = f" ({detail.get('opd')})"
    elif code == "WRITE_REQUEST":
        extra = f": {detail.get('action_type')} {detail.get('order_code')}"
    elif code in {"WRITE_DROPPED", "WRITE_REPLACED"}:
        extra = f" ({detail.get('action_id')} for {detail.get('order_code')})"
    elif code == "WRITE_ADDRESS":
        extra = f" for {detail.get('order_code')}"
    elif code == "WRITE_DATE":
        extra = f": {detail.get('target_date')} for {detail.get('order_code')}"
    elif code == "DIAGNOSIS_START":
        extra = f": {detail.get('symptom')}"
    elif code == "DIAGNOSIS_STEP":
        extra = f" (stage {detail.get('stage')})"
    elif code == "POLICY_FALLBACK":
        extra = f" (category {detail.get('category')})"
    elif code in {"REPAIR_INTAKE", "REPAIR_SYMPTOM"} and detail.get("product"):
        extra = f": {detail['product']}"
    elif code == "HANDOFF_OFFER_ACCEPTED":
        extra = f" ({detail.get('reason')})"
    elif code in {"MODEL_PLAN", "SLOT_FILL", "RESUMED_REQUEST", "KEYWORD_FALLBACK"} and detail.get("tool_calls"):
        extra = f" ({detail['tool_calls']} call{'s' if detail['tool_calls'] != 1 else ''})"
        if detail.get("reason"):
            extra += f": {detail['reason']}"
    elif code == "DIAGNOSIS_LEFT":
        extra = f" (it was at stage {detail.get('stage')}; nothing was recorded)"
    return step("route", text + extra, code=code)


def tool_step(result: Any) -> dict[str, Any]:
    arguments = pii.mask_payload({key: value for key, value in (result.arguments or {}).items() if value is not None})
    return step(
        "tool",
        f"{result.name}({_format_args(arguments)})",
        name=result.name,
        arguments=arguments,
        checks=[
            {**check, "label": CHECK_LABELS.get(check["check"], check["check"])}
            for check in getattr(result, "checks", [])
        ],
        invoked=bool(result.invoked),
        result=describe_result(result),
        status="SUCCESS" if result.ok else str(result.error_code),
    )


def describe_result(result: Any) -> str:
    envelope: Mapping[str, Any] = result.envelope
    internal: Mapping[str, Any] = result.internal or {}
    name = result.name
    if not result.ok:
        code = str(result.error_code)
        if internal.get("_throttled"):
            return "refused: bulk catalog request (FR-SEC-01); nothing returned"
        if code == "OUT_OF_SCOPE":
            return "refused: belongs to another customer; no details returned"
        if code == "UNAUTHENTICATED_SESSION":
            return "refused before any read: caller not verified"
        if code == "INVALID_INPUT":
            if name == "apply_voucher" and internal.get("_reason"):
                return f"voucher not applied ({internal['_reason']})"
            return f"blocked before the call: {internal.get('_reason') or envelope.get('message') or 'invalid input'}"
        if code == "INSUFFICIENT_EVIDENCE":
            return (
                f"no clause reached the threshold (best {envelope.get('highest_similarity_score')} "
                f"< {envelope.get('threshold_required')})"
            )
        if code == "NOT_FOUND":
            return "no matching record"
        return f"failed: {code}"
    if name == "lookup_order_details":
        order = envelope["order"]
        return (
            f"{order['order_code']}: {order['order_status']}, version {order['version']}, {len(order['items'])} item(s)"
        )
    if name == "track_shipment_status":
        shipment = envelope["shipment"]
        status = shipment.get("shipment_status") or "no shipment yet"
        eta = shipment.get("estimated_delivery_date") or "not provided"
        return f"{status}, estimated delivery {eta}, {len(envelope['events'])} carrier event(s)"
    if name == "list_customer_orders":
        codes = [order["order_code"] for order in envelope["orders"]]
        return f"{len(codes)} order(s) on the account: {', '.join(codes) or 'none'}"
    if name == "retrieve_policy_clauses":
        clauses = ", ".join(
            f"{clause['clause_id']} ({clause['similarity_score']:.2f})" for clause in envelope["clauses"]
        )
        return f"grounded on {clauses}"
    if name == "search_catalog":
        skus = [row["sku"] for row in envelope["results"]]
        return f"{len(skus)} product(s): {', '.join(skus) or 'none'}"
    if name == "compare_products":
        skus = [row["sku"] for row in envelope["products"]]
        missing = envelope.get("not_recorded") or {}
        tail = f"; not recorded: {missing}" if missing else ""
        return f"compared {', '.join(skus)}{tail}"
    if name == "apply_voucher":
        return f"valid: discount {int(envelope['discount_vnd']):,} VND (cap {int(envelope['capped_at_vnd']):,} VND)"
    if name == "escalate_to_human":
        return (
            f"recorded: ticket {envelope['ticket_number']}, handoff {envelope['handoff_id']}, "
            f"{envelope['transfer_status']} (no live transfer)"
        )
    return "success"


def write_step(write: Any) -> dict[str, Any]:
    action = write.action or {}
    code = action.get("target_order_code", "")
    kind = write.kind
    texts = {
        "PREVIEW": f"Change previewed, nothing written: {action.get('action_type')} {code} "
        f"({action.get('action_id')}, {action.get('steps_required', 1)} confirmation(s) needed)",
        "CONFIRM_AGAIN": f"Confirmation {action.get('confirmations')} of {action.get('steps_required')} received",
        "SUBMITTED": f"{action.get('action_id')} sealed (pending → submitted) and recorded for VoxCom; "
        f"{code} was not changed",
        "COMPLETED": f"{action.get('action_id')} executed by the fake OMS under the version check (→ completed)",
        "ALREADY_APPLIED": f"{action.get('action_id')} was already applied; nothing executed twice",
        "CANCELLED": f"{action.get('action_id') or 'The change'} withdrawn before execution; nothing was sent",
        "TOO_LATE": f"{action.get('action_id')} was already committed; a reversal is needed to undo it",
        "REFUSED": f"Change refused by the rules for {code}; nothing written",
        "EXPIRED": "The preview expired; nothing written",
        "UNCHANGED": "The requested day is the one already booked; no change previewed, nothing written",
        "OPD_03": "Already dispatched, so no cancellation preview: the request is recorded for a human (OPD-03)",
        "NEEDS_DATE": "No new day given yet; nothing previewed",
        "NEEDS_ADDRESS": "No new address given yet; nothing previewed",
        "NOT_FOUND": f"{code or 'The order'} was not found; nothing previewed",
    }
    return step(
        "write",
        texts.get(kind, f"Write result {kind}" + (f" ({action.get('status')})" if action.get("status") else "")),
        result=kind,
        action_id=action.get("action_id"),
        status=action.get("status"),
    )


def audit_step(row: Mapping[str, Any]) -> dict[str, Any]:
    request = row.get("request_payload_json") or {}
    action = str(row.get("action_name"))
    if action == "LLM_PROVIDER_FAILOVER":
        subject = "switched to the fallback conversation model for the rest of the session"
    else:
        subject = request.get("tool_name") or request.get("order_code") or ""
    status = f"{row.get('action_status')}/{row.get('outcome_status')}"
    return step(
        "audit",
        f"{action} · {status}" + (f" · {subject}" if subject else ""),
        audit_id=str(row.get("audit_id")),
        action_name=action,
        action_status=row.get("action_status"),
        outcome_status=row.get("outcome_status"),
    )


def state_steps(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = []
    for key, describe in _STATE_TEXT.items():
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        described = describe(old, new)
        for text in described if isinstance(described, list) else [described]:
            if text:
                steps.append(step("state", text, key=key))
    return steps


def _format_args(arguments: Mapping[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = repr(value) if isinstance(value, str) else str(value)
        if len(rendered) > _ARG_CHARS:
            rendered = rendered[: _ARG_CHARS - 1] + "…'"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------- export


def to_markdown(session_id: str, customer_id: str | None, entries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"# VoxCare session activity: {session_id}",
        "",
        f"Customer binding: {customer_id or 'not verified'}. Values are PII-masked as stored. Fictional demo data.",
        "",
    ]
    for entry in entries:
        if entry["kind"] == "turn":
            lines.append(f"## Turn {entry['turn']} ({entry['source']}): {entry['summary']}")
            lines.append("")
            lines.append(f"> {entry['caller_text']}")
        else:
            lines.append(f"## {entry['summary']}")
        lines.append("")
        for step in entry["steps"]:
            lines.append(f"- **{step['kind']}**: {step['text']}")
            if step["kind"] == "tool":
                checks = " · ".join(f"{check['label']} {_mark(check)}" for check in step.get("checks", []))
                if checks:
                    lines.append(f"  - checks: {checks}")
                lines.append(f"  - result: {step['result']}")
            elif step["kind"] == "audit":
                lines.append(f"  - audit_id: `{step['audit_id']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _mark(check: Mapping[str, Any]) -> str:
    symbol = {"pass": "✓", "fail": "✗"}.get(str(check.get("result")), "•")
    return f"{symbol} ({check['detail']})" if check.get("detail") else symbol


__all__ = [
    "MAX_ENTRIES",
    "OUTCOME_LABELS",
    "ROUTE_TEXT",
    "STATE_KEYS",
    "audit_step",
    "describe_result",
    "event_entry",
    "outcome_label",
    "session_entry",
    "state_snapshot",
    "step",
    "support_entry",
    "to_markdown",
    "tool_step",
    "turn_entry",
    "write_step",
]
