"""Dialogue manager: one caller turn → tool calls through the middleware → one grounded reply.

Order of decisions for a turn (first match wins):

1. a pending **resume question** (UC-23) is answered yes/no;
2. the **verification sub-dialogue** (FR-AUTH-01) takes the phone number;
3. a pending **handoff offer** is accepted;
4. a previewed **write** is confirmed or withdrawn (record-only in Phase 1);
5. an active **after-sales diagnosis** (≤ 3 diagnostic turns, D§9.3) continues;
6. an explicit request for a **person**, or for RMA status (UC-17), records a handoff;
7. an **open-policy-decision** topic gets its fallback line, the only permitted answer (D§8.4);
8. a **write request** (cancel, reschedule, address change, reinstatement) is previewed with zero mutations;
9. a reported **symptom** starts the diagnosis;
10. otherwise the conversation model **plans read tools**; the middleware runs them and
    :mod:`.render` speaks the result.

Writes are never proposed by, or visible to, the model.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..contracts import MAX_TOOL_CALLS_PER_TURN, PlannedToolCall
from ..domain import copy
from ..domain.clock import Clock, local_today
from ..domain.enums import Actor
from ..domain.rules import classify_fault, detect_symptom, product_categories_in, within_return_window
from ..domain.services.writes import WriteOutcome, WriteService
from ..middleware.registry import TOOLS
from ..middleware.tool_middleware import ToolContext, ToolMiddleware, ToolResult
from ..providers.keyword import KeywordPlanner
from ..security import pii
from ..storage.db import Repository
from . import intents
from .render import Rendered, render_results, write_guard
from .speech import speakable

logger = logging.getLogger("voxcare.dialogue")

S_ORDER_AWAITING_IDENTIFIER = "S_ORDER_AWAITING_IDENTIFIER"
S_AUTH_AWAITING_PHONE = "S_AUTH_AWAITING_PHONE"
MAX_CLARIFICATIONS = 3
MAX_HISTORY_MESSAGES = 24
_META_QUESTION = re.compile(
    r"\b(?:what\s+do\s+you\s+mean|repeat|say\s+(?:that|it)\s+again|pardon|sorry\?|come\s+again)\b", re.I
)


@dataclass(frozen=True)
class Flags:
    fr_auth_01: bool = True  # verification sub-dialogue
    fr_auth_02: bool = True  # Scenario 3 full refusal wording
    fr_sec_01: bool = True  # anti-scraping throttle (applied in the middleware)
    write_mode: str = "record_only"  # record_only | fake_oms


@dataclass
class TurnReply:
    text: str
    voice: str
    outcome: str
    results: list[ToolResult] = field(default_factory=list)
    plans: list[PlannedToolCall] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    binding_changed: bool = False
    provider_error: BaseException | None = None
    topic: str | None = None
    write: WriteOutcome | None = None
    # How the turn was handled, in order: [{"code": ROUTE, ...details}] (read by the activity log).
    route: list[dict[str, Any]] = field(default_factory=list)


def _at(code: str, reply: TurnReply, **detail: Any) -> TurnReply:
    """Record the branch that handled this turn before any branch it delegated to."""
    reply.route.insert(0, {"code": code, **{key: value for key, value in detail.items() if value is not None}})
    return reply


def _reply(text: str, outcome: str, *, topic: str | None = None, **fields: Any) -> TurnReply:
    return TurnReply(text=text, voice=speakable(text), outcome=outcome, topic=topic, **fields)


class DialogueManager:
    def __init__(
        self,
        *,
        middleware: ToolMiddleware,
        writes: WriteService,
        repository: Repository,
        clock: Clock,
        flags: Flags | None = None,
    ) -> None:
        self.middleware = middleware
        self.writes = writes
        self.repository = repository
        self.clock = clock
        self.flags = flags or Flags()

    # ------------------------------------------------------------------ entry point

    async def respond(
        self,
        state: dict[str, Any],
        *,
        raw_text: str,
        text: str,
        planner: Any,
        is_current: Callable[[], bool],
    ) -> TurnReply:
        reply = await self._decide(state, raw_text=raw_text, text=text, planner=planner, is_current=is_current)
        if reply.outcome in {"STALE_DISCARDED", "DEPENDENCY_UNAVAILABLE"}:
            return reply
        if reply.write is not None:  # the write guard covers write-flow replies (reads may report a cancelled order)
            guarded = write_guard(reply.text, execution_confirmed=reply.write.kind in {"COMPLETED", "ALREADY_APPLIED"})
            if guarded != reply.text:
                reply.text, reply.voice = guarded, speakable(guarded)
        if reply.outcome not in {"CLARIFICATION", "CLARIFICATION_LIMIT_REACHED"}:
            state["clarifications"] = 0  # three unclear turns in a row, not three over the whole call
        self._maybe_ask_resume(state, reply)
        state["last_reply"] = reply.text
        if reply.topic:
            state["last_topic"] = reply.topic
        return reply

    async def _decide(
        self, state: dict[str, Any], *, raw_text: str, text: str, planner: Any, is_current: Callable[[], bool]
    ) -> TurnReply:
        """``raw_text`` feeds only the in-process rule detectors; ``text`` (masked) is what is stored or sent."""
        probe = raw_text
        intake = state.pop("repair_intake", None)  # the repair question is answered on the very next turn
        if state.get("resume_pending"):
            state["resume_pending"] = False
            paused = state.pop("paused_topic", None)
            if intents.YES.search(probe) and paused:
                reply = _reply(paused.get("reply") or copy.RESUME_NOTHING, "RESUMED", topic=paused.get("label"))
                return _at("RESUME_ANSWER", reply, answer="yes", topic=paused.get("label"))
            if intents.NO.search(probe):
                return _at("RESUME_ANSWER", _reply(copy.RESUME_DECLINED, "RESUME_DECLINED"), answer="no")

        if state.get("slot_state") == S_AUTH_AWAITING_PHONE:
            phone = pii.find_phone_number(raw_text)
            if phone:
                return await self._verify(state, phone, planner=planner, is_current=is_current)

        offered = state.pop("handoff_offered", None)
        if offered and intents.YES.search(probe) and not state.get("pending_action_id"):
            return _at("HANDOFF_OFFER_ACCEPTED", await self._escalate(state, text, offered, "NORMAL"), reason=offered)

        dropped: WriteOutcome | None = None
        if state.get("pending_action_id"):
            if intents.YES.search(probe):
                return _at("WRITE_CONFIRMATION", await self._confirm_write(state))
            if intents.NO.search(probe) or intents.WITHDRAW.search(probe):
                return _at("WRITE_WITHDRAWAL", await self._withdraw(state, state.pop("pending_action_id")))
            # Anything else: a preview is confirmed only by the very next answer, so a later, unrelated "yes"
            # (to a diagnosis question, say) can never seal it (implementation choice).
            dropped = await self.writes.withdraw(state.pop("pending_action_id"), actor=Actor.SYSTEM)
        if state.get("submitted_action_id") and intents.WITHDRAW.search(probe):
            return _at("WRITE_WITHDRAWAL", await self._withdraw(state, state["submitted_action_id"]))

        reply = await self._decide_request(
            state, probe=probe, text=text, intake=intake, planner=planner, is_current=is_current
        )
        if dropped is not None and dropped.kind == "CANCELLED" and dropped.action:
            code = dropped.action["target_order_code"]
            new_action = (reply.write.action or {}) if reply.write else {}
            # "Actually make it Wednesday": a new preview of the same change already says what will happen.
            same_change = new_action.get("target_order_code") == code and new_action.get(
                "action_type"
            ) == dropped.action.get("action_type")
            if reply.outcome not in {"STALE_DISCARDED", "DEPENDENCY_UNAVAILABLE"} and not same_change:
                note = copy.pending_dropped(code)
                reply.text = f"{note} {reply.text}".strip()
                reply.voice = f"{speakable(note)} {reply.voice}".strip()
            route = "WRITE_REPLACED" if same_change else "WRITE_DROPPED"
            _at(route, reply, action_id=dropped.action["action_id"], order_code=code)
        return reply

    async def _decide_request(
        self,
        state: dict[str, Any],
        *,
        probe: str,
        text: str,
        intake: Mapping[str, Any] | None,
        planner: Any,
        is_current: Callable[[], bool],
    ) -> TurnReply:
        """Everything after the yes/no answers to a pending question: the caller's new request."""
        if state.get("diagnosis"):
            stage = state["diagnosis"].get("turn")
            if _leaves_diagnosis(probe, stage):
                # A new request is not an answer to the diagnosis question (logs SES-a70870f2, SES-7c2df43a):
                # the diagnosis ends unrecorded and the request is handled as usual.
                state.pop("diagnosis", None)
                reply = await self._decide_request(
                    state, probe=probe, text=text, intake=intake, planner=planner, is_current=is_current
                )
                return _at("DIAGNOSIS_LEFT", reply, stage=stage)
            return _at("DIAGNOSIS_STEP", await self._diagnosis_step(state, text, probe=probe), stage=stage)

        if intents.HUMAN.search(probe):
            return _at("HUMAN_REQUEST", await self._escalate(state, text, "CUSTOMER_REQUESTED_ESCALATION", "NORMAL"))
        if intents.RMA_STATUS.search(probe):
            reply = await self._escalate(state, text, "COMPLEX_EXCEPTION", "NORMAL", spoken=copy.RMA_STATUS_HANDOFF)
            return _at("RMA_STATUS", reply)

        topic = intents.opd_topic(probe)
        if topic is not None:
            line = {
                "OPD-01": copy.OPD_01_RETURN_FREIGHT,
                "OPD-02": copy.OPD_02_REFUND_TURNAROUND,
                "OPD-05": copy.OPD_05_COSMETIC,
                "OPD-06": copy.OPD_06_BEYOND_30KM,
            }[topic]
            reply = _reply(
                line, "OPD_FALLBACK", topic="your question", observations=[{"event_type": "OPD_FALLBACK", "opd": topic}]
            )
            return _at("OPD_FALLBACK", reply, opd=topic)

        if state.get("awaiting_write_address"):
            pending_address = state.pop("awaiting_write_address")
            address = intents.new_address_in(probe) or (probe.strip() if intents.ADDRESS_SHAPE.search(probe) else None)
            if address:
                reply = await self._preview_write(
                    state, "AddressChange", pending_address["order_code"], text, probe=f"address to {address}"
                )
                return _at("WRITE_ADDRESS", reply, order_code=pending_address["order_code"])

        if state.get("awaiting_write_date"):
            pending = state.pop("awaiting_write_date")
            target = intents.parse_target_date(probe, local_today(self.clock))
            if target is not None:
                reply = await self._preview_write(
                    state, "Reschedule", pending["order_code"], text, target=target, probe=probe
                )
                return _at("WRITE_DATE", reply, order_code=pending["order_code"], target_date=target.isoformat())

        intent = intents.write_intent(probe)
        order_code = intents.order_code_in(probe) or (state.get("active_order_id") if intent else None)
        if intent and order_code:
            target = intents.parse_target_date(probe, local_today(self.clock)) if intent == "Reschedule" else None
            reply = await self._preview_write(state, intent, order_code, text, target=target, probe=probe)
            return _at("WRITE_REQUEST", reply, action_type=intent, order_code=order_code)

        symptom = detect_symptom(probe)
        washer_intake = bool(intake) and intake.get("category") == "WASHING_MACHINE"
        if symptom and (intents.WASHER_CONTEXT.search(probe) or symptom == "SPIN_CYCLE_ERROR_E4" or washer_intake):
            return _at("DIAGNOSIS_START", await self._start_diagnosis(state, text, symptom), symptom=symptom)
        if intake and not intents.order_code_in(probe) and intents.policy_category(probe) is None:
            # implementation choice: only the washer E4/spin tree is diagnosable by phone in Phase 1
            state["handoff_offered"] = "COMPLEX_EXCEPTION"
            reply = _reply(copy.DIAGNOSIS_UNKNOWN_SYMPTOM, "REPAIR_UNKNOWN_SYMPTOM", topic=intake.get("topic"))
            return _at("REPAIR_SYMPTOM", reply, product=intake.get("label"))
        if intents.vague_policy_question(probe):
            return _at("POLICY_TOPICS", _reply(copy.POLICY_TOPICS, "CAPABILITY_ANSWER", topic="VoxCom policies"))
        if intents.repair_request(probe):
            category = product_categories_in(probe)[0]
            label = _PRODUCT_LABELS.get(category, "appliance")
            state["repair_intake"] = {"category": category, "label": label, "topic": f"the {label} repair"}
            reply = _reply(copy.repair_symptom_question(label), "REPAIR_QUESTION", topic=f"the {label} repair")
            return _at("REPAIR_INTAKE", reply, product=label)

        if intents.VOUCHER_LIST.search(probe) and not intents.VOUCHER_CODE_IN_TEXT.search(probe):
            # No tool lists promotions; the model otherwise re-quotes a code from earlier turns (log SES-7c2df43a).
            reply = _reply(copy.VOUCHER_LIST_UNAVAILABLE, "CAPABILITY_ANSWER", topic="vouchers")
            return _at("VOUCHER_LIST", reply)
        return await self._plan_and_run(state, text, planner=planner, is_current=is_current)

    # ------------------------------------------------------------------ model-planned reads

    async def _plan_and_run(
        self, state: dict[str, Any], text: str, *, planner: Any, is_current: Callable[[], bool]
    ) -> TurnReply:
        plans: list[PlannedToolCall] = []
        route = "MODEL_PLAN"
        if state.get("slot_state") == S_ORDER_AWAITING_IDENTIFIER:
            # The caller is answering "Could you please tell me your order code?": repeat the order reads
            # deterministically instead of letting the model re-guess the original intent.
            plans, route = self._slot_fill_plan(state, text), "SLOT_FILL"
        if not plans:
            if planner is None:
                return TurnReply("", "", "DEPENDENCY_UNAVAILABLE")
            try:
                raw = await planner.plan_turn(
                    user_text=text, history=tuple(state.get("history", ())), available_tools=TOOLS
                )
            except Exception as exc:  # the boundary logs and reports the provider failure
                return TurnReply("", "", "DEPENDENCY_UNAVAILABLE", provider_error=exc)
            if not is_current():
                return TurnReply("", "", "STALE_DISCARDED")
            note = getattr(raw, "note", None)
            plans, route = _normalise(raw), "MODEL_PLAN"
            if not plans:
                return await self._no_tool(state, text, is_current=is_current, note=note)
            plans, completed = _complete_order_reads(plans, text, active_order=state.get("active_order_id"))
            keyword_plan = _normalise(
                await _KEYWORD_PLANNER.plan_turn(user_text=text, history=(), available_tools=TOOLS)
            )
            keyword_names = {plan.name for plan in keyword_plan}
            usable_keyword_plan = bool(keyword_plan) and not any(_missing_arguments(plan) for plan in keyword_plan)
            model_names = {plan.name for plan in plans}
            empty = [plan.name for plan in plans if _missing_arguments(plan)]
            reason = None
            if empty and set(empty) <= keyword_names and usable_keyword_plan:
                # A call with no usable arguments (seen when the model's output was cut inside its JSON).
                reason = f"{', '.join(empty)} had no arguments"
            elif "compare_products" in keyword_names - model_names and usable_keyword_plan:
                # The caller asked to compare and the model planned no comparison (log SES-7c2df43a, turn 13).
                reason = "the caller asked for a comparison and the model planned none"
            if reason:
                return await self._run_plans(
                    state,
                    text,
                    keyword_plan,
                    route="KEYWORD_FALLBACK",
                    is_current=is_current,
                    reason=reason,
                    model_note=note,
                )
            if (
                intents.VOUCHER_CODE_IN_TEXT.search(text)
                and all(plan.name != "apply_voucher" for plan in plans)
                and len(plans) < MAX_TOOL_CALLS_PER_TURN
            ):
                # The caller named a voucher code but the model did not check it (log SES-36e8a973, Scenario 7):
                # the keyword planner's voucher check, built from the caller's words, is appended.
                voucher = next((plan for plan in keyword_plan if plan.name == "apply_voucher"), None)
                if voucher is not None:
                    plans = [*plans, voucher]
                    completed.append(
                        f"apply_voucher for {voucher.arguments.get('voucher_code')} added: the caller named it"
                    )
            return await self._run_plans(
                state, text, plans, route=route, is_current=is_current, completed=completed or None, model_note=note
            )
        return await self._run_plans(state, text, plans, route=route, is_current=is_current)

    async def _run_plans(
        self,
        state: dict[str, Any],
        text: str,
        plans: list[PlannedToolCall],
        *,
        route: str,
        is_current: Callable[[], bool],
        **detail: Any,
    ) -> TurnReply:
        results: list[ToolResult] = []
        for plan in plans:
            result = await self.middleware.handle(self._context(state, text), plan.name, plan.arguments)
            if not is_current():
                return TurnReply("", "", "STALE_DISCARDED")
            results.append(result)
        rendered = render_results(results, user_text=text, active_order=state.get("active_order_id"))
        return _at(route, self._finish(state, text, rendered, results, plans), tool_calls=len(plans), **detail)

    async def _no_tool(
        self, state: dict[str, Any], text: str, *, is_current: Callable[[], bool], note: str | None = None
    ) -> TurnReply:
        """The model proposed no tool. A named policy topic still goes to the policy tool (with its category);
        then the offline keyword planner is tried (same tools, same middleware); a request for a list of
        promotions gets a capability answer; anything else is a clarification (implementation choices)."""
        category = intents.policy_category(text)
        if category is not None:
            plan = PlannedToolCall("retrieve_policy_clauses", {"query": text, "category": category}, "policy-fallback")
            return await self._run_plans(
                state,
                text,
                [plan],
                route="POLICY_FALLBACK",
                is_current=is_current,
                category=category,
                model_note=note,
            )
        fallback = _normalise(await _KEYWORD_PLANNER.plan_turn(user_text=text, history=(), available_tools=TOOLS))
        if fallback:
            return await self._run_plans(
                state, text, fallback, route="KEYWORD_FALLBACK", is_current=is_current, model_note=note
            )
        if intents.VOUCHER_LIST.search(text) and not intents.VOUCHER_CODE_IN_TEXT.search(text):
            reply = _reply(copy.VOUCHER_LIST_UNAVAILABLE, "CAPABILITY_ANSWER", topic="vouchers")
            return _at("CAPABILITY", reply, model_note=note)
        return _at("CLARIFY", self._clarify(state), model_note=note)

    def _slot_fill_plan(self, state: Mapping[str, Any], text: str) -> list[PlannedToolCall]:
        """After "Could you please tell me your order code?", a bare code re-issues the order reads."""
        if state.get("slot_state") != S_ORDER_AWAITING_IDENTIFIER:
            return []
        code = intents.order_code_in(text)
        if code:
            return [
                PlannedToolCall("lookup_order_details", {"order_code": code}, "slot-fill-1"),
                PlannedToolCall("track_shipment_status", {"order_code": code, "tracking_code": None}, "slot-fill-2"),
            ]
        tracking = intents.TRACKING_CODE_IN_TEXT.search(text)
        if tracking:
            return [
                PlannedToolCall(
                    "track_shipment_status",
                    {"order_code": None, "tracking_code": tracking.group(0).upper()},
                    "slot-fill",
                )
            ]
        return []

    def _finish(
        self,
        state: dict[str, Any],
        text: str,
        rendered: Rendered,
        results: list[ToolResult],
        plans: list[PlannedToolCall],
    ) -> TurnReply:
        if rendered.needs_verification:
            catalog = any(r.name == "search_catalog" and r.error_code == "UNAUTHENTICATED_SESSION" for r in results)
            if self.flags.fr_auth_01:
                state["slot_state"] = S_AUTH_AWAITING_PHONE
                state["verification"] = {
                    "pending_request": text,
                    # the calls the gate refused, re-run as they were once the caller is verified
                    "pending_plans": [{"name": plan.name, "arguments": dict(plan.arguments)} for plan in plans],
                    "attempts": 0,
                }
                line = copy.CATALOG_UNAUTHENTICATED if catalog else copy.VERIFY_ACCOUNT_PROMPT
            else:
                line = copy.CATALOG_UNAUTHENTICATED if catalog else copy.ORDER_UNAUTHENTICATED
            others = rendered.text.strip()
            full = f"{others} {line}".strip() if others else line
            return TurnReply(full, speakable(full), "NEEDS_VERIFICATION", results=results, plans=plans)
        if rendered.missing_slot:
            state["slot_state"] = S_ORDER_AWAITING_IDENTIFIER
        elif state.get("slot_state") == S_ORDER_AWAITING_IDENTIFIER:
            state["slot_state"] = None
        text_out, voice_out = rendered.text, rendered.voice
        if not self.flags.fr_auth_02:
            for result in results:
                if result.name == "lookup_order_details" and result.error_code == "OUT_OF_SCOPE":
                    code = result.internal.get("_order_code", "")
                    text_out = text_out.replace(copy.order_out_of_scope_full(code), copy.ORDER_OUT_OF_SCOPE)
                    voice_out = speakable(text_out)
        for result in results:
            if result.ok and result.name in {"lookup_order_details", "track_shipment_status"}:
                code = (result.envelope.get("order") or result.envelope.get("shipment") or {}).get("order_code")
                if code:
                    state["active_order_id"] = code
        if rendered.outcome in {"SERVICE_TIMEOUT", "TOOL_ERROR"}:
            state["handoff_offered"] = "COMPLEX_EXCEPTION"
        elif rendered.outcome == "INSUFFICIENT_EVIDENCE":
            state["handoff_offered"] = "LOW_CONFIDENCE_THRESHOLD"
        state["clarifications"] = 0
        return TurnReply(
            text_out,
            voice_out,
            rendered.outcome,
            results=results,
            plans=plans,
            evidence=rendered.evidence,
            topic=rendered.topic,
        )

    def _clarify(self, state: dict[str, Any]) -> TurnReply:
        state["clarifications"] = int(state.get("clarifications", 0)) + 1
        if state["clarifications"] >= MAX_CLARIFICATIONS:
            state["handoff_offered"] = "CLARIFICATION_LIMIT_EXCEEDED"
            return _reply(
                copy.CLARIFY_LIMIT,
                "CLARIFICATION_LIMIT_REACHED",
                observations=[{"event_type": "CLARIFICATION_REQUESTED", "count": state["clarifications"]}],
            )
        return _reply(
            copy.CLARIFY,
            "CLARIFICATION",
            observations=[{"event_type": "CLARIFICATION_REQUESTED", "count": state["clarifications"]}],
        )

    # ------------------------------------------------------------------ verification (FR-AUTH-01)

    async def _verify(
        self, state: dict[str, Any], phone: str, *, planner: Any, is_current: Callable[[], bool]
    ) -> TurnReply:
        customer = await self.repository.find_customer_by_phone(phone)
        verification = state.setdefault("verification", {"attempts": 0})
        if customer is None:
            verification["attempts"] = int(verification.get("attempts", 0)) + 1
            attempt = verification["attempts"]
            if attempt >= 2:
                state["slot_state"] = None
                state["handoff_offered"] = "LOW_CONFIDENCE_THRESHOLD"
                reply = _reply(f"{copy.VERIFY_FAILED} {copy.OFFER_HANDOFF}", "VERIFICATION_FAILED")
                return _at("VERIFICATION", reply, matched=False, attempt=attempt, phone=pii.mask_phone_number(phone))
            reply = _reply(copy.VERIFY_FAILED, "VERIFICATION_FAILED")
            return _at("VERIFICATION", reply, matched=False, attempt=attempt, phone=pii.mask_phone_number(phone))
        state["customer_id"] = customer["customer_id"]
        state["verified"] = True
        state["slot_state"] = None
        pending = verification.get("pending_request")
        pending_plans = verification.get("pending_plans") or []
        state.pop("verification", None)
        observation = {
            "event_type": "CALLER_VERIFIED_SIMULATED",
            "customer_id": customer["customer_id"],
            "note": copy.OPD_04_NOTE,
        }
        detail = {"matched": True, "customer_id": customer["customer_id"], "phone": pii.mask_phone_number(phone)}
        if not pending and not pending_plans:
            reply = _reply(copy.VERIFY_SUCCESS, "VERIFIED", binding_changed=True, observations=[observation])
            return _at("VERIFICATION", reply, **detail)
        if pending_plans:
            plans = [
                PlannedToolCall(item["name"], dict(item["arguments"]), f"resume-{index}")
                for index, item in enumerate(pending_plans)
            ]
            follow = await self._run_plans(state, pending or "", plans, route="RESUMED_REQUEST", is_current=is_current)
        else:
            follow = await self._decide(state, raw_text=pending, text=pending, planner=planner, is_current=is_current)
        follow.text = f"{copy.VERIFY_SUCCESS} {follow.text}".strip()
        follow.voice = f"{speakable(copy.VERIFY_SUCCESS)} {follow.voice}".strip()
        follow.binding_changed = True
        follow.observations.insert(0, observation)
        return _at("VERIFICATION", follow, **detail, resumed_request=pending)

    # ------------------------------------------------------------------ handoff

    async def _escalate(
        self, state: dict[str, Any], text: str, trigger: str, priority: str, *, spoken: str | None = None
    ) -> TurnReply:
        result = await self.middleware.handle(
            self._context(state, text), "escalate_to_human", {"trigger_reason": trigger, "priority": priority}
        )
        if result.error_code == "UNAUTHENTICATED_SESSION":
            return self._finish(
                state,
                text,
                render_results([result], user_text=text),
                [result],
                [PlannedToolCall("escalate_to_human", {"trigger_reason": trigger, "priority": priority})],
            )
        rendered = render_results([result], user_text=text)
        line = spoken if (spoken and result.ok) else rendered.text
        return TurnReply(
            line,
            speakable(line),
            rendered.outcome,
            results=[result],
            topic=None,
            plans=[PlannedToolCall("escalate_to_human", {"trigger_reason": trigger, "priority": priority})],
        )

    # ------------------------------------------------------------------ writes (record-only in Phase 1)

    async def _preview_write(
        self,
        state: dict[str, Any],
        intent: str,
        order_code: str,
        text: str,
        *,
        target: date | None = None,
        probe: str | None = None,
    ) -> TurnReply:
        if not state.get("customer_id"):
            return self._needs_verification(state, text)
        if intent == "Reschedule" and target is None:
            state["awaiting_write_date"] = {"order_code": order_code}
            return _reply(
                "Which day would you like the delivery moved to?", "NEEDS_DATE", topic=f"the change to {order_code}"
            )
        reversal_of = state.get("last_completed_action") if intent == "Reversal" else None
        outcome = await self.writes.preview(
            session_id=state["session_id"],
            customer_id=state.get("customer_id"),
            action_type=intent,
            order_code=order_code,
            target_date=target,
            # the new address is stored masked, like every value in the stores ("PostgreSQL (masked values only)")
            new_address=intents.new_address_in(probe or text) and pii.mask(intents.new_address_in(probe or text))
            if intent == "AddressChange"
            else None,
            reversal_of=reversal_of,
        )
        if intent == "AddressChange" and outcome.kind == "NEEDS_ADDRESS":
            state["awaiting_write_address"] = {"order_code": order_code}
            state["active_order_id"] = order_code
            return _reply(copy.NEEDS_ADDRESS, "NEEDS_ADDRESS", topic=f"the change to {order_code}")
        observations = [{"event_type": "WRITE_PREVIEW", "action_type": intent, "result": outcome.kind}]
        if outcome.kind not in {"OUT_OF_SCOPE", "NOT_FOUND"}:
            state["active_order_id"] = order_code  # "move it to Monday" next refers to this order
        if outcome.kind == "PREVIEW" and outcome.action:
            state["pending_action_id"] = outcome.action["action_id"]
            return _reply(
                outcome.spoken,
                "CONFIRMATION_REQUESTED",
                topic=f"the change to {order_code}",
                write=outcome,
                observations=observations,
            )
        if outcome.kind == "OPD_03":
            escalation = await self.middleware.handle(
                self._context(state, text),
                "escalate_to_human",
                {"trigger_reason": "COMPLEX_EXCEPTION", "priority": "NORMAL"},
            )
            return _reply(
                outcome.spoken,
                "OPD_FALLBACK",
                topic=f"your order {order_code}",
                results=[escalation],
                write=outcome,
                observations=[*observations, {"event_type": "OPD_FALLBACK", "opd": "OPD-03"}],
            )
        if outcome.kind == "REFUSED":
            state["handoff_offered"] = "COMPLEX_EXCEPTION"
        return _reply(outcome.spoken, outcome.kind, write=outcome, observations=observations)

    async def _confirm_write(self, state: dict[str, Any]) -> TurnReply:
        action_id = state["pending_action_id"]
        outcome = await self.writes.confirm(action_id)
        if outcome.kind == "CONFIRM_AGAIN":
            return _reply(outcome.spoken, "CONFIRMATION_REQUESTED", write=outcome)
        state.pop("pending_action_id", None)
        if outcome.kind != "SUBMITTED":
            return _reply(outcome.spoken, outcome.kind, write=outcome)
        state["submitted_action_id"] = action_id
        if self.flags.write_mode == "fake_oms" and self.writes.oms is not None:
            done = await self.writes.execute(action_id)
            if done.kind == "COMPLETED" and done.action:
                state.pop("submitted_action_id", None)
                state["last_completed_action"] = done.action
            return _reply(done.spoken, f"WRITE_{done.kind}", write=done)
        return _reply(outcome.spoken, "WRITE_REQUEST_RECORDED", write=outcome)

    async def _withdraw(self, state: dict[str, Any], action_id: str) -> TurnReply:
        outcome = await self.writes.withdraw(action_id)
        if outcome.kind in {"CANCELLED", "TOO_LATE"}:
            state.pop("submitted_action_id", None)
            state.pop("pending_action_id", None)
        if outcome.kind == "TOO_LATE" and outcome.action:
            state["last_completed_action"] = outcome.action
        return _reply(outcome.spoken, "WITHDRAWN" if outcome.kind == "CANCELLED" else outcome.kind, write=outcome)

    def _needs_verification(self, state: dict[str, Any], text: str) -> TurnReply:
        if self.flags.fr_auth_01:
            state["slot_state"] = S_AUTH_AWAITING_PHONE
            state["verification"] = {"pending_request": text, "attempts": 0}
            return _reply(copy.VERIFY_ACCOUNT_PROMPT, "NEEDS_VERIFICATION")
        return _reply(copy.ORDER_UNAUTHENTICATED, "NEEDS_VERIFICATION")

    # ------------------------------------------------------------------ after-sales diagnosis (D§9.3)

    async def _start_diagnosis(self, state: dict[str, Any], text: str, symptom: str) -> TurnReply:
        if not state.get("customer_id"):
            return self._needs_verification(state, text)
        listing = await self.middleware.handle(self._context(state, text), "list_customer_orders", {})
        order_code, delivered_on = None, None
        if listing.ok:
            for order in listing.envelope["orders"]:
                names = " ".join(order["item_names"]).lower()
                if order["order_status"] == "delivered" and ("washer" in names or "washing" in names):
                    order_code, delivered_on = order["order_code"], order["delivered_on"]
        state["diagnosis"] = {
            "symptom_code": symptom,
            "turn": 1,
            "answers": [],
            "order_code": order_code,
            "delivered_on": delivered_on,
            "symptom_text": pii.mask(text),
        }
        if order_code:
            state["active_order_id"] = order_code
        return _reply(copy.DIAGNOSIS_Q1, "DIAGNOSIS_QUESTION", topic="the washer diagnosis", results=[listing])

    async def _diagnosis_step(self, state: dict[str, Any], text: str, *, probe: str | None = None) -> TurnReply:
        probe = probe or text
        diagnosis = state["diagnosis"]
        stage = diagnosis["turn"]
        if stage in (1, 2) and _META_QUESTION.search(probe):  # meta-interactions do not consume the budget
            return _reply(
                copy.DIAGNOSIS_Q1 if stage == 1 else copy.DIAGNOSIS_Q2,
                "DIAGNOSIS_QUESTION",
                topic="the washer diagnosis",
            )
        if stage == 1:
            diagnosis["answers"].append({"question": copy.DIAGNOSIS_Q1, "answer": pii.mask(text)})
            diagnosis["turn"] = 2
            return _reply(copy.DIAGNOSIS_Q2, "DIAGNOSIS_QUESTION", topic="the washer diagnosis")
        if stage == 2:
            diagnosis["answers"].append({"question": copy.DIAGNOSIS_Q2, "answer": pii.mask(text)})
            diagnosis["turn"] = 3
            fault = classify_fault(diagnosis["symptom_code"])
            if fault is None:  # turn 3: escalate immediately
                state.pop("diagnosis", None)
                return await self._escalate(state, text, "COMPLEX_EXCEPTION", "NORMAL", spoken=copy.DIAGNOSIS_RECORDED)
            diagnosis["diagnosis_class"] = fault
            policy = await self.middleware.handle(
                self._context(state, text),
                "retrieve_policy_clauses",
                {
                    "query": "Can I get a 1-for-1 replacement for a washing machine with a manufacturing defect?",
                    "category": "RETURNS",
                },
            )
            grounded = policy.ok and any(clause["clause_id"] == "POL-RET-001" for clause in policy.envelope["clauses"])
            if not grounded:
                state.pop("diagnosis", None)
                state["handoff_offered"] = "COMPLEX_EXCEPTION"
                rendered = render_results([policy], user_text=text)
                return TurnReply(
                    f"{rendered.text} {copy.OFFER_HANDOFF}".strip(),
                    speakable(rendered.text),
                    rendered.outcome,
                    results=[policy],
                )
            delivered = diagnosis.get("delivered_on")
            inside = bool(delivered) and within_return_window(date.fromisoformat(delivered), local_today(self.clock))
            diagnosis["inside_window"] = inside
            diagnosis["turn"] = "options"
            lead = copy.DIAGNOSIS_CONDITIONAL if inside else copy.DIAGNOSIS_OUTSIDE_WINDOW
            rendered = render_results([policy], user_text=text)
            return TurnReply(
                f"{lead} {copy.DIAGNOSIS_OPTIONS}",
                speakable(f"{lead} {copy.DIAGNOSIS_OPTIONS}"),
                "DIAGNOSIS_OPTIONS",
                results=[policy],
                evidence=rendered.evidence,
                topic="the washer diagnosis",
            )
        choice = intents.diagnosis_choice(probe)
        if choice is None:
            if intents.NO.search(probe) or intents.WITHDRAW.search(probe):
                state.pop("diagnosis", None)
                return _reply(copy.DIAGNOSIS_DECLINED, "DIAGNOSIS_DECLINED")
            diagnosis["option_reasks"] = int(diagnosis.get("option_reasks", 0)) + 1
            if diagnosis["option_reasks"] >= 2:  # implementation choice: the options are asked at most twice more
                state.pop("diagnosis", None)
                state["handoff_offered"] = "COMPLEX_EXCEPTION"
                return _reply(copy.CLARIFY_LIMIT, "CLARIFICATION_LIMIT")
            return _reply(copy.DIAGNOSIS_OPTIONS, "DIAGNOSIS_OPTIONS", topic="the washer diagnosis")
        diagnosis["choice"] = choice
        recommendation = (
            f"Caller requests {choice.replace('_', ' ').lower()} for {diagnosis.get('order_code') or 'the washer'}; "
            f"diagnosis {diagnosis['diagnosis_class']} from {diagnosis['symptom_code']}; technician certification "
            "required before any RMA; no RMA ticket created."
        )
        state["recommendation"] = recommendation
        state["symptom_history"] = [
            {"symptom_code": diagnosis["symptom_code"], "report": diagnosis["symptom_text"]},
            *diagnosis["answers"],
            {"diagnosis_class": diagnosis["diagnosis_class"], "choice": choice},
        ]
        inside = diagnosis.get("inside_window")
        state.pop("diagnosis", None)
        return await self._escalate(
            state,
            text,
            "COMPLEX_EXCEPTION",
            "NORMAL",
            spoken=copy.DIAGNOSIS_FINAL if inside else copy.DIAGNOSIS_RECORDED,
        )

    # ------------------------------------------------------------------ helpers

    def _context(self, state: Mapping[str, Any], text: str) -> ToolContext:
        history = list(state.get("history", ()))[-MAX_HISTORY_MESSAGES:]
        return ToolContext(
            session_id=state["session_id"],
            customer_id=state.get("customer_id"),
            user_text=text,
            turn_epoch=int(state.get("turn_epoch", 0)),
            fixture_session=bool(state.get("fixture")),
            active_order_code=state.get("active_order_id"),
            transcript=[*history, {"role": "user", "content": text}],
            symptom_history=list(state.get("symptom_history", ())),
            recommendation=state.get("recommendation"),
            summary=_summary(state, text),
        )

    def _maybe_ask_resume(self, state: dict[str, Any], reply: TurnReply) -> None:
        """UC-23: after handling the interruption, ask whether to resume the paused topic."""
        paused = state.get("paused_topic")
        if (
            not paused
            or state.get("resume_pending")
            or reply.outcome
            in {
                "RESUMED",
                "RESUME_DECLINED",
                "CONFIRMATION_REQUESTED",
                "NEEDS_VERIFICATION",
                "NEEDS_ORDER_ID",
                "DIAGNOSIS_QUESTION",
                "DIAGNOSIS_OPTIONS",
                "NEEDS_DATE",
                "REPAIR_QUESTION",
            }
        ):
            return
        if paused.get("label") and paused.get("label") == reply.topic:
            state.pop("paused_topic", None)  # the caller came back to the same topic on their own
            return
        question = copy.resume_question(paused.get("label") or "what we were discussing")
        reply.text = f"{reply.text} {question}".strip()
        reply.voice = f"{reply.voice} {speakable(question)}".strip()
        state["resume_pending"] = True


_KEYWORD_PLANNER = KeywordPlanner()
_ORDER_TOOLS = ("lookup_order_details", "track_shipment_status")
_ORDER_WORDS = re.compile(
    r"\b(?:orders?|where|track\w*|deliver\w*|arriv\w*|package|parcel|shipment|shipping|eta|status|bought|purchase\w*)\b",
    re.I,
)
_DELIVERY_WORDS = re.compile(
    r"\b(?:where|track\w*|deliver\w*|arriv\w*|package|parcel|shipment|shipping|eta|status)\b", re.I
)


def _complete_order_reads(
    plans: list[PlannedToolCall], text: str, *, active_order: str | None = None
) -> tuple[list[PlannedToolCall], list[str]]:
    """Deterministic repairs of the model's order reads, from the caller's own words only (implementation choice).

    - an order tool with an empty code gets the one ORD code the caller said;
    - a delivery question answered with lookup alone also gets the tracking read for that code (the prompt asks
      for both; Scenario 1).
    """
    codes = sorted({code.upper() for code in intents.ORDER_CODE_IN_TEXT.findall(text)})
    notes: list[str] = []
    repaired: list[PlannedToolCall] = []
    has_other_tools = any(plan.name not in _ORDER_TOOLS for plan in plans)
    for plan in plans:
        arguments = dict(plan.arguments or {})
        planned_code = str(arguments.get("order_code") or "").upper()
        if plan.name in _ORDER_TOOLS and planned_code:
            if len(codes) == 1 and planned_code != codes[0]:
                # The caller named one order; a read of another one is the model carrying an earlier order over
                # (log SES-ba23ffdc, turn 3: ORD-2002 asked, ORD-1002 tracked).
                notes.append(f"{plan.name} read {codes[0]} as named, not {planned_code} from an earlier turn")
                arguments["order_code"] = codes[0]
            elif not codes and has_other_tools and not _ORDER_WORDS.search(text):
                # A policy or product question with an order read nobody asked for (log SES-7c2df43a, turn 2).
                notes.append(f"{plan.name}({planned_code}) dropped: the caller did not ask about an order")
                continue
        if plan.name in _ORDER_TOOLS and not arguments.get("order_code") and not arguments.get("tracking_code"):
            if len(codes) == 1:
                arguments["order_code"] = codes[0]
                if plan.name == "track_shipment_status":
                    arguments.setdefault("tracking_code", None)
                notes.append(f"order_code {codes[0]} for {plan.name} taken from the caller's words")
        repaired.append(PlannedToolCall(plan.name, arguments, plan.call_id))
    unique: list[PlannedToolCall] = []
    for plan in repaired:
        if not any(p.name == plan.name and p.arguments == plan.arguments for p in unique):
            unique.append(plan)
    repaired = unique
    tracked = {p.arguments.get("order_code") for p in repaired if p.name == "track_shipment_status"}
    for plan in list(repaired):
        code = plan.arguments.get("order_code")
        if (
            plan.name == "lookup_order_details"
            and code
            and code not in tracked
            and _DELIVERY_WORDS.search(text)
            and len(repaired) < MAX_TOOL_CALLS_PER_TURN
        ):
            repaired.insert(
                repaired.index(plan) + 1,
                PlannedToolCall("track_shipment_status", {"order_code": code, "tracking_code": None}, "paired"),
            )
            tracked.add(code)
            notes.append(f"track_shipment_status added for {code}: the caller asked about delivery")
    return repaired, notes


def _leaves_diagnosis(probe: str, stage: Any) -> bool:
    """True when the caller makes a new request instead of answering the diagnosis (implementation choice)."""
    if stage in (1, 2) and (intents.YES.search(probe) or intents.NO.search(probe)):
        return False
    if stage == "options" and (intents.diagnosis_choice(probe) or intents.NO.search(probe)):
        return False
    if (
        intents.HUMAN.search(probe)
        or intents.ORDER_CODE_IN_TEXT.search(probe)
        or intents.write_intent(probe)
        or intents.opd_topic(probe)
        or intents.VOUCHER_CODE_IN_TEXT.search(probe)
    ):
        return True
    other_products = set(product_categories_in(probe)) - {"WASHING_MACHINE"}
    if other_products:  # "my TV is broken" is not an answer about the washer
        return True
    return stage == "options" and detect_symptom(probe) is not None


_NO_ARGUMENT_TOOLS = frozenset({"list_customer_orders"})


def _missing_arguments(plan: PlannedToolCall) -> bool:
    if plan.name in _NO_ARGUMENT_TOOLS:
        return False
    return not any(value not in (None, "", [], {}) for value in (plan.arguments or {}).values())


_PRODUCT_LABELS = {
    "WASHING_MACHINE": "washing machine",
    "TV_AV": "TV",
    "REFRIGERATOR": "refrigerator",
    "AIR_CONDITIONER": "air conditioner",
    "KITCHEN": "appliance",
}


def _normalise(raw: PlannedToolCall | Sequence[PlannedToolCall] | None) -> list[PlannedToolCall]:
    if raw is None:
        return []
    if isinstance(raw, PlannedToolCall):
        return [raw]
    return list(raw)[:MAX_TOOL_CALLS_PER_TURN]


def _summary(state: Mapping[str, Any], text: str) -> str:
    """Deterministic AI summary for the handoff packet (implementation choice: extractive, masked)."""
    users = [item["content"] for item in state.get("history", ()) if item.get("role") == "user"][-3:]
    parts = [f"Caller requests: {pii.mask(text)}"]
    if users:
        parts.append("Earlier in the call: " + " | ".join(pii.mask(item) for item in users))
    if state.get("active_order_id"):
        parts.append(f"Active order: {state['active_order_id']}")
    if state.get("last_reply"):
        parts.append(f"Last answer: {pii.mask(state['last_reply'])}")
    return ". ".join(parts)[:1200]


__all__ = [
    "MAX_CLARIFICATIONS",
    "MAX_HISTORY_MESSAGES",
    "S_AUTH_AWAITING_PHONE",
    "S_ORDER_AWAITING_IDENTIFIER",
    "DialogueManager",
    "Flags",
    "TurnReply",
]
