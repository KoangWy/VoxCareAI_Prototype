import asyncio
import json
import unittest

from voxcare.contracts import PlannedToolCall
from voxcare.domain.seed import SeedOMS
from voxcare.session import TOOLS, SessionBoundary


class ScriptedDialogue:
    def __init__(self, *plans):
        self.plans = list(plans)
        self.histories = []
        self.available_tools = []

    async def plan_turn(self, *, user_text, history, available_tools):
        self.histories.append(tuple(history))
        self.available_tools.append(tuple(available_tools))
        return self.plans.pop(0)


class SessionBoundaryTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_owned_order_answer_matches_seeded_order(self):
        dialogue = ScriptedDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
        oms = SeedOMS()
        service = SessionBoundary(dialogue=dialogue, oms=oms)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Where is order ORD-1002?", source="manual"))

        self.assertEqual(result.outcome, "ANSWERED")
        self.assertEqual(result.tool_outcomes[0].record["status"], "shipped")
        self.assertIn("ORD-1002", result.assistant_text)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_registry_exposes_seven_demo_tools_and_rejects_future_writes(self):
        names = {tool["name"] for tool in TOOLS}
        self.assertEqual(
            names,
            {
                "lookup_order_details",
                "track_shipment_status",
                "retrieve_policy_clauses",
                "search_catalog",
                "compare_products",
                "apply_voucher",
                "escalate_to_human",
            },
        )
        for plan in (
            PlannedToolCall("cancel_order", {"order_code": "ORD-1001"}),
            PlannedToolCall("create_rma_request", {"order_code": "ORD-1003"}),
        ):
            with self.subTest(plan=plan):
                service = SessionBoundary(dialogue=ScriptedDialogue(plan), oms=SeedOMS())
                session = self.run_async(service.create_session())
                result = self.run_async(service.handle_turn(session.session_id, "change ORD-1001", source="manual"))
                self.assertEqual(result.outcome, "TOOL_ERROR")
                self.assertEqual(result.tool_outcomes[0].error_code, "TOOL_NOT_ALLOWLISTED")

    def test_allowlisted_tool_schemas_meet_strict_function_requirements(self):
        for tool in TOOLS:
            with self.subTest(tool=tool["name"]):
                parameters = tool["parameters"]
                self.assertEqual(parameters["type"], "object")
                self.assertFalse(parameters["additionalProperties"])
                self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
                for value in parameters["properties"].values():
                    self.assertNotIn(
                        "object", value.get("type", []) if isinstance(value.get("type"), list) else [value.get("type")]
                    )

    def test_cross_customer_record_is_denied_without_leaking_details(self):
        dialogue = ScriptedDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-2002"}))
        oms = SeedOMS()
        service = SessionBoundary(dialogue=dialogue, oms=oms)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Track ORD-2002", source="manual"))

        self.assertEqual(result.outcome, "OUT_OF_SCOPE")
        self.assertNotIn("private", result.assistant_text.lower())
        self.assertNotIn("ORD-2002", result.assistant_text)
        self.assertIsNone(result.tool_outcomes[0].record)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_unknown_order_returns_not_found_without_inventing_record(self):
        dialogue = ScriptedDialogue(PlannedToolCall("track_shipment_status", {"order_code": "ORD-9999"}))
        oms = SeedOMS()
        service = SessionBoundary(dialogue=dialogue, oms=oms)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Delivery for ORD-9999?", source="manual"))

        self.assertEqual(result.outcome, "NOT_FOUND")
        self.assertIsNone(result.tool_outcomes[0].record)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_missing_order_code_prompts_before_any_lookup(self):
        dialogue = ScriptedDialogue(PlannedToolCall("lookup_order_details", {"order_code": ""}))
        oms = SeedOMS()
        service = SessionBoundary(dialogue=dialogue, oms=oms)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "What is my order status?", source="manual"))

        self.assertEqual(result.outcome, "NEEDS_ORDER_ID")
        self.assertEqual(result.tool_outcomes, ())
        self.assertIn("order code", result.assistant_text.lower())
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_no_conversation_provider_is_explicitly_unavailable(self):
        service = SessionBoundary(dialogue=None, oms=SeedOMS())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Track ORD-1002", source="manual"))

        self.assertEqual(result.outcome, "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(result.tool_outcomes, ())
        self.assertIsNone(result.assistant_text)

    def test_missing_shipment_does_not_invent_carrier_eta(self):
        dialogue = ScriptedDialogue(PlannedToolCall("track_shipment_status", {"order_code": "ORD-1001"}))
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
        session = self.run_async(service.create_session())

        result = self.run_async(
            service.handle_turn(session.session_id, "When will order ORD-1001 arrive?", source="asr_final")
        )

        self.assertEqual(result.outcome, "ANSWERED")
        self.assertIn("no shipment record", result.assistant_text.lower())
        self.assertNotIn("2026-10-15", result.assistant_text)

    def test_followup_after_clarification_keeps_history_and_customer_scope(self):
        dialogue = ScriptedDialogue(None, PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
        session = self.run_async(service.create_session())

        first = self.run_async(service.handle_turn(session.session_id, "What is my order status?", source="manual"))
        second = self.run_async(service.handle_turn(session.session_id, "ORD-1002", source="asr_final"))

        self.assertEqual(first.outcome, "NEEDS_ORDER_ID")
        self.assertEqual(second.outcome, "ANSWERED")
        self.assertIn("CUST-001", session.customer_id)
        self.assertEqual(dialogue.histories[1][-1]["content"], first.assistant_text)
        self.assertEqual(second.tool_outcomes[0].record["status"], "shipped")

    def test_unknown_or_write_tool_and_extra_scope_args_are_rejected(self):
        for plan in (
            PlannedToolCall("cancel_order", {"order_code": "ORD-1002"}),
            PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002", "customer_id": "CUST-002"}),
        ):
            with self.subTest(plan=plan):
                oms = SeedOMS()
                service = SessionBoundary(dialogue=ScriptedDialogue(plan), oms=oms)
                session = self.run_async(service.create_session())
                result = self.run_async(service.handle_turn(session.session_id, "Cancel ORD-1002", source="manual"))
                self.assertEqual(result.outcome, "TOOL_ERROR")
                if plan.name == "cancel_order":
                    self.assertIn("can't perform", result.assistant_text.lower())
                else:
                    self.assertIn("safely process", result.assistant_text.lower())
                self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_reset_discards_pending_provider_result(self):
        class PausingDialogue(ScriptedDialogue):
            def __init__(self):
                super().__init__(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def plan_turn(self, **kwargs):
                self.started.set()
                await self.release.wait()
                return await super().plan_turn(**kwargs)

        async def scenario():
            dialogue = PausingDialogue()
            service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
            old = await service.create_session()
            pending = asyncio.create_task(service.handle_turn(old.session_id, "ORD-1002", source="manual"))
            await dialogue.started.wait()
            new = await service.reset_session(old.session_id)
            dialogue.release.set()
            result = await pending
            return old, new, result

        old, new, result = self.run_async(scenario())
        self.assertNotEqual(old.session_id, new.session_id)
        self.assertEqual(result.outcome, "STALE_DISCARDED")
        self.assertIsNone(result.assistant_text)

    def test_oms_timeout_is_retried_once_then_returns_error(self):
        class TimingOutOMS(SeedOMS):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def lookup_order_details(self, **kwargs):
                self.calls += 1
                raise TimeoutError("sensitive transport detail must not leak")

        oms = TimingOutOMS()
        service = SessionBoundary(
            dialogue=ScriptedDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"})), oms=oms
        )
        session = self.run_async(service.create_session())
        result = self.run_async(service.handle_turn(session.session_id, "ORD-1002", source="manual"))

        self.assertEqual(oms.calls, 2)
        self.assertEqual(result.outcome, "TOOL_ERROR")
        self.assertNotIn("sensitive", result.assistant_text.lower())
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_policy_tool_schema_includes_spec_installation_category(self):
        names = {tool["name"]: tool for tool in TOOLS}
        category = names["retrieve_policy_clauses"]["parameters"]["properties"]["category"]
        self.assertEqual(category["enum"], ["SHIPPING", "INSTALLATION", "WARRANTY", "RETURNS", "ALL"])
        product = names["retrieve_policy_clauses"]["parameters"]["properties"]["product_category"]
        self.assertEqual(
            product["enum"], ["TV_AV", "REFRIGERATOR", "WASHING_MACHINE", "AIR_CONDITIONER", "KITCHEN", None]
        )

    def test_order_lookup_answer_names_purchased_items(self):
        dialogue = ScriptedDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1003"}))
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "What did I buy in ORD-1003?"))

        self.assertEqual(result.outcome, "ANSWERED")
        self.assertIn("Panasonic Inverter 9 kg washer", result.assistant_text)

        from voxcare.domain.policy import PolicyRetriever

        plan = PlannedToolCall(
            "retrieve_policy_clauses",
            {
                "query": "LG washer warranty terms",
                "category": "WARRANTY",
                "product_category": "WASHING_MACHINE",
                "brand": "LG",
            },
        )
        dialogue = ScriptedDialogue(plan)
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS(), policy_retriever=PolicyRetriever())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "What warranty applies to my LG washer?"))

        self.assertIn("retrieve_policy_clauses", {tool["name"] for tool in dialogue.available_tools[0]})
        self.assertEqual(result.outcome, "ANSWERED")
        self.assertTrue(result.evidence)
        self.assertTrue(all(item["is_fictional"] for item in result.evidence))
        self.assertEqual(result.corpus_id, "VOXCOM-DEMO-POLICY-V1")
        self.assertTrue(result.is_fictional)
        self.assertEqual(result.clause_version, "1.0-demo")
        self.assertIn("fictional", result.assistant_text.lower())

    def test_policy_read_with_missing_applicability_context_returns_no_clause_promise(self):
        from voxcare.domain.policy import PolicyRetriever

        dialogue = ScriptedDialogue(
            PlannedToolCall("retrieve_policy_clauses", {"query": "warranty period", "category": "WARRANTY"})
        )
        service = SessionBoundary(dialogue=dialogue, policy_retriever=PolicyRetriever())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "What is the warranty period?"))

        self.assertEqual(result.outcome, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.policy_outcome, "NEEDS_CONTEXT")
        self.assertEqual(result.missing_context, ("brand",))
        self.assertIn("need more information", result.assistant_text.lower())
        self.assertNotIn("24 months", result.assistant_text.lower())

    def test_sessions_evict_oldest_beyond_bound(self):
        """The session registry stays bounded; oldest-touched evicts first."""
        from voxcare.session import MAX_SESSIONS

        service = SessionBoundary(dialogue=None)
        first = self.run_async(service.create_session())
        for _ in range(MAX_SESSIONS):
            self.run_async(service.create_session())
        self.assertEqual(len(service._sessions), MAX_SESSIONS)
        self.assertIsNone(self.run_async(service.get_session(first.session_id)))

    def test_reset_discards_old_session_without_growth(self):
        """Reset deletes the old session; the registry does not accumulate."""
        service = SessionBoundary(dialogue=None)
        session = self.run_async(service.create_session())
        before = len(service._sessions)
        self.run_async(service.reset_session(session.session_id))
        self.assertEqual(len(service._sessions), before)
        self.assertIsNone(self.run_async(service.get_session(session.session_id)))

    def test_trace_ledger_drops_oldest_beyond_bound(self):
        """Per-session traces stay bounded; oldest events drop first."""
        from voxcare.trace import TraceLedger

        ledger = TraceLedger()
        for index in range(TraceLedger.MAX_EVENTS + 50):
            ledger.record_observation(session_id="s", event_type="X", index=index)
        snapshot = ledger.snapshot()
        self.assertEqual(len(snapshot), TraceLedger.MAX_EVENTS)
        self.assertEqual(snapshot[0]["index"], 50)

    def test_policy_service_failure_is_reported_as_unavailable_not_no_evidence(self):
        class FailingPolicy:
            async def retrieve_policy_clauses(self, _query, _filters):
                raise RuntimeError("private connection details")

        dialogue = ScriptedDialogue(
            PlannedToolCall("retrieve_policy_clauses", {"query": "return delivery fee", "category": "RETURNS"})
        )
        service = SessionBoundary(dialogue=dialogue, policy_retriever=FailingPolicy())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Can I return this?"))

        self.assertEqual(result.outcome, "TOOL_ERROR")
        self.assertEqual(result.tool_outcomes[0].status, "SERVICE_ERROR")
        self.assertIn("unavailable right now", result.assistant_text)
        self.assertNotIn("no applicable policy evidence", result.assistant_text.lower())
        self.assertNotIn("private connection details", result.assistant_text)

    def test_missing_policy_dependency_is_reported_as_unavailable(self):
        dialogue = ScriptedDialogue(
            PlannedToolCall("retrieve_policy_clauses", {"query": "return delivery fee", "category": "RETURNS"})
        )
        service = SessionBoundary(dialogue=dialogue, policy_retriever=None)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Can I return this?"))

        self.assertEqual(result.outcome, "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(result.tool_outcomes[0].status, "DEPENDENCY_UNAVAILABLE")
        self.assertIn("unavailable right now", result.assistant_text)
        self.assertNotIn("no applicable policy evidence", result.assistant_text.lower())

    def test_policy_query_pii_is_not_written_to_audit_payload(self):
        from voxcare.domain.policy import PolicyRetriever

        sentinel = "person@example.com +1 (415) 555-0134"
        dialogue = ScriptedDialogue(
            PlannedToolCall(
                "retrieve_policy_clauses",
                {"query": f"return question from {sentinel}", "category": "RETURNS"},
            )
        )
        service = SessionBoundary(dialogue=dialogue, policy_retriever=PolicyRetriever())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Can I return this?"))

        action = next(event for event in result.events if event.get("action_name") == "TOOL_INVOCATION")
        audit_payload = json.dumps(
            {"request": action["request_payload_json"], "response": action["response_payload_json"]}
        )
        self.assertNotIn("person@example.com", audit_payload)
        self.assertNotIn("415", audit_payload)
        self.assertEqual(
            action["request_payload_json"]["arguments"]["query_char_count"], len(f"return question from {sentinel}")
        )

    def test_explicit_handoff_records_request_and_never_claims_transfer(self):
        dialogue = ScriptedDialogue(PlannedToolCall("escalate_to_human", {"reason": "customer_requested_support"}))
        service = SessionBoundary(dialogue=dialogue)
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Please let me talk to a person."))

        self.assertEqual(result.outcome, "HANDOFF_REQUEST_RECORDED")
        self.assertIn("was recorded", result.assistant_text.lower())
        self.assertNotIn("transferred", result.assistant_text.lower())
        self.assertTrue(any(event.get("action_name") == "HANDOFF_REQUEST_RECORDED" for event in result.events))

    def test_two_tool_calls_execute_sequentially_and_compose_one_answer(self):
        dialogue = ScriptedDialogue(
            [
                PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}),
                PlannedToolCall("track_shipment_status", {"order_code": "ORD-1002"}),
            ]
        )
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Where is my order ORD-1002?"))

        self.assertEqual(result.outcome, "ANSWERED")
        self.assertEqual(len(result.tool_outcomes), 2)
        self.assertEqual(
            [tool.tool_name for tool in result.tool_outcomes], ["lookup_order_details", "track_shipment_status"]
        )
        self.assertIn("ORD-1002", result.assistant_text)
        self.assertIn("2026-10-15", result.assistant_text)
        self.assertEqual(len([event for event in result.events if event.get("action_name") == "TOOL_INVOCATION"]), 2)

    def test_four_tool_calls_capped_at_three_with_unknown_tool_rejected(self):
        dialogue = ScriptedDialogue(
            [
                PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}),
                PlannedToolCall("cancel_order", {"order_code": "ORD-1002"}),
                PlannedToolCall("track_shipment_status", {"order_code": "ORD-1002"}),
                PlannedToolCall("lookup_order_details", {"order_code": "ORD-1001"}),
            ]
        )
        service = SessionBoundary(dialogue=dialogue, oms=SeedOMS())
        session = self.run_async(service.create_session())

        result = self.run_async(service.handle_turn(session.session_id, "Where is my order ORD-1002?"))

        self.assertEqual(len(result.tool_outcomes), 3)
        self.assertEqual(result.tool_outcomes[1].error_code, "TOOL_NOT_ALLOWLISTED")
        self.assertEqual(result.outcome, "TOOL_ERROR")
        self.assertIn("ORD-1002", result.assistant_text)


class GroundedAnswerTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def answer(self, *plans, text="question"):
        service = SessionBoundary(dialogue=ScriptedDialogue(*plans))

        async def scenario():
            session = await service.create_session()
            return [await service.handle_turn(session.session_id, text) for _ in plans]

        return self.run_async(scenario())

    def test_spoken_order_code_variants_resolve_to_the_canonical_code(self):
        for spoken in ("ord 1002", "ORD1002", " ord_1002 "):
            with self.subTest(spoken=spoken):
                (result,) = self.answer(PlannedToolCall("lookup_order_details", {"order_code": spoken}))
                self.assertEqual(result.outcome, "ANSWERED")
                self.assertIn("ORD-1002 is shipped", result.assistant_text)

    def test_clarification_names_every_supported_intent(self):
        (result,) = self.answer(None)
        self.assertEqual(result.outcome, "NEEDS_ORDER_ID")
        for phrase in ("order code", "compare products", "voucher", "warranty"):
            self.assertIn(phrase, result.assistant_text)

    def test_history_sent_to_the_model_stays_bounded(self):
        from voxcare.session.boundary import MAX_HISTORY_MESSAGES

        turns = 20
        dialogue = ScriptedDialogue(*([None] * turns))
        service = SessionBoundary(dialogue=dialogue)

        async def scenario():
            session = await service.create_session()
            for index in range(turns):
                await service.handle_turn(session.session_id, f"hello {index}")

        self.run_async(scenario())
        self.assertEqual(len(dialogue.histories[-1]), MAX_HISTORY_MESSAGES)
        self.assertEqual(dialogue.histories[-1][-1]["role"], "assistant")

    def test_unanswered_turn_is_not_left_for_the_model_to_answer_later(self):
        class FlakyDialogue(ScriptedDialogue):
            async def plan_turn(self, *, user_text, history, available_tools):
                self.histories.append(tuple(history))
                if user_text == "first":
                    raise TimeoutError
                return None

        dialogue = FlakyDialogue()
        service = SessionBoundary(dialogue=dialogue)

        async def scenario():
            session = await service.create_session()
            failed = await service.handle_turn(session.session_id, "first")
            await service.handle_turn(session.session_id, "second")
            return failed

        failed = self.run_async(scenario())
        self.assertEqual(failed.outcome, "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(dialogue.histories[-1], ())

    def test_catalog_answer_speaks_product_names_and_prices(self):
        (result,) = self.answer(
            PlannedToolCall(
                "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": None}
            )
        )
        self.assertEqual(result.outcome, "ANSWERED")
        self.assertIn("Samsung 55-inch 4K TV (TV-SAM-55) at 11,490,000 VND", result.assistant_text)

    def test_inapplicable_voucher_is_explained_in_plain_english(self):
        (result,) = self.answer(
            PlannedToolCall(
                "apply_voucher", {"voucher_code": "VOXCOM_500K", "order_value_vnd": 5_000_000, "category": "TV_AV"}
            )
        )
        self.assertEqual(result.outcome, "INVALID_INPUT")
        self.assertIn("needs a higher order value", result.assistant_text)
        self.assertNotIn("BELOW_MINIMUM", result.assistant_text)


if __name__ == "__main__":
    unittest.main()
