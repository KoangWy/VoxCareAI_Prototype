import asyncio
import unittest

from voxcare.contracts import PlannedToolCall
from voxcare.domain import PolicyRetriever, SeedOMS
from voxcare.session import SessionBoundary


class ScriptedDialogue:
    def __init__(self, *plans):
        self.plans = list(plans)

    async def plan_turn(self, *, user_text, history, available_tools):
        return self.plans.pop(0)


class PolicySessionIntegrationTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_policy_tool_returns_fictional_evidence_and_customer_scope_stays_fixed(self):
        oms = SeedOMS()
        service = SessionBoundary(
            dialogue=ScriptedDialogue(
                PlannedToolCall(
                    "retrieve_policy_clauses",
                    {
                        "query": "LG washer warranty terms",
                        "category": "WARRANTY",
                        "product_category": "WASHING_MACHINE",
                        "brand": "LG",
                    },
                ),
                PlannedToolCall("lookup_order_details", {"order_code": "ORD-2002"}),
            ),
            oms=oms,
            policy_retriever=PolicyRetriever(),
        )

        async def scenario():
            session = await service.create_session()
            policy = await service.handle_turn(session.session_id, "What is the LG washer warranty?")
            followup = await service.handle_turn(session.session_id, "What about order ORD-2002?")
            return session, policy, followup

        session, policy, followup = self.run_async(scenario())

        self.assertEqual(policy.outcome, "ANSWERED")
        self.assertEqual(policy.policy_outcome, "SUCCESS")
        self.assertEqual(policy.turn_epoch, 1)
        self.assertEqual([item["policy_id"] for item in policy.evidence], ["POL-WAR-002"])
        self.assertTrue(policy.is_fictional)
        self.assertIn("POL-WAR-002", policy.assistant_text)
        self.assertEqual(session.customer_id, "CUST-001")
        self.assertEqual(followup.outcome, "OUT_OF_SCOPE")
        self.assertIsNone(followup.tool_outcomes[0].record)
        self.assertNotIn("ORD-2002", followup.assistant_text)
        self.assertEqual(oms.business_snapshot(), oms.initial_snapshot)

    def test_expired_session_does_not_reach_the_policy_tool(self):
        service = SessionBoundary(dialogue=ScriptedDialogue(), policy_retriever=None)

        result = self.run_async(service.handle_turn("missing-session", "return fee?"))

        self.assertEqual(result.outcome, "SESSION_EXPIRED")
        self.assertEqual(result.evidence, ())

    def test_missing_policy_dependency_is_reported_without_evidence(self):
        service = SessionBoundary(
            dialogue=ScriptedDialogue(
                PlannedToolCall(
                    "retrieve_policy_clauses",
                    {"query": "return fee", "category": "RETURNS", "product_category": None, "brand": None},
                )
            ),
            policy_retriever=None,
        )

        async def scenario():
            session = await service.create_session()
            return await service.handle_turn(session.session_id, "Is there a return fee?")

        result = self.run_async(scenario())

        self.assertEqual(result.outcome, "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(result.policy_outcome, "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(result.evidence, ())
        self.assertIn("unavailable", result.assistant_text.lower())


if __name__ == "__main__":
    unittest.main()
