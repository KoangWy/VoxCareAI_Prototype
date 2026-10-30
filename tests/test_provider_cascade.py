import os
import unittest
from unittest.mock import patch

from voxcare.contracts import PlannedToolCall
from voxcare.providers import CascadeDialogueProvider, provider_status


class FakePrimary:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def plan_turn(self, **kwargs):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeFallback:
    def __init__(self):
        self.calls = 0

    async def plan_turn(self, **kwargs):
        self.calls += 1
        return [PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"})]


class CascadeTests(unittest.IsolatedAsyncioTestCase):
    def test_generic_host_model_environment_does_not_override_voxcare_provider_defaults(self):
        with patch.dict(
            os.environ, {"ANTHROPIC_MODEL": "unrelated-host-model", "OPENAI_MODEL": "host-model"}, clear=True
        ):
            status = provider_status()
        self.assertEqual(status["conversation"].model, "gpt-6-luna")
        self.assertEqual(status["conversation_fallback"].model, "claude-haiku-5-5")

    async def test_switches_after_two_consecutive_retryable_failures_and_sticks(self):
        primary = FakePrimary(
            TimeoutError(), TimeoutError(), PlannedToolCall("lookup_order_details", {"order_code": "ORD-1001"})
        )
        fallback = FakeFallback()
        cascade = CascadeDialogueProvider(primary, fallback)
        call = {"user_text": "ORD-1002", "history": (), "available_tools": ()}

        with self.assertRaises(TimeoutError):
            await cascade.plan_turn(**call)
        result = await cascade.plan_turn(**call)
        result_after_reset = await cascade.plan_turn(**call)

        self.assertEqual(result[0].name, "lookup_order_details")
        self.assertEqual(result_after_reset[0].arguments["order_code"], "ORD-1002")
        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 2)

    async def test_success_resets_failure_streak_before_fallback(self):
        primary = FakePrimary(
            TimeoutError(), PlannedToolCall("lookup_order_details", {"order_code": "ORD-1001"}), TimeoutError()
        )
        fallback = FakeFallback()
        cascade = CascadeDialogueProvider(primary, fallback)
        call = {"user_text": "ORD-1002", "history": (), "available_tools": ()}

        with self.assertRaises(TimeoutError):
            await cascade.plan_turn(**call)
        await cascade.plan_turn(**call)
        with self.assertRaises(TimeoutError):
            await cascade.plan_turn(**call)

        self.assertEqual(fallback.calls, 0)


if __name__ == "__main__":
    unittest.main()
