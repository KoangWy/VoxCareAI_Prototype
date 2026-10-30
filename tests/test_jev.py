import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from voxcare.turns import JevConfigurationError, JevProviderError, TurnInput
from voxcare.turns.jev import TypeSafeJevJudge


class FakeNoul:
    def __init__(self, *, instructions, criteria):
        self.instructions = instructions
        self.criteria = criteria


class FakeNoulCriteria:
    def __init__(self, *, true, false):
        self.true = true
        self.false = false


class FakeRetryPolicy:
    def __init__(self, *, max_retries):
        self.max_retries = max_retries


class JevAdapterTests(unittest.IsolatedAsyncioTestCase):
    def make_turn(self):
        return TurnInput(
            session_id="session-1",
            turn_epoch=4,
            transcript="Can you track ORD-1002? Call person@example.com at +1 (415) 555-0134.",
            conversation_history=(
                {"role": "user", "text": "My email is private@example.org and my address is 42 Nguyen Trai Street"},
                {"role": "assistant", "text": "What can I help with?"},
            ),
            assistant_last_message="What can I help with?",
            silence_ms=650,
            asr_final=True,
            stable_word_ratio=1.0,
            local_vad_quiet=True,
        )

    async def test_missing_key_is_configuration_error_before_client_creation(self):
        judge = TypeSafeJevJudge()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(JevConfigurationError):
            await judge(self.make_turn())

    async def test_uses_current_async_sdk_and_sends_text_not_audio(self):
        calls = {}

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                calls["client_kwargs"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def system_one(self, *, state, questions, timeout, retry):
                calls["state"] = state
                calls["questions"] = questions
                calls["call_timeout"] = timeout
                calls["call_retry"] = retry
                return SimpleNamespace(
                    answers={
                        "turn_complete": SimpleNamespace(noul=0.84),
                        "holding_floor": SimpleNamespace(noul=0.08),
                        "addressed_to_agent": SimpleNamespace(noul=0.93),
                    },
                    model="jev-test",
                    usage=SimpleNamespace(input_tokens=31, output_tokens=8),
                )

        sdk = types.ModuleType("typesafe_sdk")
        sdk.AsyncTypeSafeClient = FakeAsyncClient
        sdk.Noul = FakeNoul
        sdk.NoulCriteria = FakeNoulCriteria
        sdk.RetryPolicy = FakeRetryPolicy
        with (
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "dummy-test-value"}),
            patch.dict(sys.modules, {"typesafe_sdk": sdk}),
        ):
            result = await TypeSafeJevJudge(timeout_ms=150)(self.make_turn())

        self.assertEqual(result.turn_complete, 0.84)
        self.assertEqual(result.holding_floor, 0.08)
        self.assertEqual(result.addressed_to_agent, 0.93)
        # No taking_floor question while the assistant is silent: scored 0.0, never an error.
        self.assertEqual(result.taking_floor, 0.0)
        self.assertEqual(result.model, "jev-test")
        self.assertEqual(result.input_tokens, 31)
        self.assertEqual(calls["client_kwargs"]["model"], "jev-latest")
        self.assertEqual(calls["client_kwargs"]["retry"].max_retries, 0)
        self.assertEqual(calls["call_retry"].max_retries, 0)
        self.assertEqual(calls["client_kwargs"]["timeout"], 0.15)
        self.assertEqual(calls["call_timeout"], 0.15)
        self.assertEqual(
            calls["state"]["current_user_transcript"],
            "Can you track ORD-1002? Call [REDACTED_EMAIL] at [REDACTED_PHONE].",
        )
        self.assertEqual(
            calls["state"]["conversation_history"][0]["text"],
            "My email is [REDACTED_EMAIL] and my address is [REDACTED_ADDRESS]",
        )
        serialized_state = str(calls["state"])
        self.assertNotIn("person@example.com", serialized_state)
        self.assertNotIn("private@example.org", serialized_state)
        self.assertNotIn("415", serialized_state)
        self.assertNotIn("42 Nguyen Trai Street", serialized_state)
        self.assertNotIn("audio", calls["state"])
        # Assistant is silent here, so taking_floor is not asked.
        self.assertEqual(set(calls["questions"]), {"turn_complete", "holding_floor", "addressed_to_agent"})
        # reference calibration travels in the shared prompts/state.
        self.assertIn("timing_notes", calls["state"])
        self.assertIn("silence_since_last_word_ms", calls["state"])
        self.assertIn("asr_final", calls["state"])
        self.assertIn("stable_word_ratio", calls["state"])
        instructions = calls["questions"]["turn_complete"].instructions
        self.assertIn("Within-turn pauses are usually under ~800 ms", instructions)
        self.assertIn("Web. Actually", instructions)
        self.assertIn("assistant_current_speech", calls["state"])

    async def test_invalid_judgment_is_rejected_without_echoing_response(self):
        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def system_one(self, **kwargs):
                return SimpleNamespace(answers={"turn_complete": SimpleNamespace(noul=1.5)})

        sdk = types.ModuleType("typesafe_sdk")
        sdk.AsyncTypeSafeClient = FakeAsyncClient
        sdk.Noul = FakeNoul
        sdk.NoulCriteria = FakeNoulCriteria
        sdk.RetryPolicy = FakeRetryPolicy
        with (
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "dummy-test-value"}),
            patch.dict(sys.modules, {"typesafe_sdk": sdk}),
            self.assertRaisesRegex(JevProviderError, "invalid_response"),
        ):
            await TypeSafeJevJudge()(self.make_turn())


if __name__ == "__main__":
    unittest.main()
