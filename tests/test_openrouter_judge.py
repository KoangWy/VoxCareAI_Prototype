import json
import os
import unittest
from unittest.mock import patch

import httpx

from voxcare.turns import JevConfigurationError, JevProviderError, TurnInput
from voxcare.turns.openrouter import DEFAULT_OPENROUTER_DECISIONS_URL, OpenRouterDecisionsJudge


class OpenRouterAdapterTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_missing_key_is_configuration_error_before_any_request(self):
        def handler(request):
            raise AssertionError("network call attempted without OPENROUTER_API_KEY")

        judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(JevConfigurationError):
            await judge(self.make_turn())

    async def test_sends_text_state_and_two_noul_questions_then_parses_judgment(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "inception/mercury-decide-20260930",
                    "answers": {
                        "turn_complete": {"type": "noul", "noul": 0.84},
                        "holding_floor": {"type": "noul", "noul": 0.08},
                        "addressed_to_agent": {"type": "noul", "noul": 0.93},
                    },
                    "usage": {"input_tokens": 494, "output_tokens": 2},
                },
            )

        judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(handler), timeout_ms=1500)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}):
            judgment = await judge(self.make_turn())

        self.assertEqual(captured["url"], DEFAULT_OPENROUTER_DECISIONS_URL)
        self.assertEqual(captured["auth"], "Bearer or-test-key")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "inception/mercury-decide")
        # PII masking must run before any text leaves the process.
        self.assertNotIn("@", payload["state"]["current_user_transcript"])
        self.assertNotIn("@", payload["state"]["conversation_history"][0]["text"])
        self.assertEqual(payload["state"]["silence_since_last_word_ms"], 650)
        self.assertIs(payload["state"]["asr_final"], True)
        self.assertEqual(payload["questions"]["turn_complete"]["type"], "noul")
        self.assertIn("true", payload["questions"]["turn_complete"]["criteria"])
        self.assertIn("false", payload["questions"]["holding_floor"]["criteria"])
        self.assertEqual({"turn_complete", "holding_floor", "addressed_to_agent"}, set(payload["questions"]))
        self.assertEqual(judgment.turn_complete, 0.84)
        self.assertEqual(judgment.holding_floor, 0.08)
        self.assertEqual(judgment.addressed_to_agent, 0.93)
        # No taking_floor question while the assistant is silent: scored 0.0, never an error.
        self.assertEqual(judgment.taking_floor, 0.0)
        self.assertEqual(judgment.model, "inception/mercury-decide-20260930")
        self.assertEqual(judgment.input_tokens, 494)
        self.assertEqual(judgment.output_tokens, 2)
        self.assertIsNotNone(judgment.latency_ms)

    async def test_http_error_status_becomes_provider_error_without_body_details(self):
        def handler(request):
            return httpx.Response(400, json={"error": "bad request: transcript would be echoed here"})

        judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(handler))
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}),
            self.assertRaises(JevProviderError) as ctx,
        ):
            await judge(self.make_turn())
        self.assertEqual(str(ctx.exception), "http_400")

    async def test_retryable_statuses_map_to_retryable_provider_error(self):
        # 429/5xx (congested route) is distinct from a malformed
        # response; the response body is never surfaced.
        for status in (429, 500, 503):

            def handler(request, status=status):
                return httpx.Response(status, json={"error": "congested: transcript would be echoed here"})

            judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(handler))
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}),
                self.assertRaises(JevProviderError) as ctx,
            ):
                await judge(self.make_turn())
            self.assertEqual(str(ctx.exception), "retryable_provider_error", f"status={status}")

    async def test_transport_failure_is_sanitized_provider_error(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(handler))
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}),
            self.assertRaises(JevProviderError) as ctx,
        ):
            await judge(self.make_turn())
        self.assertEqual(str(ctx.exception), "ConnectError")

    async def test_malformed_or_out_of_range_responses_are_provider_errors(self):
        cases = [
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"answers": {}}),
            httpx.Response(
                200,
                json={
                    "answers": {
                        "turn_complete": {"noul": 1.5},
                        "holding_floor": {"noul": 0.1},
                        "addressed_to_agent": {"noul": 0.9},
                    }
                },
            ),
        ]
        for response in cases:
            with self.subTest(body=getattr(response, "_content", b"")[:40]):
                judge = OpenRouterDecisionsJudge(transport=httpx.MockTransport(lambda request, r=response: r))
                with (
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"}),
                    self.assertRaises(JevProviderError),
                ):
                    await judge(self.make_turn())


if __name__ == "__main__":
    unittest.main()
