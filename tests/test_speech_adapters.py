import asyncio
import json
import unittest
from types import SimpleNamespace

from voxcare.providers import DeepgramConnection, ElevenLabsTTS, OpenAIResponsesAdapter


class FakeSocket:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        async def iterator():
            for message in self.messages:
                yield message

        return iterator()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_adapter_streams_with_strict_allowlisted_functions(self):
        class FakeStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(
                            type="function_call",
                            name="lookup_order_details",
                            arguments='{"order_code":"ORD-1002"}',
                            call_id="call-safe",
                        ),
                    )

                return iterator()

        class FakeResponses:
            def __init__(self):
                self.payload = None

            async def create(self, **payload):
                self.payload = payload
                return FakeStream()

        adapter = OpenAIResponsesAdapter(api_key="unit-test-secret", model="gpt-6-luna")
        fake = FakeResponses()
        adapter.client = SimpleNamespace(responses=fake)
        tools = [
            {
                "name": "lookup_order_details",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"order_code": {"type": "string"}},
                    "required": ["order_code"],
                    "additionalProperties": False,
                },
            }
        ]

        plans = await adapter.plan_turn(
            user_text="ORD-1002 contact person@example.com",
            history=({"role": "user", "content": "Call +1 (415) 555-0134"},),
            available_tools=tools,
        )
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.arguments, {"order_code": "ORD-1002"})
        self.assertEqual(fake.payload["model"], "gpt-6-luna")
        self.assertTrue(fake.payload["stream"])
        self.assertTrue(fake.payload["tools"][0]["strict"])
        self.assertFalse(fake.payload["tools"][0]["parameters"]["additionalProperties"])
        sent_text = " ".join(item["content"] for item in fake.payload["input"])
        self.assertNotIn("person@example.com", sent_text)
        self.assertNotIn("415", sent_text)
        self.assertIn("[REDACTED_EMAIL]", sent_text)
        self.assertIn("[REDACTED_PHONE]", sent_text)


class SpeechAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepgram_adapter_connects_to_nova2_english_interim_stream(self):
        socket = FakeSocket(
            [
                json.dumps(
                    {
                        "type": "Results",
                        "channel": {"alternatives": [{"transcript": "where is ORD-1002"}]},
                        "start": 1.25,
                        "duration": 0.6,
                        "is_final": False,
                        "speech_final": False,
                    }
                )
            ]
        )
        captured = {}

        async def connector(uri, **kwargs):
            captured.update(uri=uri, **kwargs)
            return socket

        connection = DeepgramConnection(api_key="unit-test-secret", connector=connector)
        await connection.connect()
        await connection.send_audio(b"\x00\x00")
        event = await asyncio.wait_for(connection.events().__anext__(), timeout=1)
        await connection.close()

        self.assertIn("model=nova-2", captured["uri"])
        self.assertIn("language=en-US", captured["uri"])
        self.assertIn("interim_results=true", captured["uri"])
        self.assertIn("sample_rate=16000", captured["uri"])
        self.assertEqual(captured["additional_headers"]["Authorization"], "Token unit-test-secret")
        self.assertEqual(event["type"], "transcript")
        self.assertEqual(event["start"], 1.25)
        self.assertEqual(event["duration"], 0.6)
        self.assertEqual(event["text"], "where is ORD-1002")
        self.assertEqual(socket.sent[0], b"\x00\x00")

    async def test_elevenlabs_adapter_streams_pcm_chunks_and_uses_flash_model(self):
        socket = FakeSocket(
            [json.dumps({"audio": "AQIDBA==", "isFinal": False}), json.dumps({"audio": "BQY=", "isFinal": True})]
        )
        captured = {}

        def connector(uri, **kwargs):
            captured.update(uri=uri, **kwargs)
            return socket

        adapter = ElevenLabsTTS(api_key="unit-test-secret", voice_id="VOICE-DEMO", connector=connector)
        chunks = [
            chunk
            async for chunk in adapter.stream(
                "Order ORD-1002 is in transit; tracking 1Z999AA10123456784, ETA 2026-10-15. "
                "Call person@example.com at +1 (415) 555-0134."
            )
        ]

        self.assertEqual(chunks, [b"\x01\x02\x03\x04", b"\x05\x06"])
        self.assertIn("model_id=eleven_flash_v2_5", captured["uri"])
        self.assertIn("output_format=pcm_16000", captured["uri"])
        self.assertEqual(captured["additional_headers"]["xi-api-key"], "unit-test-secret")
        self.assertEqual(json.loads(socket.sent[0])["text"], " ")
        sent_text = json.loads(socket.sent[1])["text"]
        self.assertIn("ORD-1002", sent_text)
        self.assertIn("1Z999AA10123456784", sent_text)
        self.assertIn("2026-10-15", sent_text)
        self.assertNotIn("person@example.com", sent_text)
        self.assertNotIn("415", sent_text)
        self.assertEqual(json.loads(socket.sent[2]), {"text": "", "flush": True})


if __name__ == "__main__":
    unittest.main()
