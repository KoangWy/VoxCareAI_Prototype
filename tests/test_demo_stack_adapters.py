import asyncio
import json
import unittest
from types import SimpleNamespace

from voxcare.providers import (
    CascadeDialogueProvider,
    OpenAIChatCompletionsAdapter,
    PiperTTS,
    ProviderUnavailable,
    VoskConnection,
    asr_connection_from_env,
    conversation_factory_from_env,
    provider_status,
    tts_from_env,
)
from voxcare.providers.tts import resample_pcm16_to_16k, split_sentences

TOOLS = (
    {
        "name": "lookup_order_details",
        "description": "read",
        "parameters": {
            "type": "object",
            "properties": {"order_code": {"type": "string"}},
            "additionalProperties": False,
        },
    },
)


def _chunk_delta(tool_calls=None, content=None):
    delta = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class ChatCompletionsAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, **kwargs):
        adapter = OpenAIChatCompletionsAdapter(api_key="unit-test", model="deepseek-v4.1-flash", **kwargs)
        fake = _FakeChatCompletions()
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return adapter, fake

    async def test_streams_tool_call_fragments_into_one_plan(self):
        adapter, fake = self._adapter()
        fake.events = [
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=0, id="call-1", function=SimpleNamespace(name="lookup_order_", arguments='{"order')
                    )
                ]
            ),
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=0, id=None, function=SimpleNamespace(name="details", arguments='_code":"ORD-1002"}')
                    )
                ]
            ),
        ]
        plans = await adapter.plan_turn(user_text="Track ORD-1002", history=(), available_tools=TOOLS)
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.name, "lookup_order_details")
        payload = fake.payload
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["tools"][0]["function"]["name"], "lookup_order_details")
        self.assertNotIn("strict", payload["tools"][0]["function"])

    async def test_masks_pii_before_sending(self):
        adapter, fake = self._adapter()
        fake.events = [_chunk_delta(content="hello")]
        await adapter.plan_turn(user_text="email me at person@example.com", history=(), available_tools=TOOLS)
        sent = json.dumps(fake.payload["messages"])
        self.assertNotIn("person@example.com", sent)

    async def test_reasoning_effort_only_when_configured(self):
        adapter, fake = self._adapter(reasoning_effort="low")
        fake.events = [_chunk_delta(content="hi")]
        await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS)
        self.assertEqual(fake.payload["extra_body"], {"reasoning_effort": "low"})

        adapter_none, fake_none = self._adapter()
        fake_none.events = [_chunk_delta(content="hi")]
        await adapter_none.plan_turn(user_text="hi", history=(), available_tools=TOOLS)
        self.assertNotIn("extra_body", fake_none.payload)

    async def test_plain_content_without_tool_call_returns_empty_list(self):
        adapter, fake = self._adapter()
        fake.events = [_chunk_delta(content="Here is your answer.")]
        self.assertEqual(await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS), [])

    async def test_two_tool_calls_returned_in_order(self):
        adapter, fake = self._adapter()
        fake.events = [
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=0, id="a", function=SimpleNamespace(name="lookup_order_details", arguments="{}")
                    )
                ]
            ),
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=1, id="b", function=SimpleNamespace(name="track_shipment_status", arguments="{}")
                    )
                ]
            ),
        ]
        plans = await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS)
        self.assertEqual([plan.name for plan in plans], ["lookup_order_details", "track_shipment_status"])
        self.assertEqual([plan.call_id for plan in plans], ["a", "b"])

    async def test_four_tool_calls_truncated_to_cap_of_three(self):
        adapter, fake = self._adapter()
        fake.events = [
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=i,
                        id=f"call-{i}",
                        function=SimpleNamespace(name="lookup_order_details", arguments="{}"),
                    )
                ]
            )
            for i in range(4)
        ]
        with self.assertLogs("voxcare.providers", level="INFO"):
            plans = await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS)
        self.assertEqual(len(plans), 3)
        self.assertEqual([plan.call_id for plan in plans], ["call-0", "call-1", "call-2"])

    async def test_parallel_tool_calls_false_sent_and_gateway_fallback(self):
        adapter, fake = self._adapter()
        fake.events = [_chunk_delta(content="hi")]
        await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS)
        self.assertFalse(fake.payload["parallel_tool_calls"])

        class RejectingCompletions(_FakeChatCompletions):
            async def create(self, **payload):
                if "parallel_tool_calls" in payload:
                    raise TypeError("create() got an unexpected keyword argument 'parallel_tool_calls'")
                self.payload = payload

                class Stream:
                    def __aiter__(self):
                        async def iterator():
                            return
                            yield

                        return iterator()

                return Stream()

        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=RejectingCompletions()))
        self.assertEqual(await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS), [])

    async def test_invalid_arguments_json_rejected(self):
        adapter, fake = self._adapter()
        fake.events = [
            _chunk_delta(
                tool_calls=[
                    SimpleNamespace(
                        index=0, id="a", function=SimpleNamespace(name="lookup_order_details", arguments="{broken")
                    )
                ]
            )
        ]
        with self.assertRaises(ValueError):
            await adapter.plan_turn(user_text="hi", history=(), available_tools=TOOLS)


class _FakeChatCompletions:
    def __init__(self):
        self.payload = None

    async def create(self, **payload):
        self.payload = payload

        class Stream:
            def __init__(self, events):
                self._events = events

            def __aiter__(self):
                async def iterator():
                    for event in self._events:
                        yield event

                return iterator()

        return Stream(self.events)


class PiperTests(unittest.IsolatedAsyncioTestCase):
    def test_split_sentences_keeps_boundaries_and_wraps_long_text(self):
        parts = split_sentences("Order ORD-1002 shipped. ETA is Friday. Thanks!")
        self.assertEqual(parts, ["Order ORD-1002 shipped.", "ETA is Friday.", "Thanks!"])
        long = "word " * 120
        wrapped = split_sentences(long.strip())
        self.assertTrue(all(len(part) <= 300 for part in wrapped))
        self.assertTrue(len(wrapped) > 1)

    def test_resample_22050_to_16k_length_and_passthrough(self):
        pcm = b"\x01\x00" * 22050
        resampled = resample_pcm16_to_16k(pcm, 22_050)
        self.assertEqual(len(resampled), 2 * 16_000)
        self.assertIs(pcm, resample_pcm16_to_16k(pcm, 16_000))

    async def test_stream_yields_16k_pcm_per_sentence(self):
        calls = []

        def runner(sentence):
            calls.append(sentence)
            return b"\x00\x01" * 22_050  # one second at 22050 Hz

        tts = PiperTTS(model_path="voice.onnx", voice_sample_rate=22_050, runner=runner)
        chunks = [chunk async for chunk in tts.stream("First sentence. Second sentence.")]
        self.assertEqual(calls, ["First sentence.", "Second sentence."])
        self.assertEqual(sum(len(chunk) for chunk in chunks), 2 * 2 * 16_000)

    async def test_in_process_voice_is_loaded_once_and_reused(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from voxcare.providers import tts as tts_module

        loads = []

        class FakeVoice:
            def synthesize(self, sentence):
                yield SimpleNamespace(audio_int16_bytes=b"\x00\x01" * 16_000)

        def load(path):
            loads.append(path)
            return FakeVoice()

        tts_module._load_piper_voice.cache_clear()
        with patch.object(tts_module, "_load_piper_voice", side_effect=load):
            tts = PiperTTS(model_path="voice.onnx", voice_sample_rate=16_000)
            await tts.warm()
            chunks = [chunk async for chunk in tts.stream("One. Two.")]
        self.assertEqual(sum(len(chunk) for chunk in chunks), 2 * 2 * 16_000)
        self.assertEqual(loads, ["voice.onnx"] * 3)  # warm + one call per sentence, all served by the cache

    async def test_stream_rejects_invalid_text(self):
        tts = PiperTTS(model_path="voice.onnx", runner=lambda _s: b"")
        with self.assertRaises(ValueError):
            async for _chunk in tts.stream("   "):
                pass

    def test_from_env_reports_missing_model(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"VOXCARE_PIPER_MODEL": ""}, clear=False):
            with self.assertRaises(ProviderUnavailable) as missing:
                PiperTTS.from_env()
            self.assertEqual(missing.exception.code, "PIPER_MODEL_NOT_CONFIGURED")
        with patch.dict("os.environ", {"VOXCARE_PIPER_MODEL": "/nonexistent/voice.onnx"}, clear=False):
            with self.assertRaises(ProviderUnavailable) as absent:
                PiperTTS.from_env()
            self.assertEqual(absent.exception.code, "PIPER_MODEL_MISSING")


class FakeRecognizer:
    def __init__(self):
        self.audio = b""
        self._final_delivered = True

    def AcceptWaveform(self, pcm: bytes) -> bool:
        self.audio += pcm
        # Endpoint after ~0.5 s of audio (8000 samples = 16000 bytes), mimicking Kaldi.
        return len(self.audio) >= 16_000

    def PartialResult(self) -> str:
        return json.dumps({"partial": "where is my order"})

    def Result(self) -> str:
        return json.dumps({"text": "where is my order ORD 1002"})

    def FinalResult(self) -> str:
        return json.dumps({"text": ""})


class VoskConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_then_final_events_match_deepgram_contract(self):
        connection = VoskConnection(model_path="ignored", recognizer_factory=lambda _p: FakeRecognizer())
        await connection.connect()
        collected: list[dict] = []

        async def consume():
            async for event in connection.events():
                collected.append(event)

        consumer = asyncio.create_task(consume())
        await connection.send_audio(b"\x00\x00" * 4000)  # 4000 samples -> partial
        await connection.send_audio(b"\x00\x00" * 4000)  # 8000 samples -> endpoint -> final
        await asyncio.sleep(0)
        await connection.close()
        await asyncio.wait_for(consumer, timeout=2)

        partials = [event for event in collected if event["type"] == "transcript" and not event["is_final"]]
        finals = [event for event in collected if event["type"] == "transcript" and event["is_final"]]
        self.assertEqual(len(partials), 1)
        self.assertEqual(partials[0]["text"], "where is my order")
        self.assertFalse(partials[0]["speech_final"])
        self.assertEqual(partials[0]["start"], 1.0)  # upcoming segment id
        self.assertEqual(len(finals), 1)
        self.assertTrue(finals[0]["speech_final"])
        self.assertEqual(finals[0]["start"], 1.0)
        self.assertEqual(finals[0]["text"], "where is my order ORD 1002")
        self.assertIn({"type": "SpeechStarted"}, collected)

    async def test_send_audio_requires_open_connection(self):
        connection = VoskConnection(model_path="ignored")
        with self.assertRaises(ProviderUnavailable):
            await connection.send_audio(b"\x00\x00")

    async def test_finalize_emits_trailing_final(self):
        class TrailingRecognizer(FakeRecognizer):
            def AcceptWaveform(self, pcm: bytes) -> bool:
                return False

            def FinalResult(self) -> str:
                return json.dumps({"text": "trailing words"})

        connection = VoskConnection(model_path="ignored", recognizer_factory=lambda _p: TrailingRecognizer())
        await connection.connect()
        collected: list[dict] = []

        async def consume():
            async for event in connection.events():
                collected.append(event)

        consumer = asyncio.create_task(consume())
        await connection.send_audio(b"\x00\x00" * 100)
        await connection.finalize()
        await connection.close()
        await asyncio.wait_for(consumer, timeout=2)
        finals = [event for event in collected if event["type"] == "transcript" and event["is_final"]]
        self.assertEqual([event["text"] for event in finals], ["trailing words"])
        self.assertTrue(finals[0]["speech_final"])


class DemoStackSelectionTests(unittest.TestCase):
    def test_provider_status_reflects_offline_selections(self):
        import os

        env = {
            "VOXCARE_ASR_PROVIDER": "vosk",
            "VOXCARE_VOSK_MODEL": "models/missing-vosk",
            "VOXCARE_TTS_PROVIDER": "piper",
            "VOXCARE_PIPER_MODEL": "models/missing.onnx",
            "VOXCARE_OPENAI_TRANSPORT": "chat",
            "OPENAI_API_KEY": "unit-test",
        }
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            status = provider_status()
            self.assertEqual(status["asr"].name, "Vosk (offline)")
            self.assertFalse(status["asr"].configured)
            self.assertEqual(status["tts"].name, "Piper (offline)")
            self.assertFalse(status["tts"].configured)
            self.assertEqual(status["conversation"].name, "OpenAI Chat Completions (gateway)")
            self.assertTrue(status["conversation"].configured)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_unsupported_providers_reported_as_configuration_failure(self):
        import os

        for provider_env, factory, code in (
            ({"VOXCARE_ASR_PROVIDER": "whispery"}, asr_connection_from_env, "ASR_PROVIDER_UNSUPPORTED"),
            ({"VOXCARE_TTS_PROVIDER": "espeak"}, tts_from_env, "TTS_PROVIDER_UNSUPPORTED"),
        ):
            previous = {key: os.environ.get(key) for key in provider_env}
            os.environ.update(provider_env)
            try:
                with self.assertRaises(ProviderUnavailable) as caught:
                    factory()
                self.assertEqual(caught.exception.code, code)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_vosk_missing_directory_reports_model_missing(self):
        import os

        env = {"VOXCARE_ASR_PROVIDER": "vosk", "VOXCARE_VOSK_MODEL": "models/nope"}
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            with self.assertRaises(ProviderUnavailable) as caught:
                asr_connection_from_env()
            self.assertEqual(caught.exception.code, "VOSK_MODEL_MISSING")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_conversation_factory_uses_chat_transport(self):
        import os

        env = {
            "VOXCARE_OPENAI_TRANSPORT": "chat",
            "OPENAI_API_KEY": "unit-test",
            "VOXCARE_OPENAI_MODEL": "deepseek-v4.1-flash",
        }
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            cascade = conversation_factory_from_env()
            self.assertIsInstance(cascade, CascadeDialogueProvider)
            self.assertIsInstance(cascade.primary, OpenAIChatCompletionsAdapter)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_conversation_factory_rejects_unknown_transport(self):
        import os

        env = {"VOXCARE_OPENAI_TRANSPORT": "grpc", "OPENAI_API_KEY": "unit-test"}
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            with self.assertRaises(ProviderUnavailable) as caught:
                conversation_factory_from_env()
            self.assertEqual(caught.exception.code, "CONVERSATION_TRANSPORT_UNSUPPORTED")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
