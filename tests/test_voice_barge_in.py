"""Voice-socket behaviour: barge-in matrix, re-judge, close codes, protocol contract."""

import asyncio
import json
import unittest
from queue import Queue
from threading import Event

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voxcare.app as api
from voxcare import providers as voxcare_providers
from voxcare.contracts import PlannedToolCall
from voxcare.turns import JevJudgment, TurnController
from voxcare.voice import protocol as vm


class FakeASR:
    def __init__(self):
        self.queue = Queue()
        self.audio_frames = []

    async def connect(self):
        return None

    async def send_audio(self, frame):
        self.audio_frames.append(frame)

    async def finalize(self):
        return None

    async def close(self):
        self.queue.put(None)

    async def events(self):
        while True:
            event = await asyncio.to_thread(self.queue.get)
            if event is None:
                return
            yield event


class QuickTTS:
    @classmethod
    def from_env(cls):
        return cls()

    async def stream(self, text):
        yield b"\x00\x00\x01\x00"


class BlockingTTS:
    """TTS that stalls mid-stream so the assistant stays 'speaking' on cue."""

    last = None

    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.cancelled = Event()

    @classmethod
    def from_env(cls):
        instance = cls()
        cls.last = instance
        return instance

    async def stream(self, text):
        self.started.set()
        try:
            await asyncio.to_thread(self.release.wait, 10)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield b"\x00\x00\x01\x00"


class FakeDialogue:
    def __init__(self):
        self.user_texts = []

    async def plan_turn(self, **kwargs):
        self.user_texts.append(kwargs.get("user_text"))
        return PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"})


class ScriptedJev:
    """Yield one scripted score-set per call, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_transcripts = []

    async def __call__(self, turn):
        self.calls += 1
        self.seen_transcripts.append(turn.transcript)
        scores = self.script[min(self.calls - 1, len(self.script) - 1)]
        return JevJudgment(**scores)


def receive_until(websocket, predicate, limit=60):
    messages = []
    for _ in range(limit):
        message = websocket.receive_json()
        messages.append(message)
        if predicate(message):
            return message, messages
    raise AssertionError(f"event not received; saw {[item.get('type') for item in messages]}")


class BargeInMatrixTests(unittest.TestCase):
    def setUp(self):
        BlockingTTS.last = None

    def open_session(self, monkeypatch, client, *, judge, tts_cls, extra_env=None):
        asr = FakeASR()
        monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
        monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", tts_cls.from_env)
        monkeypatch.setattr(api.boundary, "dialogue_factory", lambda: FakeDialogue())
        monkeypatch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
        monkeypatch.setenv("VOXCARE_TURN_MODE", "jev")
        for key, value in (extra_env or {}).items():
            monkeypatch.setenv(key, value)
        session = client.post("/api/sessions").json()
        return session, asr

    def commit_a_turn(self, socket, asr, *, segment="s1"):
        socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
        started, _ = receive_until(socket, lambda m: m.get("type") in ("voice_started", "provider_unavailable"))
        if started["type"] == "provider_unavailable":
            assert socket.receive_json()["type"] == "voice_started"
        socket.send_json({"type": "vad_speech_start", "segment_id": segment, "client_monotonic_ms": 100})
        receive_until(socket, lambda m: m.get("type") == "turn_control")
        asr.queue.put(
            {
                "type": "transcript",
                "text": "Track ORD-1002",
                "is_final": True,
                "speech_final": True,
                "sequence": 1,
            }
        )
        receive_until(socket, lambda m: m.get("type") == "transcript_final")
        socket.send_json(
            {
                "type": "vad_quiet",
                "segment_id": segment,
                "client_monotonic_ms": 900,
                "speech_end_monotonic_ms": 300,
                "silence_ms": 600,
            }
        )
        _, seen = receive_until(socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT")
        audio_start, _ = receive_until(socket, lambda m: m.get("type") == "audio_start")
        return audio_start, seen

    def test_semantic_backchannel_keeps_audio_without_interrupt(self):
        """'uh-huh' during TTS holds the turn; audio untouched, no interrupt."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                judge = ScriptedJev(
                    [
                        {"turn_complete": 0.95, "holding_floor": 0.0},
                        {"turn_complete": 0.9, "holding_floor": 0.0, "taking_floor": 0.1},
                    ]
                )
                session, asr = self.open_session(
                    patch, client, judge=judge, tts_cls=BlockingTTS, extra_env={"VOXCARE_BARGE_IN": "semantic"}
                )
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    self.commit_a_turn(socket, asr)
                    assert BlockingTTS.last.started.wait(2)
                    socket.send_json(
                        {
                            "type": "vad_speech_start",
                            "segment_id": "s2",
                            "client_monotonic_ms": 1200,
                            "agent_speaking": True,
                        }
                    )
                    _, seen = receive_until(
                        socket,
                        lambda m: m.get("type") == "turn_control" and m.get("reason") == "genuine_vad_speech",
                    )
                    assert not any(m.get("type") == "interrupt" for m in seen)
                    # The held words arrive, then quiet: the judge rules backchannel.
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "uh-huh",
                            "is_final": True,
                            "speech_final": True,
                            "sequence": 2,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_final")
                    socket.send_json(
                        {
                            "type": "vad_quiet",
                            "segment_id": "s2",
                            "client_monotonic_ms": 1900,
                            "speech_end_monotonic_ms": 1300,
                            "silence_ms": 600,
                        }
                    )
                    held, seen2 = receive_until(
                        socket,
                        lambda m: m.get("type") == "turn_control" and m.get("reason") == "backchannel_hold",
                    )
                    assert held["state"] == "HOLD"
                    assert not any(m.get("type") == "interrupt" for m in seen2)
                    assert not BlockingTTS.last.cancelled.is_set()
                    BlockingTTS.last.release.set()

    def test_semantic_real_interruption_sends_interrupt_and_stops_tts(self):
        """'wait stop' during TTS sends interrupt and cancels the stream."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                judge = ScriptedJev(
                    [
                        {"turn_complete": 0.95, "holding_floor": 0.0},
                        {"turn_complete": 0.9, "holding_floor": 0.0, "taking_floor": 0.95},
                    ]
                )
                session, asr = self.open_session(
                    patch, client, judge=judge, tts_cls=BlockingTTS, extra_env={"VOXCARE_BARGE_IN": "semantic"}
                )
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    self.commit_a_turn(socket, asr)
                    assert BlockingTTS.last.started.wait(2)
                    socket.send_json(
                        {
                            "type": "vad_speech_start",
                            "segment_id": "s2",
                            "client_monotonic_ms": 1200,
                            "agent_speaking": True,
                        }
                    )
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "Wait stop",
                            "is_final": True,
                            "speech_final": True,
                            "sequence": 2,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_final")
                    socket.send_json(
                        {
                            "type": "vad_quiet",
                            "segment_id": "s2",
                            "client_monotonic_ms": 1600,
                            "speech_end_monotonic_ms": 1000,
                            "silence_ms": 600,
                        }
                    )
                    interrupt, _ = receive_until(socket, lambda m: m.get("type") == "interrupt")
                    assert interrupt["reason"] == "taking_floor"
                    assert BlockingTTS.last.cancelled.wait(2)
                    BlockingTTS.last.release.set()

    def test_default_mode_cancels_tts_immediately_without_interrupt(self):
        """Default (off): any onset cancels TTS at once; no interrupt message exists."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                judge = ScriptedJev([{"turn_complete": 0.95, "holding_floor": 0.0}])
                session, asr = self.open_session(patch, client, judge=judge, tts_cls=BlockingTTS)
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    self.commit_a_turn(socket, asr)
                    assert BlockingTTS.last.started.wait(2)
                    socket.send_json({"type": "vad_speech_start", "segment_id": "s2", "client_monotonic_ms": 1200})
                    held, seen = receive_until(
                        socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "HOLD"
                    )
                    assert held["reason"] == "genuine_vad_speech"
                    assert not any(m.get("type") == "interrupt" for m in seen)
                    assert BlockingTTS.last.cancelled.wait(2)
                    BlockingTTS.last.release.set()


class RejudgeTests(unittest.TestCase):
    def test_text_change_rejudges_after_debounce_without_new_quiet(self):
        """A revised partial re-judges (~150 ms debounce) with no new quiet event."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                asr = FakeASR()
                patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
                patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", QuickTTS.from_env)
                judge = ScriptedJev(
                    [
                        {"turn_complete": 0.3, "holding_floor": 0.0},
                        {"turn_complete": 0.95, "holding_floor": 0.0},
                    ]
                )
                patch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
                patch.setenv("VOXCARE_TURN_MODE", "jev")
                session = client.post("/api/sessions").json()
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
                    first, _ = receive_until(
                        socket, lambda m: m.get("type") in ("voice_started", "provider_unavailable")
                    )
                    if first["type"] == "provider_unavailable":
                        assert socket.receive_json()["type"] == "voice_started"
                    socket.send_json({"type": "vad_speech_start", "segment_id": "r1", "client_monotonic_ms": 100})
                    receive_until(socket, lambda m: m.get("type") == "turn_control")
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "Track order",
                            "is_final": True,
                            "speech_final": False,
                            "sequence": 1,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_partial")
                    socket.send_json(
                        {
                            "type": "vad_quiet",
                            "segment_id": "r1",
                            "client_monotonic_ms": 900,
                            "speech_end_monotonic_ms": 300,
                            "silence_ms": 600,
                        }
                    )
                    first, _ = receive_until(
                        socket, lambda m: m.get("type") == "turn_control" and m.get("reason") != "genuine_vad_speech"
                    )
                    assert first["state"] == "HOLD"
                    assert judge.calls == 1
                    # Revised text, still quiet, no new quiet event: debounce re-judges.
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "Track order ORD-1002",
                            "is_final": True,
                            "speech_final": False,
                            "sequence": 2,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_partial")
                    _second, _ = receive_until(
                        socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT"
                    )
                    assert judge.calls == 2
                    # Segment join keeps the earlier words; the re-judge runs
                    # on the revised (longer) text.
                    assert judge.seen_transcripts[-1].endswith("Track order ORD-1002")
                    assert len(judge.seen_transcripts[-1]) > len(judge.seen_transcripts[0])

    def test_extended_silence_alone_rejudges(self):
        """A second quiet with no new ASR still re-judges (borderline scores)."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                asr = FakeASR()
                patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
                patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", QuickTTS.from_env)
                judge = ScriptedJev(
                    [
                        {"turn_complete": 0.6, "holding_floor": 0.0},
                        {"turn_complete": 0.95, "holding_floor": 0.0},
                    ]
                )
                patch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
                patch.setenv("VOXCARE_TURN_MODE", "jev")
                session = client.post("/api/sessions").json()
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
                    first, _ = receive_until(
                        socket, lambda m: m.get("type") in ("voice_started", "provider_unavailable")
                    )
                    if first["type"] == "provider_unavailable":
                        assert socket.receive_json()["type"] == "voice_started"
                    socket.send_json({"type": "vad_speech_start", "segment_id": "q1", "client_monotonic_ms": 100})
                    receive_until(socket, lambda m: m.get("type") == "turn_control")
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "Track ORD-1002",
                            "is_final": True,
                            "speech_final": True,
                            "sequence": 1,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_final")
                    socket.send_json(
                        {
                            "type": "vad_quiet",
                            "segment_id": "q1",
                            "client_monotonic_ms": 900,
                            "speech_end_monotonic_ms": 300,
                            "silence_ms": 600,
                        }
                    )
                    first, _ = receive_until(
                        socket, lambda m: m.get("type") == "turn_control" and m.get("reason") != "genuine_vad_speech"
                    )
                    assert first["state"] == "HOLD"
                    assert judge.calls == 1
                    # No new ASR — extended silence alone fires the re-judge.
                    socket.send_json(
                        {
                            "type": "vad_quiet",
                            "segment_id": "q1",
                            "client_monotonic_ms": 2000,
                            "speech_end_monotonic_ms": 300,
                            "silence_ms": 1700,
                        }
                    )
                    _second, _ = receive_until(
                        socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT"
                    )
                    assert judge.calls == 2


class CloseCodeTests(unittest.TestCase):
    def test_oversize_frame_sends_error_then_close_1009(self):
        """Oversize audio names the error first, then closes with 1009."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                asr = FakeASR()
                patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
                patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", QuickTTS.from_env)
                patch.setattr(api.boundary, "dialogue_factory", lambda: FakeDialogue())
                patch.setenv("VOXCARE_TURN_MODE", "baseline")
                session = client.post("/api/sessions").json()
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
                    assert socket.receive_json()["type"] == "voice_started"
                    socket.send_bytes(b"\x00" * (65 * 1024))
                    error = socket.receive_json()
                    assert error == {"type": "error", "code": "AUDIO_FRAME_TOO_LARGE"}
                    with self.assertRaises(WebSocketDisconnect) as ctx:
                        socket.receive_json()
                    self.assertEqual(ctx.exception.code, 1009)

    def test_session_audio_budget_sends_error_then_close_1008(self):
        """The >24 MiB session budget names the error, then closes with 1008."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                asr = FakeASR()
                patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
                patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", QuickTTS.from_env)
                patch.setattr(api.boundary, "dialogue_factory", lambda: FakeDialogue())
                patch.setenv("VOXCARE_TURN_MODE", "baseline")
                session = client.post("/api/sessions").json()
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
                    assert socket.receive_json()["type"] == "voice_started"
                    for _ in range(385):
                        socket.send_bytes(b"\x00" * (64 * 1024))
                    error = socket.receive_json()
                    assert error == {"type": "error", "code": "SESSION_AUDIO_LIMIT_REACHED"}
                    with self.assertRaises(WebSocketDisconnect) as ctx:
                        socket.receive_json()
                    self.assertEqual(ctx.exception.code, 1008)

    def test_utterance_end_is_ignored_explicitly(self):
        """Provider UtteranceEnd is recorded and dropped — socket stays alive."""
        with TestClient(api.app) as client:
            import pytest

            monkeypatch = pytest.MonkeyPatch()
            with monkeypatch.context() as patch:
                asr = FakeASR()
                patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
                patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", QuickTTS.from_env)
                patch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(ScriptedJev([])))
                patch.setattr(api.boundary, "dialogue_factory", lambda: FakeDialogue())
                patch.setenv("VOXCARE_TURN_MODE", "jev")
                session = client.post("/api/sessions").json()
                with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                    assert socket.receive_json()["type"] == "session"
                    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
                    first, _ = receive_until(
                        socket, lambda m: m.get("type") in ("voice_started", "provider_unavailable")
                    )
                    if first["type"] == "provider_unavailable":
                        assert socket.receive_json()["type"] == "voice_started"
                    socket.send_json({"type": "vad_speech_start", "segment_id": "u1", "client_monotonic_ms": 100})
                    receive_until(socket, lambda m: m.get("type") == "turn_control")
                    asr.queue.put({"type": "UtteranceEnd"})
                    asr.queue.put(
                        {
                            "type": "transcript",
                            "text": "Track ORD-1002",
                            "is_final": True,
                            "speech_final": True,
                            "sequence": 1,
                        }
                    )
                    receive_until(socket, lambda m: m.get("type") == "transcript_final")
                trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()
                assert any(e.get("event_type") == "PROVIDER_UTTERANCE_END" for e in trace["events"])

    def test_chunked_body_without_content_length_is_capped(self):
        """The 64 KiB bound applies to the streamed body, not just the header."""

        def body():
            for _ in range(100):
                yield b"x" * 1024

        with TestClient(api.app) as client:
            response = client.post("/api/sessions/anything/turns", content=body())
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"error": "REQUEST_TOO_LARGE"})


class ProtocolContractTests(unittest.TestCase):
    def test_server_names_match_the_protocol_module(self):
        """Every wire name flows through the constants module's single truth."""
        import pathlib

        package = pathlib.Path(api.__file__).parent
        source = "\n".join(
            (package / name).read_text()
            for name in (
                "http/routes.py",
                "voice/connection.py",
                "voice/dispatch.py",
                "voice/asr_events.py",
                "voice/turn_control.py",
                "voice/evaluate.py",
            )
        )
        reverse = {value: attr for attr, value in vars(vm).items() if attr.isupper() and isinstance(value, str)}
        for name in (*vm.SERVER_MESSAGE_TYPES, *vm.ERROR_CODES):
            self.assertIn(f"vm.{reverse[name]}", source, f"{name} not emitted via vm.{reverse[name]}")
        for name in vm.CLIENT_MESSAGE_TYPES:
            self.assertIn(f"vm.{reverse[name]}", source, f"client message {name} unhandled via vm.{reverse[name]}")
        self.assertEqual(len(set(vm.SERVER_MESSAGE_TYPES)), len(vm.SERVER_MESSAGE_TYPES))
        self.assertEqual(len(set(vm.CLIENT_MESSAGE_TYPES)), len(vm.CLIENT_MESSAGE_TYPES))

    def test_voice_protocol_doc_covers_every_name(self):
        import pathlib

        doc = (pathlib.Path(api.__file__).parents[2] / "docs" / "VOICE_PROTOCOL.md").read_text()
        for name in (*vm.SERVER_MESSAGE_TYPES, *vm.CLIENT_MESSAGE_TYPES, *vm.ERROR_CODES):
            self.assertIn(name, doc, f"{name} missing from VOICE_PROTOCOL.md")

    def test_session_hello_announces_protocol_two(self):
        with TestClient(api.app) as client:
            session = client.post("/api/sessions").json()
            with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                hello = socket.receive_json()
        self.assertEqual(hello["type"], "session")
        self.assertEqual(hello["protocol"], vm.PROTOCOL_VERSION)
        self.assertEqual(hello["protocol"], 2)


def _open_voice(patch, client, judge, *, tts_cls=QuickTTS, env=None):
    asr = FakeASR()
    patch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    patch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", tts_cls.from_env)
    patch.setattr(api.boundary, "dialogue_factory", lambda: FakeDialogue())
    patch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
    patch.setenv("VOXCARE_TURN_MODE", "jev")
    for key, value in (env or {}).items():
        patch.setenv(key, value)
    return client.post("/api/sessions").json(), asr


def _start_and_speak(socket, asr, text, *, segment="v1"):
    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
    receive_until(socket, lambda m: m.get("type") == "voice_started")
    socket.send_json({"type": "vad_speech_start", "segment_id": segment, "client_monotonic_ms": 100})
    receive_until(socket, lambda m: m.get("type") == "turn_control")
    asr.queue.put({"type": "transcript", "text": text, "is_final": True, "speech_final": True, "sequence": 1})
    receive_until(socket, lambda m: m.get("type") == "transcript_final")
    socket.send_json(
        {
            "type": "vad_quiet",
            "segment_id": segment,
            "client_monotonic_ms": 900,
            "speech_end_monotonic_ms": 300,
            "silence_ms": 650,
        }
    )


class VoiceFlowTests(unittest.TestCase):
    def test_session_hello_announces_barge_in_mode(self):
        import pytest

        with TestClient(api.app) as client, pytest.MonkeyPatch().context() as patch:
            session, _ = _open_voice(patch, client, ScriptedJev([]), env={"VOXCARE_BARGE_IN": "semantic"})
            with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                hello = socket.receive_json()
        self.assertEqual(hello["type"], "session")
        self.assertEqual(hello["barge_in"], "semantic")

    def test_hold_verdict_rejudges_after_more_silence_without_client_prompting(self):
        import pytest

        judge = ScriptedJev(
            [{"turn_complete": 0.3, "holding_floor": 0.0}, {"turn_complete": 0.95, "holding_floor": 0.0}]
        )
        with TestClient(api.app) as client, pytest.MonkeyPatch().context() as patch:
            session, asr = _open_voice(patch, client, judge)
            with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                assert socket.receive_json()["type"] == "session"
                _start_and_speak(socket, asr, "I want to check my order")
                held, _ = receive_until(
                    socket, lambda m: m.get("type") == "turn_control" and m["reason"] != "genuine_vad_speech"
                )
                receive_until(socket, lambda m: m.get("type") == "turn_control" and m["state"] == "COMMIT")
        self.assertEqual(held["reason"], "turn_not_complete")
        self.assertEqual(judge.calls, 2)
        self.assertGreater(len(judge.seen_transcripts), 1)

    def test_caller_retry_discards_the_degraded_words(self):
        import pytest

        judge = ScriptedJev([{"turn_complete": 0.2, "holding_floor": 0.0}])
        with TestClient(api.app) as client, pytest.MonkeyPatch().context() as patch:
            session, asr = _open_voice(patch, client, judge)
            with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                assert socket.receive_json()["type"] == "session"
                _start_and_speak(socket, asr, "Um so the thing is")
                recovery, _ = receive_until(socket, lambda m: m.get("type") == "recovery_required")
                socket.send_json(
                    {
                        "type": "caller_retry",
                        "transcript_version": recovery["transcript_version"],
                        "hold_episode_id": recovery["hold_episode_id"],
                    }
                )
                receive_until(socket, lambda m: m.get("type") == "retry_ready")
                asr.queue.put(
                    {
                        "type": "transcript",
                        "text": "Track ORD-1002",
                        "is_final": False,
                        "speech_final": False,
                        "sequence": 2,
                    }
                )
                partial, _ = receive_until(socket, lambda m: m.get("type") == "transcript_partial")
        self.assertEqual(partial["text"], "Track ORD-1002")

    def test_backchannel_during_playback_is_discarded_not_left_to_degrade(self):
        import pytest

        judge = ScriptedJev(
            [
                {"turn_complete": 0.95, "holding_floor": 0.0},
                {"turn_complete": 0.9, "holding_floor": 0.0, "taking_floor": 0.1},
            ]
        )
        BlockingTTS.last = None
        with TestClient(api.app) as client, pytest.MonkeyPatch().context() as patch:
            session, asr = _open_voice(patch, client, judge, tts_cls=BlockingTTS, env={"VOXCARE_BARGE_IN": "semantic"})
            with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
                assert socket.receive_json()["type"] == "session"
                _start_and_speak(socket, asr, "Track ORD-1002")
                receive_until(socket, lambda m: m.get("type") == "audio_start")
                assert BlockingTTS.last.started.wait(2)
                socket.send_json(
                    {
                        "type": "vad_speech_start",
                        "segment_id": "b2",
                        "client_monotonic_ms": 1200,
                        "agent_speaking": True,
                    }
                )
                asr.queue.put(
                    {"type": "transcript", "text": "uh-huh", "is_final": True, "speech_final": True, "sequence": 2}
                )
                receive_until(socket, lambda m: m.get("type") == "transcript_final")
                socket.send_json(
                    {
                        "type": "vad_quiet",
                        "segment_id": "b2",
                        "client_monotonic_ms": 1900,
                        "speech_end_monotonic_ms": 1300,
                        "silence_ms": 600,
                    }
                )
                receive_until(socket, lambda m: m.get("reason") == "backchannel_hold")
                BlockingTTS.last.release.set()
                for _ in range(20):  # the released stream sends one binary batch, then audio_end
                    message = socket.receive()
                    if message.get("text") and json.loads(message["text"]).get("type") == "audio_end":
                        break
                else:
                    raise AssertionError("audio_end not received")
            trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()["events"]
        self.assertTrue(any(event.get("event_type") == "BACKCHANNEL_DISCARDED" for event in trace))


class VoiceStateTests(unittest.TestCase):
    def test_silence_grows_from_the_quiet_report(self):
        from voxcare.voice.state import VoiceState

        state = VoiceState()
        self.assertIsNone(state.silence_ms())
        state.mark_quiet(650)
        state.quiet_received_ms -= 1000
        self.assertGreaterEqual(state.silence_ms(), 1650)
        state.mark_speaking()
        self.assertIsNone(state.silence_ms())


class DeadlineWatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_host_timer_follows_a_rearmed_deadline_then_degrades(self):
        import time
        from types import SimpleNamespace

        from voxcare.turns import TurnDecision
        from voxcare.voice.state import VoiceState
        from voxcare.voice.turn_control import send_turn_control

        now = int(time.monotonic() * 1000)
        replies = [
            TurnDecision("HOLD", "deadline_not_reached", 1, "ep", "req", 1, 50, now + 60),
            TurnDecision("DEGRADED", "hold_deadline", 1, "ep", "req", 1, 120, now + 60),
        ]
        asked = []

        class Controller:
            async def on_hold_deadline(self, session_id, epoch, episode_id):
                asked.append(episode_id)
                return replies.pop(0)

        boundary = SimpleNamespace(
            get_session=lambda _sid: asyncio.sleep(0, result=object()),
            get_turn_controller=lambda _sid: asyncio.sleep(0, result=Controller()),
            record_audit_event=lambda *_args: asyncio.sleep(0),
        )
        sent = []

        async def send_json(payload):
            sent.append(payload)

        state = VoiceState()
        state.mark_quiet(700)
        first = TurnDecision("HOLD", "turn_not_complete", 1, "ep", "req", 1, 0, now + 20)
        await send_turn_control(boundary, send_json, "SES", state, first, version=0, generation=0)
        await asyncio.wait_for(asyncio.gather(*state.deadline_tasks.values(), return_exceptions=True), 2)
        await asyncio.sleep(0.2)

        self.assertEqual(asked, ["ep", "ep"])
        self.assertEqual([m.get("state") for m in sent if m["type"] == "turn_control"], ["HOLD", "DEGRADED"])
        self.assertEqual([m["type"] for m in sent][-1], "recovery_required")
        self.assertEqual(state.deadline_tasks, {})


if __name__ == "__main__":
    unittest.main()
