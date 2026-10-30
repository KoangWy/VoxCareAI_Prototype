import asyncio
import json
import time
from queue import Queue
from threading import Event

from fastapi.testclient import TestClient

import voxcare.app as api
from voxcare import providers as voxcare_providers
from voxcare.contracts import PlannedToolCall
from voxcare.turns import JevJudgment, TurnController


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


class FakeTTS:
    @classmethod
    def from_env(cls):
        return cls()

    async def stream(self, text):
        yield b"\x00\x00\x01\x00"


class FakeDialogue:
    def __init__(self, plan):
        self.plan = plan
        self.user_texts = []

    async def plan_turn(self, **kwargs):
        self.user_texts.append(kwargs.get("user_text"))
        return self.plan


class SlowDialogue(FakeDialogue):
    def __init__(self, plan):
        super().__init__(plan)
        self.started = Event()
        self.release = Event()
        self.cancelled = Event()

    async def plan_turn(self, **_kwargs):
        self.started.set()
        try:
            await asyncio.to_thread(self.release.wait, 5)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return self.plan


class RecordingJev:
    def __init__(self):
        self.turns = []
        self.scores = [0.95, 0.95]

    async def __call__(self, turn):
        self.turns.append(turn)
        score = self.scores.pop(0) if self.scores else 0.95
        return JevJudgment(turn_complete=score, holding_floor=0.0)


class RevisionAwareJev:
    def __init__(self):
        self.calls = 0
        self.first_started = Event()
        self.first_cancelled = Event()

    async def __call__(self, _turn):
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.first_cancelled.set()
                raise
        return JevJudgment(turn_complete=0.95, holding_floor=0.0)


def receive_until(websocket, predicate, limit=30):
    messages = []
    for _ in range(limit):
        message = websocket.receive_json()
        messages.append(message)
        if predicate(message):
            return message, messages
    raise AssertionError(f"event not received; saw {[item.get('type') for item in messages]}")


def configure_fake_voice(monkeypatch, dialogue):
    asr = FakeASR()
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(api.boundary, "dialogue_factory", lambda: dialogue)
    monkeypatch.setenv("VOXCARE_TURN_MODE", "baseline")
    return asr


def start_fake_voice(client, session_id, socket):
    assert socket.receive_json()["type"] == "session"
    socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
    assert socket.receive_json()["type"] == "voice_started"
    socket.send_json({"type": "vad_speech_start", "segment_id": "slow-seg", "client_monotonic_ms": 100})
    receive_until(socket, lambda m: m.get("type") == "turn_control")


def send_final_and_quiet(asr, socket, text="Track ORD-1002"):
    asr.queue.put({"type": "transcript", "text": text, "is_final": True, "speech_final": True, "sequence": 1})
    receive_until(socket, lambda m: m.get("type") == "transcript_final")
    socket.send_json(
        {
            "type": "vad_quiet",
            "segment_id": "slow-seg",
            "client_monotonic_ms": 900,
            "speech_end_monotonic_ms": 300,
            "silence_ms": 600,
        }
    )
    decision, _ = receive_until(socket, lambda m: m.get("type") == "turn_control")
    assert decision["state"] == "COMMIT"


def test_ws_voice_to_policy_evidence_tts_and_playback_ack(monkeypatch):
    asr = FakeASR()
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(
        api.boundary,
        "dialogue_factory",
        lambda: FakeDialogue(
            PlannedToolCall(
                "retrieve_policy_clauses",
                {
                    "query": "LG washing machine warranty",
                    "category": "WARRANTY",
                    "product_category": "WASHING_MACHINE",
                    "brand": "LG",
                },
            )
        ),
    )
    monkeypatch.setenv("VOXCARE_TURN_MODE", "baseline")

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json(
                {
                    "type": "start",
                    "sample_rate": 16_000,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "client_clock": {"name": "performance.now", "resolution_ms": 0.01},
                }
            )
            assert socket.receive_json()["type"] == "voice_started"
            socket.send_bytes(b"\x00\x00\x01\x00")
            socket.send_json({"type": "vad_speech_start", "segment_id": "seg-1", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "What warranty applies to my LG washing machine?",
                    "is_final": True,
                    "speech_final": True,
                    "sequence": 1,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_final")
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "seg-1",
                    "client_monotonic_ms": 900,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 600,
                }
            )
            decision, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT"
            )
            assert decision["turn_mode"] == "baseline"
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            result = turn["result"]
            assert result["outcome"] == "ANSWERED"
            assert result["corpus_id"] == "VOXCOM-DEMO-POLICY-V1"
            assert result["evidence"][0]["is_fictional"] is True
            audio_start, _ = receive_until(socket, lambda m: m.get("type") == "audio_start")
            # every binary TTS batch is preceded by a JSON audio_chunk
            # envelope carrying turn/chunk identity (protocol 2).
            chunk_envelope, _ = receive_until(socket, lambda m: m.get("type") == "audio_chunk")
            assert chunk_envelope["turn_id"] == audio_start["turn_id"]
            assert chunk_envelope["audio_chunk_id"] == audio_start["audio_chunk_id"]
            assert chunk_envelope["chunk_index"] == 0
            assert chunk_envelope["byte_length"] == 4
            assert socket.receive_bytes() == b"\x00\x00\x01\x00"
            assert socket.receive_json()["type"] == "audio_end"
            socket.send_json(
                {
                    "type": "playback_started",
                    "turn_id": result["turn_id"],
                    "audio_chunk_id": audio_start["audio_chunk_id"],
                    "client_monotonic_ms": 1500,
                    "clock_name": "performance.now",
                    "acoustic_confirmed": False,
                }
            )
            ack = socket.receive_json()
            assert ack["type"] == "playback_ack_recorded"
            assert ack["ttfa_ms"] is None
            assert ack["ttfa_status"] == "unavailable"

        trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()

    assert asr.audio_frames == [b"\x00\x00\x01\x00"]
    assert trace["persistence"] == "in_memory_for_session_lifetime"
    actions = [event for event in trace["events"] if event.get("action_name")]
    tool = next(event for event in actions if event["action_name"] == "TOOL_INVOCATION")
    assert tool["request_payload_json"]["tool_name"] == "retrieve_policy_clauses"
    assert tool["outcome_status"] == "SUCCESS"
    assert any(event.get("event_type") == "PLAYBACK_STARTED" for event in trace["events"])


def test_ws_jev_configuration_failure_requires_explicit_finish_and_does_not_fake_success(monkeypatch):
    asr = FakeASR()
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(
        api.boundary,
        "dialogue_factory",
        lambda: FakeDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"})),
    )
    monkeypatch.setenv("VOXCARE_TURN_MODE", "jev")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            assert socket.receive_json()["type"] == "provider_unavailable"
            assert socket.receive_json()["type"] == "voice_started"
            socket.send_json({"type": "vad_speech_start", "segment_id": "seg-2", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "Track order ORD-1002",
                    "is_final": True,
                    "speech_final": True,
                    "sequence": 1,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_final")
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "seg-2",
                    "client_monotonic_ms": 900,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 600,
                }
            )
            first, _ = receive_until(socket, lambda m: m.get("type") == "turn_control")
            assert first["reason"] == "jev_unconfigured"
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "seg-2",
                    "client_monotonic_ms": 1100,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 800,
                }
            )
            degraded, _ = receive_until(socket, lambda m: m.get("type") == "recovery_required")
            socket.send_json(
                {
                    "type": "manual_finish",
                    "transcript_version": degraded["transcript_version"],
                    "hold_episode_id": degraded["hold_episode_id"],
                }
            )
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            assert turn["result"]["outcome"] == "ANSWERED"
            assert "ORD-1002" in turn["result"]["assistant_text"]

        trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()

    assert any(event.get("event_type") == "PROVIDER_UNAVAILABLE" for event in trace["events"])
    degraded_audits = [event for event in trace["events"] if event.get("action_name") == "TURN_COMPLETION_DEGRADED"]
    assert degraded_audits
    fallback = [event for event in trace["events"] if event.get("action_name") == "TURN_COMPLETION_FALLBACK_RESOLVED"]
    assert fallback and fallback[-1]["request_payload_json"]["resolution"] == "MANUAL_FINISH"


def test_http_handoff_records_latest_server_observed_asr_context(monkeypatch):
    asr = configure_fake_voice(
        monkeypatch, FakeDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    )

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            assert socket.receive_json()["type"] == "voice_started"
            socket.send_json({"type": "vad_speech_start", "segment_id": "support-seg", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "Please contact caller@example.com or +1 (415) 555-0134",
                    "is_final": False,
                    "speech_final": False,
                    "sequence": 1,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_partial")
            response = client.post(
                f"/api/sessions/{session['session_id']}/handoff", json={"reason": "Please get me a person"}
            )
            assert response.status_code == 200
        trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()

    handoff = next(item for item in trace["events"] if item.get("action_name") == "HANDOFF_REQUEST_RECORDED")
    serialized = str(handoff["request_payload_json"])
    assert "caller@example.com" not in serialized
    assert "415" not in serialized
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_PHONE]" in serialized


def test_stalled_conversation_dispatch_times_out_with_explicit_unavailable(monkeypatch):
    """A gateway stall must not hang the socket: 30 s bound, then provider_unavailable, no turn."""
    import voxcare.voice.dispatch as dispatch

    class HangingDialogue(FakeDialogue):
        async def plan_turn(self, **_kwargs):
            await asyncio.Future()
            raise AssertionError("unreachable")

    dialogue = HangingDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    asr = configure_fake_voice(monkeypatch, dialogue)
    monkeypatch.setattr(dispatch, "_DISPATCH_TURN_TIMEOUT_S", 0.2)

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            start_fake_voice(client, session["session_id"], socket)
            asr.queue.put(
                {"type": "transcript", "text": "Track ORD-1002", "is_final": True, "speech_final": True, "sequence": 1}
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_final")
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "slow-seg",
                    "client_monotonic_ms": 900,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 600,
                }
            )
            decision, _ = receive_until(socket, lambda m: m.get("type") == "turn_control")
            assert decision["state"] == "COMMIT"
            unavailable, _ = receive_until(socket, lambda m: m.get("type") == "provider_unavailable")
            assert unavailable["provider"] == "conversation"
            assert unavailable["error"] == "CONVERSATION_TIMEOUT"

    dialogue = SlowDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    asr = configure_fake_voice(monkeypatch, dialogue)

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            start_fake_voice(client, session["session_id"], socket)
            send_final_and_quiet(asr, socket)
            assert dialogue.started.wait(2)
            socket.send_json({"type": "vad_speech_start", "segment_id": "next-seg", "client_monotonic_ms": 1200})
            next_turn, _ = receive_until(socket, lambda m: m.get("type") == "turn_control")
            assert next_turn["turn_id"] != session["session_id"]
            assert dialogue.cancelled.wait(2)
            dialogue.release.set()

    assert dialogue.cancelled.is_set()


def test_reset_cancels_slow_provider_and_invalidates_old_session(monkeypatch):
    dialogue = SlowDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    asr = configure_fake_voice(monkeypatch, dialogue)

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            start_fake_voice(client, session["session_id"], socket)
            send_final_and_quiet(asr, socket)
            assert dialogue.started.wait(2)
            socket.send_json({"type": "reset"})
            reset = receive_until(socket, lambda m: m.get("type") == "session_reset")[0]
            assert reset["session"]["session_id"] != session["session_id"]
            assert dialogue.cancelled.wait(2)
            dialogue.release.set()
        old = client.get(f"/api/sessions/{session['session_id']}")

    assert old.status_code == 404


def test_jev_receives_masked_prior_turn_and_can_evaluate_stable_partial(monkeypatch):
    asr = FakeASR()
    judge = RecordingJev()
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(
        api.boundary,
        "dialogue_factory",
        lambda: FakeDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"})),
    )
    monkeypatch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
    monkeypatch.setenv("VOXCARE_TURN_MODE", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "controlled-test-key")

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        prior = client.post(
            f"/api/sessions/{session['session_id']}/turns",
            json={"text": "My email is caller@example.com"},
        )
        assert prior.status_code == 200
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            receive_until(socket, lambda m: m.get("type") == "voice_started")
            socket.send_json({"type": "vad_speech_start", "segment_id": "partial-seg", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "And can you check",
                    "is_final": True,
                    "speech_final": False,
                    "sequence": 1,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_partial")
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "partial-seg",
                    "client_monotonic_ms": 700,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 400,
                }
            )
            decision, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("reason") != "genuine_vad_speech"
            )
            assert decision["state"] == "COMMIT"

    assert judge.turns
    turn = judge.turns[-1]
    assert turn.asr_final is False
    assert turn.stable_word_ratio == 1.0
    assert turn.conversation_history[-2]["role"] == "user"
    assert turn.conversation_history[-1]["role"] == "assistant"
    assert "caller@example.com" not in str(turn.conversation_history)
    assert turn.assistant_last_message == turn.conversation_history[-1]["text"]


def test_jev_holds_and_joins_resumed_deepgram_segments_then_discards_late_final(monkeypatch):
    asr = FakeASR()
    judge = RecordingJev()
    judge.scores = [0.1, 0.95]
    dialogue = FakeDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(api.boundary, "dialogue_factory", lambda: dialogue)
    monkeypatch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
    monkeypatch.setenv("VOXCARE_TURN_MODE", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "controlled-test-key")

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            receive_until(socket, lambda m: m.get("type") == "voice_started")
            socket.send_json({"type": "vad_speech_start", "segment_id": "first-fragment", "client_monotonic_ms": 100})
            first_control, _ = receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "I want order status for",
                    "is_final": True,
                    "speech_final": False,
                    "start": 0.0,
                    "duration": 1.0,
                    "sequence": 1,
                }
            )
            partial, _ = receive_until(socket, lambda m: m.get("type") == "transcript_partial")
            assert partial["text"] == "I want order status for"
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "first-fragment",
                    "client_monotonic_ms": 500,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 200,
                }
            )
            hold, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("reason") != "genuine_vad_speech"
            )
            assert hold["state"] == "HOLD"

            socket.send_json({"type": "vad_speech_start", "segment_id": "resumed-fragment", "client_monotonic_ms": 700})
            resumed_control, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("reason") == "genuine_vad_speech"
            )
            assert resumed_control["turn_id"] == first_control["turn_id"]
            for text, final, sequence in (("ORD-10", False, 2), ("ORD-1002", False, 3), ("ORD-1002", True, 4)):
                asr.queue.put(
                    {
                        "type": "transcript",
                        "text": text,
                        "is_final": final,
                        "speech_final": False,
                        "start": 1.0,
                        "duration": 0.7,
                        "sequence": sequence,
                    }
                )
                receive_until(
                    socket,
                    lambda m, seq=sequence: m.get("sequence") == seq and m.get("type", "").startswith("transcript_"),
                )
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "resumed-fragment",
                    "client_monotonic_ms": 1500,
                    "speech_end_monotonic_ms": 900,
                    "silence_ms": 600,
                }
            )
            decision, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT"
            )
            assert decision["transcript_version"] > hold["transcript_version"]
            receive_until(socket, lambda m: m.get("type") == "turn")
            while True:
                message = socket.receive()
                if message.get("text") and json.loads(message["text"]).get("type") == "audio_end":
                    break
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "ORD-1002",
                    "is_final": True,
                    "speech_final": True,
                    "start": 1.0,
                    "duration": 0.7,
                    "sequence": 5,
                }
            )
            for _ in range(20):
                trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()
                if any(event.get("event_type") == "ASR_TRANSCRIPT_AFTER_COMMIT_DISCARDED" for event in trace["events"]):
                    break
                time.sleep(0.05)

    assert dialogue.user_texts == ["I want order status for ORD-1002"]
    assert any(event.get("event_type") == "ASR_TRANSCRIPT_AFTER_COMMIT_DISCARDED" for event in trace["events"])


def test_new_quiet_stable_transcript_cancels_old_jev_call_and_uses_latest_version(monkeypatch):
    asr = FakeASR()
    judge = RevisionAwareJev()
    dialogue = FakeDialogue(PlannedToolCall("lookup_order_details", {"order_code": "ORD-1002"}))
    monkeypatch.setattr(voxcare_providers.DeepgramConnection, "from_env", lambda: asr)
    monkeypatch.setattr(voxcare_providers.ElevenLabsTTS, "from_env", FakeTTS.from_env)
    monkeypatch.setattr(api.boundary, "dialogue_factory", lambda: dialogue)
    monkeypatch.setattr(api.boundary, "turn_controller_factory", lambda: TurnController(judge))
    monkeypatch.setenv("VOXCARE_TURN_MODE", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "controlled-test-key")

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            assert socket.receive_json()["type"] == "session"
            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            receive_until(socket, lambda m: m.get("type") == "voice_started")
            socket.send_json({"type": "vad_speech_start", "segment_id": "revision", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "Track order ORD-1002",
                    "is_final": True,
                    "speech_final": False,
                    "start": 1.0,
                    "duration": 0.6,
                    "sequence": 1,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_partial")
            socket.send_json(
                {
                    "type": "vad_quiet",
                    "segment_id": "revision",
                    "client_monotonic_ms": 900,
                    "speech_end_monotonic_ms": 300,
                    "silence_ms": 600,
                }
            )
            assert judge.first_started.wait(2)
            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "please",
                    "is_final": True,
                    "speech_final": False,
                    "start": 1.7,
                    "duration": 0.3,
                    "sequence": 2,
                }
            )
            receive_until(socket, lambda m: m.get("type") == "transcript_partial" and m.get("sequence") == 2)
            assert judge.first_cancelled.wait(2)
            decision, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("state") == "COMMIT"
            )
            receive_until(socket, lambda m: m.get("type") == "turn")

    assert decision["transcript_version"] == 3
    assert judge.calls == 2
    assert dialogue.user_texts == ["Track order ORD-1002 please"]
