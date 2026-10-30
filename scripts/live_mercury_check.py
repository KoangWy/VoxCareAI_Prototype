"""Live Mercury check: real OpenRouter Decisions calls through the app seams.

Two phases, both hitting the real `inception/mercury-decide` endpoint:

A. Judge every transcript of `evaluation/jev_cases.json` with
   `OpenRouterDecisionsJudge` and report the scores next to the controller
   thresholds (0.72 turn-complete / 0.50 holding-floor). This is calibration
   evidence, not an accuracy claim: N is 6 transcripts, not N>=30/intent.

B. Run the real WebSocket voice flow under FastAPI TestClient with a faked
   ASR/TTS/dialogue and the real OpenRouter judge in `TurnController`,
   asserting the loop reaches COMMIT and dispatches the turn.

Requires OPENROUTER_API_KEY in the environment (or repo `.env`). Exit code is
non-zero when any phase fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

import voxcare.app as api
from voxcare import providers as voxcare_providers
from voxcare.contracts import PlannedToolCall
from voxcare.turns import JevConfigurationError, JevProviderError, TurnInput
from voxcare.turns.openrouter import OpenRouterDecisionsJudge

TURN_COMPLETE_THRESHOLD = 0.72
HOLDING_FLOOR_THRESHOLD = 0.50


def load_cases() -> list[dict]:
    path = APP_ROOT / "evaluation" / "jev_cases.json"
    return json.loads(path.read_text())["cases"]


async def phase_a(judge: OpenRouterDecisionsJudge) -> bool:
    print("=== Phase A: jev_cases transcripts through the real OpenRouter judge ===")
    ok = True
    for case in load_cases():
        assistant_last = case["assistant_last_message"]
        history: list[dict[str, str]] = [{"role": "assistant", "text": assistant_last}]
        assistant_speech = case.get("assistant_current_speech")
        for index, transcript in enumerate(case["transcript_sequence"]):
            turn = TurnInput(
                session_id="live-check",
                turn_epoch=1,
                transcript=transcript,
                conversation_history=tuple(history),
                assistant_last_message=assistant_last,
                silence_ms=700,
                asr_final=True,
                stable_word_ratio=1.0,
                local_vad_quiet=True,
                assistant_current_speech=assistant_speech,
            )
            try:
                judgment = await judge(turn)
            except (JevConfigurationError, JevProviderError) as exc:
                print(f"  {case['case_id']!r} [{index}] PROVIDER ERROR: {exc}")
                ok = False
                continue
            commit = (
                judgment.turn_complete >= TURN_COMPLETE_THRESHOLD
                and judgment.holding_floor < HOLDING_FLOOR_THRESHOLD
                and judgment.addressed_to_agent >= 0.5
            )
            print(
                f"  {case['case_id']!r} [{index}] {transcript!r} -> "
                f"turn_complete={judgment.turn_complete:.4f} holding_floor={judgment.holding_floor:.4f} "
                f"addressed={judgment.addressed_to_agent:.4f} taking={judgment.taking_floor} "
                f"latency={judgment.latency_ms}ms tokens={judgment.input_tokens}/{judgment.output_tokens} "
                f"=> {'COMMIT' if commit else 'HOLD'}"
            )
            history.append({"role": "user", "text": transcript})
    return ok


class FakeASR:
    def __init__(self):
        import queue

        self.queue: queue.Queue = queue.Queue()
        self.audio_frames: list[bytes] = []

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

    async def plan_turn(self, **kwargs):
        return self.plan


def receive_until(websocket, predicate, limit=40):
    messages = []
    for _ in range(limit):
        message = websocket.receive_json()
        messages.append(message)
        if predicate(message):
            return message, messages
    raise AssertionError(f"event not received; saw {[item.get('type') for item in messages]}")


def phase_b() -> bool:
    print("=== Phase B: real WebSocket flow with the real OpenRouter judge (ASR/TTS/dialogue faked) ===", flush=True)
    from fastapi.testclient import TestClient

    # The fakes feed transcript events through DeepgramConnection.from_env; force
    # the provider dispatch onto that path even when the local .env selects Vosk.
    os.environ["VOXCARE_ASR_PROVIDER"] = "deepgram"
    asr = FakeASR()
    voxcare_providers.DeepgramConnection.from_env = lambda: asr
    voxcare_providers.ElevenLabsTTS.from_env = FakeTTS.from_env
    api.boundary.dialogue_factory = lambda: FakeDialogue(
        PlannedToolCall(
            "retrieve_policy_clauses",
            {
                "query": "LG washing machine warranty",
                "category": "WARRANTY",
                "product_category": "WASHING_MACHINE",
                "brand": "LG",
            },
        )
    )

    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
            first = socket.receive_json()
            turn_provider = first.get("providers", {}).get("turn_decision", {})
            print(
                f"  turn_decision provider: name={turn_provider.get('name')!r} "
                f"mode={turn_provider.get('mode')!r} configured={turn_provider.get('configured')} "
                f"model={turn_provider.get('model')!r}"
            )
            if not turn_provider.get("configured"):
                print("  FAIL: turn_decision provider is not configured")
                return False

            socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            assert socket.receive_json()["type"] == "voice_started"
            socket.send_json({"type": "vad_speech_start", "segment_id": "seg-1", "client_monotonic_ms": 100})
            receive_until(socket, lambda m: m.get("type") == "turn_control")

            asr.queue.put(
                {
                    "type": "transcript",
                    "text": "Track ORD-1002.",
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
            decision, seen = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("state") in {"COMMIT", "DEGRADED"}
            )
            scores = decision.get("scores") or {}
            print(
                f"  decision: state={decision['state']} reason={decision['reason']} "
                f"scores={ {k: round(v, 4) for k, v in scores.items()} }"
            )
            if decision["state"] != "COMMIT":
                print(f"  FAIL: expected COMMIT; messages seen: {[m.get('type') for m in seen]}")
                return False
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            result = turn["result"]
            print(
                f"  turn dispatched: outcome={result['outcome']} tool_outcomes="
                f"{[o['tool_name'] for o in result.get('tool_outcomes', [])]}"
            )
            audio_start, _ = receive_until(socket, lambda m: m.get("type") == "audio_start")
            print(f"  audio streaming started: sample_rate={audio_start.get('sample_rate')}")
            print("  PASS: real Mercury judgment committed the turn inside the voice flow")
            return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env", override=False)
    judge = OpenRouterDecisionsJudge(
        model=os.getenv("VOXCARE_JEV_MODEL", "inception/mercury-decide"),
        timeout_ms=int(os.getenv("VOXCARE_JEV_TIMEOUT_MS", "1500")),
    )

    results = []
    if args.phase in ("a", "both"):
        results.append(("phase A (case scores)", asyncio.run(phase_a(judge))))
    if args.phase in ("b", "both"):
        results.append(("phase B (ws flow)", phase_b()))
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
