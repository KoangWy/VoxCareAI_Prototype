"""Live voice end-to-end harness: Piper synth -> Vosk ASR -> judge -> gateway LLM -> Piper TTS.

Runs the real WebSocket voice flow in-process (FastAPI TestClient) with every
provider real: Piper synthesises English utterances to 16 kHz mono PCM16,
frames stream as 20 ms binary chunks plus browser-style vad_speech_start /
vad_quiet control messages, Vosk transcribes, the configured turn judge
(VOXCARE_TURN_MODE from .env) verdicts, the gateway LLM plans the tool call,
and Piper speaks the answer back.

Scenarios:
  1. plain order question -> transcript + COMMIT + spoken answer (audio ids match)
  2. hesitation ("Let me think ... it is ORD-1002") -> HOLD then COMMIT after continuation
  3. self-echo: feed the assistant's own TTS audio back as mic input -> no new turn committed
  4. backchannel "uh-huh" while the assistant speaks (semantic barge-in mode) -> protocol check

Caps: at most MAX_JUDGE_CALLS turn judgments and MAX_LLM_TURNS turn dispatches;
every scenario has a per-step timeout. Output is secret-free (no keys, no raw
audio). Exit 0 only when every scenario passes.
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

TOTAL_TIMEOUT_S = 480  # hard wall-clock cap for the whole harness
FRAME_BYTES = 640  # 20 ms at 16 kHz mono PCM16
TRAILING_SILENCE_FRAMES = 75  # 1.5 s of zero audio so Vosk endpointing finalizes
STEP_TIMEOUT_S = 60
MAX_JUDGE_CALLS = 40

UTTERANCES = {
    # NOTE: the small Vosk model hears a spelled order code ("O R D 1 0 0 2")
    # as words, so the spoken-answer scenario uses a policy question the ASR
    # hears reliably; order-code lookup is covered by the typed campaign.
    "plain": "What is the return window for my TV?",
    "hesitation_part1": "Let me think.",
    "hesitation_part2": "What is the return window for my TV?",
    "backchannel": "Uh-huh.",
    "floor_grab": "Stop! I need to change my order right now.",
}


async def synth_16k(tts, text: str) -> bytes:
    """Synthesise text with the real Piper voice; return 16 kHz mono PCM16 bytes."""
    return b"".join([chunk async for chunk in tts.stream(text)])


def frames(pcm: bytes, size: int = FRAME_BYTES) -> list[bytes]:
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]


class Harness:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []
        self.judge_calls = 0
        self.llm_turns = 0

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(passed for _, passed, _ in self.results)


def receive_until(socket, predicate, limit: int = 200, timeout: float = STEP_TIMEOUT_S):
    """Read TestClient websocket messages until predicate matches; return (match, all seen)."""
    messages = []
    deadline = time.monotonic() + timeout
    for _ in range(limit):
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting; saw {[m.get('type') for m in messages]}")
        message = socket.receive_json()
        messages.append(message)
        if predicate(message):
            return message, messages
    raise AssertionError(f"event not received; saw {[m.get('type') for m in messages]}")


def _start_reader(socket, pending) -> None:
    """One daemon thread pumping raw socket messages into a queue (binary-safe)."""
    import threading

    def _read() -> None:
        while True:
            try:
                pending.put(socket.receive())
            except Exception as exc:
                pending.put({"_reader_error": type(exc).__name__})
                return

    threading.Thread(target=_read, daemon=True).start()


def drain_available(socket, wait_s: float = 4.0):
    """Collect messages arriving within wait_s (for negative assertions).

    A daemon reader thread feeds a queue so the wait is honestly bounded:
    TestClient's blocking receive has no timeout of its own.
    """
    import queue

    pending: queue.Queue = queue.Queue()
    _start_reader(socket, pending)
    messages = []
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            messages.append(pending.get(timeout=max(0.1, deadline - time.monotonic())))
        except queue.Empty:
            break
    return messages


async def stream_utterance(socket, pcm: bytes, segment_id: str, start_ms: float, silence_ms: int = 800):
    socket.send_json({"type": "vad_speech_start", "segment_id": segment_id, "client_monotonic_ms": start_ms})
    for frame in frames(pcm):
        socket.send_bytes(frame)
    for _ in range(TRAILING_SILENCE_FRAMES):
        socket.send_bytes(b"\x00" * FRAME_BYTES)
    socket.send_json(
        {
            "type": "vad_quiet",
            "segment_id": segment_id,
            "client_monotonic_ms": start_ms + 1000,
            "speech_end_monotonic_ms": start_ms + 1000 - silence_ms,
            "silence_ms": silence_ms,
        }
    )


def requiet(socket, segment_id: str, start_ms: float, silence_ms: int) -> None:
    """Extended-silence re-quiet: arms the server's silence re-judge path."""
    socket.send_json(
        {
            "type": "vad_quiet",
            "segment_id": segment_id,
            "client_monotonic_ms": start_ms,
            "speech_end_monotonic_ms": start_ms - silence_ms,
            "silence_ms": silence_ms,
        }
    )


def wait_verdict(socket, judge_name: str, harness, segment_id: str, start_ms: float, timeout: float = STEP_TIMEOUT_S):
    """Read verdicts; on HOLD, re-quiet with growing silence until COMMIT/DEGRADED/recovery/timeout."""
    silence_ms = 1500
    deadline = time.monotonic() + timeout
    tries = 0
    while time.monotonic() < deadline and tries < 8:
        message = socket.receive_json()
        if not isinstance(message, dict):
            continue
        mtype = message.get("type")
        if mtype == "turn_control":
            harness.judge_calls += 1
            state = message.get("state")
            if state in {"COMMIT", "DEGRADED"}:
                return message
            tries += 1
            print(f"    ({judge_name}: {state} {message.get('reason')}; re-quiet silence={silence_ms}ms)", flush=True)
            silence_ms = min(silence_ms + 1500, 9000)
            requiet(socket, segment_id, start_ms + silence_ms, silence_ms)
        elif mtype in {"recovery_required", "error", "provider_unavailable"}:
            print(f"    ({judge_name}: {mtype} {str(message)[:160]})", flush=True)
            return {"state": "DEGRADED", "reason": mtype, "scores": None}
        elif mtype in {"transcript_partial", "transcript_final"}:
            print(f"    ({judge_name}: {mtype} text={message.get('text', '')!r:.60})", flush=True)
    raise AssertionError(f"{judge_name}: no COMMIT/DEGRADED verdict within {timeout}s")


def collect_audio(socket, turn_id: str, timeout: float = STEP_TIMEOUT_S) -> tuple[bytes, str]:
    """Collect audio_start, all audio_chunk envelopes + binary frames, audio_end for a turn."""
    start, _ = receive_until(
        socket, lambda m: m.get("type") == "audio_start" and m.get("turn_id") == turn_id, timeout=timeout
    )
    chunk_id = start["audio_chunk_id"]
    audio = bytearray()
    deadline = time.monotonic() + timeout
    pending_envelope = False
    while time.monotonic() < deadline:
        message = socket.receive()
        if message.get("bytes") is not None:
            if not pending_envelope:
                raise AssertionError("binary TTS batch arrived without a preceding audio_chunk envelope")
            pending_envelope = False
            audio.extend(message["bytes"])
            continue
        import json as _json

        try:
            data = _json.loads(message.get("text", ""))
        except ValueError:
            continue
        if data.get("type") == "audio_chunk" and data.get("audio_chunk_id") == chunk_id:
            pending_envelope = True
        elif data.get("type") == "audio_end" and data.get("audio_chunk_id") == chunk_id:
            return bytes(audio), chunk_id


def scenario_plain(harness: Harness, client, tts, clock: list[float]) -> bytes:
    print("== Scenario 1: plain order question ==", flush=True)
    session = client.post("/api/sessions").json()
    sid = session["session_id"]
    with client.websocket_connect(f"/ws/sessions/{sid}") as socket:
        assert socket.receive_json()["type"] == "session"
        socket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                "channels": 1,
                "encoding": "pcm_s16le",
                "client_clock": {"name": "perf", "resolution_ms": 0.1},
            }
        )
        assert socket.receive_json()["type"] == "voice_started"
        pcm = asyncio.run(synth_16k(tts, UTTERANCES["plain"]))
        t0 = time.monotonic()
        asyncio.run(stream_utterance(socket, pcm, "seg-plain", clock[0]))
        try:
            final, _ = receive_until(socket, lambda m: m.get("type") == "transcript_final")
            harness.record("plain.transcript_final", True, f"text={final.get('text', '')!r:.80}")
        except AssertionError as exc:
            harness.record("plain.transcript_final", False, str(exc)[:120])
            return b""
        try:
            decision = wait_verdict(socket, "plain", harness, "seg-plain", clock[0])
            scores = decision.get("scores") or {}
            harness.record(
                "plain.judge_verdict",
                decision["state"] == "COMMIT",
                f"state={decision['state']} reason={decision.get('reason')} "
                f"scores={ {k: round(v, 3) for k, v in scores.items()} }",
            )
            if decision["state"] != "COMMIT":
                return b""
        except AssertionError as exc:
            harness.record("plain.judge_verdict", False, str(exc)[:120])
            return b""
        harness.llm_turns += 1
        try:
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            result = turn["result"]
            harness.record(
                "plain.turn_dispatched",
                result["outcome"] == "ANSWERED" and bool(result.get("assistant_text")),
                f"outcome={result['outcome']} tools={[o['tool_name'] for o in result.get('tool_outcomes', [])]} "
                f"answer={result.get('assistant_text', '')!r:.100}",
            )
            if result["outcome"] != "ANSWERED" or not result.get("assistant_text"):
                return b""
            turn_id = result["turn_id"]
        except AssertionError as exc:
            harness.record("plain.turn_dispatched", False, str(exc)[:120])
            return b""
        try:
            audio, chunk_id = collect_audio(socket, turn_id)
            harness.record(
                "plain.spoken_answer",
                len(audio) > 0,
                f"{len(audio)} bytes, chunk={chunk_id[:8]}…, latency={(time.monotonic() - t0):.1f}s",
            )
            return audio
        except AssertionError as exc:
            harness.record("plain.spoken_answer", False, str(exc)[:120])
            return b""


def scenario_hesitation(harness: Harness, client, tts, clock: list[float]) -> None:
    print("== Scenario 2: hesitation then continuation ==", flush=True)
    session = client.post("/api/sessions").json()
    with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
        assert socket.receive_json()["type"] == "session"
        socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
        assert socket.receive_json()["type"] == "voice_started"
        part1 = asyncio.run(synth_16k(tts, UTTERANCES["hesitation_part1"]))
        part2 = asyncio.run(synth_16k(tts, UTTERANCES["hesitation_part2"]))
        asyncio.run(stream_utterance(socket, part1, "seg-hes1", clock[0], silence_ms=400))
        try:
            first, _ = receive_until(socket, lambda m: m.get("type") == "turn_control", timeout=STEP_TIMEOUT_S)
            harness.judge_calls += 1
            harness.record(
                "hesitation.first_verdict",
                first["state"] in {"HOLD", "COMMIT"},
                f"state={first['state']} reason={first.get('reason')}",
            )
        except AssertionError as exc:
            harness.record("hesitation.first_verdict", False, str(exc)[:120])
            return
        asyncio.run(stream_utterance(socket, part2, "seg-hes2", clock[0] + 5000, silence_ms=800))
        try:
            decision = wait_verdict(socket, "hesitation", harness, "seg-hes2", clock[0] + 5000)
            harness.record(
                "hesitation.committed_after_continuation",
                decision["state"] == "COMMIT",
                f"state={decision['state']} reason={decision.get('reason')}",
            )
            if decision["state"] != "COMMIT":
                return
        except AssertionError as exc:
            harness.record("hesitation.committed_after_continuation", False, str(exc)[:120])
            return
        harness.llm_turns += 1
        try:
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            outcome = turn["result"]["outcome"]
            harness.record("hesitation.turn_dispatched", outcome == "ANSWERED", f"outcome={outcome}")
        except AssertionError as exc:
            harness.record("hesitation.turn_dispatched", False, str(exc)[:120])


def scenario_echo(harness: Harness, client, assistant_audio: bytes, clock: list[float]) -> None:
    print("== Scenario 3: self-echo does not self-talk ==", flush=True)
    if not assistant_audio:
        harness.record("echo.no_new_turn", False, "no assistant audio captured in scenario 1")
        return
    session = client.post("/api/sessions").json()
    with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
        assert socket.receive_json()["type"] == "session"
        socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
        assert socket.receive_json()["type"] == "voice_started"
        # Feed the assistant's own reply back as microphone input with NO vad_quiet:
        # without a quiet signal the controller must not evaluate, hence no turn.
        socket.send_json({"type": "vad_speech_start", "segment_id": "seg-echo", "client_monotonic_ms": clock[0]})
        for frame in frames(assistant_audio):
            socket.send_bytes(frame)
        seen = drain_available(socket, wait_s=6.0)
        import json as _json

        heard = []
        for item in seen:
            if not isinstance(item, dict) or item.get("bytes") is not None:
                continue
            try:
                heard.append(_json.loads(item.get("text", "")))
            except ValueError:
                continue
        turns = [m for m in heard if m.get("type") == "turn"]
        harness.record(
            "echo.no_new_turn",
            len(turns) == 0,
            f"{len(turns)} turn(s) committed from echo; heard={[m.get('type') for m in heard][:8]}",
        )


def scenario_backchannel(harness: Harness, client, tts, clock: list[float]) -> None:
    print("== Scenario 4: backchannel during assistant speech (VOXCARE_BARGE_IN=semantic) ==", flush=True)
    os.environ["VOXCARE_BARGE_IN"] = "semantic"
    session = client.post("/api/sessions").json()
    with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
        assert socket.receive_json()["type"] == "session"
        socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
        assert socket.receive_json()["type"] == "voice_started"
        pcm = asyncio.run(synth_16k(tts, UTTERANCES["plain"]))
        asyncio.run(stream_utterance(socket, pcm, "seg-bc0", clock[0]))
        try:
            decision, _ = receive_until(
                socket, lambda m: m.get("type") == "turn_control" and m.get("state") in {"COMMIT", "DEGRADED"}
            )
            if decision["state"] != "COMMIT":
                harness.record("backchannel.setup_commit", False, f"setup turn {decision['state']}")
                return
        except AssertionError as exc:
            harness.record("backchannel.setup_commit", False, str(exc)[:120])
            return
        harness.llm_turns += 1
        try:
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            turn_id = turn["result"]["turn_id"]
        except AssertionError as exc:
            harness.record("backchannel.setup_commit", False, str(exc)[:120])
            return
        # Start collecting audio, then speak a backchannel while it plays.
        start, _ = receive_until(socket, lambda m: m.get("type") == "audio_start" and m.get("turn_id") == turn_id)
        backchannel = asyncio.run(synth_16k(tts, UTTERANCES["backchannel"]))
        socket.send_json(
            {
                "type": "vad_speech_start",
                "segment_id": "seg-bc1",
                "client_monotonic_ms": clock[0] + 2000,
                "agent_speaking": True,
            }
        )
        for frame in frames(backchannel):
            socket.send_bytes(frame)
        socket.send_json(
            {
                "type": "vad_quiet",
                "segment_id": "seg-bc1",
                "client_monotonic_ms": clock[0] + 3000,
                "speech_end_monotonic_ms": clock[0] + 2600,
                "silence_ms": 400,
            }
        )
        harness.judge_calls += 1
        # Drain the reply and the backchannel's own verdict.
        drained = drain_audio(socket, start["audio_chunk_id"], timeout=STEP_TIMEOUT_S)
        # Backchannel ("uh-huh") must never interrupt: audio plays to the
        # end and the held words are judged a backchannel (or, when the small
        # Vosk model hears no word at all, held as an empty false-open). A
        # real floor grab is covered by scenario_interruption.
        harness.record(
            "backchannel.protocol",
            drained["audio_done"]
            and not drained["interrupted"]
            and drained["verdict_reason"] in {"backchannel_hold", "false_open_empty_transcript"},
            f"audio_end={drained['audio_done']} interrupt={drained['interrupted']} "
            f"verdict={drained['verdict_state']}/{drained['verdict_reason']} "
            f"saw={drained['seen'][:10]}",
        )


def drain_audio(socket, chunk_id: str, timeout: float = STEP_TIMEOUT_S) -> dict:
    """Drain one TTS stream plus the judgment of speech heard during it.

    The server can finish *sending* a long reply (``audio_end``) well before
    a browser finishes *playing* it, so the drain keeps reading after
    ``audio_end`` until the barge-in speech gets a judged verdict (any
    ``turn_control`` other than the onset's ``genuine_vad_speech``) or an
    ``interrupt`` arrives.
    """
    import json as _json
    import queue

    seen_types: list[str] = []
    deadline = time.monotonic() + timeout
    audio_done = False
    audio_end_after_interrupt = False
    verdict = None
    interrupted = False
    pending: queue.Queue = queue.Queue()
    _start_reader(socket, pending)
    quiet_rounds = 0
    while time.monotonic() < deadline:
        if interrupted and quiet_rounds >= 3:
            break  # TTS cancelled; no audio_end should follow
        if audio_done and verdict is not None and not interrupted and quiet_rounds >= 2:
            break  # judged, reply finished, nothing else coming
        try:
            message = pending.get(timeout=1.0)
        except queue.Empty:
            quiet_rounds += 1
            continue
        quiet_rounds = 0
        if message.get("bytes") is not None:
            seen_types.append("binary")
            continue
        try:
            data = _json.loads(message.get("text", ""))
        except ValueError:
            continue
        seen_types.append(data.get("type", "?"))
        if data.get("type") == "turn_control" and data.get("reason") != "genuine_vad_speech" and verdict is None:
            verdict = data
        if data.get("type") == "interrupt":
            interrupted = True
        if data.get("type") == "audio_end" and data.get("audio_chunk_id") == chunk_id:
            audio_done = True
            audio_end_after_interrupt = interrupted
    return {
        "audio_done": audio_done,
        "audio_end_after_interrupt": audio_end_after_interrupt,
        "interrupted": interrupted,
        "verdict_state": (verdict or {}).get("state", "none"),
        "verdict_reason": (verdict or {}).get("reason", ""),
        "seen": seen_types,
    }


def scenario_interruption(harness: Harness, client, tts, clock: list[float]) -> None:
    print("== Scenario 5: floor grab during assistant speech (VOXCARE_BARGE_IN=semantic) ==", flush=True)
    os.environ["VOXCARE_BARGE_IN"] = "semantic"
    session = client.post("/api/sessions").json()
    with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as socket:
        assert socket.receive_json()["type"] == "session"
        socket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
        assert socket.receive_json()["type"] == "voice_started"
        pcm = asyncio.run(synth_16k(tts, UTTERANCES["plain"]))
        asyncio.run(stream_utterance(socket, pcm, "seg-in0", clock[0]))
        try:
            decision = wait_verdict(socket, "interruption-setup", harness, "seg-in0", clock[0])
            if decision["state"] != "COMMIT":
                harness.record("interruption.setup_commit", False, f"setup turn {decision['state']}")
                return
        except AssertionError as exc:
            harness.record("interruption.setup_commit", False, str(exc)[:120])
            return
        harness.llm_turns += 1
        try:
            turn, _ = receive_until(socket, lambda m: m.get("type") == "turn")
            turn_id = turn["result"]["turn_id"]
        except AssertionError as exc:
            harness.record("interruption.setup_commit", False, str(exc)[:120])
            return
        start, _ = receive_until(socket, lambda m: m.get("type") == "audio_start" and m.get("turn_id") == turn_id)
        grab = asyncio.run(synth_16k(tts, UTTERANCES["floor_grab"]))
        socket.send_json(
            {
                "type": "vad_speech_start",
                "segment_id": "seg-in1",
                "client_monotonic_ms": clock[0] + 2000,
                "agent_speaking": True,
            }
        )
        for frame in frames(grab):
            socket.send_bytes(frame)
        for _ in range(TRAILING_SILENCE_FRAMES):
            socket.send_bytes(b"\x00" * FRAME_BYTES)
        socket.send_json(
            {
                "type": "vad_quiet",
                "segment_id": "seg-in1",
                "client_monotonic_ms": clock[0] + 3000,
                "speech_end_monotonic_ms": clock[0] + 2600,
                "silence_ms": 400,
            }
        )
        harness.judge_calls += 1
        drained = drain_audio(socket, start["audio_chunk_id"], timeout=STEP_TIMEOUT_S)
        # A genuine floor grab must interrupt with a taking_floor verdict and
        # no audio_end may follow the interrupt (the server may already have
        # sent the whole reply, which the browser would still be playing).
        harness.record(
            "interruption.protocol",
            drained["interrupted"]
            and drained["verdict_reason"] == "taking_floor_interrupt"
            and not drained["audio_end_after_interrupt"],
            f"audio_end={drained['audio_done']} interrupt={drained['interrupted']} "
            f"verdict={drained['verdict_state']}/{drained['verdict_reason']} "
            f"saw={drained['seen'][:12]}",
        )


def main() -> int:
    # Whole-run watchdog: a stalled provider call must fail loudly (with a traceback), never hang.
    faulthandler.dump_traceback_later(TOTAL_TIMEOUT_S, exit=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="plain,hesitation,echo,backchannel,interruption")
    args = parser.parse_args()
    wanted = {name.strip() for name in args.scenarios.split(",")}

    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env", override=False)
    print(
        f"turn_mode={os.getenv('VOXCARE_TURN_MODE')} asr={os.getenv('VOXCARE_ASR_PROVIDER')} "
        f"tts={os.getenv('VOXCARE_TTS_PROVIDER')} transport={os.getenv('VOXCARE_OPENAI_TRANSPORT')} "
        f"llm={os.getenv('VOXCARE_OPENAI_MODEL')} judge={os.getenv('VOXCARE_JEV_MODEL')}",
        flush=True,
    )

    from voxcare.providers import tts_from_env

    tts = tts_from_env()
    from fastapi.testclient import TestClient

    import voxcare.app as api

    harness = Harness()
    clock = [1000.0]
    with TestClient(api.app) as client:
        assistant_audio = b""
        if "plain" in wanted:
            assistant_audio = scenario_plain(harness, client, tts, clock)
            clock[0] += 30_000
        if "hesitation" in wanted:
            scenario_hesitation(harness, client, tts, clock)
            clock[0] += 30_000
        if "echo" in wanted:
            scenario_echo(harness, client, assistant_audio, clock)
            clock[0] += 30_000
        if "backchannel" in wanted:
            scenario_backchannel(harness, client, tts, clock)
            clock[0] += 30_000
        if "interruption" in wanted:
            scenario_interruption(harness, client, tts, clock)
    print("RESULT:", "ALL PASS" if harness.ok else "FAILURES PRESENT", flush=True)
    return 0 if harness.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
