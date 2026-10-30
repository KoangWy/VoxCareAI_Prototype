"""Response dispatch: a committed voice turn becomes a ``turn`` result and streamed TTS audio.

Every binary TTS batch is preceded by an ``audio_chunk`` JSON envelope with
the turn/chunk identity (protocol 2); protocol-1 clients ignore the envelope
and play every frame. The stream is fenced by ``audio_generation`` so only a
newer reply, an off-mode onset, an interrupt, or a reset retires it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict

from . import protocol as vm
from .state import VoiceState, monotonic_ms
from .turn_control import is_current

logger = logging.getLogger("voxcare.voice")

# Host bound for one voice turn's dialogue call (plan + tools). The LLM SDKs
# carry a 12 s read timeout, but a stalled gateway stream has outlived it; on
# timeout the client gets `provider_unavailable` / `CONVERSATION_TIMEOUT`.
_DISPATCH_TURN_TIMEOUT_S = 30.0
MAX_SPOKEN_CHARS = 2_000


def _turn_timing(result, state: VoiceState, turn_id: str) -> dict:  # type: ignore[no-untyped-def]
    speaking = bool(result.assistant_text and state.tts)
    return {
        **dict(result.timing),
        "T0": state.t0_by_turn.get(turn_id),
        "T1": state.t1_by_turn.get(turn_id, {"status": "unavailable", "reason": "asr_final_not_observed"}),
        "T2": state.t2_by_turn.get(turn_id, {"status": "unavailable", "reason": "turn_decision_not_observed"}),
        "T3": {"status": "unavailable", "reason": "provider_first_token_not_instrumented"},
        "T4": {
            "status": "pending" if speaking else "unavailable",
            "reason": "awaiting_text_to_tts" if speaking else "tts_unavailable",
        },
        "T5": {
            "status": "pending" if speaking else "unavailable",
            "reason": "awaiting_tts_first_chunk" if speaking else "tts_unavailable",
        },
        "T6": {"status": "unavailable", "reason": "awaiting_browser_playback_ack"},
        "ttfa_ms": None,
        "ttfa_status": "unavailable_until_client_T0_T6",
    }


async def dispatch_turn(  # type: ignore[no-untyped-def]
    boundary,
    send_json,
    send_binary,
    session_id: str,
    state: VoiceState,
    turn_text: str,
    version: int,
    generation: int,
    turn_id: str,
) -> None:
    try:
        result = await asyncio.wait_for(
            boundary.handle_turn(session_id, turn_text, source="asr_final", correlation_id=turn_id),
            timeout=_DISPATCH_TURN_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning("voice turn dispatch timed out", extra={"turn_id": turn_id})
        await boundary.record_observation(session_id, "conversation_timeout", turn_id=turn_id)
        await send_json(
            {
                "type": vm.SERVER_PROVIDER_UNAVAILABLE,
                "provider": "conversation",
                "error": "CONVERSATION_TIMEOUT",
                "turn_id": turn_id,
            }
        )
        return
    if not await is_current(boundary, session_id, state, generation, version) or result.outcome == "STALE_DISCARDED":
        await boundary.record_observation(
            session_id,
            "stale_turn_discarded",
            turn_id=turn_id,
            turn_epoch=result.turn_epoch,
            transcript_version=version,
        )
        return
    payload = asdict(result)
    payload["timing"] = _turn_timing(result, state, turn_id)
    await send_json({"type": vm.SERVER_TURN, "result": payload})
    if not result.assistant_text:
        return
    if state.tts is None:
        await send_json(
            {
                "type": vm.SERVER_PROVIDER_UNAVAILABLE,
                "provider": "tts",
                "error": "TTS_NOT_CONFIGURED",
                "turn_id": result.turn_id,
            }
        )
        await boundary.record_observation(session_id, "tts_unavailable", turn_id=result.turn_id)
        return
    await _speak(boundary, send_json, send_binary, session_id, state, result.turn_id, result.assistant_text)


async def _speak(boundary, send_json, send_binary, session_id: str, state: VoiceState, turn_id: str, text: str) -> None:  # type: ignore[no-untyped-def]
    # Take over the audio line: this reply retires any older stream.
    state.retire_audio()
    audio_generation = state.audio_generation
    audio_chunk_id = str(uuid.uuid4())
    state.active_audio_turn_id = turn_id
    state.active_audio_chunk_id = audio_chunk_id
    state.assistant_speaking = True
    state.assistant_speech_text = text[:MAX_SPOKEN_CHARS]

    async def still_ours() -> bool:
        return audio_generation == state.audio_generation and bool(await boundary.get_session(session_id))

    try:
        await send_json(
            {"type": vm.SERVER_AUDIO_START, "sample_rate": 16_000, "turn_id": turn_id, "audio_chunk_id": audio_chunk_id}
        )
        await boundary.record_observation(
            session_id,
            "text_sent_to_tts",
            turn_id=turn_id,
            audio_chunk_id=audio_chunk_id,
            T4_server_monotonic_ms=monotonic_ms(),
        )
        chunk_index = 0
        async for chunk in state.tts.stream(text):
            if not await still_ours():
                return
            if not chunk:
                continue
            if chunk_index == 0:
                await boundary.record_observation(
                    session_id,
                    "tts_first_chunk",
                    turn_id=turn_id,
                    audio_chunk_id=audio_chunk_id,
                    T5_server_monotonic_ms=monotonic_ms(),
                )
            await send_json(
                {
                    "type": vm.SERVER_AUDIO_CHUNK,
                    "turn_id": turn_id,
                    "audio_chunk_id": audio_chunk_id,
                    "chunk_index": chunk_index,
                    "byte_length": len(chunk),
                    "protocol": vm.PROTOCOL_VERSION,
                }
            )
            chunk_index += 1
            await send_binary(chunk)
        if not await still_ours():
            return
        await send_json({"type": vm.SERVER_AUDIO_END, "turn_id": turn_id, "audio_chunk_id": audio_chunk_id})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("TTS stream failed", extra={"error_type": type(exc).__name__})
        if audio_generation == state.audio_generation:
            await send_json(
                {
                    "type": vm.SERVER_PROVIDER_UNAVAILABLE,
                    "provider": "tts",
                    "error": "SPEECH_OUTPUT_UNAVAILABLE",
                    "turn_id": turn_id,
                }
            )
        await boundary.record_observation(session_id, "tts_failed", turn_id=turn_id, error_type=type(exc).__name__)
    finally:
        if audio_generation == state.audio_generation:
            # The server finished sending (or failed). The browser may still be
            # playing buffered audio, so the reply text stays available for a
            # semantic barge-in judgment until the next onset decides.
            state.assistant_speaking = False


async def interrupt_for_taking_floor(boundary, send_json, session_id: str, state: VoiceState, decision) -> None:  # type: ignore[no-untyped-def]
    """Semantic barge-in: a taking_floor verdict stops playback now; the mic stays open."""
    await send_json(
        {
            "type": vm.SERVER_INTERRUPT,
            "session_id": session_id,
            "turn_id": state.current_turn_id,
            "turn_epoch": decision.turn_epoch,
            "hold_episode_id": decision.hold_episode_id,
            "reason": "taking_floor",
            "taking_floor": (decision.scores or {}).get("taking_floor"),
        }
    )
    await boundary.record_observation(
        session_id, "barge_in_interrupted", turn_id=state.current_turn_id, turn_epoch=decision.turn_epoch
    )
    for task in tuple(state.response_tasks):
        task.cancel()
    state.retire_audio()


__all__ = ["dispatch_turn", "interrupt_for_taking_floor"]
