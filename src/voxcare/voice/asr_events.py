"""ASR event fan-in: normalized provider events become versioned turn state.

Consumes the event dicts both ASR adapters emit, folds transcripts into the
bounded active utterance, broadcasts ``transcript_partial`` /
``transcript_final``, and re-judges (debounced) when finalized text changes
while the caller is quiet. Provider ``SpeechStarted`` / ``UtteranceEnd`` are
diagnostic only — the browser's local VAD is the turn authority.
"""

from __future__ import annotations

from . import protocol as vm
from .evaluate import REJUDGE_DEBOUNCE_MS, Evaluate, cancel_evaluations, request_evaluation
from .state import MAX_TRANSCRIPT_CHARS, VoiceState, monotonic_ms


async def handle_asr_events(
    boundary, send_json, session_id: str, state: VoiceState, connection, evaluate: Evaluate
) -> None:  # type: ignore[no-untyped-def]
    async for event in connection.events():
        kind = event.get("type")
        if not await boundary.get_session(session_id):
            return
        if kind == "transcript":
            await _on_transcript(boundary, send_json, session_id, state, event, evaluate)
        elif kind == "provider_error":
            await send_json({"type": vm.SERVER_PROVIDER_UNAVAILABLE, "provider": "asr", "error": "ASR_STREAM_FAILED"})
            await boundary.record_observation(session_id, "asr_failed", error_type=event.get("error_type"))
        elif kind == "SpeechStarted":
            await boundary.record_observation(session_id, "provider_speech_started")
        elif kind == "UtteranceEnd":
            await boundary.record_observation(session_id, "provider_utterance_end")


async def _on_transcript(boundary, send_json, session_id: str, state: VoiceState, event, evaluate: Evaluate) -> None:  # type: ignore[no-untyped-def]
    if state.turn_completed:
        await boundary.record_observation(
            session_id,
            "asr_transcript_after_commit_discarded",
            turn_id=state.current_turn_id,
            sequence=event.get("sequence"),
        )
        return
    text = str(event.get("text", ""))[:MAX_TRANSCRIPT_CHARS].strip()
    is_final_segment = bool(event.get("is_final"))
    start = event.get("start")
    segment_id = float(start) if isinstance(start, (int, float)) and not isinstance(start, bool) else None
    state.apply_transcript(text, is_final=is_final_segment, segment_id=segment_id)
    state.asr_final = bool(event.get("speech_final"))
    # New text makes any pending or in-flight verdict stale.
    cancel_evaluations(state)
    if state.turn_controller is not None:
        state.turn_controller.invalidate_pending(session_id)
    await boundary.set_live_transcript(session_id, state.transcript)
    if state.asr_final:
        state.t1_by_turn[state.current_turn_id] = {
            "status": "available",
            "clock": "server_monotonic",
            "ms": monotonic_ms(),
        }
    await boundary.record_observation(
        session_id,
        "asr_transcript",
        turn_id=state.current_turn_id,
        transcript_version=state.transcript_version,
        sequence=event.get("sequence"),
        is_final=is_final_segment,
        speech_final=state.asr_final,
        character_count=len(state.transcript),
    )
    await send_json(
        {
            "type": vm.SERVER_TRANSCRIPT_FINAL if state.asr_final else vm.SERVER_TRANSCRIPT_PARTIAL,
            "session_id": session_id,
            "turn_id": state.current_turn_id,
            "transcript_version": state.transcript_version,
            "sequence": event.get("sequence"),
            "text": state.transcript,
            "is_final": is_final_segment,
            "speech_final": state.asr_final,
        }
    )
    if state.vad_quiet and (state.asr_final or is_final_segment):
        request_evaluation(state, evaluate, delay_ms=REJUDGE_DEBOUNCE_MS)


__all__ = ["handle_asr_events"]
