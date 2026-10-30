"""Turn-control emission for the voice socket: fencing, ``turn_control`` messages, deadlines, recovery.

The controller owns the absolute HOLD deadline; this module owns the host
timer that asks the controller about it. The timer follows the deadline as
the controller re-arms it, so a HOLD episode always ends in a verdict or a
caller action — never a silent wait.
"""

from __future__ import annotations

import asyncio
import time

from . import protocol as vm
from .state import VoiceState, monotonic_ms

RECOVERY_CHOICES = ["manual_finish", "caller_retry", "handoff_request"]


async def is_current(boundary, session_id: str, state: VoiceState, generation: int, version: int | None = None) -> bool:  # type: ignore[no-untyped-def]
    if generation != state.response_generation or not await boundary.get_session(session_id):
        return False
    return version is None or version == state.transcript_version


def pending_timing() -> dict[str, dict[str, str]]:
    """T3-T6 slots before the reply exists (see docs/VOICE_PROTOCOL.md section 4)."""
    return {
        "T3": {"status": "unavailable", "reason": "provider_first_token_not_instrumented"},
        "T4": {"status": "unavailable", "reason": "provider_text_emit_not_instrumented"},
        "T5": {"status": "pending", "reason": "awaiting_tts_first_chunk"},
        "T6": {"status": "unavailable", "reason": "awaiting_browser_playback_ack"},
    }


async def send_turn_control(  # type: ignore[no-untyped-def]
    boundary,
    send_json,
    session_id: str,
    state: VoiceState,
    decision,
    *,
    version: int,
    generation: int,
) -> None:
    if not await is_current(boundary, session_id, state, generation, version):
        return
    turn_id = state.current_turn_id
    state.t2_by_turn[turn_id] = {"status": "available", "clock": "server_monotonic", "ms": monotonic_ms()}
    await send_json(
        {
            "type": vm.SERVER_TURN_CONTROL,
            "session_id": session_id,
            "turn_id": turn_id,
            "turn_epoch": decision.turn_epoch,
            "transcript_version": version,
            "decision_request_id": decision.decision_request_id,
            "hold_episode_id": decision.hold_episode_id,
            "state": decision.state,
            "reason": decision.reason,
            "jeval_noncommit_count": decision.jeval_noncommit_count,
            "hold_elapsed_ms": decision.hold_elapsed_ms,
            "hold_deadline_ms": decision.hold_deadline_ms,
            "should_interrupt": decision.should_interrupt,
            "scores": dict(decision.scores) if decision.scores else None,
            "turn_mode": state.turn_mode,
            "timing": {
                "T0": state.t0_by_turn.get(turn_id),
                "T2": state.t2_by_turn[turn_id],
                **pending_timing(),
                "ttfa_ms": None,
                "ttfa_status": "unavailable_until_client_T0_T6",
            },
        }
    )
    if decision.audit_event:
        await boundary.record_audit_event(session_id, decision.audit_event)
    episode_id = decision.hold_episode_id
    if decision.state == "HOLD" and decision.hold_deadline_ms is not None and episode_id:
        if episode_id not in state.deadline_tasks:
            state.deadline_tasks[episode_id] = asyncio.create_task(
                _watch_deadline(boundary, send_json, session_id, state, decision, generation)
            )
    elif episode_id:
        cancel_deadline(state, episode_id)
    if decision.state == "DEGRADED":
        state.pending_recovery = True
        await maybe_offer_recovery(send_json, session_id, state, decision, generation)


async def _watch_deadline(boundary, send_json, session_id: str, state: VoiceState, decision, generation: int) -> None:  # type: ignore[no-untyped-def]
    episode_id = decision.hold_episode_id
    deadline_ms = decision.hold_deadline_ms
    current = asyncio.current_task()
    try:
        while True:
            await asyncio.sleep(max(0.0, deadline_ms / 1000 - time.monotonic()))
            controller = await boundary.get_turn_controller(session_id)
            if controller is None or generation != state.response_generation:
                return
            result = await controller.on_hold_deadline(session_id, decision.turn_epoch, episode_id)
            if result.state == "HOLD" and result.reason == "deadline_not_reached" and result.hold_deadline_ms:
                deadline_ms = result.hold_deadline_ms  # re-armed by a later HOLD verdict or onset
                continue
            if result.state == "DEGRADED" and result.reason == "hold_deadline":
                if state.deadline_tasks.get(episode_id) is current:
                    del state.deadline_tasks[episode_id]
                await send_turn_control(
                    boundary,
                    send_json,
                    session_id,
                    state,
                    result,
                    version=state.transcript_version,
                    generation=generation,
                )
            return
    finally:
        if state.deadline_tasks.get(episode_id) is current:
            del state.deadline_tasks[episode_id]


def cancel_deadline(state: VoiceState, episode_id: str) -> None:
    task = state.deadline_tasks.pop(episode_id, None)
    if task is not None and task is not asyncio.current_task():
        task.cancel()


def cancel_all_deadlines(state: VoiceState) -> None:
    for episode_id in tuple(state.deadline_tasks):
        cancel_deadline(state, episode_id)


async def maybe_offer_recovery(send_json, session_id: str, state: VoiceState, decision, generation: int) -> None:  # type: ignore[no-untyped-def]
    """Offer the explicit recovery choices once the caller is quiet (never while they talk)."""
    if state.pending_recovery and state.vad_quiet and generation == state.response_generation:
        state.pending_recovery = False
        await send_json(
            {
                "type": vm.SERVER_RECOVERY_REQUIRED,
                "session_id": session_id,
                "turn_id": state.current_turn_id,
                "turn_epoch": decision.turn_epoch,
                "transcript_version": state.transcript_version,
                "hold_episode_id": decision.hold_episode_id,
                "choices": list(RECOVERY_CHOICES),
            }
        )


__all__ = [
    "RECOVERY_CHOICES",
    "cancel_all_deadlines",
    "cancel_deadline",
    "is_current",
    "maybe_offer_recovery",
    "pending_timing",
    "send_turn_control",
]
