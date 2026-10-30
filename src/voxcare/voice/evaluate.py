"""Turn evaluation: judge the current quiet episode and route COMMIT / HOLD / DEGRADED.

Scheduling uses one evaluation slot per connection: at most one judgment is
in flight and at most one is pending on a timer. A transcript change replaces
an in-flight judgment (its result would be stale anyway); a silence-only
re-judge never cancels one. Triggers:

- local-VAD quiet: judge now when the transcript is empty (arms the episode),
  after a short debounce when ASR is already final, otherwise after
  ``ASR_FINAL_WAIT_MS`` so finality has a chance to land first;
- an ASR final while quiet: debounced re-judge on the new text;
- a HOLD verdict that more silence can change (``REJUDGE_REASONS``): one
  re-judge after more quiet, so an episode reaches COMMIT or recovery
  without waiting for the absolute deadline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..privacy import mask_pii
from ..turns import TurnInput, confirm_quiet_ms_for
from ..turns.prompts import trajectory_from_partials
from .dispatch import dispatch_turn, interrupt_for_taking_floor
from .state import VoiceState, monotonic_ms
from .turn_control import is_current, send_turn_control

REJUDGE_DEBOUNCE_MS = 150
ASR_FINAL_WAIT_MS = 700
REJUDGE_SILENCE_MS = 1000
# HOLD reasons that a later judgment with more mic silence can turn into a COMMIT.
REJUDGE_REASONS = frozenset(
    {"turn_not_complete", "confirming", "asr_not_stable", "missing_mic_silence", "jev_timeout", "jev_error"}
)

Evaluate = Callable[[], Awaitable[None]]


def request_evaluation(state: VoiceState, evaluate: Evaluate, *, delay_ms: int = 0, text_changed: bool = True) -> None:
    """Schedule a judgment in the connection's single evaluation slot (replaces any pending timer)."""
    cancel_pending_evaluation(state)
    if delay_ms <= 0:
        _start(state, evaluate, replace=text_changed)
        return

    async def later() -> None:
        await asyncio.sleep(delay_ms / 1000)
        if state.pending_evaluation is asyncio.current_task():
            state.pending_evaluation = None
        _start(state, evaluate, replace=text_changed)

    task = asyncio.create_task(later())
    state.pending_evaluation = task
    state.evaluation_tasks.add(task)
    task.add_done_callback(state.evaluation_tasks.discard)


def _start(state: VoiceState, evaluate: Evaluate, *, replace: bool) -> None:
    running = state.latest_evaluation_task
    if running is not None and not running.done():
        if not replace:
            return  # its verdict is still coming; silence alone does not restart it
        running.cancel()
    task = asyncio.create_task(evaluate())
    state.latest_evaluation_task = task
    state.evaluation_tasks.add(task)
    task.add_done_callback(state.evaluation_tasks.discard)


def cancel_pending_evaluation(state: VoiceState) -> None:
    if state.pending_evaluation is not None and state.pending_evaluation is not asyncio.current_task():
        state.pending_evaluation.cancel()
    state.pending_evaluation = None


def cancel_evaluations(state: VoiceState) -> None:
    """Drop the pending timer and the in-flight judgment (new speech, reset, retry)."""
    cancel_pending_evaluation(state)
    running = state.latest_evaluation_task
    if running is not None and not running.done() and running is not asyncio.current_task():
        running.cancel()


def schedule_after_quiet(state: VoiceState, evaluate: Evaluate) -> None:
    if not state.transcript.strip() and not state.barge_in_held:
        request_evaluation(state, evaluate)
    elif state.asr_final:
        request_evaluation(state, evaluate, delay_ms=REJUDGE_DEBOUNCE_MS)
    else:
        request_evaluation(state, evaluate, delay_ms=ASR_FINAL_WAIT_MS)


async def evaluate_current_turn(boundary, send_json, send_binary, session_id: str, state: VoiceState) -> None:  # type: ignore[no-untyped-def]
    if not state.vad_quiet:
        return
    current = await boundary.get_session(session_id)
    controller = await boundary.get_turn_controller(session_id)
    if current is None or controller is None:
        return
    version = state.transcript_version
    generation = state.response_generation
    turn_id = state.current_turn_id
    now_ms = monotonic_ms()
    history = await boundary.get_conversation_history(session_id)
    safe_history = tuple(
        {"role": item.get("role", "unknown"), "text": mask_pii(str(item.get("content", "")))}
        for item in history
        if item.get("role") in {"user", "assistant"}
    )
    assistant_last_message = next(
        (item["text"] for item in reversed(safe_history) if item["role"] == "assistant"), None
    )
    opened = state.turn_opened_monotonic_ms
    silence = state.silence_ms(now_ms)
    turn = TurnInput(
        session_id=session_id,
        turn_epoch=current.turn_epoch + 1,
        transcript=state.transcript,
        conversation_history=safe_history,
        assistant_last_message=assistant_last_message,
        silence_ms=silence,
        asr_final=state.asr_final,
        stable_word_ratio=state.transcript_stable_ratio,
        local_vad_quiet=state.vad_quiet,
        source="asr",
        turn_elapsed_ms=now_ms - opened if opened is not None else None,
        transcript_trajectory=tuple(
            trajectory_from_partials(state.partial_snapshots, now_ms=now_ms, current_text=state.transcript)
        ),
        # Only a held barge-in is judged against the assistant's speech.
        assistant_current_speech=state.assistant_speech_text if state.barge_in_held else None,
    )
    decision = await controller.evaluate(turn)
    if not await is_current(boundary, session_id, state, generation, version):
        await boundary.record_observation(
            session_id,
            "stale_turn_decision_discarded",
            turn_id=turn_id,
            turn_epoch=decision.turn_epoch,
            transcript_version=version,
        )
        return
    if decision.state == "COMMIT":
        # Freeze the utterance before any further await: late ASR text must
        # neither change it nor cancel this task before the turn is dispatched.
        state.turn_completed = True
        cancel_pending_evaluation(state)
    await send_turn_control(boundary, send_json, session_id, state, decision, version=version, generation=generation)

    async def evaluate_again() -> None:
        await evaluate_current_turn(boundary, send_json, send_binary, session_id, state)

    if decision.state == "COMMIT":
        if generation != state.response_generation:
            return  # new speech retired this turn while the verdict was being sent
        task = asyncio.create_task(
            dispatch_turn(
                boundary, send_json, send_binary, session_id, state, turn.transcript, version, generation, turn_id
            )
        )
        state.response_tasks.add(task)
        task.add_done_callback(state.response_tasks.discard)
    elif decision.state == "HOLD" and decision.should_interrupt and state.barge_in_held:
        # Semantic barge-in: a taking_floor verdict stops the reply; the words
        # then get one normal judgment as the caller's new turn.
        await interrupt_for_taking_floor(boundary, send_json, session_id, state, decision)
        request_evaluation(state, evaluate_again, delay_ms=REJUDGE_DEBOUNCE_MS, text_changed=False)
    elif decision.state == "HOLD" and decision.reason == "backchannel_hold":
        # "uh-huh" while the assistant speaks is not a turn: drop it so the
        # held episode cannot later degrade into a recovery prompt.
        await _discard_backchannel(boundary, session_id, state, decision)
    elif decision.state == "HOLD" and decision.reason in REJUDGE_REASONS and state.vad_quiet:
        wait = REJUDGE_SILENCE_MS
        if decision.reason == "confirming" and decision.scores and silence is not None:
            need = confirm_quiet_ms_for(decision.scores.get("turn_complete", 0.0), controller.config)
            wait = max(REJUDGE_DEBOUNCE_MS, need - silence)
        request_evaluation(state, evaluate_again, delay_ms=wait, text_changed=False)


async def _discard_backchannel(boundary, session_id: str, state: VoiceState, decision) -> None:  # type: ignore[no-untyped-def]
    from .turn_control import cancel_deadline

    if decision.hold_episode_id:
        cancel_deadline(state, decision.hold_episode_id)
    await boundary.record_observation(
        session_id, "backchannel_discarded", turn_id=state.current_turn_id, transcript_version=state.transcript_version
    )
    state.barge_in_held = False
    state.reset_turn_buffers(new_turn_id=True)
    state.turn_completed = True  # ignore the rest of this backchannel until the next onset
    await boundary.set_live_transcript(session_id, "")


__all__ = [
    "ASR_FINAL_WAIT_MS",
    "REJUDGE_DEBOUNCE_MS",
    "REJUDGE_REASONS",
    "REJUDGE_SILENCE_MS",
    "cancel_evaluations",
    "cancel_pending_evaluation",
    "evaluate_current_turn",
    "request_evaluation",
    "schedule_after_quiet",
]
