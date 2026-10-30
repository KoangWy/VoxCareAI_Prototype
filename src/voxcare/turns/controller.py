"""Application-owned turn control around an optional text-only end-of-turn judge.

The controller never treats a timeout, error, or HOLD deadline as turn
completion. The host owns its timers and supplies events tagged from the
browser's local-VAD path; client messages are not authenticated.

Episode model (see docs/TURN_CONTROL.md): an episode opens at a local-VAD
onset, its absolute HOLD deadline is anchored at the first quiet+final
observation and re-armed on every non-commit verdict observed during mic
quiet (including a timed-out judgment). Only a new local-VAD onset resets
the two-non-commit budget; ASR partials, timers and provider callbacks
never do. The first of two non-commit verdicts or the deadline degrades the
episode to an explicit caller choice.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

FallbackAction = Literal["MANUAL_FINISH", "CALLER_RETRY", "CALLER_REQUESTED_HANDOFF"]


@dataclass(frozen=True)
class TurnInput:
    session_id: str
    turn_epoch: int
    transcript: str
    conversation_history: Sequence[Mapping[str, str]] = field(default_factory=tuple)
    assistant_last_message: str | None = None
    silence_ms: int | None = None
    asr_final: bool | None = None
    stable_word_ratio: float | None = None
    local_vad_quiet: bool = False
    now_ms: int | None = None
    source: str = "asr"
    turn_elapsed_ms: int | None = None
    transcript_trajectory: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    assistant_current_speech: str | None = None

    def timestamp_ms(self) -> int:
        return self.now_ms if self.now_ms is not None else int(time.monotonic() * 1000)


@dataclass(frozen=True)
class JevJudgment:
    turn_complete: float
    holding_floor: float
    addressed_to_agent: float = 1.0
    taking_floor: float | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class JevJudge(Protocol):
    def __call__(self, turn: TurnInput) -> Awaitable[JevJudgment]: ...


class BaselineTurnJudge:
    """Deterministic baseline: complete once ASR is final and the mic has been quiet >= 500 ms.

    Never presented as a Jev judgment; it answers instantly and has no
    notion of holding the floor or side-talk.
    """

    async def __call__(self, turn: TurnInput) -> JevJudgment:
        ready = turn.asr_final is True and turn.local_vad_quiet and (turn.silence_ms or 0) >= 500
        return JevJudgment(turn_complete=0.99 if ready else 0.0, holding_floor=0.0)


class JevConfigurationError(RuntimeError):
    """Jev is not configured; this is distinct from a provider outage."""


class JevProviderError(RuntimeError):
    """Jev could not return a judgment; details are intentionally not logged."""


@dataclass(frozen=True)
class TurnConfig:
    jev_timeout_ms: int = 150
    hold_deadline_ms: int = 3000
    max_noncommit_evaluations: int = 2
    turn_complete_threshold: float = 0.72
    holding_floor_threshold: float = 0.50
    addressed_to_agent_threshold: float = 0.50
    taking_floor_threshold: float = 0.50
    min_stable_word_ratio: float = 0.90
    eos_threshold_floor: float = 0.55
    eos_decay_ms: int = 3000
    silence_cap_ms: int = 5000
    silence_cap_holding_ms: int = 12000
    confirm_quiet_min_ms: int = 350
    confirm_quiet_max_ms: int = 1000

    def __post_init__(self) -> None:
        if self.jev_timeout_ms <= 0 or self.hold_deadline_ms <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_noncommit_evaluations != 2:
            raise ValueError("Spec #12 fixes recovery at two non-commit evaluations")
        for name in (
            "turn_complete_threshold",
            "holding_floor_threshold",
            "addressed_to_agent_threshold",
            "taking_floor_threshold",
            "min_stable_word_ratio",
            "eos_threshold_floor",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.eos_threshold_floor > self.turn_complete_threshold:
            raise ValueError("eos_threshold_floor must not exceed turn_complete_threshold")
        for name in (
            "eos_decay_ms",
            "silence_cap_ms",
            "silence_cap_holding_ms",
            "confirm_quiet_min_ms",
            "confirm_quiet_max_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.confirm_quiet_min_ms > self.confirm_quiet_max_ms:
            raise ValueError("confirm_quiet_min_ms must not exceed confirm_quiet_max_ms")
        if self.silence_cap_ms > self.silence_cap_holding_ms:
            raise ValueError("silence_cap_ms must not exceed silence_cap_holding_ms")


@dataclass(frozen=True)
class TurnDecision:
    state: Literal["COMMIT", "HOLD", "DEGRADED", "FALLBACK_RESOLVED"]
    reason: str
    turn_epoch: int
    hold_episode_id: str | None
    decision_request_id: str | None
    jeval_noncommit_count: int
    hold_elapsed_ms: int
    hold_deadline_ms: int | None
    audit_event: Mapping[str, Any] | None = None
    scores: Mapping[str, float] | None = None
    should_interrupt: bool = False


@dataclass
class _Episode:
    session_id: str
    turn_epoch: int
    hold_episode_id: str
    started_ms: int
    deadline_ms: int
    anchored: bool = False
    quiet_armed_ms: int | None = None
    noncommit_count: int = 0
    active_request_id: str | None = None
    latest_request_id: str | None = None
    latest_input: TurnInput | None = None
    degraded: bool = False


def effective_turn_complete_threshold(quiet_ms: int | None, config: TurnConfig, *, holding: bool = False) -> float:
    """Silence-relaxed `turn_complete` bar: full threshold while the caller asks
    for time in words, relaxing linearly toward the floor as mic quiet grows."""
    if holding:
        return config.turn_complete_threshold
    if not quiet_ms or quiet_ms <= 0 or config.eos_decay_ms <= 0:
        return config.turn_complete_threshold
    progress = min(1.0, quiet_ms / config.eos_decay_ms)
    relaxed = config.turn_complete_threshold - (config.turn_complete_threshold - config.eos_threshold_floor) * progress
    return round(relaxed, 3)


def confirm_quiet_ms_for(turn_complete: float, config: TurnConfig) -> int:
    """Mic quiet required after a complete verdict, scaled by confidence: a
    certain 1.0 needs the minimum window, an unsure score needs the maximum."""
    unsure = min(1.0, max(0.0, 1.0 - turn_complete))
    return round(config.confirm_quiet_min_ms + (config.confirm_quiet_max_ms - config.confirm_quiet_min_ms) * unsure)


class TurnController:
    """Stateful evaluator, instantiated per session (or shared safely by session id).

    `evaluate` sends only transcript/context/timing to the injected judge, and
    only when local VAD reports quiet. The caller must invoke
    `on_genuine_speech` exclusively from the designated local-VAD observation
    path; this tag is not a security guarantee. ASR partials and timer events
    cannot reset the recovery budget.
    """

    def __init__(self, judge: JevJudge, config: TurnConfig | None = None) -> None:
        self._judge = judge
        self.config = config or TurnConfig()
        self._episodes: dict[str, _Episode] = {}

    async def evaluate(self, turn: TurnInput) -> TurnDecision:
        now = turn.timestamp_ms()
        episode = self._get_or_start(turn, now)
        if episode.turn_epoch != turn.turn_epoch:
            return self._decision(episode, "HOLD", "stale_turn", now)
        episode.latest_input = turn
        if episode.degraded:
            return self._decision(episode, "DEGRADED", "awaiting_caller_choice", now)
        if now >= episode.deadline_ms:
            return self._degrade(episode, "hold_deadline", now)
        if not turn.transcript.strip():
            # silence with no words is a free false-open hold — it must
            # not consume the two-non-commit recovery budget. The onset
            # snapshot (latest_input) is kept as the false-open baseline.
            return self._free_hold(episode, "false_open_empty_transcript", now)
        if not turn.local_vad_quiet:
            return self._decision(episode, "HOLD", "caller_speaking", now)

        request_id = str(uuid.uuid4())
        episode.active_request_id = request_id
        episode.latest_request_id = request_id
        guard = (turn.session_id, turn.turn_epoch, episode.hold_episode_id, request_id)
        try:
            task = asyncio.create_task(self._judge(turn))
            done, _ = await asyncio.wait({task}, timeout=self.config.jev_timeout_ms / 1000)
            if not done:
                task.cancel()
                task.add_done_callback(_consume_task_result)
                if not self._is_current(episode, guard):
                    return self._decision(episode, "HOLD", "stale_result", now)
                return self._noncommit(episode, "jev_timeout", _now_from_turn(turn))
            judgment = task.result()
        except asyncio.CancelledError:
            if episode.active_request_id == request_id:
                episode.active_request_id = None
            task.cancel()
            task.add_done_callback(_consume_task_result)
            raise
        except JevConfigurationError:
            if not self._is_current(episode, guard):
                return self._decision(episode, "HOLD", "stale_result", now)
            return self._noncommit(episode, "jev_unconfigured", _now_from_turn(turn))
        except Exception:
            if not self._is_current(episode, guard):
                return self._decision(episode, "HOLD", "stale_result", now)
            return self._noncommit(episode, "jev_error", _now_from_turn(turn))

        completed_at = _now_from_turn(turn)
        if not self._is_current(episode, guard):
            return self._decision(episode, "HOLD", "stale_result", completed_at)
        if completed_at >= episode.deadline_ms:
            return self._degrade(episode, "hold_deadline", completed_at)

        scores = {
            "turn_complete": _clamp(judgment.turn_complete),
            "holding_floor": _clamp(judgment.holding_floor),
            "addressed_to_agent": _clamp(judgment.addressed_to_agent),
            "taking_floor": _clamp(judgment.taking_floor) if judgment.taking_floor is not None else 0.0,
        }
        quiet_ms = turn.silence_ms if turn.silence_ms and turn.silence_ms > 0 else 0
        holding = scores["holding_floor"] >= self.config.holding_floor_threshold
        self._anchor_deadline(episode, turn, completed_at)

        cap = self.config.silence_cap_holding_ms if holding else self.config.silence_cap_ms
        if quiet_ms >= cap:
            # Sustained mic silence is itself the evidence: close the turn
            # when nothing points away (side-talk or an explicit hold), else
            # degrade so the caller picks.
            if scores["addressed_to_agent"] >= self.config.addressed_to_agent_threshold and not holding:
                episode.active_request_id = None
                return self._decision(episode, "COMMIT", "quiet_cap_commit", completed_at, scores=scores)
            episode.active_request_id = None
            return self._noncommit(episode, "quiet_cap_unaddressed_or_holding", completed_at, scores)

        stable = turn.asr_final is True or (
            turn.stable_word_ratio is not None and turn.stable_word_ratio >= self.config.min_stable_word_ratio
        )
        if not stable:
            return self._noncommit(episode, "asr_not_stable", completed_at, scores)
        if not quiet_ms:
            return self._noncommit(episode, "missing_mic_silence", completed_at, scores)
        if holding and scores["turn_complete"] < self.config.turn_complete_threshold:
            # Patient mode: the caller is asking for time in words — keep the
            # full bar, no relaxation, and re-check after more quiet.
            return self._noncommit(episode, "caller_holding_floor", completed_at, scores)
        bar = effective_turn_complete_threshold(quiet_ms, self.config, holding=False)
        if scores["turn_complete"] < bar:
            return self._noncommit(episode, "turn_not_complete", completed_at, scores)
        if scores["addressed_to_agent"] < self.config.addressed_to_agent_threshold:
            # Complete but meant for someone else in the room — hold for the
            # caller, never dispatch a side-talk turn to the assistant.
            return self._noncommit(episode, "side_talk_hold", completed_at, scores)
        if (turn.assistant_current_speech or "").strip():
            # Words heard while the assistant is speaking are judged as a
            # potential interruption first (stage 2): a backchannel holds
            # with the audio untouched, a real floor grab holds and the host
            # interrupts. Neither ever COMMITs mid-playback.
            if self._should_interrupt(turn, scores):
                return self._noncommit(episode, "taking_floor_interrupt", completed_at, scores)
            return self._noncommit(episode, "backchannel_hold", completed_at, scores)
        need = confirm_quiet_ms_for(scores["turn_complete"], self.config)
        if quiet_ms < need:
            # Complete-looking and addressed, but the mic has not been quiet
            # long enough for this confidence: hold for a confirming re-judge
            # after more silence instead of committing on one sample.
            return self._noncommit(episode, "confirming", completed_at, scores)

        episode.active_request_id = None
        return self._decision(episode, "COMMIT", "jev_complete_and_vad_quiet", completed_at, scores=scores)

    def _should_interrupt(self, turn: TurnInput, scores: Mapping[str, float]) -> bool:
        """Stage 2 of semantic barge-in: only a deliberate floor grab while the
        assistant is speaking interrupts; backchannels never do."""
        assistant_speaking = bool((turn.assistant_current_speech or "").strip())
        return assistant_speaking and scores.get("taking_floor", 0.0) >= self.config.taking_floor_threshold

    async def on_genuine_speech(self, turn: TurnInput) -> TurnDecision:
        """Reset only from a local VAD speech onset, in the current episode.

        A genuine onset after degradation re-arms a fresh episode: the caller
        retrying by speaking is the clearest recovery signal there is, and the
        pending degraded audit stays PENDING exactly as on a caller-selected
        retry. Completion is still never inferred from the onset itself.
        """
        now = turn.timestamp_ms()
        episode = self._get_or_start(turn, now)
        if turn.source != "local_vad":
            return self._decision(episode, "HOLD", "ignored_untrusted_speech_reset", now)
        if episode.turn_epoch != turn.turn_epoch:
            return self._decision(episode, "HOLD", "stale_speech_reset", now)
        if episode.degraded:
            episode = _Episode(
                session_id=turn.session_id,
                turn_epoch=turn.turn_epoch,
                hold_episode_id=str(uuid.uuid4()),
                started_ms=now,
                deadline_ms=now + self.config.hold_deadline_ms,
                latest_request_id=str(uuid.uuid4()),
                latest_input=turn,
            )
            self._episodes[turn.session_id] = episode
            return self._decision(episode, "HOLD", "genuine_vad_speech", now)
        if now >= episode.deadline_ms:
            return self._degrade(episode, "hold_deadline", now)
        episode.noncommit_count = 0
        episode.started_ms = now
        episode.deadline_ms = now + self.config.hold_deadline_ms
        episode.anchored = False
        episode.quiet_armed_ms = None
        episode.active_request_id = None
        episode.latest_request_id = str(uuid.uuid4())
        episode.latest_input = turn
        return self._decision(episode, "HOLD", "genuine_vad_speech", now)

    def invalidate_pending(self, session_id: str) -> None:
        """Fence any in-flight judgment when a newer transcript version arrives."""
        episode = self._episodes.get(session_id)
        if episode is not None:
            episode.active_request_id = None

    def episode_size(self, session_id: str) -> int:
        """Number of tracked episodes for a session (must stay 0/1)."""
        return 1 if session_id in self._episodes else 0

    async def on_hold_deadline(
        self,
        session_id: str,
        turn_epoch: int,
        hold_episode_id: str,
        *,
        now_ms: int | None = None,
    ) -> TurnDecision:
        now = now_ms if now_ms is not None else int(time.monotonic() * 1000)
        episode = self._episodes.get(session_id)
        if episode is None or episode.turn_epoch != turn_epoch or episode.hold_episode_id != hold_episode_id:
            return TurnDecision("HOLD", "stale_deadline", turn_epoch, hold_episode_id, None, 0, 0, None)
        if episode.degraded:
            return self._decision(episode, "DEGRADED", "awaiting_caller_choice", now)
        if now < episode.deadline_ms:
            return self._decision(episode, "HOLD", "deadline_not_reached", now)
        return self._degrade(episode, "hold_deadline", now)

    async def resolve_fallback(
        self,
        session_id: str,
        turn_epoch: int,
        hold_episode_id: str,
        action: FallbackAction,
        *,
        now_ms: int | None = None,
    ) -> TurnDecision:
        now = now_ms if now_ms is not None else int(time.monotonic() * 1000)
        episode = self._episodes.get(session_id)
        if episode is None or episode.turn_epoch != turn_epoch or episode.hold_episode_id != hold_episode_id:
            return TurnDecision("HOLD", "stale_fallback", turn_epoch, hold_episode_id, None, 0, 0, None)
        if not episode.degraded:
            return self._decision(episode, "HOLD", "fallback_requires_degraded_state", now)
        if action not in ("MANUAL_FINISH", "CALLER_RETRY", "CALLER_REQUESTED_HANDOFF"):
            raise ValueError("unsupported caller-selected fallback")

        outcome = "PENDING" if action == "CALLER_RETRY" else "SUCCESS"
        event = _audit_event(
            session_id=session_id,
            epoch=turn_epoch,
            episode=episode,
            action_name="TURN_COMPLETION_FALLBACK_RESOLVED",
            outcome_status=outcome,
            payload={
                "resolution": action,
                "hold_episode_id": episode.hold_episode_id,
                "decision_request_id": episode.latest_request_id,
                "turn_epoch": turn_epoch,
            },
            now_ms=now,
        )
        if action == "CALLER_RETRY":
            next_episode = _Episode(
                session_id=session_id,
                turn_epoch=turn_epoch,
                hold_episode_id=str(uuid.uuid4()),
                started_ms=now,
                deadline_ms=now + self.config.hold_deadline_ms,
                latest_request_id=str(uuid.uuid4()),
                latest_input=episode.latest_input,
            )
            self._episodes[session_id] = next_episode
            return self._decision(next_episode, "HOLD", "caller_selected_retry", now, audit_event=event)

        episode.active_request_id = None
        return self._decision(episode, "FALLBACK_RESOLVED", action.lower(), now, audit_event=event)

    def _get_or_start(self, turn: TurnInput, now: int) -> _Episode:
        episode = self._episodes.get(turn.session_id)
        if episode is None or episode.turn_epoch != turn.turn_epoch:
            episode = _Episode(
                session_id=turn.session_id,
                turn_epoch=turn.turn_epoch,
                hold_episode_id=str(uuid.uuid4()),
                started_ms=now,
                deadline_ms=now + self.config.hold_deadline_ms,
                latest_request_id=str(uuid.uuid4()),
                latest_input=turn,
            )
            self._episodes[turn.session_id] = episode
        return episode

    def _anchor_deadline(self, episode: _Episode, turn: TurnInput, now: int) -> None:
        """Anchor the absolute deadline at the first quiet+final observation;
        the HOLD deadline runs from when the caller actually yielded, not from
        speech onset (whose ASR + model latency would otherwise eat it)."""
        if episode.anchored:
            return
        if turn.local_vad_quiet and (turn.asr_final is True or (turn.silence_ms or 0) > 0):
            episode.anchored = True
            episode.quiet_armed_ms = now
            episode.deadline_ms = now + self.config.hold_deadline_ms

    def _is_current(self, episode: _Episode, guard: tuple[str, int, str, str]) -> bool:
        session_id, epoch, episode_id, request_id = guard
        active = self._episodes.get(session_id)
        return (
            active is episode
            and active.turn_epoch == epoch
            and active.hold_episode_id == episode_id
            and active.active_request_id == request_id
        )

    def _free_hold(self, episode: _Episode, reason: str, now: int) -> TurnDecision:
        """A HOLD that consumes no recovery budget (empty-transcript false-open)."""
        episode.active_request_id = None
        return self._decision(episode, "HOLD", reason, now)

    def _noncommit(
        self,
        episode: _Episode,
        reason: str,
        now: int,
        scores: Mapping[str, float] | None = None,
    ) -> TurnDecision:
        episode.active_request_id = None
        episode.noncommit_count += 1
        if episode.noncommit_count >= self.config.max_noncommit_evaluations:
            return self._degrade(episode, "noncommit_limit", now, scores)
        # Re-arm the absolute deadline on each quiet HOLD verdict (reference
        # caps behaviour): a slow-but-healthy judgment restarts the clock
        # instead of inheriting an onset-anchored deadline it cannot meet.
        latest = episode.latest_input
        if latest is not None and latest.local_vad_quiet and (latest.silence_ms or 0) > 0:
            episode.deadline_ms = now + self.config.hold_deadline_ms
            episode.anchored = True
            episode.quiet_armed_ms = now
        if now >= episode.deadline_ms:
            return self._degrade(episode, "hold_deadline", now, scores)
        return self._decision(episode, "HOLD", reason, now, scores=scores)

    def _degrade(
        self,
        episode: _Episode,
        reason: str,
        now: int,
        scores: Mapping[str, float] | None = None,
    ) -> TurnDecision:
        if episode.degraded:
            return self._decision(episode, "DEGRADED", "awaiting_caller_choice", now, scores=scores)
        episode.degraded = True
        episode.active_request_id = None
        event = _audit_event(
            session_id=episode.session_id,
            epoch=episode.turn_epoch,
            episode=episode,
            action_name="TURN_COMPLETION_DEGRADED",
            outcome_status="PENDING",
            payload={
                "reason": reason,
                "hold_episode_id": episode.hold_episode_id,
                "decision_request_id": episode.latest_request_id,
                "jeval_noncommit_count": episode.noncommit_count,
                "hold_elapsed_ms": max(0, now - episode.started_ms),
                "turn_epoch": episode.turn_epoch,
            },
            now_ms=now,
        )
        return self._decision(episode, "DEGRADED", reason, now, audit_event=event, scores=scores)

    def _decision(
        self,
        episode: _Episode,
        state: Literal["COMMIT", "HOLD", "DEGRADED", "FALLBACK_RESOLVED"],
        reason: str,
        now: int,
        *,
        audit_event: Mapping[str, Any] | None = None,
        scores: Mapping[str, float] | None = None,
    ) -> TurnDecision:
        interrupt = False
        if scores is not None and episode.latest_input is not None:
            interrupt = self._should_interrupt(episode.latest_input, scores)
        return TurnDecision(
            state=state,
            reason=reason,
            turn_epoch=episode.turn_epoch,
            hold_episode_id=episode.hold_episode_id,
            decision_request_id=episode.latest_request_id,
            jeval_noncommit_count=episode.noncommit_count,
            hold_elapsed_ms=max(0, now - episode.started_ms),
            hold_deadline_ms=episode.deadline_ms,
            audit_event=audit_event,
            scores=scores,
            should_interrupt=interrupt,
        )


def _now_from_turn(turn: TurnInput) -> int:
    # For production callers omit now_ms to use monotonic time. A supplied
    # deterministic time remains fixed across an awaited judge for tests; the
    # deadline is also enforced independently by the host's deadline callback.
    return turn.timestamp_ms()


def _clamp(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        return


def _audit_event(
    *,
    session_id: str,
    epoch: int,
    episode: _Episode,
    action_name: str,
    outcome_status: str,
    payload: Mapping[str, Any],
    now_ms: int,
) -> Mapping[str, Any]:
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "audit_id": str(uuid.uuid4()),
        "session_id": session_id,
        "order_id": None,
        "mutation_idempotency_key": None,
        "actor": "System",
        "action_name": action_name,
        "action_status": "Success",
        "outcome_status": outcome_status,
        "request_payload_json": dict(payload),
        "response_payload_json": {"observed_at_monotonic_ms": now_ms},
        "timestamp": timestamp,
    }
