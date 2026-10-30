"""Per-connection voice state: one mutable instance per WebSocket.

Three independent fences guard late work:

- ``response_generation`` — bumped when a new caller onset (or reset/retry)
  retires everything in flight for the previous turn;
- ``transcript_version`` — bumped on every transcript change and onset, so a
  verdict computed for older text is never emitted;
- ``audio_generation`` — bumped only when assistant audio is deliberately
  retired (newer reply, off-mode onset, interrupt, reset), so a semantic
  barge-in hold can bump the version while the reply keeps playing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_TRANSCRIPT_CHARS = 2_000
MAX_FINALIZED_SEGMENTS = 80
MAX_PARTIAL_SNAPSHOTS = 8


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass
class VoiceState:
    # Settings resolved once per connection.
    turn_mode: str = "baseline"
    barge_in: str = "off"
    # Providers.
    asr: Any = None
    tts: Any = None
    asr_task: asyncio.Task | None = None
    audio_bytes: int = 0
    # Fences.
    response_generation: int = 0
    transcript_version: int = 0
    audio_generation: int = 0
    # Active caller turn.
    current_turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transcript: str = ""
    finalized_segments: list[dict[str, Any]] = field(default_factory=list)
    interim_segment: str = ""
    interim_segment_id: float | None = None
    seen_segment_ids: set[float] = field(default_factory=set)
    transcript_stable_ratio: float | None = None
    partial_snapshots: list[dict[str, Any]] = field(default_factory=list)
    turn_opened_monotonic_ms: int | None = None
    turn_completed: bool = False
    asr_final: bool = False
    # Local-VAD observation: silence reported by the browser at quiet, plus
    # the server time it arrived, so later judgments see the silence grow.
    vad_quiet: bool = False
    silence_at_quiet_ms: int | None = None
    quiet_received_ms: int | None = None
    pending_recovery: bool = False
    # Assistant output.
    active_audio_turn_id: str | None = None
    active_audio_chunk_id: str | None = None
    assistant_speaking: bool = False
    assistant_speech_text: str | None = None
    barge_in_held: bool = False
    # Background work.
    response_tasks: set[asyncio.Task] = field(default_factory=set)
    evaluation_tasks: set[asyncio.Task] = field(default_factory=set)
    latest_evaluation_task: asyncio.Task | None = None
    pending_evaluation: asyncio.Task | None = None
    deadline_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    # Timing observations (T0-T2) keyed by turn id, plus the client clock.
    client_clock: dict[str, Any] = field(default_factory=dict)
    t0_by_turn: dict[str, dict[str, Any]] = field(default_factory=dict)
    t1_by_turn: dict[str, dict[str, Any]] = field(default_factory=dict)
    t2_by_turn: dict[str, dict[str, Any]] = field(default_factory=dict)
    turn_controller: Any = None

    def silence_ms(self, now_ms: int | None = None) -> int | None:
        """Mic silence now: the browser's figure at quiet plus server time elapsed since."""
        if not self.vad_quiet or self.silence_at_quiet_ms is None:
            return None
        elapsed = 0 if self.quiet_received_ms is None else max(0, (now_ms or monotonic_ms()) - self.quiet_received_ms)
        return self.silence_at_quiet_ms + elapsed

    def mark_quiet(self, silence_ms: int | None) -> None:
        self.vad_quiet = True
        self.silence_at_quiet_ms = silence_ms
        self.quiet_received_ms = monotonic_ms()

    def mark_speaking(self) -> None:
        self.vad_quiet = False
        self.silence_at_quiet_ms = None
        self.quiet_received_ms = None

    def reset_turn_buffers(self, *, new_turn_id: bool) -> None:
        self.finalized_segments.clear()
        self.seen_segment_ids.clear()
        self.interim_segment = ""
        self.interim_segment_id = None
        self.transcript = ""
        self.transcript_stable_ratio = None
        self.partial_snapshots.clear()
        self.turn_opened_monotonic_ms = monotonic_ms()
        self.turn_completed = False
        self.asr_final = False
        if new_turn_id:
            self.current_turn_id = str(uuid.uuid4())

    def retire_audio(self) -> None:
        """Invalidate the assistant reply currently streaming (if any)."""
        self.audio_generation += 1
        self.active_audio_turn_id = None
        self.active_audio_chunk_id = None
        self.assistant_speaking = False
        self.assistant_speech_text = None
        self.barge_in_held = False

    def apply_transcript(self, text: str, *, is_final: bool, segment_id: float | None) -> None:
        """Fold one ASR update into the bounded active transcript.

        Finalized segments accumulate (deduplicated by segment identity); the
        current interim segment is replaced as it changes. A new interim with a
        different identity promotes the previous interim to finalized.
        """
        promote = self.interim_segment and self.interim_segment_id is not None
        if is_final:
            if promote and self.interim_segment_id != segment_id:
                self._retain_segment(self.interim_segment, self.interim_segment_id)
            self._retain_segment(text, segment_id)
            self.interim_segment = ""
            self.interim_segment_id = None
            self.transcript_stable_ratio = 1.0
        else:
            if promote and segment_id is not None and self.interim_segment_id != segment_id:
                self._retain_segment(self.interim_segment, self.interim_segment_id)
            self.interim_segment = text
            self.interim_segment_id = segment_id
            self.transcript_stable_ratio = None
        parts = [segment["text"] for segment in self.finalized_segments]
        if self.interim_segment:
            parts.append(self.interim_segment)
        self.transcript = " ".join(parts)[-MAX_TRANSCRIPT_CHARS:]
        self.transcript_version += 1
        self.partial_snapshots.append({"text": self.transcript, "at": monotonic_ms()})
        del self.partial_snapshots[:-MAX_PARTIAL_SNAPSHOTS]

    def _retain_segment(self, text: str, segment_id: float | None) -> None:
        if not text or (segment_id is not None and segment_id in self.seen_segment_ids):
            return
        self.finalized_segments.append({"text": text, "identity": segment_id})
        if segment_id is not None:
            self.seen_segment_ids.add(segment_id)
        while len(self.finalized_segments) > MAX_FINALIZED_SEGMENTS or (
            self.finalized_segments
            and sum(len(item["text"]) + 1 for item in self.finalized_segments) > MAX_TRANSCRIPT_CHARS
        ):
            removed = self.finalized_segments.pop(0)
            if removed["identity"] is not None:
                self.seen_segment_ids.discard(removed["identity"])


__all__ = ["VoiceState", "monotonic_ms"]
