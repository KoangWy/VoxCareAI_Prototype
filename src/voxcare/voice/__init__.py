"""WebSocket voice flow: per-connection state, ASR fan-in, turn evaluation, and TTS dispatch.

- :mod:`.protocol` — every wire message name and error code (``docs/VOICE_PROTOCOL.md``).
- :mod:`.connection` — the socket loop and client-message handlers.
- :mod:`.asr_events` — provider transcripts into the versioned active utterance.
- :mod:`.evaluate` — the single evaluation slot and verdict routing.
- :mod:`.turn_control` — ``turn_control`` emission, HOLD deadlines, recovery offers.
- :mod:`.dispatch` — committed turn into ``turn`` + streamed audio; barge-in interrupt.
- :mod:`.state` — :class:`VoiceState`.
"""

from __future__ import annotations

from .connection import MAX_AUDIO_FRAME_BYTES, MAX_SESSION_AUDIO_BYTES, VoiceConnection, register, voice_socket
from .state import VoiceState

__all__ = [
    "MAX_AUDIO_FRAME_BYTES",
    "MAX_SESSION_AUDIO_BYTES",
    "VoiceConnection",
    "VoiceState",
    "register",
    "voice_socket",
]
