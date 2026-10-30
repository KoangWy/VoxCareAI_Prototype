"""Streaming TTS adapters: ElevenLabs Flash v2.5 and offline Piper.

Both stream PCM16 mono 16 kHz chunks — the playout contract the browser
expects — after applying the shared deterministic PII mask.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
from array import array
from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import websockets

from ..config import ConfigError, load_provider_settings
from ..privacy import mask_pii
from .base import ProviderUnavailable

logger = logging.getLogger("voxcare.providers")

MAX_TTS_CHARS = 4_000
OUTPUT_SAMPLE_RATE = 16_000


def _settings():  # type: ignore[no-untyped-def]
    try:
        return load_provider_settings()
    except ConfigError as exc:
        raise ProviderUnavailable(str(exc), code="TTS_PROVIDER_UNSUPPORTED") from None


class ElevenLabsTTS:
    """Streaming PCM16 TTS adapter; output bytes are forwarded to browser playout."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        connector: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self._connector = connector or websockets.connect

    @classmethod
    def from_env(cls) -> ElevenLabsTTS:
        key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise ProviderUnavailable("ElevenLabs API key is not configured", code="ELEVENLABS_NOT_CONFIGURED")
        voice = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
        if not voice:
            raise ProviderUnavailable("ElevenLabs voice ID is not configured", code="ELEVENLABS_VOICE_NOT_CONFIGURED")
        return cls(api_key=key, voice_id=voice, model=_settings().elevenlabs_model)

    async def warm(self) -> None:
        """Nothing to preload for a hosted voice."""

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        text = mask_pii(text).strip()
        if not text or len(text) > MAX_TTS_CHARS:
            raise ValueError("TTS text length is invalid")
        query = urlencode({"model_id": self.model, "output_format": "pcm_16000", "auto_mode": "true"})
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?{query}"
        async with self._connector(
            uri, additional_headers={"xi-api-key": self.api_key}, max_size=1_048_576, open_timeout=8
        ) as socket:
            await socket.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}))
            await socket.send(json.dumps({"text": text, "try_trigger_generation": True}))
            await socket.send(json.dumps({"text": "", "flush": True}))
            async for raw in socket:
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                encoded = message.get("audio")
                if encoded:
                    yield base64.b64decode(encoded, validate=True)
                if message.get("isFinal"):
                    break


def split_sentences(text: str, *, limit: int = 300) -> list[str]:
    """Sentence-sized synthesis units; over-long sentences wrap at a word boundary."""
    parts = [part.strip() for part in re.findall(r"[^.!?]+[.!?]+[\"')\]]*\s*|[^.!?]+$", text) if part.strip()]
    sentences: list[str] = []
    for part in parts:
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            sentences.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            sentences.append(part)
    return sentences or [text.strip()]


def pcm_chunks(pcm: bytes, *, chunk_bytes: int = 8192) -> list[bytes]:
    return [pcm[offset : offset + chunk_bytes] for offset in range(0, len(pcm), chunk_bytes)]


def resample_pcm16_to_16k(pcm: bytes, source_rate: int) -> bytes:
    """Linear-interpolation resample of PCM16 mono to 16 kHz (identity at 16 kHz)."""
    if source_rate == OUTPUT_SAMPLE_RATE:
        return pcm
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    target_count = int(len(samples) * OUTPUT_SAMPLE_RATE / source_rate)
    if target_count <= 0:
        return b""
    step = source_rate / OUTPUT_SAMPLE_RATE
    resampled = array("h", bytes(2 * target_count))
    last = len(samples) - 1
    for index in range(target_count):
        position = index * step
        left = int(position)
        right = min(left + 1, last)
        fraction = position - left
        value = samples[left] * (1.0 - fraction) + samples[right] * fraction
        resampled[index] = int(max(-32768, min(32767, round(value))))
    return resampled.tobytes()


@lru_cache(maxsize=2)
def _load_piper_voice(model_path: str) -> Any:
    from piper import PiperVoice  # optional demo dependency

    return PiperVoice.load(model_path)


class PiperTTS:
    """Offline Piper TTS (demo deviation from the design provider set).

    The voice model is loaded once per process and synthesis runs in a worker
    thread, sentence by sentence, so the first sentence plays while later ones
    are still being synthesized.
    """

    def __init__(
        self,
        *,
        model_path: str,
        voice_sample_rate: int | None = None,
        runner: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.voice_sample_rate = voice_sample_rate or self._read_sample_rate(model_path)
        self._runner = runner

    @staticmethod
    def _read_sample_rate(model_path: str) -> int:
        try:
            config = json.loads(Path(f"{model_path}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return OUTPUT_SAMPLE_RATE
        rate = (config.get("audio") or {}).get("sample_rate") if isinstance(config, dict) else None
        return int(rate) if isinstance(rate, (int, float)) and rate > 0 else OUTPUT_SAMPLE_RATE

    @classmethod
    def from_env(cls) -> PiperTTS:
        model_path = _settings().piper_model
        if not model_path:
            raise ProviderUnavailable("Piper voice model is not configured", code="PIPER_MODEL_NOT_CONFIGURED")
        if not Path(model_path).is_file():
            raise ProviderUnavailable("Piper voice model file is missing", code="PIPER_MODEL_MISSING")
        if importlib.util.find_spec("piper") is None:
            raise ProviderUnavailable("piper-tts is not installed (uv sync --group demo)", code="PIPER_NOT_INSTALLED")
        return cls(model_path=model_path)

    async def warm(self) -> None:
        """Load the voice model ahead of the first answer (about 1-2 s on first use)."""
        if self._runner is None:
            await asyncio.to_thread(_load_piper_voice, self.model_path)

    async def _synthesize(self, sentence: str) -> bytes:
        if self._runner is not None:
            result = self._runner(sentence)
            return await result if asyncio.iscoroutine(result) else result

        def run() -> bytes:
            voice = _load_piper_voice(self.model_path)
            return b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(sentence))

        return await asyncio.to_thread(run)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        text = mask_pii(text).strip()
        if not text or len(text) > MAX_TTS_CHARS:
            raise ValueError("TTS text length is invalid")
        for sentence in split_sentences(text):
            pcm = await self._synthesize(sentence)
            for chunk in pcm_chunks(resample_pcm16_to_16k(pcm, self.voice_sample_rate)):
                yield chunk


SpeechSynthesizer = ElevenLabsTTS | PiperTTS

__all__ = [
    "ElevenLabsTTS",
    "PiperTTS",
    "SpeechSynthesizer",
    "pcm_chunks",
    "resample_pcm16_to_16k",
    "split_sentences",
]
