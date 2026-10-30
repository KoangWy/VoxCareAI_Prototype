"""Streaming ASR connections: Deepgram Nova-2 live adapter + offline Vosk adapter.

Both emit the same normalized event dicts (``transcript`` /
``provider_error`` / ``SpeechStarted`` / ``UtteranceEnd``) so the voice
socket flow is unchanged whichever provider is selected.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import websockets

from ..config import ConfigError, load_provider_settings
from .base import ProviderUnavailable

logger = logging.getLogger("voxcare.providers")


class DeepgramConnection:
    """Live Nova-2 WebSocket adapter; audio is PCM16 mono at 16 kHz."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        connector: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self._connector = connector or websockets.connect
        self.socket = None
        self._receive_task: asyncio.Task | None = None
        self._events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=100)
        self._sequence = 0

    @classmethod
    def from_env(cls) -> DeepgramConnection:
        key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not key:
            raise ProviderUnavailable("Deepgram API key is not configured", code="DEEPGRAM_NOT_CONFIGURED")
        return cls(api_key=key)

    async def connect(self) -> None:
        query = urlencode(
            {
                "model": self.model,
                "language": self.language,
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
                "interim_results": "true",
                "smart_format": "true",
                "endpointing": 1000,
                "utterance_end_ms": 1000,
                "vad_events": "true",
            }
        )
        self.socket = await self._connector(
            f"wss://api.deepgram.com/v1/listen?{query}",
            additional_headers={"Authorization": f"Token {self.api_key}"},
            max_size=1_048_576,
            ping_interval=20,
            open_timeout=8,
        )
        self._receive_task = asyncio.create_task(self._receive())

    async def send_audio(self, pcm16: bytes) -> None:
        if self.socket is None:
            raise ProviderUnavailable("Deepgram connection is not open")
        if len(pcm16) > 64 * 1024:
            raise ValueError("audio frame too large")
        await self.socket.send(pcm16)

    async def finalize(self) -> None:
        if self.socket is not None:
            await self.socket.send(json.dumps({"type": "Finalize"}))

    async def events(self) -> AsyncIterator[dict[str, object]]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        if self.socket is not None:
            try:
                await self.socket.send(json.dumps({"type": "CloseStream"}))
                await self.socket.close()
            finally:
                self.socket = None
        if self._receive_task is not None:
            self._receive_task.cancel()
            await asyncio.gather(self._receive_task, return_exceptions=True)
            self._receive_task = None

    async def _receive(self) -> None:
        try:
            assert self.socket is not None  # noqa: S101 - connect() invariant, not a security check
            async for raw in self.socket:
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = message.get("type")
                if kind == "Results":
                    channel = message.get("channel") or {}
                    alternatives = channel.get("alternatives") or []
                    transcript = alternatives[0].get("transcript", "") if alternatives else ""
                    if transcript:
                        self._sequence += 1
                        await self._events.put(
                            {
                                "type": "transcript",
                                "text": transcript,
                                "is_final": bool(message.get("is_final")),
                                "speech_final": bool(message.get("speech_final")),
                                "start": message.get("start"),
                                "duration": message.get("duration"),
                                "sequence": self._sequence,
                            }
                        )
                elif kind in {"SpeechStarted", "UtteranceEnd"}:
                    await self._events.put({"type": kind})
                elif kind == "Error":
                    await self._events.put({"type": "provider_error", "error_type": "DeepgramStreamError"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Deepgram stream ended", extra={"error_type": type(exc).__name__})
            await self._events.put({"type": "provider_error", "error_type": type(exc).__name__})
        finally:
            await self._events.put(None)


@lru_cache(maxsize=2)
def _load_vosk_model(model_path: str) -> Any:
    """One loaded Vosk model per path, shared by every connection (recognizers stay per-connection)."""
    from vosk import Model, SetLogLevel  # optional demo dependency

    SetLogLevel(-1)
    return Model(model_path)


class VoskConnection:
    """Offline Vosk streaming ASR (demo stack deviation from design §2.1).

    Emits the same normalized event dicts as DeepgramConnection so the voice
    socket flow is unchanged. Vosk final results stand in for speech_final;
    turn commit still requires the browser-local VAD quiet signal.
    """

    def __init__(
        self,
        *,
        model_path: str,
        recognizer_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self._recognizer_factory = recognizer_factory
        self._recognizer: Any = None
        self._events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=100)
        self._sequence = 0
        self._segment_id = 0.0
        self._last_partial = ""

    @classmethod
    def from_env(cls) -> VoskConnection:
        try:
            model_path = load_provider_settings().vosk_model
        except ConfigError as exc:
            raise ProviderUnavailable(str(exc), code="ASR_PROVIDER_UNSUPPORTED") from None
        if not model_path:
            raise ProviderUnavailable("Vosk model path is not configured", code="VOSK_MODEL_NOT_CONFIGURED")
        if not Path(model_path).is_dir():
            raise ProviderUnavailable("Vosk model directory is missing", code="VOSK_MODEL_MISSING")
        if importlib.util.find_spec("vosk") is None:
            raise ProviderUnavailable("vosk is not installed (uv sync --group demo)", code="VOSK_NOT_INSTALLED")
        return cls(model_path=model_path)

    async def connect(self) -> None:
        if self._recognizer_factory is not None:
            self._recognizer = self._recognizer_factory(self.model_path)
            return

        def build() -> Any:
            from vosk import KaldiRecognizer  # optional demo dependency

            return KaldiRecognizer(_load_vosk_model(self.model_path), 16_000)

        self._recognizer = await asyncio.to_thread(build)

    async def send_audio(self, pcm16: bytes) -> None:
        recognizer = self._recognizer
        if recognizer is None:
            raise ProviderUnavailable("Vosk connection is not open", code="VOSK_NOT_CONNECTED")
        if len(pcm16) > 64 * 1024:
            raise ValueError("audio frame too large")
        try:
            finalized = await asyncio.to_thread(recognizer.AcceptWaveform, pcm16)
            if finalized:
                await self._emit_final()
            else:
                await self._emit_partial()
        except ProviderUnavailable:
            raise
        except Exception as exc:
            logger.warning("Vosk stream failed", extra={"error_type": type(exc).__name__})
            await self._events.put({"type": "provider_error", "error_type": type(exc).__name__})

    async def _emit_final(self, raw: str | None = None) -> None:
        assert self._recognizer is not None  # noqa: S101 - connect()/guard invariant, not a security check
        text = str(json.loads(raw if raw is not None else self._recognizer.Result()).get("text", "")).strip()
        self._last_partial = ""
        if not text:
            return
        self._segment_id += 1.0
        self._sequence += 1
        await self._events.put(
            {
                "type": "transcript",
                "text": text,
                "is_final": True,
                "speech_final": True,
                "start": self._segment_id,
                "duration": None,
                "sequence": self._sequence,
            }
        )

    async def _emit_partial(self) -> None:
        assert self._recognizer is not None  # noqa: S101 - connect()/guard invariant, not a security check
        text = str(json.loads(self._recognizer.PartialResult()).get("partial", "")).strip()
        if not text or text == self._last_partial:
            return
        started = not self._last_partial
        self._last_partial = text
        self._sequence += 1
        if started:
            await self._events.put({"type": "SpeechStarted"})
        await self._events.put(
            {
                "type": "transcript",
                "text": text,
                "is_final": False,
                "speech_final": False,
                "start": self._segment_id + 1.0,
                "duration": None,
                "sequence": self._sequence,
            }
        )

    async def finalize(self) -> None:
        if self._recognizer is not None:
            await self._emit_final(self._recognizer.FinalResult())

    async def events(self) -> AsyncIterator[dict[str, object]]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self._recognizer = None
        await self._events.put(None)


AsrConnection = DeepgramConnection | VoskConnection
