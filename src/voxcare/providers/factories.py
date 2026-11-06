"""Provider factories: per-session dialogue cascades and per-connection speech adapters.

Factories read validated :mod:`voxcare.config` settings; API keys are only
checked for presence. A provider/transport value that fails validation raises
:exc:`ProviderUnavailable` with a stable machine-readable ``code``.
"""

from __future__ import annotations

import os

from ..config import ConfigError, ProviderSettings, load_provider_settings
from .asr import AsrConnection, DeepgramConnection, VoskConnection
from .base import ProviderUnavailable
from .keyword import KeywordPlanner
from .llm import AnthropicMessagesAdapter, CascadeDialogueProvider, OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from .tts import ElevenLabsTTS, PiperTTS, SpeechSynthesizer


def _settings(code: str) -> ProviderSettings:
    try:
        return load_provider_settings()
    except ConfigError as exc:
        raise ProviderUnavailable(str(exc), code=code) from None


def asr_connection_from_env() -> AsrConnection:
    if _settings("ASR_PROVIDER_UNSUPPORTED").asr_provider == "vosk":
        return VoskConnection.from_env()
    return DeepgramConnection.from_env()


def tts_from_env() -> SpeechSynthesizer:
    if _settings("TTS_PROVIDER_UNSUPPORTED").tts_provider == "piper":
        return PiperTTS.from_env()
    return ElevenLabsTTS.from_env()


def conversation_factory_from_env() -> CascadeDialogueProvider | KeywordPlanner | None:
    """Create an independent cascade for each session, or None when the primary key is absent."""
    settings = _settings("CONVERSATION_TRANSPORT_UNSUPPORTED")
    if settings.conversation_provider == "keyword":
        return KeywordPlanner()  # offline demo stand-in: no model, no network
    if settings.conversation_provider == "anthropic":
        # Anthropic as the primary model; there is no fallback in this mode.
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not anthropic_key:
            return None
        primary_anthropic = AnthropicMessagesAdapter(
            api_key=anthropic_key, model=settings.anthropic_model, base_url=settings.anthropic_base_url
        )
        return CascadeDialogueProvider(primary_anthropic, None)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    primary: OpenAIResponsesAdapter | OpenAIChatCompletionsAdapter
    if settings.openai_transport == "chat":
        primary = OpenAIChatCompletionsAdapter(
            api_key=api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            session_header=settings.session_header,
            max_tokens=settings.openai_max_tokens,
            reasoning_effort=settings.openai_reasoning_effort,
        )
    else:
        primary = OpenAIResponsesAdapter(
            api_key=api_key, model=settings.openai_model, base_url=settings.openai_base_url
        )
    fallback_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    fallback = (
        AnthropicMessagesAdapter(
            api_key=fallback_key, model=settings.anthropic_model, base_url=settings.anthropic_base_url
        )
        if fallback_key
        else None
    )
    ttft_ms = os.getenv("VOXCARE_LLM_TTFT_TIMEOUT_MS", "").strip()
    timeout_s = int(ttft_ms) / 1000 if ttft_ms.isdigit() and int(ttft_ms) > 0 else 1.5
    return CascadeDialogueProvider(primary, fallback, primary_timeout_s=timeout_s)


__all__ = ["asr_connection_from_env", "conversation_factory_from_env", "tts_from_env"]
