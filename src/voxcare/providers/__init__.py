"""Provider adapters (ASR, conversation, TTS) and the factories that select them from the environment."""

from __future__ import annotations

from .asr import AsrConnection, DeepgramConnection, VoskConnection
from .base import SYSTEM_PROMPT, ProviderStatus, ProviderUnavailable, provider_status
from .factories import asr_connection_from_env, conversation_factory_from_env, tts_from_env
from .llm import AnthropicMessagesAdapter, CascadeDialogueProvider, OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from .tts import ElevenLabsTTS, PiperTTS, SpeechSynthesizer

__all__ = [
    "SYSTEM_PROMPT",
    "AnthropicMessagesAdapter",
    "AsrConnection",
    "CascadeDialogueProvider",
    "DeepgramConnection",
    "ElevenLabsTTS",
    "OpenAIChatCompletionsAdapter",
    "OpenAIResponsesAdapter",
    "PiperTTS",
    "ProviderStatus",
    "ProviderUnavailable",
    "SpeechSynthesizer",
    "VoskConnection",
    "asr_connection_from_env",
    "conversation_factory_from_env",
    "provider_status",
    "tts_from_env",
]
