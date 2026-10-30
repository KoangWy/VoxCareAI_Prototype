"""Shared provider foundation: errors, status reporting, and LLM request helpers.

:class:`ProviderUnavailable` marks a missing or misconfigured provider with a
stable machine-readable ``code``; :func:`provider_status` builds the
credential-free ``/api/config`` view; the remaining helpers shape the system
prompt, tool schemas, and PII-masked message history for the LLM adapters.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import ConfigError, ProviderSettings, load_provider_settings
from ..privacy import mask_pii

SYSTEM_PROMPT = (
    "You are VoxCare, an English voice support assistant for the fictional retailer VoxCom. "
    "Your only job is to choose which supplied read-only tool(s) answer the caller's latest request; "
    "the application speaks the answer from the tool results, so never write facts yourself. "
    "Never invent order, status, shipment, ETA, price, discount, fee, return eligibility, refund, or warranty facts, "
    "and never request or attempt a business write (cancel, change, refund, reschedule). "
    "Orders: order codes have the form ORD- plus four digits. Speech recognition may spell one out, e.g. "
    "'o r d four five six seven' or 'ord forty five sixty seven' for ORD-4567; convert the caller's own spoken "
    "digits to that form before calling lookup_order_details (status and items) or track_shipment_status "
    "(carrier, ETA), and never substitute a code the caller did not say. "
    "A status plus a delivery question may need both tools. If no order code was given, call no tool. "
    "Products: use search_catalog for browsing by category/brand/budget (prices in VND), compare_products "
    "for two or three SKUs from earlier results, and apply_voucher to quote a demo voucher against an order value. "
    "Policies: use retrieve_policy_clauses with the caller's question, a category (SHIPPING, INSTALLATION, "
    "WARRANTY, RETURNS, or ALL), and only the brand/product category the caller explicitly gave (otherwise null). "
    "If the caller explicitly asks for a person, call escalate_to_human; it only records the request. "
    "For greetings, small talk, or anything the tools cannot answer, call no tool. "
    "Never include customer IDs or arguments the tool schema does not define."
)


class ProviderUnavailable(RuntimeError):
    """Raised when a required external speech/conversation provider is not configured."""

    def __init__(self, message: str, *, code: str = "PROVIDER_NOT_CONFIGURED") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    configured: bool
    model: str | None
    detail: str


def _has_key(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def provider_status() -> dict[str, ProviderStatus]:
    """Credential-free provider view for ``/api/config``; never raises on bad configuration."""
    try:
        settings = load_provider_settings()
    except ConfigError as exc:
        broken = ProviderStatus("Provider configuration error", False, None, str(exc))
        return {"asr": broken, "conversation": broken, "conversation_fallback": broken, "tts": broken}
    return {
        "asr": _asr_status(settings),
        "conversation": _conversation_status(settings),
        "conversation_fallback": ProviderStatus(
            "Anthropic Messages", _has_key("ANTHROPIC_API_KEY"), settings.anthropic_model, "Sticky session fallback"
        ),
        "tts": _tts_status(settings),
    }


def _asr_status(settings: ProviderSettings) -> ProviderStatus:
    if settings.asr_provider == "vosk":
        path = settings.vosk_model
        return ProviderStatus(
            "Vosk (offline)",
            bool(path) and Path(path).is_dir(),
            Path(path).name if path else None,
            "Offline streaming English ASR; demo deviation from the design provider set",
        )
    return ProviderStatus(
        "Deepgram Nova-2", _has_key("DEEPGRAM_API_KEY"), "nova-2", "Streaming English ASR over WebSocket"
    )


def _conversation_status(settings: ProviderSettings) -> ProviderStatus:
    if settings.openai_transport == "chat":
        name = "OpenAI Chat Completions (gateway)"
        detail = "OpenAI-compatible gateway via chat completions; demo deviation from the design Responses transport"
    else:
        name, detail = "OpenAI Responses", "Primary model; Anthropic fallback after two retryable failures"
    return ProviderStatus(name, _has_key("OPENAI_API_KEY"), settings.openai_model, detail)


def _tts_status(settings: ProviderSettings) -> ProviderStatus:
    if settings.tts_provider == "piper":
        path = settings.piper_model
        return ProviderStatus(
            "Piper (offline)",
            bool(path) and Path(path).is_file(),
            Path(path).name if path else None,
            "Offline neural TTS, per-sentence PCM16; demo deviation from the design provider set",
        )
    return ProviderStatus(
        "ElevenLabs Flash v2.5",
        _has_key("ELEVENLABS_API_KEY") and _has_key("ELEVENLABS_VOICE_ID"),
        settings.elevenlabs_model,
        "Streaming PCM audio over WebSocket",
    )


def responses_tools(tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Tool schemas in the OpenAI Responses (strict function) shape."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": True,
        }
        for tool in tools
    ]


def chat_completions_tools(tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Tool schemas in the chat-completions ``{"type": "function", "function": ...}`` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def chat_messages(user_text: str, history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """User/assistant history plus the new user message, PII-masked for the external model."""
    result = [
        {"role": item["role"], "content": mask_pii(item["content"])}
        for item in history
        if item.get("role") in {"user", "assistant"}
    ]
    result.append({"role": "user", "content": mask_pii(user_text)})
    return result


__all__ = [
    "SYSTEM_PROMPT",
    "ProviderStatus",
    "ProviderUnavailable",
    "chat_completions_tools",
    "chat_messages",
    "provider_status",
    "responses_tools",
]
