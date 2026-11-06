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
    "Your only job is to choose which supplied tool(s) answer the caller's latest request; the application speaks "
    "the answer verbatim from the tool results, so never write facts yourself. Rules: "
    "(1) Use only the supplied tools. You cannot change orders: requests to cancel, reschedule, change an address or "
    "reinstate an order are handled by the application, so never call a tool for them and never claim a change "
    "happened. Questions about refund, return, exchange, warranty, shipping or installation rules are policy "
    "questions (rule 3), not change requests. "
    "(2) Never state an ETA, date or arrival word yourself; track_shipment_status returns the carrier ETA or null. "
    "(3) Policy questions go to retrieve_policy_clauses with the caller's question as query and a category "
    "(SHIPPING, INSTALLATION, WARRANTY, RETURNS, or ALL when it spans several or you are unsure); never answer "
    "a policy question from general knowledge. "
    "(4) Policy answers cite clauses and are never an approval for the caller's own order. "
    "(5) Return or exchange eligibility always depends on technician certification. "
    "(6) Never ask for or pass customer ids, stock counts or internal ids; the server injects the caller's scope. "
    "Orders: codes have the form ORD- plus four digits. Speech recognition may spell one out ('o r d one zero zero "
    "two' is ORD-1002); convert only digits the caller actually said, never invent a code. For order status call "
    "lookup_order_details; for delivery, tracking or 'where is my package' call lookup_order_details and "
    "track_shipment_status for the same code. If the caller asks about an order but gives no code, still call "
    "lookup_order_details with an empty order_code so the application can ask for it. For earlier purchases or "
    "order history call list_customer_orders. "
    "Products: search_catalog for browsing by category, brand, budget (VND) or keyword; compare_products for two or "
    "three SKUs from the same category (TV-SAM-55, TV-LG-55, WM-PAN-09 are SKU shapes); apply_voucher to quote a "
    "voucher against an order value and category (a quote only). A compare-and-voucher request may need "
    "search_catalog, compare_products and apply_voucher in that order. "
    "If the caller asks for a person, call escalate_to_human with trigger_reason CUSTOMER_REQUESTED_ESCALATION and "
    "priority NORMAL; it records a request and never transfers the call. "
    "For greetings, small talk, or anything the tools cannot answer, call no tool."
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
        "conversation_fallback": _fallback_status(settings),
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


def _fallback_status(settings: ProviderSettings) -> ProviderStatus:
    if settings.conversation_provider == "keyword":
        return ProviderStatus("No fallback", False, None, "Offline keyword planner; no model is used")
    if settings.conversation_provider == "anthropic":
        return ProviderStatus("No fallback", False, None, "Anthropic is the primary model; no fallback is used")
    return ProviderStatus(
        "Anthropic Messages", _has_key("ANTHROPIC_API_KEY"), settings.anthropic_model, "Sticky session fallback"
    )


def _conversation_status(settings: ProviderSettings) -> ProviderStatus:
    if settings.conversation_provider == "keyword":
        return ProviderStatus(
            "Offline keyword planner",
            True,
            "keyword-planner",
            "Deterministic stand-in for the LLM (VOXCARE_CONVERSATION_PROVIDER=keyword); demo deviation, not a model",
        )
    if settings.conversation_provider == "anthropic":
        return ProviderStatus(
            "Anthropic Messages",
            _has_key("ANTHROPIC_API_KEY"),
            settings.anthropic_model,
            "Primary model (VOXCARE_CONVERSATION_PROVIDER=anthropic); one tool call per turn",
        )
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
