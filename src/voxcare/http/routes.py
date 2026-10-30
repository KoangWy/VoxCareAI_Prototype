"""REST surface: config/health plus the session, turn, trace, handoff, and reset routes.

Typed turns are judgment-free by design: ``POST /turns`` carries no
microphone timing to judge. The judged path is the WebSocket voice flow in
:mod:`voxcare.voice`.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from ..config import ConfigError, config_error_code, load_provider_settings, load_turn_settings, turn_mode_from_env
from ..providers import provider_status
from ..session.render import HANDOFF_TEXT
from .bootstrap import STATIC_DIR, boundary
from .schemas import HandoffRequest, TextTurnRequest

# Judged turn modes and the provider key each one requires.
JUDGED_TURN_MODES = {"jev": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def turn_decision_status() -> dict[str, Any]:
    """``/api/config`` view of the turn judge; an invalid turn variable is a config error, never baseline."""
    try:
        settings = load_turn_settings()
    except ConfigError as exc:
        return {
            "name": "Turn configuration error",
            "mode": turn_mode_from_env(),
            "configured": False,
            "model": None,
            "error": config_error_code(exc),
            "detail": str(exc),
        }
    key_name = JUDGED_TURN_MODES.get(settings.mode)
    if key_name is None:
        return {"name": "Application baseline", "mode": settings.mode, "configured": True, "model": None}
    try:
        model: str | None = load_provider_settings().judge_model(settings.mode)
    except ConfigError:
        model = None
    return {
        "name": "TypeSafe Jev" if settings.mode == "jev" else "OpenRouter Decisions",
        "mode": settings.mode,
        "configured": bool(os.getenv(key_name, "").strip()),
        "model": model,
    }


def provider_payload() -> dict[str, Any]:
    result: dict[str, Any] = {
        name: {"name": state.name, "configured": state.configured, "model": state.model, "detail": state.detail}
        for name, state in provider_status().items()
    }
    result["turn_decision"] = turn_decision_status()
    return result


def barge_in_mode() -> str:
    try:
        return load_turn_settings().barge_in
    except ConfigError:
        return "off"


def session_payload(session) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    payload = asdict(session)
    payload["data_notice"] = "Fictional seeded data; this is not production authentication."
    return payload


def turn_payload(result) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return asdict(result)


def register(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index():  # type: ignore[no-untyped-def]
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def config():  # type: ignore[no-untyped-def]
        providers = provider_payload()
        return {
            "locale": "en-US",
            "session_customer": "CUST-001 (simulated)",
            "turn_mode": turn_mode_from_env(),
            "barge_in": barge_in_mode(),
            "providers": providers,
            "voice_ready": all(providers[name]["configured"] for name in ("asr", "conversation", "tts")),
            "limitations": [
                "English only; simulated identity and fictional read-only OMS data.",
                "Latency and accuracy benchmarks are not measured by this demo.",
            ],
        }

    @app.get("/healthz")
    async def healthz():  # type: ignore[no-untyped-def]
        return {"status": "ok"}

    @app.post("/api/sessions")
    async def create_session():  # type: ignore[no-untyped-def]
        # Invalid configuration fails loudly here with the variable name, never mid-turn.
        try:
            load_turn_settings()
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=config_error_code(exc)) from None
        try:
            load_provider_settings()
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=config_error_code(exc, "PROVIDER_CONFIG_INVALID")) from None
        return session_payload(await boundary.create_session())

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):  # type: ignore[no-untyped-def]
        session = await boundary.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return session_payload(session)

    @app.get("/api/sessions/{session_id}/trace")
    async def session_trace(session_id: str):  # type: ignore[no-untyped-def]
        events = await boundary.session_trace(session_id)
        if events is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {"session_id": session_id, "persistence": "in_memory_for_session_lifetime", "events": events}

    @app.post("/api/sessions/{session_id}/handoff")
    async def request_handoff(session_id: str, request: HandoffRequest):  # type: ignore[no-untyped-def]
        handoff = await boundary.record_handoff(session_id, reason=request.reason)
        if handoff is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {
            "outcome": "HANDOFF_REQUEST_RECORDED",
            "request_id": handoff["request_id"],
            "status": handoff["status"],
            "message": HANDOFF_TEXT,
        }

    @app.post("/api/sessions/{session_id}/turns")
    async def text_turn(session_id: str, request: TextTurnRequest):  # type: ignore[no-untyped-def]
        if request.source not in {"manual", "asr_final"}:
            raise HTTPException(status_code=422, detail="INVALID_SOURCE")
        result = await boundary.handle_turn(session_id, request.text, source=request.source)
        if result.outcome == "SESSION_EXPIRED":
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return turn_payload(result)

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str):  # type: ignore[no-untyped-def]
        session = await boundary.reset_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return session_payload(session)

    from ..voice import connection

    connection.register(app)


__all__ = [
    "JUDGED_TURN_MODES",
    "barge_in_mode",
    "provider_payload",
    "register",
    "session_payload",
    "turn_decision_status",
    "turn_payload",
]
