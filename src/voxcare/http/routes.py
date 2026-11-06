"""REST surface: config/health, the session token, turns, trace/audit/spans, handoff and reset.

Every session route requires the ``VoxCare-Session-Token`` (header, or
``Authorization: Bearer``); it is issued by ``POST /session`` (alias
``POST /api/sessions``). Typed turns are judgment-free by design: they carry no
microphone timing to judge. The judged path is the voice WebSocket.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ..config import ConfigError, config_error_code, load_provider_settings, load_turn_settings, turn_mode_from_env
from ..observability.otel import RECENT_SPANS
from ..providers import provider_status
from ..security.tokens import HEADER_NAME, InvalidToken, SessionClaims, bearer_token
from ..session.activity import to_markdown
from .bootstrap import STATIC_DIR, boundary
from .schemas import HandoffRequest, SessionCreateRequest, TextTurnRequest

# Judged turn modes and the provider key each one requires.
JUDGED_TURN_MODES = {"jev": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
WSS_HANDSHAKE_BUDGET_S = 2.0  # NFR-04


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


def session_payload(session, *, token: str | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    payload = asdict(session)
    payload["data_notice"] = (
        "Fictional seeded data. Caller verification is simulated by binding a fixture customer (OPD-04); "
        "production needs an OTP challenge."
    )
    if token is not None:
        payload["token"] = token
        payload["token_header"] = HEADER_NAME
    return payload


def turn_payload(result) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return asdict(result)


async def session_claims(
    session_id: str,
    voxcare_session_token: str | None = Header(default=None, alias=HEADER_NAME),
    authorization: str | None = Header(default=None),
) -> SessionClaims:
    """The verified token for ``session_id`` (401 when missing, expired, forged or for another session)."""
    try:
        return await boundary.verify_token(bearer_token(authorization, voxcare_session_token), session_id=session_id)
    except InvalidToken:
        if await boundary.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND") from None
        raise HTTPException(status_code=401, detail="UNAUTHENTICATED_SESSION") from None


async def _wss_handshake(request: Request) -> dict[str, Any]:
    """Open and close a voice WebSocket against this server with a probe token (NFR-04, ≤ 2.0 s)."""
    import websockets

    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host") or f"{request.url.hostname}:{request.url.port}"
    token = boundary.runtime.tokens.issue("SES-HEALTH-PROBE", None, now=datetime.now(UTC), probe=True)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(WSS_HANDSHAKE_BUDGET_S):
            async with websockets.connect(f"{scheme}://{host}/ws/voice?token={token}", open_timeout=2) as socket:
                await socket.recv()
        return {"ok": True, "handshake_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def register(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index():  # type: ignore[no-untyped-def]
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def config():  # type: ignore[no-untyped-def]
        providers = provider_payload()
        return {
            "locale": "en-US",
            "session_customer": "CUST-001 (simulated verification, OPD-04)",
            "turn_mode": turn_mode_from_env(),
            "barge_in": barge_in_mode(),
            "providers": providers,
            "data": boundary.runtime.describe(),
            "voice_ready": all(providers[name]["configured"] for name in ("asr", "conversation", "tts")),
            "limitations": [
                "English only; simulated caller verification and fictional VoxCom data.",
                "Writes are previewed and recorded for VoxCom, never executed (Phase 1).",
                "Latency and accuracy targets are not measured by this demo.",
            ],
        }

    @app.get("/healthz")
    async def healthz(request: Request, wss: int = 0):  # type: ignore[no-untyped-def]
        components = await boundary.runtime.health()
        if wss:
            components["wss_handshake"] = await _wss_handshake(request)
        healthy = all(item.get("ok") for item in components.values())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "degraded", "components": components},
        )

    async def _create(request: SessionCreateRequest | None) -> dict[str, Any]:
        # Invalid configuration fails loudly here with the variable name, never mid-turn.
        try:
            load_turn_settings()
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=config_error_code(exc)) from None
        try:
            load_provider_settings()
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=config_error_code(exc, "PROVIDER_CONFIG_INVALID")) from None
        options = request or SessionCreateRequest()
        try:
            session = await boundary.create_session(
                verified=options.verified, fixture_session_id=options.fixture_session_id
            )
        except ValueError:
            raise HTTPException(status_code=422, detail="UNKNOWN_FIXTURE_SESSION") from None
        return session_payload(session, token=boundary.issue_token(session))

    @app.post("/session")
    async def create_session_alias(request: SessionCreateRequest | None = None):  # type: ignore[no-untyped-def]
        return await _create(request)

    @app.post("/api/sessions")
    async def create_session(request: SessionCreateRequest | None = None):  # type: ignore[no-untyped-def]
        return await _create(request)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        session = await boundary.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return session_payload(session)

    @app.post("/api/sessions/{session_id}/token")
    async def refresh_token(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        session = await boundary.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {"token": boundary.issue_token(session), "token_header": HEADER_NAME}

    @app.get("/api/sessions/{session_id}/trace")
    async def session_trace(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        events = await boundary.session_trace(session_id)
        if events is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {
            "session_id": session_id,
            "persistence": "audit rows in audit_log; observations in memory",
            "events": events,
        }

    @app.get("/api/sessions/{session_id}/audit")
    async def session_audit(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        rows = await boundary.session_audit(session_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {"session_id": session_id, "audit_log": rows}

    @app.get("/api/sessions/{session_id}/handoffs")
    async def session_handoffs(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        packets = await boundary.session_handoffs(session_id)
        if packets is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {"session_id": session_id, "handoff_context_packets": packets}

    @app.get("/api/sessions/{session_id}/activity")
    async def session_activity(  # type: ignore[no-untyped-def]
        session_id: str, format: str = "json", _: SessionClaims = Depends(session_claims)
    ):
        log = await boundary.session_activity(session_id)
        if log is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        if format == "markdown":
            body = to_markdown(log["session_id"], log["customer_id"], log["entries"])
            return PlainTextResponse(
                body,
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="voxcare-activity-{session_id}.md"'},
            )
        if format != "json":
            raise HTTPException(status_code=400, detail="FORMAT_INVALID")
        return log

    @app.get("/api/sessions/{session_id}/spans")
    async def session_spans(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        return {"session_id": session_id, "spans": RECENT_SPANS.for_session(session_id)}

    @app.post("/api/sessions/{session_id}/handoff")
    async def request_handoff(  # type: ignore[no-untyped-def]
        session_id: str, request: HandoffRequest, _: SessionClaims = Depends(session_claims)
    ):
        handoff = await boundary.record_handoff(session_id, reason=request.reason)
        if handoff is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return {
            "outcome": handoff["status"],
            "request_id": handoff["request_id"],
            "ticket_number": handoff.get("ticket_number"),
            "status": handoff["status"],
            "message": handoff["message"],
            "activity": handoff.get("activity"),
        }

    @app.post("/api/sessions/{session_id}/turns")
    async def text_turn(  # type: ignore[no-untyped-def]
        session_id: str, request: TextTurnRequest, _: SessionClaims = Depends(session_claims)
    ):
        if request.source not in {"manual", "asr_final"}:
            raise HTTPException(status_code=422, detail="INVALID_SOURCE")
        result = await boundary.handle_turn(session_id, request.text, source=request.source)
        if result.outcome == "SESSION_EXPIRED":
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        payload = turn_payload(result)
        if result.binding_changed:  # the caller was verified: the token now carries the bound customer
            session = await boundary.get_session(session_id)
            if session is not None:
                payload["session_token"] = boundary.issue_token(session)
        return payload

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str, _: SessionClaims = Depends(session_claims)):  # type: ignore[no-untyped-def]
        session = await boundary.reset_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        return session_payload(session, token=boundary.issue_token(session))

    from ..voice import connection

    connection.register(app)


__all__ = [
    "JUDGED_TURN_MODES",
    "barge_in_mode",
    "provider_payload",
    "register",
    "session_claims",
    "session_payload",
    "turn_decision_status",
    "turn_payload",
]
