"""HTTP bootstrap: app factory, lifespan, turn-controller factory, and request-size bound.

Owns the FastAPI application object and the per-process
:class:`~voxcare.session.SessionBoundary` (the only shared mutable state).
Request bodies are capped at 64 KiB on the streamed body, not just on
``Content-Length``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import TurnSettings, load_provider_settings, load_turn_settings
from ..domain.seed import SeedOMS
from ..providers import conversation_factory_from_env
from ..session import SessionBoundary
from ..turns import BaselineTurnJudge, TurnController
from ..turns.jev import TypeSafeJevJudge
from ..turns.openrouter import OpenRouterDecisionsJudge

APP_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = APP_ROOT / "static"
MAX_REQUEST_BYTES = 64 * 1024

load_dotenv(APP_ROOT / ".env", override=False)
logging.basicConfig(
    level=os.getenv("VOXCARE_LOG_LEVEL", "INFO").strip().upper() or "INFO",
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

oms = SeedOMS()

# OpenRouter judges hold an httpx connection pool but no per-session state, so
# sessions share one judge per (model, budget); shutdown closes them.
_openrouter_judges: dict[tuple[str, int], OpenRouterDecisionsJudge] = {}


def build_turn_controller(settings: TurnSettings | None = None) -> TurnController:
    """A fresh per-session controller for the configured turn mode."""
    resolved = settings or load_turn_settings()
    config = resolved.config
    if resolved.mode == "baseline":
        return TurnController(judge=BaselineTurnJudge(), config=config)
    model = load_provider_settings().judge_model(resolved.mode)
    if resolved.mode == "jev":
        return TurnController(judge=TypeSafeJevJudge(model=model, timeout_ms=config.jev_timeout_ms), config=config)
    key = (model, config.jev_timeout_ms)
    judge = _openrouter_judges.get(key)
    if judge is None:
        judge = _openrouter_judges[key] = OpenRouterDecisionsJudge(model=model, timeout_ms=config.jev_timeout_ms)
    return TurnController(judge=judge, config=config)


boundary = SessionBoundary(
    dialogue=None,
    dialogue_factory=conversation_factory_from_env,
    oms=oms,
    turn_controller_factory=build_turn_controller,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    judges = list(_openrouter_judges.values())
    _openrouter_judges.clear()
    for judge in judges:
        with contextlib.suppress(Exception):
            await judge.aclose()


def create_app() -> FastAPI:
    from .. import __version__
    from . import routes

    app = FastAPI(title="VoxCare AI voice-support prototype", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def bound_request_size(request: Request, call_next):  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"error": "REQUEST_TOO_LARGE"})
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"error": "REQUEST_TOO_LARGE"})
        request._body = bytes(body)
        return await call_next(request)

    routes.register(app)
    return app


app = create_app()

__all__ = ["APP_ROOT", "MAX_REQUEST_BYTES", "STATIC_DIR", "app", "boundary", "build_turn_controller", "create_app"]
