"""ASGI entry point: ``uvicorn voxcare.app:app --app-dir src``.

``boundary`` is the process-wide :class:`~voxcare.session.SessionBoundary`
behind every REST and WebSocket route.
"""

from __future__ import annotations

from .http import app, boundary

__all__ = ["app", "boundary"]
