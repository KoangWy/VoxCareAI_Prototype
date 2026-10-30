"""Session boundary: per-session state, the read-only tool registry, and grounded answers."""

from __future__ import annotations

from .boundary import MAX_SESSIONS, MAX_TEXT_CHARS, SessionBoundary
from .tools import MAX_HANDOFF_REASON_CHARS, TOOLS

__all__ = ["MAX_HANDOFF_REASON_CHARS", "MAX_SESSIONS", "MAX_TEXT_CHARS", "TOOLS", "SessionBoundary"]
