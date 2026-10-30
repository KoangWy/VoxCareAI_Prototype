"""Turn control: the judged end-of-turn state machine and its judge adapters.

- :mod:`.controller` — :class:`TurnController`, :class:`TurnConfig`, the
  deterministic :class:`BaselineTurnJudge`, and the decision/episode model.
- :mod:`.prompts` — the shared judge state and four Noul questions.
- :mod:`.jev` — TypeSafe Jev adapter (direct SDK transport).
- :mod:`.openrouter` — OpenRouter Decisions adapter (same questions over HTTP).
"""

from __future__ import annotations

from .controller import (
    BaselineTurnJudge,
    FallbackAction,
    JevConfigurationError,
    JevJudge,
    JevJudgment,
    JevProviderError,
    TurnConfig,
    TurnController,
    TurnDecision,
    TurnInput,
    confirm_quiet_ms_for,
    effective_turn_complete_threshold,
)

__all__ = [
    "BaselineTurnJudge",
    "FallbackAction",
    "JevConfigurationError",
    "JevJudge",
    "JevJudgment",
    "JevProviderError",
    "TurnConfig",
    "TurnController",
    "TurnDecision",
    "TurnInput",
    "confirm_quiet_ms_for",
    "effective_turn_complete_threshold",
]
