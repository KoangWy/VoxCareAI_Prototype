"""Centralized ``VOXCARE_*`` environment configuration with fail-fast validation.

Every prototype setting is parsed here. :func:`load_turn_settings` covers turn
control (mode, budgets, thresholds, barge-in) and :func:`load_provider_settings`
covers provider selection (ASR/TTS/LLM names, models, endpoints, local model
paths). Invalid values raise :exc:`ConfigError` whose message starts with the
offending variable name, so misconfiguration fails loudly at session creation
instead of as a bare ``ValueError`` mid-turn. API keys are never parsed here:
providers only check their presence.

Turn-decision budgets keep per-judge defaults:

- TypeSafe-direct (``jev``) and ``baseline``: 300 ms judge budget, 3000 ms HOLD deadline.
- OpenRouter decisions: 2500 ms judge budget, 9000 ms HOLD deadline (the
  route measures 1-2 s per judgment, and the first judgment can only start
  after browser-VAD quiet plus ASR finality).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields

from .turns.controller import TurnConfig

VALID_TURN_MODES = ("baseline", "jev", "openrouter")
VALID_BARGE_IN_MODES = ("off", "semantic")
VALID_ASR_PROVIDERS = ("deepgram", "vosk")
VALID_TTS_PROVIDERS = ("elevenlabs", "piper")
VALID_OPENAI_TRANSPORTS = ("responses", "chat")

TYPESAFE_JUDGE_TIMEOUT_MS = 300
TYPESAFE_HOLD_DEADLINE_MS = 3000
OPENROUTER_JUDGE_TIMEOUT_MS = 2500
OPENROUTER_HOLD_DEADLINE_MS = 9000

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_OPENROUTER_MODEL = "inception/mercury-decide"

# TurnConfig field -> environment variable (fields without an entry are fixed by spec).
_TURN_ENV = {
    "jev_timeout_ms": "VOXCARE_JEV_TIMEOUT_MS",
    "hold_deadline_ms": "VOXCARE_HOLD_DEADLINE_MS",
    "turn_complete_threshold": "VOXCARE_TURN_COMPLETE_THRESHOLD",
    "holding_floor_threshold": "VOXCARE_HOLDING_FLOOR_THRESHOLD",
    "addressed_to_agent_threshold": "VOXCARE_ADDRESSED_TO_AGENT_THRESHOLD",
    "taking_floor_threshold": "VOXCARE_TAKING_FLOOR_THRESHOLD",
    "min_stable_word_ratio": "VOXCARE_MIN_STABLE_WORD_RATIO",
    "eos_threshold_floor": "VOXCARE_EOS_THRESHOLD_FLOOR",
    "eos_decay_ms": "VOXCARE_EOS_DECAY_MS",
    "silence_cap_ms": "VOXCARE_EOS_MAX_SILENCE_MS",
    "silence_cap_holding_ms": "VOXCARE_EOS_MAX_SILENCE_HOLDING_MS",
    "confirm_quiet_min_ms": "VOXCARE_EOS_CONFIRM_QUIET_MIN_MS",
    "confirm_quiet_max_ms": "VOXCARE_EOS_CONFIRM_QUIET_MAX_MS",
}


class ConfigError(ValueError):
    """One environment variable failed validation; the message starts with its name."""


def _parse_positive_int(raw: str | None, *, field: str, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{field} must be an integer, got {text!r}") from None
    if value <= 0:
        raise ConfigError(f"{field} must be positive, got {value}")
    return value


def _parse_unit_float(raw: str | None, *, field: str, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"{field} must be a number, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{field} must be between 0 and 1, got {value}")
    return value


def _parse_choice(raw: str | None, *, field: str, default: str, valid: tuple[str, ...]) -> str:
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in valid:
        raise ConfigError(f"{field} must be one of {', '.join(valid)}, got {raw.strip()!r}")
    return value


def _parse_text(raw: str | None, default: str | None = None) -> str | None:
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass(frozen=True)
class TurnSettings:
    """Resolved, validated settings for one turn mode."""

    mode: str
    config: TurnConfig
    barge_in: str


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved provider selection; API keys stay unread."""

    asr_provider: str = "deepgram"
    tts_provider: str = "elevenlabs"
    openai_transport: str = "responses"
    openai_model: str = "gpt-6-luna"
    openai_base_url: str | None = None
    openai_max_tokens: int = 1024
    openai_reasoning_effort: str | None = None
    session_header: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_base_url: str | None = None
    elevenlabs_model: str = "eleven_flash_v2_5"
    vosk_model: str | None = None
    piper_model: str | None = None
    jev_model: str | None = None

    def judge_model(self, mode: str) -> str:
        """The decision model for a judged turn mode (explicit override or the per-mode default)."""
        return self.jev_model or (DEFAULT_OPENROUTER_MODEL if mode == "openrouter" else DEFAULT_JEV_MODEL)


def turn_mode_from_env(env: Mapping[str, str] | None = None) -> str:
    """Raw turn mode without validation (for error reporting)."""
    source = env if env is not None else os.environ
    return (source.get("VOXCARE_TURN_MODE") or "baseline").strip().lower()


def load_turn_settings(env: Mapping[str, str] | None = None) -> TurnSettings:
    """Parse and validate every turn variable; raises :exc:`ConfigError`."""
    source = env if env is not None else os.environ
    mode = _parse_choice(
        source.get("VOXCARE_TURN_MODE"), field="VOXCARE_TURN_MODE", default="baseline", valid=VALID_TURN_MODES
    )
    defaults = {field.name: field.default for field in fields(TurnConfig)}
    if mode == "openrouter":
        defaults["jev_timeout_ms"], defaults["hold_deadline_ms"] = (
            OPENROUTER_JUDGE_TIMEOUT_MS,
            OPENROUTER_HOLD_DEADLINE_MS,
        )
    else:
        defaults["jev_timeout_ms"], defaults["hold_deadline_ms"] = TYPESAFE_JUDGE_TIMEOUT_MS, TYPESAFE_HOLD_DEADLINE_MS

    values: dict[str, int | float] = {}
    for name, env_name in _TURN_ENV.items():
        default = defaults[name]
        if isinstance(default, float):
            values[name] = _parse_unit_float(source.get(env_name), field=env_name, default=default)
        else:
            values[name] = _parse_positive_int(source.get(env_name), field=env_name, default=int(default))
    try:
        config = TurnConfig(**values)  # type: ignore[arg-type]
    except ValueError as exc:
        # Cross-field rules (floor <= threshold, min <= max): name the variable, not the field.
        name, _, rule = str(exc).partition(" ")
        raise ConfigError(f"{_TURN_ENV.get(name, name)} {rule}") from None
    barge_in = _parse_choice(
        source.get("VOXCARE_BARGE_IN"), field="VOXCARE_BARGE_IN", default="off", valid=VALID_BARGE_IN_MODES
    )
    return TurnSettings(mode=mode, config=config, barge_in=barge_in)


def load_provider_settings(env: Mapping[str, str] | None = None) -> ProviderSettings:
    """Parse and validate provider-selection variables; raises :exc:`ConfigError`."""
    source = env if env is not None else os.environ
    get = source.get
    defaults = ProviderSettings()
    return ProviderSettings(
        asr_provider=_parse_choice(
            get("VOXCARE_ASR_PROVIDER"), field="VOXCARE_ASR_PROVIDER", default="deepgram", valid=VALID_ASR_PROVIDERS
        ),
        tts_provider=_parse_choice(
            get("VOXCARE_TTS_PROVIDER"), field="VOXCARE_TTS_PROVIDER", default="elevenlabs", valid=VALID_TTS_PROVIDERS
        ),
        openai_transport=_parse_choice(
            get("VOXCARE_OPENAI_TRANSPORT"),
            field="VOXCARE_OPENAI_TRANSPORT",
            default="responses",
            valid=VALID_OPENAI_TRANSPORTS,
        ),
        openai_model=_parse_text(get("VOXCARE_OPENAI_MODEL"), defaults.openai_model) or defaults.openai_model,
        openai_base_url=_parse_text(get("VOXCARE_OPENAI_BASE_URL")),
        openai_max_tokens=_parse_positive_int(
            get("VOXCARE_OPENAI_MAX_TOKENS"), field="VOXCARE_OPENAI_MAX_TOKENS", default=defaults.openai_max_tokens
        ),
        openai_reasoning_effort=_parse_text(get("VOXCARE_OPENAI_REASONING_EFFORT")),
        session_header=_parse_text(get("VOXCARE_OPENAI_SESSION_HEADER")),
        anthropic_model=_parse_text(get("VOXCARE_ANTHROPIC_MODEL"), defaults.anthropic_model)
        or defaults.anthropic_model,
        anthropic_base_url=_parse_text(get("VOXCARE_ANTHROPIC_BASE_URL")),
        elevenlabs_model=_parse_text(get("VOXCARE_ELEVENLABS_MODEL"), defaults.elevenlabs_model)
        or defaults.elevenlabs_model,
        vosk_model=_parse_text(get("VOXCARE_VOSK_MODEL")),
        piper_model=_parse_text(get("VOXCARE_PIPER_MODEL")),
        jev_model=_parse_text(get("VOXCARE_JEV_MODEL")),
    )


def config_error_code(error: ConfigError, prefix: str = "TURN_CONFIG_INVALID") -> str:
    """Stable ``<PREFIX>:<VARIABLE>`` code for a settings failure."""
    return f"{prefix}:{str(error).split(' ', 1)[0]}"


__all__ = [
    "DEFAULT_JEV_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "OPENROUTER_HOLD_DEADLINE_MS",
    "OPENROUTER_JUDGE_TIMEOUT_MS",
    "TYPESAFE_HOLD_DEADLINE_MS",
    "TYPESAFE_JUDGE_TIMEOUT_MS",
    "VALID_BARGE_IN_MODES",
    "VALID_TURN_MODES",
    "ConfigError",
    "ProviderSettings",
    "TurnSettings",
    "config_error_code",
    "load_provider_settings",
    "load_turn_settings",
    "turn_mode_from_env",
]
