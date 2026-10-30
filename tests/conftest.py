import os

import pytest

# The app loads the developer's .env at import time. Every test starts from a
# clean provider environment instead, so no test can reach a real provider or
# depend on local demo-stack settings; tests set exactly what they need.
_PROVIDER_KEYS = (
    "DEEPGRAM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "TYPESAFE_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def hermetic_provider_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VOXCARE_") or key in _PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
