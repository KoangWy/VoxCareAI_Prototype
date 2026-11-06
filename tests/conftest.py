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
# Store settings every app-level test runs with: the in-memory relational twin,
# in-process fakeredis, an ephemeral Chroma collection and the offline hashing
# embedding. tests/integration/ runs the same suites against PostgreSQL/Redis.
_STORE_KEYS = ("DATABASE_URL", "REDIS_URL", "CHROMA_PATH", "SESSION_TOKEN_SECRET")
TEST_STORE_ENV = {
    "VOXCARE_DATA_BACKEND": "memory",
    "VOXCARE_POLICY_EMBEDDING": "hashing",
    "CHROMA_PATH": "",
    "SESSION_TOKEN_SECRET": "test-secret-for-voxcare-unit-tests-0123456789",
    "VOXCARE_LOG_FORMAT": "text",
}


@pytest.fixture(autouse=True)
def hermetic_provider_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VOXCARE_") or key in _PROVIDER_KEYS or key in _STORE_KEYS:
            monkeypatch.delenv(key, raising=False)
    for key, value in TEST_STORE_ENV.items():
        monkeypatch.setenv(key, value)
