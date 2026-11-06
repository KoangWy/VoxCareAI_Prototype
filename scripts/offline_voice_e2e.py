"""Offline voice end-to-end check: real Piper + Vosk + tools, no API keys, no network.

Runs the scenarios of ``scripts/live_voice_e2e.py`` with the two network
dependencies replaced: the conversation model by a small keyword planner and
the turn judge by the application baseline. Everything else is the real voice
flow (WebSocket protocol, ASR fan-in, turn control, session boundary, grounded
rendering, voice rendering, TTS streaming).

    uv run --group demo python scripts/fetch_demo_models.py   # once
    uv run --group demo python scripts/offline_voice_e2e.py

Needs VOXCARE_VOSK_MODEL / VOXCARE_PIPER_MODEL (from .env or the defaults
under models/). Semantic barge-in scenarios need a real judge, so they are
not in the default list. Exit 0 only when every scenario passes.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = "plain,hesitation,echo"


def main() -> int:
    os.environ["VOXCARE_TURN_MODE"] = "baseline"  # wins over .env (loaded with override=False)
    os.environ["VOXCARE_BARGE_IN"] = "off"
    os.environ["VOXCARE_ASR_PROVIDER"] = "vosk"
    os.environ["VOXCARE_TTS_PROVIDER"] = "piper"
    os.environ.setdefault("VOXCARE_VOSK_MODEL", str(APP_ROOT / "models" / "vosk-model-small-en-us-0.15"))
    os.environ.setdefault("VOXCARE_PIPER_MODEL", str(APP_ROOT / "models" / "en_US-lessac-medium.onnx"))
    sys.path.insert(0, str(APP_ROOT / "src"))

    import voxcare.app as api
    from voxcare.providers.keyword import KeywordPlanner  # the shared offline stand-in for the model

    os.environ.setdefault("VOXCARE_DATA_BACKEND", "memory")  # no database needed for the offline harness
    api.boundary.dialogue_factory = KeywordPlanner
    scenarios = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENARIOS
    sys.argv = ["live_voice_e2e.py", "--scenarios", scenarios]
    try:
        runpy.run_path(str(APP_ROOT / "scripts" / "live_voice_e2e.py"), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
