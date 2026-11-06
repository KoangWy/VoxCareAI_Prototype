"""Download the offline demo models (Vosk ASR + Piper voice) into models/.

Demo-stack deviation from design section 2.1; the files are git-ignored and
this script is the only supported way to fetch them.

Usage: uv run --group demo python scripts/fetch_demo_models.py [--vosk small|lgraph|large]

Vosk English models (word error rates as published by the Vosk project,
LibriSpeech test-clean / TED-LIUM):
  small   vosk-model-small-en-us-0.15   ~40 MB   9.85 / 10.38  (default)
  lgraph  vosk-model-en-us-0.22-lgraph  ~128 MB  7.82 / 8.20
  large   vosk-model-en-us-0.22         ~1.8 GB  5.69 / 6.05   (needs several GB of RAM)
Point VOXCARE_VOSK_MODEL at the directory the script prints.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

VOSK_BASE_URL = "https://alphacephei.com/vosk/models"
VOSK_MODELS = {
    "small": "vosk-model-small-en-us-0.15",
    "lgraph": "vosk-model-en-us-0.22-lgraph",
    "large": "vosk-model-en-us-0.22",
}
PIPER_VOICE_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium"
)
PIPER_FILES = (".onnx", ".onnx.json")


def _fetch(url: str, target: Path) -> None:
    print(f"downloading {url} -> {target}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:  # noqa: S310 - fixed model hosts
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)


def fetch_vosk(variant: str = "small") -> None:
    dirname = VOSK_MODELS[variant]
    target_dir = MODELS_DIR / dirname
    if (target_dir / "final.mdl").exists() or (target_dir / "am" / "final.mdl").exists():
        print(f"vosk model already present: {target_dir}")
    else:
        archive = MODELS_DIR / f"{dirname}.zip"
        _fetch(f"{VOSK_BASE_URL}/{dirname}.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(MODELS_DIR)
        archive.unlink()
        print(f"vosk model ready: {target_dir}")
    print(f"set in .env: VOXCARE_VOSK_MODEL=models/{dirname}")


def fetch_piper_voice() -> None:
    for suffix in PIPER_FILES:
        target = MODELS_DIR / f"en_US-lessac-medium{suffix}"
        if target.exists():
            print(f"piper voice file already present: {target}")
            continue
        _fetch(f"{PIPER_VOICE_BASE}{suffix}", target)
    print("piper voice ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the offline demo models into models/.")
    parser.add_argument("--vosk", choices=sorted(VOSK_MODELS), default="small", help="Vosk English model size")
    args = parser.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    try:
        fetch_vosk(args.vosk)
        fetch_piper_voice()
    except Exception as exc:
        print(f"failed to fetch demo models: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
