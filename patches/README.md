# Saved patches

## bilingual-vi.patch

Vietnamese + English support, set aside so the English prototype can be finished first.
It adds a per-session language (vi-VN / en-US, chosen in the page header, Vietnamese by default),
Vietnamese Vosk/Piper models, Vietnamese answer templates and policy translations, Vietnamese
spoken-number rendering, Vietnamese spoken order codes, bilingual prompts, tests and docs.
Built and tested against the English version that replaced it (261 backend + 74 frontend tests).

Apply it later from the project root:

```sh
git apply --check --ignore-whitespace patches/bilingual-vi.patch   # dry run
git apply --ignore-whitespace patches/bilingual-vi.patch
uv run python scripts/fetch_demo_models.py --lang vi              # Vietnamese Vosk + Piper models
```

Then add to `.env`:

```sh
VOXCARE_VOSK_MODEL_VI=models/vosk-model-vn-0.4
VOXCARE_PIPER_MODEL_VI=models/vi_VN-vais1000-medium.onnx
```

`--ignore-whitespace` lets the patch (LF line endings) apply to a Windows checkout with CRLF files.
If the English code has changed a lot by then, apply what still fits and use the patch as the reference.
