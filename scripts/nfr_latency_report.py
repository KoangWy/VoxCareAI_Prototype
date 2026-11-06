"""NFR-01 / NFR-02 report from recorded OTel spans: distributions only, no claims beyond the data.

    # during a voice test run, for each session id:
    curl -H "VoxCare-Session-Token: $TOKEN" http://127.0.0.1:8000/api/sessions/$SID/spans > spans_$SID.json
    uv run python scripts/nfr_latency_report.py spans_*.json

Reports, with N and p50/p90/p95/max:

- **full turn** (NFR-02, target p50 ≤ 2.0 s, p95 ≤ 3.5 s): duration of each ``turn`` span
  (T1 commit → reply ready) plus its ``tts`` span up to T5;
- **TTFA** (NFR-01b, adopted ceiling 755 ms): ``ttfa_playback_start_ms`` = T6 - T0 on the browser clock
  (guide §9.5); the browser's playback start is not an acoustic confirmation, which is stated with the number;
- **LLM** (T2 → T3) and **tool** span durations, for the per-link budget.

NFR-01a needs ≥ 150 English turns (planning assumption); the report prints whether N reaches it.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def summary(name: str, values: list[float], target: str = "") -> str:
    if not values:
        return f"{name:28s} N=0 (no data)"
    return (
        f"{name:28s} N={len(values):<4d} p50={percentile(values, 50):8.1f} ms  p90={percentile(values, 90):8.1f}  "
        f"p95={percentile(values, 95):8.1f}  max={max(values):8.1f}  {target}"
    )


def load(paths: list[str]) -> list[dict]:
    spans = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        spans.extend(data["spans"] if isinstance(data, dict) else data)
    return spans


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    spans = load(argv)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        by_name[span["name"]].append(span)
    turn_ms = [span["duration_ms"] for span in by_name["turn"] if span.get("duration_ms") is not None]
    llm_ms = [span["duration_ms"] for span in by_name["llm"] if span.get("duration_ms") is not None]
    tool_ms = [span["duration_ms"] for name, items in by_name.items() if name.startswith("tool:") for span in items]
    tts_first = []
    for span in by_name["tts"]:
        events = {event["name"]: event["time_ns"] for event in span["events"]}
        if "T4_text_forwarded_to_tts" in events and "T5_first_audio_chunk" in events:
            tts_first.append((events["T5_first_audio_chunk"] - events["T4_text_forwarded_to_tts"]) / 1e6)
    ttfa = [
        span["attributes"]["ttfa_playback_start_ms"]
        for span in by_name["client_playback"]
        if "ttfa_playback_start_ms" in span["attributes"]
    ]
    print(f"spans: {len(spans)} from {len(argv)} file(s)")
    print(summary("NFR-02 turn (T1→reply)", turn_ms, "target p50 ≤ 2000, p95 ≤ 3500 (not yet measured)"))
    print(summary("LLM plan (T2→T3)", llm_ms, "budget 225→650"))
    print(summary("tool calls", tool_ms))
    print(summary("TTS first audio (T4→T5)", tts_first, "budget 80→185"))
    print(summary("NFR-01b TTFA (T6-T0)", ttfa, "ceiling 755 (adopted external benchmark); browser playback start"))
    print(
        f"NFR-01a sample size: {len(ttfa)} turns with T0 and T6 (planning assumption ≥ 150): "
        f"{'reached' if len(ttfa) >= 150 else 'NOT reached — report as a partial measurement'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
