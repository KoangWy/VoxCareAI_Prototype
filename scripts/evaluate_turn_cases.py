#!/usr/bin/env python3
"""Summarize paired baseline/Jev endpoint observations from JSONL.

This script does not generate sample observations. With no input it reports
that no measured comparison is available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MODES = {"baseline", "jev"}
REQUIRED = {
    "case_id",
    "condition_id",
    "mode",
    "config_id",
    "annotated_yield_ms",
    "natural_commit_ms",
    "degraded_ms",
    "transcript_sha256",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(item, dict) or REQUIRED - item.keys():
            missing = sorted(REQUIRED - item.keys()) if isinstance(item, dict) else sorted(REQUIRED)
            raise ValueError(f"line {line_no}: missing required fields: {', '.join(missing)}")
        if item["mode"] not in MODES:
            raise ValueError(f"line {line_no}: mode must be baseline or jev")
        rows.append(item)
    return rows


def validate_pairs(rows: Iterable[dict[str, Any]]) -> None:
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["case_id"], row["condition_id"])
        if row["mode"] in pairs[key]:
            raise ValueError(f"duplicate {row['mode']} record for {key}")
        pairs[key].add(row["mode"])
        hashes[key].add(row["transcript_sha256"])
    incomplete = [key for key, modes in pairs.items() if modes != MODES]
    if incomplete:
        raise ValueError(f"unpaired conditions: {incomplete}")
    mismatched = [key for key, values in hashes.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"transcript hashes differ between modes: {mismatched}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_pairs(rows)
    result: dict[str, Any] = {
        "label": "observed results only; no inference beyond supplied cases",
        "case_count": len(rows) // 2,
    }
    for mode in sorted(MODES):
        selected = [row for row in rows if row["mode"] == mode]
        premature = 0
        missed = 0
        delays: list[int] = []
        for row in selected:
            expected = row["annotated_yield_ms"]
            committed = row["natural_commit_ms"]
            if committed is None:
                if expected is not None:
                    missed += 1
                continue
            if expected is None or committed < expected:
                premature += 1
            elif expected is not None:
                delays.append(int(committed) - int(expected))
        result[mode] = {
            "observed_turns": len(selected),
            "premature_endpoint_count": premature,
            "premature_endpoint_rate": premature / len(selected) if selected else None,
            "yielded_but_no_natural_commit_count": missed,
            "yield_to_commit_delay_ms": delays,
            "median_yield_to_commit_delay_ms": statistics.median(delays) if delays else None,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL actual observations; each case/condition needs both modes")
    args = parser.parse_args(argv)
    if args.input is None:
        print("No trial observations supplied. No accuracy or latency result is available.", file=sys.stderr)
        print("Use --input <observed-results.jsonl> after collecting paired runs.", file=sys.stderr)
        return 0
    try:
        summary = summarize(load_jsonl(args.input))
    except (OSError, ValueError) as exc:
        print(f"Cannot summarize comparison: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
