"""NFR-04 uptime probe: ``GET /healthz?wss=1`` every 60 s, with the WSS handshake bounded to 2.0 s.

    uv run python scripts/uptime_probe.py --url http://127.0.0.1:8000 --interval 60 --duration 3600 --out uptime.jsonl

A probe counts as up only when the stores respond and the voice WebSocket
handshake completes within 2.0 s. The report is the measured uptime over the run
(target ≥ 99.5 %, not yet measured); nothing is extrapolated beyond the run.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import httpx


def probe(url: str) -> dict:
    started = time.perf_counter()
    try:
        response = httpx.get(f"{url}/healthz", params={"wss": 1}, timeout=3.0)
        body = response.json()
        handshake = body.get("components", {}).get("wss_handshake", {})
        up = response.status_code == 200 and handshake.get("ok") is True and handshake.get("handshake_ms", 9e9) <= 2000
        return {
            "up": up,
            "status": response.status_code,
            "components": body.get("components"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:
        return {
            "up": False,
            "error": type(exc).__name__,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=3600.0, help="seconds to run")
    parser.add_argument("--out", default=None, help="append one JSON line per probe")
    args = parser.parse_args()
    deadline = time.monotonic() + args.duration
    probes = up = 0
    while True:
        result = {"ts": datetime.now(UTC).isoformat(), **probe(args.url)}
        probes += 1
        up += bool(result["up"])
        print(json.dumps(result))
        if args.out:
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
        if time.monotonic() + args.interval > deadline:
            break
        time.sleep(args.interval)
    print(f"uptime {up}/{probes} probes = {100 * up / probes:.2f} % (target >= 99.5 %; measured over this run only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
