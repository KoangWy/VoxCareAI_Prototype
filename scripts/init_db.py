"""Create (or reset) the PostgreSQL schema and seed fixtures, and build the Chroma policy index.

    uv run python scripts/init_db.py            # create when missing, then report
    uv run python scripts/init_db.py --reset    # drop the schema and reseed (fresh demo fixtures)

Reads the same settings as the app (``DATABASE_URL``, ``VOXCARE_DB_SCHEMA``,
``CHROMA_PATH``, ``VOXCARE_POLICY_EMBEDDING``; ``.env`` is loaded). Without
``DATABASE_URL`` it starts the embedded PostgreSQL in ``data/pg`` when the
optional ``localdb`` group is installed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

from voxcare.config import load_data_settings  # noqa: E402
from voxcare.runtime import build_runtime  # noqa: E402


async def main(reset: bool) -> int:
    settings = load_data_settings()
    if reset:
        settings = type(settings)(**{**settings.__dict__, "db_reset_on_start": True})
    runtime = await build_runtime(settings)
    try:
        orders = [
            await runtime.repository.get_order_by_code(code)
            for code in ("ORD-1001", "ORD-1002", "ORD-1003", "ORD-2002")
        ]
        print("stores:", runtime.describe())
        print("orders:", [(o["order_code"], o["status"], o["version"]) for o in orders if o])
        print("health:", await runtime.health())
    finally:
        await runtime.aclose()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reset", action="store_true", help="drop and reseed the schema")
    raise SystemExit(asyncio.run(main(parser.parse_args().reset)))
