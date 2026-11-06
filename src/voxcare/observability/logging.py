"""Structured JSON logs, emitted only after PII masking (guide §16, NFR-05).

``configure_logging`` installs one handler on the root logger with
:class:`voxcare.security.pii.MaskingLogFilter` (every message is masked before
any formatter sees it) and a JSON formatter (``VOXCARE_LOG_FORMAT=json``, the
default) or the plain text format (``text``).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from ..security.pii import MaskingLogFilter

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value if isinstance(value, str | int | float | bool | type(None)) else str(value)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_voxcare", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler._voxcare = True  # type: ignore[attr-defined]
    handler.addFilter(MaskingLogFilter())
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    for name in ("uvicorn.access", "uvicorn.error"):  # uvicorn logs WebSocket paths, which carry ?token=
        target = logging.getLogger(name)
        if not any(isinstance(item, MaskingLogFilter) for item in target.filters):
            target.addFilter(MaskingLogFilter())
    # The embedded PostgreSQL logs every pg_ctl call at INFO; keep only its warnings and errors.
    logging.getLogger("pgserver").setLevel(logging.WARNING)


__all__ = ["JsonFormatter", "configure_logging"]
