"""OpenTelemetry spans per turn (guide §9.5, §16; NFR-02).

A ``turn`` span (attributes ``session_id``, ``turn_epoch``) contains ``asr``,
``endpointing``, ``llm``, ``tool:<name>`` and ``tts`` children; T1…T5 are span
events, and the client-reported T0/T6 arrive as events on the same turn.

``setup_tracing`` installs an SDK tracer provider with an in-memory ring buffer
(``RECENT_SPANS``) that ``GET /api/sessions/{id}/spans`` and
``scripts/nfr_latency_report.py`` read. Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
install ``opentelemetry-exporter-otlp`` to also ship spans to a collector.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace

logger = logging.getLogger("voxcare.otel")
TRACER_NAME = "voxcare"
_MAX_SPANS = 5000


class RecentSpanStore:
    """Thread-safe ring buffer of finished spans as plain dicts."""

    def __init__(self, capacity: int = _MAX_SPANS) -> None:
        self._spans: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, span: dict[str, Any]) -> None:
        with self._lock:
            self._spans.append(span)

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [span for span in self._spans if span["attributes"].get("session_id") == session_id]

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


RECENT_SPANS = RecentSpanStore()
_installed = False


def setup_tracing() -> None:
    """Install the SDK provider once per process (idempotent)."""
    global _installed
    if _installed:
        return
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _RingExporter(SpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            for span in spans:
                parent = span.parent.span_id if span.parent is not None else None
                RECENT_SPANS.add(
                    {
                        "name": span.name,
                        "trace_id": f"{span.context.trace_id:032x}",
                        "span_id": f"{span.context.span_id:016x}",
                        "parent_span_id": f"{parent:016x}" if parent else None,
                        "start_ns": span.start_time,
                        "end_ns": span.end_time,
                        "duration_ms": (span.end_time - span.start_time) / 1e6 if span.end_time else None,
                        "attributes": dict(span.attributes or {}),
                        "events": [
                            {"name": event.name, "time_ns": event.timestamp, "attributes": dict(event.attributes or {})}
                            for event in span.events
                        ],
                    }
                )
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_RingExporter()))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp is not installed")
    trace.set_tracer_provider(provider)
    _installed = True


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


__all__ = ["RECENT_SPANS", "RecentSpanStore", "setup_tracing", "tracer"]
