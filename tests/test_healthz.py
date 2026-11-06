"""``/healthz`` (guide §16, NFR-04): 200 when the stores respond, plus a WSS handshake within 2.0 s."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

import voxcare.app as api


def test_healthz_reports_each_store():
    with TestClient(api.app) as client:
        body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {"relational_memory", "fakeredis", "chroma"}
    assert all(item["ok"] for item in body["components"].values())


def test_healthz_is_503_when_a_store_is_down():
    with TestClient(api.app) as client:
        api.boundary.runtime.repository.fail_with = RuntimeError("down")
        response = client.get("/healthz")
        api.boundary.runtime.repository.fail_with = None
    assert response.status_code == 503 and response.json()["status"] == "degraded"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_healthz_confirms_a_wss_handshake_within_two_seconds(live_server):
    started = time.perf_counter()
    response = httpx.get(f"{live_server}/healthz?wss=1", timeout=5)
    elapsed = time.perf_counter() - started
    body = response.json()
    assert response.status_code == 200, body
    assert body["components"]["wss_handshake"]["ok"] is True
    assert body["components"]["wss_handshake"]["handshake_ms"] < 2000 and elapsed < 2.0
