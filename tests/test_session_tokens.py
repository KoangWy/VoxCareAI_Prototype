"""``VoxCare-Session-Token`` (guide §9.1, §15): HS256 JWT on every REST call and on the voice handshake."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voxcare.app as api
from voxcare.security.tokens import InvalidToken, TokenService

NOW = datetime(2026, 11, 3, tzinfo=UTC)
SECRET = "unit-test-secret-0123456789abcdefghijklmnop"


def test_token_round_trip_expiry_tamper_and_mismatch():
    service = TokenService(SECRET, ttl_s=1800)
    token = service.issue("SES-1", "CUST-001", now=NOW)
    claims = service.verify(token, now=NOW + timedelta(seconds=10), session_id="SES-1")
    assert (claims.session_id, claims.customer_id) == ("SES-1", "CUST-001")
    assert claims.exp == int((NOW + timedelta(seconds=1800)).timestamp())  # exp ≤ session TTL
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    with pytest.raises(InvalidToken):
        service.verify(token, now=NOW + timedelta(seconds=1801))
    with pytest.raises(InvalidToken):
        service.verify(token, now=NOW, session_id="SES-2")
    with pytest.raises(InvalidToken):
        TokenService("another-secret-0123456789abcdefghijklmn").verify(token, now=NOW)
    forged = jwt.encode(
        {"session_id": "SES-1", "customer_id": "CUST-002", "exp": 9_999_999_999}, "x" * 40, algorithm="HS256"
    )
    with pytest.raises(InvalidToken):
        service.verify(forged, now=NOW)
    unsigned = jwt.encode({"session_id": "SES-1", "exp": 9_999_999_999}, None, algorithm="none")
    with pytest.raises(InvalidToken):
        service.verify(unsigned, now=NOW)
    assert service.verify(service.issue("SES-1", None, now=NOW), now=NOW).customer_id is None
    with pytest.raises(ValueError):
        TokenService("short")


def test_rest_routes_require_the_session_token():
    with TestClient(api.app) as client:
        session = client.post("/session").json()
        sid, token = session["session_id"], session["token"]
        assert session["token_header"] == "VoxCare-Session-Token" and session["customer_id"] == "CUST-001"
        assert client.get(f"/api/sessions/{sid}").status_code == 401
        assert client.get(f"/api/sessions/{sid}", headers={"VoxCare-Session-Token": "nope"}).status_code == 401
        assert client.get(f"/api/sessions/{sid}", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        other = client.post("/api/sessions").json()
        assert (
            client.post(
                f"/api/sessions/{sid}/turns", json={"text": "hi"}, headers={"VoxCare-Session-Token": other["token"]}
            ).status_code
            == 401
        )
        assert client.get("/api/sessions/SES-missing", headers={"VoxCare-Session-Token": token}).status_code == 404
        refreshed = client.post(f"/api/sessions/{sid}/token", headers={"VoxCare-Session-Token": token}).json()
        assert (
            client.get(f"/api/sessions/{sid}/audit", headers={"VoxCare-Session-Token": refreshed["token"]}).status_code
            == 200
        )


def test_unverified_session_and_fixture_session_options():
    with TestClient(api.app) as client:
        unverified = client.post("/api/sessions", json={"verified": False}).json()
        assert unverified["customer_id"] is None and unverified["verified"] is False
        claims = jwt.decode(unverified["token"], options={"verify_signature": False})
        assert claims["customer_id"] is None and claims["session_id"] == unverified["session_id"]
        fixture = client.post("/api/sessions", json={"fixture_session_id": "SES-VOX-8839203"}).json()
        assert fixture["session_id"] == "SES-VOX-8839203" and fixture["fixture"] is True
        assert client.post("/api/sessions", json={"fixture_session_id": "SES-NOPE"}).status_code == 422


def test_voice_socket_rejects_without_a_valid_token():
    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        for url in (
            f"/ws/sessions/{session['session_id']}",
            f"/ws/sessions/{session['session_id']}?token=bad",
            "/ws/voice",
            "/ws/voice?token=bad",
        ):
            with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect(url) as socket:
                socket.receive_json()
            assert closed.value.code == 4401
        with client.websocket_connect(f"/ws/voice?token={session['token']}") as socket:
            hello = socket.receive_json()
            assert hello["type"] == "session" and hello["session"]["session_id"] == session["session_id"]


def test_verification_reissues_a_token_bound_to_the_customer(monkeypatch):
    from voxcare.contracts import PlannedToolCall

    class Planner:
        def __init__(self):
            self.turns = [[PlannedToolCall("lookup_order_details", {"order_code": "ORD-1001"})]] * 2

        async def plan_turn(self, **_):
            return self.turns.pop(0) if self.turns else []

    monkeypatch.setattr(api.boundary, "dialogue_factory", Planner)
    with TestClient(api.app) as client:
        session = client.post("/api/sessions", json={"verified": False}).json()
        headers = {"VoxCare-Session-Token": session["token"]}
        first = client.post(
            f"/api/sessions/{session['session_id']}/turns", json={"text": "Check ORD-1001"}, headers=headers
        ).json()
        assert first["outcome"] == "NEEDS_VERIFICATION" and "session_token" not in first
        second = client.post(
            f"/api/sessions/{session['session_id']}/turns", json={"text": "0900000001"}, headers=headers
        ).json()
        assert second["customer_id"] == "CUST-001"
        claims = jwt.decode(second["session_token"], options={"verify_signature": False})
        assert claims["customer_id"] == "CUST-001"
