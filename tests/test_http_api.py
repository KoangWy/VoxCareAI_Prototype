from fastapi.testclient import TestClient

import voxcare.app as api


def test_config_and_session_report_missing_direct_providers(monkeypatch):
    for name in (
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    with TestClient(api.app) as client:
        config = client.get("/api/config").json()
        session = client.post("/api/sessions").json()
        turn = client.post(f"/api/sessions/{session['session_id']}/turns", json={"text": "Track ORD-1002"}).json()

    assert config["voice_ready"] is False
    assert config["providers"]["conversation"]["configured"] is False
    assert session["customer_id"] == "CUST-001"
    assert session["state"] == "conversation_provider_unavailable"
    assert turn["outcome"] == "DEPENDENCY_UNAVAILABLE"
    assert turn["assistant_text"] is None


def test_reset_ends_prior_session_and_creates_fresh_scope(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(api.app) as client:
        old = client.post("/api/sessions").json()
        new = client.post(f"/api/sessions/{old['session_id']}/reset").json()
        old_lookup = client.get(f"/api/sessions/{old['session_id']}")

    assert old["session_id"] != new["session_id"]
    assert old_lookup.status_code == 404
    assert new["customer_id"] == "CUST-001"


def test_voice_websocket_reports_missing_asr_without_fake_success(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(f"/ws/sessions/{session['session_id']}") as websocket:
            assert websocket.receive_json()["type"] == "session"
            websocket.send_json({"type": "start", "sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"})
            response = websocket.receive_json()

    assert response == {"type": "provider_unavailable", "provider": "asr", "error": "DEEPGRAM_NOT_CONFIGURED"}


def test_text_request_size_is_bounded():
    with TestClient(api.app) as client:
        response = client.post(
            "/api/sessions/anything/turns", content="x" * (65 * 1024), headers={"content-type": "application/json"}
        )
    assert response.status_code == 413


def test_unrelated_host_model_environment_does_not_change_voxcare_defaults(monkeypatch):
    monkeypatch.delenv("VOXCARE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("VOXCARE_ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "host-only-model")
    monkeypatch.setenv("ANTHROPIC_MODEL", "muse-spark-1.3-contributor")
    with TestClient(api.app) as client:
        providers = client.get("/api/config").json()["providers"]
    assert providers["conversation"]["model"] == "gpt-6-luna"
    assert providers["conversation_fallback"]["model"] == "claude-haiku-5-5"


def test_handoff_endpoint_records_request_without_voice_or_provider(monkeypatch):
    for name in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(api.app) as client:
        session = client.post("/api/sessions").json()
        response = client.post(f"/api/sessions/{session['session_id']}/handoff", json={"reason": "r" * 200})
        too_long = client.post(f"/api/sessions/{session['session_id']}/handoff", json={"reason": "r" * 201})
        trace = client.get(f"/api/sessions/{session['session_id']}/trace").json()
        missing = client.post("/api/sessions/unknown/handoff", json={}).status_code

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "HANDOFF_REQUEST_RECORDED"
    assert body["status"] == "REQUEST_RECORDED"
    assert "does not make a live transfer" in body["message"]
    audit = next(item for item in trace["events"] if item.get("action_name") == "HANDOFF_REQUEST_RECORDED")
    assert audit["request_payload_json"] == {"reason": "r" * 200, "context": ""}
    assert missing == 404
    assert too_long.status_code == 422


def test_invalid_provider_variable_reports_config_error_without_crashing(monkeypatch):
    monkeypatch.setenv("VOXCARE_ASR_PROVIDER", "whispery")
    with TestClient(api.app) as client:
        config = client.get("/api/config")
        session = client.post("/api/sessions")

    assert config.status_code == 200
    assert config.json()["voice_ready"] is False
    assert config.json()["providers"]["asr"]["configured"] is False
    assert "VOXCARE_ASR_PROVIDER" in config.json()["providers"]["asr"]["detail"]
    assert session.status_code == 500
    assert session.json()["detail"] == "PROVIDER_CONFIG_INVALID:VOXCARE_ASR_PROVIDER"


def test_config_reports_barge_in_mode(monkeypatch):
    monkeypatch.setenv("VOXCARE_BARGE_IN", "semantic")
    with TestClient(api.app) as client:
        assert client.get("/api/config").json()["barge_in"] == "semantic"


def test_openrouter_sessions_share_one_judge_client(monkeypatch):
    from voxcare.http.bootstrap import build_turn_controller

    monkeypatch.setenv("VOXCARE_TURN_MODE", "openrouter")
    first, second = build_turn_controller(), build_turn_controller()
    assert first is not second
    assert first._judge is second._judge
