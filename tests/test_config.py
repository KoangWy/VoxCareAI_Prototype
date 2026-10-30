"""Validated turn settings — per-judge defaults, fail-fast errors."""

import unittest

from fastapi.testclient import TestClient

import voxcare.app as api
from voxcare.config import (
    OPENROUTER_HOLD_DEADLINE_MS,
    OPENROUTER_JUDGE_TIMEOUT_MS,
    TYPESAFE_HOLD_DEADLINE_MS,
    TYPESAFE_JUDGE_TIMEOUT_MS,
    ConfigError,
    config_error_code,
    load_turn_settings,
)


class TurnSettingsTests(unittest.TestCase):
    def test_typesafe_direct_defaults_match_spec(self):
        settings = load_turn_settings({"VOXCARE_TURN_MODE": "jev"})
        self.assertEqual(settings.config.jev_timeout_ms, TYPESAFE_JUDGE_TIMEOUT_MS)
        self.assertEqual(settings.config.jev_timeout_ms, 300)
        self.assertEqual(settings.config.hold_deadline_ms, TYPESAFE_HOLD_DEADLINE_MS)
        self.assertEqual(settings.config.hold_deadline_ms, 3000)

    def test_openrouter_defaults_cover_free_route_round_trips(self):
        # the OpenRouter free route measures 0.9-2.1 s; the default
        # budget must clear the reference model's own p50 (~275 ms) by an
        # order of magnitude, and the deadline must absorb ASR finality.
        settings = load_turn_settings({"VOXCARE_TURN_MODE": "openrouter"})
        self.assertEqual(settings.config.jev_timeout_ms, OPENROUTER_JUDGE_TIMEOUT_MS)
        self.assertEqual(settings.config.jev_timeout_ms, 2500)
        self.assertEqual(settings.config.hold_deadline_ms, OPENROUTER_HOLD_DEADLINE_MS)
        self.assertEqual(settings.config.hold_deadline_ms, 9000)
        self.assertGreaterEqual(settings.config.jev_timeout_ms, 275)

    def test_locked_thresholds_stay_documented_defaults(self):
        settings = load_turn_settings({})
        self.assertEqual(settings.mode, "baseline")
        self.assertEqual(settings.config.turn_complete_threshold, 0.72)
        self.assertEqual(settings.config.holding_floor_threshold, 0.50)
        self.assertEqual(settings.config.max_noncommit_evaluations, 2)
        self.assertEqual(settings.barge_in, "off")

    def test_invalid_values_fail_fast_with_field_names(self):
        cases = [
            ({"VOXCARE_TURN_MODE": "nope"}, "VOXCARE_TURN_MODE"),
            ({"VOXCARE_JEV_TIMEOUT_MS": "fast"}, "VOXCARE_JEV_TIMEOUT_MS"),
            ({"VOXCARE_JEV_TIMEOUT_MS": "0"}, "VOXCARE_JEV_TIMEOUT_MS"),
            ({"VOXCARE_JEV_TIMEOUT_MS": "-5"}, "VOXCARE_JEV_TIMEOUT_MS"),
            ({"VOXCARE_HOLD_DEADLINE_MS": "none"}, "VOXCARE_HOLD_DEADLINE_MS"),
            ({"VOXCARE_HOLD_DEADLINE_MS": "-1"}, "VOXCARE_HOLD_DEADLINE_MS"),
            ({"VOXCARE_TURN_COMPLETE_THRESHOLD": "2"}, "VOXCARE_TURN_COMPLETE_THRESHOLD"),
            ({"VOXCARE_TURN_COMPLETE_THRESHOLD": "nan-ish"}, "VOXCARE_TURN_COMPLETE_THRESHOLD"),
            ({"VOXCARE_BARGE_IN": "always"}, "VOXCARE_BARGE_IN"),
        ]
        for env, field in cases:
            with self.subTest(env=env), self.assertRaises(ConfigError) as ctx:
                load_turn_settings(env)
            self.assertIn(field, str(ctx.exception))

    def test_env_overrides_still_win(self):
        settings = load_turn_settings({"VOXCARE_TURN_MODE": "openrouter", "VOXCARE_JEV_TIMEOUT_MS": "2500"})
        self.assertEqual(settings.config.jev_timeout_ms, 2500)


class TurnSettingsValidationTests(unittest.TestCase):
    def test_confirm_window_variables_reach_the_controller_config(self):
        settings = load_turn_settings(
            {"VOXCARE_EOS_CONFIRM_QUIET_MIN_MS": "200", "VOXCARE_EOS_CONFIRM_QUIET_MAX_MS": "900"}
        )
        self.assertEqual((settings.config.confirm_quiet_min_ms, settings.config.confirm_quiet_max_ms), (200, 900))

    def test_cross_field_rule_names_the_variable(self):
        with self.assertRaises(ConfigError) as ctx:
            load_turn_settings({"VOXCARE_EOS_THRESHOLD_FLOOR": "0.9"})
        self.assertTrue(str(ctx.exception).startswith("VOXCARE_EOS_THRESHOLD_FLOOR "))
        self.assertEqual(config_error_code(ctx.exception), "TURN_CONFIG_INVALID:VOXCARE_EOS_THRESHOLD_FLOOR")


class TurnConfigApiTests(unittest.TestCase):
    def test_bogus_mode_is_config_error_never_application_baseline(self):
        with TestClient(api.app) as client:
            import os

            old = os.environ.get("VOXCARE_TURN_MODE")
            os.environ["VOXCARE_TURN_MODE"] = "nonsense"
            try:
                config = client.get("/api/config").json()
            finally:
                if old is None:
                    del os.environ["VOXCARE_TURN_MODE"]
                else:
                    os.environ["VOXCARE_TURN_MODE"] = old
        turn = config["providers"]["turn_decision"]
        self.assertEqual(turn["mode"], "nonsense")
        self.assertNotEqual(turn["name"], "Application baseline")
        self.assertFalse(turn["configured"])
        self.assertIn("VOXCARE_TURN_MODE", turn["error"])

    def test_bogus_timeout_rejects_session_creation_loudly(self):
        with TestClient(api.app) as client:
            import os

            old = os.environ.get("VOXCARE_JEV_TIMEOUT_MS")
            os.environ["VOXCARE_JEV_TIMEOUT_MS"] = "zero"
            try:
                response = client.post("/api/sessions")
            finally:
                if old is None:
                    del os.environ["VOXCARE_JEV_TIMEOUT_MS"]
                else:
                    os.environ["VOXCARE_JEV_TIMEOUT_MS"] = old
        self.assertEqual(response.status_code, 500)
        self.assertIn("VOXCARE_JEV_TIMEOUT_MS", response.json()["detail"])

    def test_typed_turn_stays_judgment_free_by_design(self):
        # POST /turns never routes through the judge — no mic timing
        # exists to judge. Even in a judged mode without keys, a typed turn
        # reports the dialogue dependency, never a turn-mode error.
        with TestClient(api.app) as client:
            import os

            old_mode, old_key = os.environ.get("VOXCARE_TURN_MODE"), os.environ.get("OPENROUTER_API_KEY")
            os.environ["VOXCARE_TURN_MODE"] = "openrouter"
            os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                session = client.post("/api/sessions").json()
                turn = client.post(
                    f"/api/sessions/{session['session_id']}/turns", json={"text": "Track ORD-1002"}
                ).json()
            finally:
                if old_mode is None:
                    del os.environ["VOXCARE_TURN_MODE"]
                else:
                    os.environ["VOXCARE_TURN_MODE"] = old_mode
                if old_key is not None:
                    os.environ["OPENROUTER_API_KEY"] = old_key
        self.assertEqual(turn["outcome"], "DEPENDENCY_UNAVAILABLE")
        self.assertEqual(turn["assistant_text"], None)


if __name__ == "__main__":
    unittest.main()
