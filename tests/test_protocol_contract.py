"""Protocol contract: every server message type has a client decision.

Parses the client handler table (`HANDLER_OWNER`) out of
`static/voice-protocol.js` with a small regex and asserts it matches the
machine-readable server truth (`src/voxcare/voice/protocol.py`) exactly —
both directions. Adding a server message type without a client decision (or
leaving a dead name in the client table) fails here first.

Ownership values are `reducer` (reduceMessage in voice-protocol.js) or
`app` (the DOM glue in app.js owns it deliberately: streaming
audio/transcript frames). Either counts as a client decision — the point is
that the name was considered, not that it lives in one file.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from voxcare.voice import protocol as vm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_JS = ROOT / "static" / "voice-protocol.js"

HANDLER_OWNER_RE = re.compile(
    r"export\s+const\s+HANDLER_OWNER\s*=\s*Object\.freeze\(\{(?P<body>.*?)\}\);",
    re.DOTALL,
)
ENTRY_RE = re.compile(r"(?P<name>[A-Za-z_][\w]*)\s*:\s*['\"](?P<owner>reducer|app)['\"]")


def _parse_handler_owner() -> dict[str, str]:
    source = PROTOCOL_JS.read_text(encoding="utf-8")
    match = HANDLER_OWNER_RE.search(source)
    assert match, "HANDLER_OWNER table missing from static/voice-protocol.js"
    entries = dict(ENTRY_RE.findall(match.group("body")))
    assert entries, "HANDLER_OWNER table parsed empty — regex out of sync?"
    unknown = {owner for owner in entries.values()} - {"reducer", "app"}
    assert not unknown, f"unknown handler owners: {sorted(unknown)}"
    return entries


class ProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler_owner = _parse_handler_owner()

    def test_every_server_type_has_a_client_decision(self):
        missing = [name for name in vm.SERVER_MESSAGE_TYPES if name not in self.handler_owner]
        self.assertEqual(missing, [], f"client has no case for server types: {missing}")

    def test_no_dead_client_names_without_a_server_type(self):
        dead = [name for name in self.handler_owner if name not in vm.SERVER_MESSAGE_TYPES]
        self.assertEqual(dead, [], f"dead client cases with no server type: {dead}")

    def test_retired_names_stay_retired(self):
        for dead in ("turn_deadline", "policy_result", "playback_failed"):
            self.assertNotIn(dead, self.handler_owner, f"retired name {dead!r} must not regain a handler")

    def test_protocol_version_constant_matches_server(self):
        source = PROTOCOL_JS.read_text(encoding="utf-8")
        match = re.search(r"export\s+const\s+PROTOCOL_VERSION\s*=\s*(\d+)\s*;", source)
        self.assertIsNotNone(match, "PROTOCOL_VERSION missing from static/voice-protocol.js")
        assert match is not None
        self.assertEqual(int(match.group(1)), vm.PROTOCOL_VERSION)

    def test_key_messages_are_reducer_owned(self):
        for name in ("session", "turn_control", "audio_chunk", "interrupt", "error"):
            self.assertEqual(self.handler_owner[name], "reducer", name)

    def test_streaming_frames_are_app_owned(self):
        for name in ("voice_started", "transcript_partial", "transcript_final", "audio_start", "audio_end"):
            self.assertEqual(self.handler_owner[name], "app", f"{name} must stay app-owned (streaming glue)")
