import asyncio
import unittest

from voxcare.turns import JevJudgment, TurnController, TurnInput


class TurnControllerTests(unittest.IsolatedAsyncioTestCase):
    def make_input(self, *, now=0, epoch=1, transcript="Can you track ORD-1002?", **overrides):
        values = {
            "session_id": "session-1",
            "turn_epoch": epoch,
            "transcript": transcript,
            "conversation_history": ({"role": "assistant", "text": "What can I help with?"},),
            "assistant_last_message": "What can I help with?",
            "silence_ms": 800,
            "asr_final": True,
            "stable_word_ratio": 1.0,
            "local_vad_quiet": True,
            "now_ms": now,
        }
        values.update(overrides)
        return TurnInput(**values)

    async def test_complete_grounded_turn_commits_only_with_quiet_final_input(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.91, holding_floor=0.02)

        controller = TurnController(judge=judge)
        decision = await controller.evaluate(self.make_input())

        self.assertEqual(decision.state, "COMMIT")
        self.assertEqual(decision.turn_epoch, 1)
        self.assertEqual(decision.jeval_noncommit_count, 0)

    async def test_nonquiet_partial_is_not_sent_to_jev_or_committed(self):
        async def judge(_):
            self.fail("Jev must only be prompted when local VAD reports quiet")

        controller = TurnController(judge=judge)
        decision = await controller.evaluate(self.make_input(local_vad_quiet=False, silence_ms=0, asr_final=False))

        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "caller_speaking")
        self.assertEqual(decision.jeval_noncommit_count, 0)

    async def test_second_noncommit_degrades_with_audit_and_no_commit(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.40, holding_floor=0.2)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        second = await controller.evaluate(self.make_input(now=100))

        self.assertEqual(first.state, "HOLD")
        self.assertEqual(first.jeval_noncommit_count, 1)
        self.assertEqual(second.state, "DEGRADED")
        self.assertEqual(second.jeval_noncommit_count, 2)
        self.assertEqual(second.audit_event["action_name"], "TURN_COMPLETION_DEGRADED")
        self.assertEqual(second.audit_event["outcome_status"], "PENDING")
        self.assertIsNone(second.audit_event["mutation_idempotency_key"])

    async def test_deadline_is_absolute_and_timeout_does_not_extend_it(self):
        calls = 0

        async def judge(_):
            nonlocal calls
            calls += 1
            raise TimeoutError

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=10))
        deadline = first.hold_deadline_ms
        self.assertEqual(first.state, "HOLD")
        self.assertEqual(first.jeval_noncommit_count, 1)
        self.assertEqual(deadline, 3010)

        decision = await controller.on_hold_deadline("session-1", 1, first.hold_episode_id, now_ms=deadline)
        self.assertEqual(decision.state, "DEGRADED")
        self.assertEqual(decision.jeval_noncommit_count, 1)
        self.assertEqual(calls, 1)

    async def test_hold_entry_arms_deadline_even_without_a_jev_result(self):
        async def judge(_):
            self.fail("a non-quiet HOLD must not call Jev")

        controller = TurnController(judge=judge)
        held = await controller.evaluate(self.make_input(now=100, local_vad_quiet=False, silence_ms=0, asr_final=False))
        expired = await controller.on_hold_deadline("session-1", 1, held.hold_episode_id, now_ms=3100)

        self.assertEqual(held.state, "HOLD")
        self.assertEqual(held.hold_deadline_ms, 3100)
        self.assertEqual(expired.state, "DEGRADED")
        self.assertEqual(expired.jeval_noncommit_count, 0)
        self.assertTrue(expired.audit_event["request_payload_json"]["decision_request_id"])

    async def test_genuine_vad_speech_resets_budget_in_same_episode(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.4, holding_floor=0.7)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        reset = await controller.on_genuine_speech(
            self.make_input(now=1200, transcript="Let me think", source="local_vad")
        )

        self.assertEqual(reset.state, "HOLD")
        self.assertEqual(reset.hold_episode_id, first.hold_episode_id)
        self.assertEqual(reset.jeval_noncommit_count, 0)
        self.assertEqual(reset.hold_deadline_ms, 4200)
        self.assertNotEqual(reset.decision_request_id, first.decision_request_id)

    async def test_genuine_vad_speech_after_degradation_rearms_a_fresh_episode(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.40, holding_floor=0.2)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        second = await controller.evaluate(self.make_input(now=100))
        self.assertEqual(second.state, "DEGRADED")

        rearm = await controller.on_genuine_speech(
            self.make_input(now=500, transcript="Let me try again", source="local_vad")
        )
        self.assertEqual(rearm.state, "HOLD")
        self.assertEqual(rearm.reason, "genuine_vad_speech")
        self.assertNotEqual(rearm.hold_episode_id, first.hold_episode_id)
        self.assertEqual(rearm.jeval_noncommit_count, 0)

        # The fresh episode evaluates again instead of answering from the
        # degraded cache; the low-score judge noncommits it with budget 1/2.
        again = await controller.evaluate(self.make_input(now=900))
        self.assertEqual(again.state, "HOLD")
        self.assertEqual(again.hold_episode_id, rearm.hold_episode_id)
        self.assertEqual(again.jeval_noncommit_count, 1)

    async def test_genuine_vad_speech_at_deadline_degrades_without_reset(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.4, holding_floor=0.7)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        speech = await controller.on_genuine_speech(
            self.make_input(now=3000, transcript="I have one more thing", source="local_vad")
        )

        self.assertEqual(first.state, "HOLD")
        self.assertEqual(speech.state, "DEGRADED")
        self.assertEqual(speech.reason, "hold_deadline")
        self.assertEqual(speech.hold_episode_id, first.hold_episode_id)
        self.assertEqual(speech.hold_deadline_ms, 3000)
        self.assertEqual(speech.audit_event["request_payload_json"]["reason"], "hold_deadline")

    async def test_genuine_vad_speech_after_deadline_degrades_without_reset(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.4, holding_floor=0.7)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        speech = await controller.on_genuine_speech(
            self.make_input(now=3001, transcript="I have one more thing", source="local_vad")
        )

        self.assertEqual(first.state, "HOLD")
        self.assertEqual(speech.state, "DEGRADED")
        self.assertEqual(speech.reason, "hold_deadline")
        self.assertEqual(speech.hold_episode_id, first.hold_episode_id)
        self.assertEqual(speech.hold_deadline_ms, 3000)
        self.assertEqual(speech.audit_event["request_payload_json"]["hold_elapsed_ms"], 3001)

    async def test_late_result_after_vad_reset_is_quarantined(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def judge(_):
            started.set()
            await release.wait()
            return JevJudgment(turn_complete=0.99, holding_floor=0.0)

        controller = TurnController(judge=judge)
        pending = asyncio.create_task(controller.evaluate(self.make_input(now=0)))
        await started.wait()
        reset = await controller.on_genuine_speech(
            self.make_input(now=500, transcript="and another thing", source="local_vad")
        )
        release.set()
        late = await pending

        self.assertEqual(reset.state, "HOLD")
        self.assertEqual(late.state, "HOLD")
        self.assertEqual(late.reason, "stale_result")
        self.assertEqual(late.hold_episode_id, reset.hold_episode_id)

    async def test_cancelling_evaluation_cancels_its_provider_task(self):
        started = asyncio.Event()
        judge_cancelled = asyncio.Event()

        async def judge(_):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                judge_cancelled.set()
                raise

        controller = TurnController(judge=judge)
        evaluation = asyncio.create_task(controller.evaluate(self.make_input()))
        await started.wait()
        evaluation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await evaluation
        await asyncio.sleep(0)

        self.assertTrue(judge_cancelled.is_set())

    async def test_new_transcript_version_invalidates_pending_jev_without_spending_budget(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def judge(_):
            started.set()
            await release.wait()
            return JevJudgment(turn_complete=0.1, holding_floor=0.9)

        controller = TurnController(judge=judge)
        evaluation = asyncio.create_task(controller.evaluate(self.make_input()))
        await started.wait()
        controller.invalidate_pending("session-1")
        release.set()
        result = await evaluation

        self.assertEqual(result.state, "HOLD")
        self.assertEqual(result.reason, "stale_result")
        self.assertEqual(result.jeval_noncommit_count, 0)

    async def test_caller_retry_uses_fresh_episode_and_records_fallback_resolution(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.1, holding_floor=0.9)

        controller = TurnController(judge=judge)
        first = await controller.evaluate(self.make_input(now=0))
        degraded = await controller.evaluate(self.make_input(now=10))
        retry = await controller.resolve_fallback("session-1", 1, degraded.hold_episode_id, "CALLER_RETRY", now_ms=20)

        self.assertEqual(degraded.state, "DEGRADED")
        self.assertEqual(retry.state, "HOLD")
        self.assertNotEqual(retry.hold_episode_id, first.hold_episode_id)
        self.assertEqual(retry.turn_epoch, 1)
        self.assertEqual(retry.jeval_noncommit_count, 0)
        self.assertEqual(retry.audit_event["action_name"], "TURN_COMPLETION_FALLBACK_RESOLVED")
        self.assertEqual(retry.audit_event["outcome_status"], "PENDING")

    async def test_manual_finish_is_caller_controlled_eot_not_a_write(self):
        async def judge(_):
            return JevJudgment(turn_complete=0.0, holding_floor=1.0)

        controller = TurnController(judge=judge)
        await controller.evaluate(self.make_input(now=0))
        degraded = await controller.evaluate(self.make_input(now=1))
        resolved = await controller.resolve_fallback(
            "session-1", 1, degraded.hold_episode_id, "MANUAL_FINISH", now_ms=2
        )

        self.assertEqual(resolved.state, "FALLBACK_RESOLVED")
        self.assertEqual(resolved.audit_event["action_name"], "TURN_COMPLETION_FALLBACK_RESOLVED")
        self.assertEqual(resolved.audit_event["outcome_status"], "SUCCESS")
        self.assertIsNone(resolved.audit_event["mutation_idempotency_key"])


if __name__ == "__main__":
    unittest.main()
