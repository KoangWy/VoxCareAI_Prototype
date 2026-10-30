"""Controller behaviour: deadlines, relaxed bar, addressee, barge-in, false-open."""

import asyncio
import unittest

from voxcare.config import load_turn_settings
from voxcare.turns import (
    JevJudgment,
    TurnConfig,
    TurnController,
    TurnInput,
    confirm_quiet_ms_for,
    effective_turn_complete_threshold,
)
from voxcare.turns.prompts import state_from_turn


def scripted_judgment(**scores):
    async def judge(_turn):
        values = {"turn_complete": 0.0, "holding_floor": 0.0, **scores}
        return JevJudgment(**values)

    return judge


def make_input(now=0, **overrides):
    values = {
        "session_id": "session-1",
        "turn_epoch": 1,
        "transcript": "Can you track ORD-1002?",
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


def openrouter_config(**overrides):
    settings = load_turn_settings({"VOXCARE_TURN_MODE": "openrouter"})
    values = {field: getattr(settings.config, field) for field in TurnConfig.__dataclass_fields__}
    values.update(overrides)
    return TurnConfig(**values)


class PerJudgeDefaultsTests(unittest.TestCase):
    def test_two_second_judge_commits_at_openrouter_defaults(self):
        # a healthy-but-slow verdict (2 s) must COMMIT
        # instead of self-degrading on a 2500/9000 budget.
        async def slow_judge(_turn):
            await asyncio.sleep(2.0)
            return JevJudgment(turn_complete=0.95, holding_floor=0.0)

        async def run():
            controller = TurnController(judge=slow_judge, config=openrouter_config())
            return await controller.evaluate(make_input(now=0, silence_ms=1200))

        decision = asyncio.run(run())
        self.assertEqual(decision.state, "COMMIT")
        self.assertEqual(decision.jeval_noncommit_count, 0)

    def test_two_second_judge_times_out_at_typesafe_budget(self):
        async def slow_judge(_turn):
            await asyncio.sleep(2.0)
            return JevJudgment(turn_complete=0.95, holding_floor=0.0)

        async def run():
            controller = TurnController(judge=slow_judge, config=openrouter_config(jev_timeout_ms=300))
            first = await controller.evaluate(make_input(now=0, silence_ms=1200))
            second = await controller.evaluate(make_input(now=100, silence_ms=1300))
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first.reason, "jev_timeout")
        self.assertEqual(second.state, "DEGRADED")


class DeadlineAnchoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_anchors_at_first_quiet_final_not_speech_onset(self):
        controller = TurnController(judge=scripted_judgment(turn_complete=0.4), config=openrouter_config())
        # Speech onset: non-quiet hold arms the episode but must not start the
        # judged clock at a latency the first verdict cannot meet.
        onset = await controller.evaluate(make_input(now=0, local_vad_quiet=False, silence_ms=0, asr_final=False))
        self.assertEqual(onset.hold_deadline_ms, 9000)
        # First quiet+final anchors the absolute deadline there.
        first = await controller.evaluate(make_input(now=4000, silence_ms=700))
        self.assertEqual(first.state, "HOLD")
        self.assertEqual(first.hold_deadline_ms, 13000)

    async def test_quiet_hold_rearms_deadline(self):
        # without re-arming, a 9 s deadline from onset would
        # degrade a healthy slow judgment; each quiet HOLD restarts the clock.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.4), config=openrouter_config())
        await controller.evaluate(
            make_input(now=0, local_vad_quiet=False, silence_ms=0, asr_final=False, transcript="")
        )
        # The first quiet+final anchors a fresh clock at quiet time.
        second = await controller.evaluate(make_input(now=8500, silence_ms=750))
        self.assertEqual(second.state, "HOLD")
        self.assertEqual(second.hold_deadline_ms, 17500)
        # The host timer firing at the old onset-anchored deadline must not
        # degrade: the re-armed clock still has room.
        timer = await controller.on_hold_deadline("session-1", 1, second.hold_episode_id, now_ms=9000)
        self.assertEqual(timer.state, "HOLD")
        self.assertEqual(timer.reason, "deadline_not_reached")

    async def test_episode_map_stays_bounded_across_epochs(self):
        # one entry per session no matter how many epochs pass; a late
        # deadline callback for a superseded episode is stale, never fatal.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.95))
        first = await controller.evaluate(make_input(now=0, turn_epoch=1))
        self.assertEqual(first.state, "COMMIT")
        self.assertEqual(controller.episode_size("session-1"), 1)
        second = await controller.evaluate(make_input(now=100, turn_epoch=2))
        self.assertEqual(second.state, "COMMIT")
        self.assertEqual(controller.episode_size("session-1"), 1)
        stale = await controller.on_hold_deadline("session-1", 1, first.hold_episode_id, now_ms=10_000)
        self.assertEqual(stale.reason, "stale_deadline")


class AddresseeGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_but_unaddressed_never_commits(self):
        # room speech that sounds complete must not dispatch a turn.
        controller = TurnController(
            judge=scripted_judgment(turn_complete=0.95, addressed_to_agent=0.1),
        )
        decision = await controller.evaluate(make_input(transcript="Dinner's ready"))
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "side_talk_hold")
        self.assertEqual(decision.jeval_noncommit_count, 1)

    async def test_complete_and_addressed_commits(self):
        controller = TurnController(
            judge=scripted_judgment(turn_complete=0.95, addressed_to_agent=0.9),
        )
        decision = await controller.evaluate(make_input())
        self.assertEqual(decision.state, "COMMIT")
        self.assertIn("addressed_to_agent", decision.scores or {})

    async def test_taking_floor_flag_only_when_assistant_speaks(self):
        async def judge(turn):
            self.assertEqual(turn.assistant_current_speech, "One more thing.")
            return JevJudgment(turn_complete=0.9, holding_floor=0.0, taking_floor=0.95)

        controller = TurnController(judge=judge)
        decision = await controller.evaluate(make_input(assistant_current_speech="One more thing."))
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "taking_floor_interrupt")
        self.assertTrue(decision.should_interrupt)

    async def test_backchannel_holds_without_interrupting(self):
        controller = TurnController(
            judge=scripted_judgment(turn_complete=0.9, taking_floor=0.1),
        )
        decision = await controller.evaluate(
            make_input(transcript="uh-huh", assistant_current_speech="Your warranty covers two years.")
        )
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "backchannel_hold")
        self.assertFalse(decision.should_interrupt)


class RelaxedBarTests(unittest.IsolatedAsyncioTestCase):
    async def test_mid_sixties_complete_commits_after_sustained_quiet(self):
        # 0.67-0.70 completes under the old fixed 0.72 bar must
        # yield once the mic has been quiet, not stall to DEGRADED.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.68))
        short = await controller.evaluate(make_input(now=0, silence_ms=100))
        self.assertEqual(short.state, "HOLD")
        self.assertEqual(short.reason, "turn_not_complete")
        grown = await controller.evaluate(make_input(now=2900, silence_ms=3000))
        self.assertEqual(grown.state, "COMMIT")

    async def test_holding_floor_disables_relaxation(self):
        # "Hold on a second" + 3 s of quiet must not hand over the floor.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.68, holding_floor=0.9))
        decision = await controller.evaluate(make_input(now=0, silence_ms=3000))
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "caller_holding_floor")

    def test_relaxation_curve_matches_reference(self):
        config = TurnConfig()
        self.assertEqual(effective_turn_complete_threshold(0, config), 0.72)
        self.assertEqual(effective_turn_complete_threshold(1500, config), 0.635)
        self.assertEqual(effective_turn_complete_threshold(3000, config), 0.55)
        self.assertEqual(effective_turn_complete_threshold(9000, config), 0.55)
        self.assertEqual(effective_turn_complete_threshold(3000, config, holding=True), 0.72)

    def test_confirm_window_scales_with_confidence(self):
        config = TurnConfig()
        self.assertEqual(confirm_quiet_ms_for(1.0, config), 350)
        self.assertEqual(confirm_quiet_ms_for(0.0, config), 1000)
        certain = confirm_quiet_ms_for(0.95, config)
        unsure = confirm_quiet_ms_for(0.75, config)
        self.assertLess(certain, unsure)


class ConfirmAndCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsure_complete_waits_for_confirming_quiet(self):
        # a complete-looking single sample holds for a confirming
        # re-judge after more silence instead of committing at once.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.75))
        first = await controller.evaluate(make_input(now=0, silence_ms=400))
        self.assertEqual(first.state, "HOLD")
        self.assertEqual(first.reason, "confirming")
        second = await controller.evaluate(make_input(now=600, silence_ms=1200))
        self.assertEqual(second.state, "COMMIT")

    async def test_sustained_silence_commits_without_new_asr(self):
        # extended silence alone re-judges; the cap closes the turn when
        # nothing points away (addressed, not holding).
        controller = TurnController(judge=scripted_judgment(turn_complete=0.5))
        decision = await controller.evaluate(make_input(now=0, silence_ms=5000))
        self.assertEqual(decision.state, "COMMIT")
        self.assertEqual(decision.reason, "quiet_cap_commit")

    async def test_cap_degrades_side_talk_instead_of_committing(self):
        controller = TurnController(judge=scripted_judgment(turn_complete=0.9, addressed_to_agent=0.1))
        first = await controller.evaluate(make_input(now=0, silence_ms=5000))
        self.assertEqual(first.state, "HOLD")
        self.assertEqual(first.reason, "quiet_cap_unaddressed_or_holding")
        second = await controller.evaluate(make_input(now=100, silence_ms=5100))
        self.assertEqual(second.state, "DEGRADED")

    async def test_cap_is_patient_while_caller_holds_floor(self):
        controller = TurnController(judge=scripted_judgment(turn_complete=0.5, holding_floor=0.9))
        patient = await controller.evaluate(make_input(now=0, silence_ms=12000, transcript="Let me think"))
        self.assertEqual(patient.state, "HOLD")
        self.assertEqual(patient.reason, "quiet_cap_unaddressed_or_holding")


class FalseOpenTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_transcript_never_consumes_budget(self):
        # two silent evaluations must not degrade; the onset snapshot
        # stays as the false-open baseline.
        controller = TurnController(judge=scripted_judgment(turn_complete=0.95))
        first = await controller.evaluate(make_input(now=0, transcript="   "))
        second = await controller.evaluate(make_input(now=100, transcript=""))
        for decision in (first, second):
            self.assertEqual(decision.state, "HOLD")
            self.assertEqual(decision.reason, "false_open_empty_transcript")
            self.assertEqual(decision.jeval_noncommit_count, 0)
        self.assertIsNone(first.audit_event)
        followup = await controller.evaluate(make_input(now=200))
        self.assertEqual(followup.state, "COMMIT")


class JudgeStateTests(unittest.TestCase):
    def test_state_carries_timing_trajectory_and_speech(self):
        # the judge sees trajectory, elapsed time, granular
        # stability, and the assistant's current speech.
        turn = make_input(
            silence_ms=650,
            turn_elapsed_ms=2300,
            transcript_trajectory=(
                {"text": "Track", "ms_ago": 900},
                {"text": "Track ORD", "ms_ago": 400},
            ),
            assistant_current_speech="Your warranty covers two years.",
        )
        state = state_from_turn(turn)
        self.assertIn("timing_notes", state)
        self.assertEqual(state["silence_since_last_word_ms"], 650)
        self.assertIs(state["asr_final"], True)
        self.assertEqual(state["stable_word_ratio"], 1.0)
        self.assertEqual(state["turn_elapsed_ms"], 2300)
        self.assertEqual(
            state["transcript_trajectory"],
            [{"text": "Track", "ms_ago": 900}, {"text": "Track ORD", "ms_ago": 400}],
        )
        self.assertEqual(state["assistant_current_speech"], "Your warranty covers two years.")
        # Current text is dropped from its own trajectory.
        again = state_from_turn(make_input(transcript="hi", transcript_trajectory=({"text": "hi", "ms_ago": 5},)))
        self.assertNotIn("transcript_trajectory", again)


if __name__ == "__main__":
    unittest.main()
