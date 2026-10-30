import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_turn_cases.py"
SPEC = importlib.util.spec_from_file_location("evaluate_turn_cases", MODULE_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationSummaryTests(unittest.TestCase):
    def record(self, mode, commit):
        return {
            "case_id": "unfinished_clause",
            "condition_id": "recording-1",
            "mode": mode,
            "config_id": f"{mode}-v1",
            "annotated_yield_ms": 1200,
            "natural_commit_ms": commit,
            "degraded_ms": None,
            "transcript_sha256": "same-asr-transcript-hash",
        }

    def test_reports_premature_and_late_outcomes_separately(self):
        result = evaluation.summarize([self.record("baseline", 900), self.record("jev", 1500)])

        self.assertEqual(result["baseline"]["premature_endpoint_count"], 1)
        self.assertEqual(result["jev"]["premature_endpoint_count"], 0)
        self.assertEqual(result["jev"]["yield_to_commit_delay_ms"], [300])
        self.assertEqual(result["case_count"], 1)

    def test_requires_paired_identical_transcript_conditions(self):
        baseline = self.record("baseline", 1200)
        jev = self.record("jev", 1300)
        jev["transcript_sha256"] = "different"

        with self.assertRaisesRegex(ValueError, "hashes differ"):
            evaluation.summarize([baseline, jev])


if __name__ == "__main__":
    unittest.main()
