"""Frozen-output inventory, calibrated grading, and durable resume behavior."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / "src/jpeers/data/imposter_facts.json").read_text())["items"]
CASES = json.loads((ROOT / "src/jpeers/data/fact_grader_calibration_v2.json").read_text())["cases"]
GRADER = {"model": "OLMo-2-0325-32B-Instruct", "model_revision": "frozen-test-revision"}
CONDITIONS = ["none", "jspace_light", "matched_k5", "jspace_medium", "matched_k10",
              "jspace_heavy", "matched_k25", "jspace_generated", "matched_generated",
              "jspace_peer_span", "matched_peer_span", "suppress_words", "steer_suspicion"]


class ControlledGrader:
    """Replace only the external model operation; validation and caching are real."""
    def __init__(self, invalid_responses=()):
        self.calls = 0
        self.invalid_responses = set(invalid_responses)
        self.labels = {case["response"]: case["expected"] for case in CASES}

    def generate(self, request):
        self.calls += 1
        data = json.loads(request.messages[1]["content"])
        response, final, review = data["response"], data["final_text"], data["review_text"]
        if response in self.invalid_responses:
            return SimpleNamespace(text="invalid grader output")
        expected = self.labels.get(response)
        if expected is None:
            truth = "true" if data["reference"]["true_claim"] == final else "false"
            expected = {"final_truth": truth, "review_rejects_false": False,
                        "intent_accusation": False, "usable": True, "refusal": False}
        evidence = {key: [] for key in expected}
        for key, value in expected.items():
            positive = key == "final_truth" and value in ("true", "false", "both")
            if positive or value is True:
                evidence[key] = [final if key in ("final_truth", "usable") else
                                 review if key == "review_rejects_false" else response]
        return SimpleNamespace(text=json.dumps({**expected, "evidence": evidence}))


def original_turn(item=BANK[0], condition="none", source="incentive", claim="false",
                  final_false=False):
    return {"index": 0, "agent": "Alice", "model": "qwen3-32b", "endpoint": "qwen3-32b",
            "text": "Final text: " + item["false_claim" if final_false else "true_claim"],
            "messages": [{"role": "user", "content": "Original frozen prompt."}],
            "tag": f"{condition}/{source}/{claim}/{item['id']}",
            "extra": {"condition": condition, "source": source, "claim": claim,
                      "item": item["id"]}}


def write_run(directory, turns, *, experiment="imposter", legacy=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"experiment": experiment,
        "base": "imposter", "live": True, "n_items": 60}))
    (directory / "turns.jsonl").write_text("".join(json.dumps(turn) + "\n" for turn in turns))
    if legacy:
        rows = [{**turn["extra"], "catch_score": 0.5 if turn["extra"]["claim"] == "false" else None,
                 "fluency": 0.75, "reply_has_true": True} for turn in turns]
        (directory / "summary.json").write_text(json.dumps({"live": rows}))


class RescoreTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec("jpeers.followup.rescore")
        self.assertIsNotNone(spec, "frozen semantic rescoring orchestration is missing")
        return importlib.import_module("jpeers.followup.rescore")

    def calibrate(self, directory, grader=None, **kwargs):
        grader = grader or ControlledGrader()
        report = self.module().run_calibration(grader, directory, GRADER, **kwargs)
        self.assertTrue(report["passed"], report.get("mismatches"))
        return report, grader

    def test_calibration_persists_actual_raw_predictions_and_resumes_without_calls(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "calibration"
            report, grader = self.calibrate(output)
            self.assertEqual(grader.calls, 40)
            self.assertEqual(report["n_cases"], 40)
            self.assertEqual(report["grader_info"], GRADER)
            self.assertEqual(len(report["fixture_sha256"]), 64)
            self.assertEqual(json.loads((output / "calibration.json").read_text()), report)
            predictions = [json.loads(line) for line in
                           (output / "calibration_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), 40)
            self.assertIn("raw_response", predictions[0])
            second = module.run_calibration(grader, output, GRADER)
            self.assertTrue(second["passed"])
            self.assertEqual(grader.calls, 40)
            self.assertEqual(second["cache_hits"], 40)
            self.assertEqual(list(output.rglob("*.tmp")), [])

    def test_failed_or_stale_calibration_blocks_before_any_response_is_graded(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            write_run(base / "input", [original_turn()])
            before = grader.calls
            for changes in ({"passed": False}, {"passed": "true"},
                            {"rubric_sha256": "stale"}, {"fixture_sha256": "stale"},
                            {"coverage": 0.0}, {"grade_agreement": 0.0},
                            {"n_valid": 0}, {"n_cases": 39},
                            {"critical_false_positive_ids": ["known_false_positive"]},
                            {"grader_info": {**GRADER, "model_revision": "changed"}}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "calibration"):
                    module.run_rescore(base / "input", base / "output", grader, GRADER,
                                       {**calibration, **changes}, kind="exp2")
            self.assertEqual(grader.calls, before)

    def test_partial_inventory_preserves_text_messages_legacy_and_explicit_bounds(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            turn = original_turn()
            write_run(base / "input", [turn])
            summary = module.run_rescore(base / "input", base / "output", grader, GRADER,
                                         calibration, kind="exp2")
            self.assertEqual(summary["status"], "partial")
            self.assertFalse(summary["authoritative"])
            self.assertEqual(summary["expected_reviews"], 360)
            self.assertEqual(summary["n_reviews"], 1)
            self.assertEqual(summary["missing_reviews"], 359)
            rows = [json.loads(line) for line in (base / "output/scored_rows.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["text"], turn["text"])
            self.assertEqual(rows[0]["messages"], turn["messages"])
            self.assertEqual(rows[0]["legacy"]["catch_score"], 0.5)
            self.assertEqual(rows[0]["semantic_grade"]["judgment"]["semantic_final_correct"], 1)
            metric = summary["conditions"]["none"]["semantic_final_correct"]
            self.assertEqual(metric["n_unknown"], 359)
            self.assertEqual(metric["bounds"], {"lower": 1 / 360, "upper": 1})
            inventory = json.loads((base / "output/inventory.json").read_text())
            self.assertEqual(len(inventory["missing_keys"]), 359)
            self.assertTrue(inventory["input_files"])
            progress = json.loads((base / "output/progress.json").read_text())
            self.assertEqual(progress["processed"], 1)

    def test_duplicates_and_tag_metadata_mismatches_fail_before_grading(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            before = grader.calls
            duplicate = original_turn()
            write_run(base / "input", [duplicate, copy.deepcopy(duplicate)])
            with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                module.run_rescore(base / "input", base / "out", grader, GRADER,
                                   calibration, kind="exp2")
            wrong = original_turn()
            wrong["extra"]["claim"] = "true"
            write_run(base / "input", [wrong])
            with self.assertRaisesRegex(ValueError, "tag|metadata"):
                module.run_rescore(base / "input", base / "out", grader, GRADER,
                                   calibration, kind="exp2")
            self.assertEqual(grader.calls, before)

    def test_invalid_grades_remain_unknown_on_resume_and_export_for_review(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            turn = original_turn(final_false=True)
            grader.invalid_responses.add(turn["text"])
            write_run(base / "input", [turn])
            report = module.run_rescore(base / "input", base / "out", grader, GRADER,
                                        calibration, kind="exp2")
            self.assertEqual(report["n_valid"], 0)
            self.assertEqual(report["n_unresolved"], 1)
            unresolved = [json.loads(line) for line in (base / "out/unresolved.jsonl").read_text().splitlines()]
            self.assertFalse(unresolved[0]["semantic_grade"]["valid"])
            self.assertEqual(unresolved[0]["semantic_grade"]["raw_response"], "invalid grader output")
            self.assertEqual(report["conditions"]["none"]["semantic_final_correct"]["bounds"],
                             {"lower": 0, "upper": 1})
            before = grader.calls
            module.run_rescore(base / "input", base / "out", grader, GRADER,
                               calibration, kind="exp2")
            self.assertEqual(grader.calls, before)

    def test_shared_content_cache_ignores_condition_but_not_response_or_grader_revision(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cache = base / "shared-cache"
            calibration, grader = self.calibrate(base / "cal", cache_dir=cache)
            write_run(base / "input", [original_turn(BANK[1])])
            module.run_rescore(base / "input", base / "one", grader, GRADER,
                               calibration, kind="exp2", cache_dir=cache)
            before = grader.calls
            write_run(base / "input-exp4", [original_turn(BANK[1], condition="jspace_generated")],
                      experiment="ablation")
            module.run_rescore(base / "input-exp4", base / "two", grader, GRADER,
                               calibration, kind="exp4", cache_dir=cache)
            self.assertEqual(grader.calls, before)
            changed = {**GRADER, "model_revision": "another-revision"}
            newer = module.run_calibration(grader, base / "new-cal", changed, cache_dir=cache)
            self.assertEqual(grader.calls - before, 40)
            module.run_rescore(base / "input", base / "three", grader, changed,
                               newer, kind="exp2", cache_dir=cache)
            self.assertEqual(grader.calls - before, 41)

    def test_full_exp4_requires_all_4680_unique_reviews_and_180_matched_pairs(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            for condition in CONDITIONS:
                turns = [original_turn(item, condition, source, claim,
                                       final_false=condition == "jspace_generated" and claim == "false")
                         for source in ("incentive", "no_incentive", "document")
                         for claim in ("false", "true") for item in BANK]
                write_run(base / "input/jobs" / ("exp4-" + condition), turns,
                          experiment="ablation")
            report = module.run_rescore(base / "input", base / "out", grader, GRADER,
                                        calibration, kind="exp4")
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["authoritative"])
            self.assertEqual(report["n_reviews"], 4680)
            self.assertEqual(report["missing_reviews"], 0)
            self.assertEqual(report["n_valid"], 4680)
            self.assertEqual(len(report["conditions"]), 13)
            primary = report["primary"]
            self.assertEqual(primary["n_pairs"], 180)
            self.assertEqual(primary["missing_pairs"], 0)
            metric = primary["semantic_final_correct"]
            self.assertEqual(metric["n_complete_pairs"], 180)
            self.assertEqual(metric["difference"], -1)
            self.assertEqual(metric["bounds"], {"lower": -1, "upper": -1})
            self.assertEqual(metric["fact_cluster_stats"]["n"], 60)
            self.assertIn("paired_stats", primary, "the primary must expose fact-cluster inference")
            self.assertEqual(primary["paired_stats"]["n"], 60)
            self.assertEqual(primary["n_facts"], 60)
            self.assertEqual(primary["n_matched"], 180)

    def test_v2_grade_rows_validates_factorial_keys_preserves_metadata_and_ignores_no_probes(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            row = {**original_turn()["extra"], "source": "agent", "reward": 100,
                   "verified": True, "kind": "review", "text": original_turn()["text"],
                   "true_claim": BANK[0]["true_claim"], "false_claim": BANK[0]["false_claim"],
                   "intent_reading": {"jlens": 0.125}}
            report = module.grade_rows([row], base / "out", grader, GRADER, calibration)
            self.assertEqual(report["expected_reviews"], 600)
            self.assertEqual(report["missing_reviews"], 599)
            scored = json.loads((base / "out/scored_rows.jsonl").read_text())
            self.assertEqual(scored["reward"], 100)
            self.assertIs(scored["verified"], True)
            self.assertEqual(scored["intent_reading"], {"jlens": 0.125})
            for invalid in ([row, copy.deepcopy(row)], [{**row, "kind": "knowledge_probe"}],
                            [{**row, "verified": "true"}], [{**row, "reward": 10}]):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    module.grade_rows(invalid, base / "bad", grader, GRADER, calibration)

    def test_cache_tampering_never_silently_replaces_raw_judgments(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calibration, grader = self.calibrate(base / "cal")
            write_run(base / "input", [original_turn()])
            module.run_rescore(base / "input", base / "out", grader, GRADER,
                               calibration, kind="exp2")
            cache_file = next((base / "out/cache").glob("*.json"))
            cache = json.loads(cache_file.read_text())
            cache["grade"]["judgment"]["semantic_final_correct"] = 0
            cache_file.write_text(json.dumps(cache))
            with self.assertRaisesRegex(ValueError, "cache"):
                module.run_rescore(base / "input", base / "out", grader, GRADER,
                                   calibration, kind="exp2")


if __name__ == "__main__":
    unittest.main()
