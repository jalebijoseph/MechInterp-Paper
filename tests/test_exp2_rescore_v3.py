"""CPU-only checks for frozen Exp2 inventory and exact v3 calibration reuse."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("exp2_v3", ROOT / "scripts/run_exp2_rescore_v3.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
shared, rescore = worker.shared, worker.rescore
import queue_exp4_rescore_v3 as manager


class NoHTTP:
    def get(self, *args, **kwargs):
        raise AssertionError("Preflight must not make HTTP requests")

    post = get


def receipt(raw: str) -> dict:
    return {"received": True, "raw_response": raw, "finish_reason": "stop",
            "response_id": "mock", "reasoning_tokens": 0, "completion_tokens": 1,
            "prompt_tokens": 1, "protocol_error": None, "transport_error": None}


class Exp2V3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "original" / "summary.json"
        rows = []
        for number, key in enumerate(rescore._expected_keys("exp2_v2", rescore._facts())):
            condition, source, claim, item, reward, verified = key
            fact = rescore._facts()[item]
            rows.append({"kind": "review", "condition": condition, "source": source,
                         "claim": claim, "item": item, "reward": reward, "verified": verified,
                         "text": f"Final Text: Saved review marker {number:04d}.",
                         "true_claim": fact["true_claim"], "false_claim": fact["false_claim"]})
        self.summary = {"protocol": "audited-followup-v2", "counts": {"recorded_reviews": 600},
                        "reviews": rows, "knowledge_probes": [{"kind": "knowledge_probe"}] * 480}
        rescore._atomic_json(self.input, self.summary)
        self.args = SimpleNamespace(
            input_summary=self.input, input_summary_sha256=rescore._file_hash(self.input),
            output_dir=self.root / "new", calibration_source=self.root / "prior",
            calibration_source_sha256=None, model="nemotron-3.5-super-vl-exp4-v3",
            model_revision="a" * 40, generation_profile="nemotron-3.5-super-vl",
            fixture=ROOT / "src/jpeers/data/fact_grader_holdout_v3.json",
            fixture_sha256=rescore._file_hash(ROOT / "src/jpeers/data/fact_grader_holdout_v3.json"),
            base_url="http://mock", concurrency=4, resume=False, prepare_only=True)
        identity = worker.grader_identity(self.args)
        development, development_hash = rescore._fixtures()
        holdout, holdout_hash = shared.load_holdout(self.args.fixture, self.args.fixture_sha256,
                                                   development["cases"])
        prior = {"grader_info": identity, "input_root": "/original-exp4",
                 "development_fixture_sha256": development_hash,
                 "holdout_fixture_sha256": holdout_hash,
                 "holdout_fixture_scorer_version": shared.HOLDOUT_FIXTURE_VERSION,
                 "development_only": False, "source_files": worker.source_files()}
        rescore._atomic_json(self.args.calibration_source / "run.json", prior)
        for phase, cases in (("calibration", development["cases"]), ("holdout", holdout)):
            for case in cases:
                payload = shared.request_payload(identity["model"], case["item"], case["response"],
                                                 identity["generation"], identity["checklist"])
                schema = payload["response_format"]["json_schema"]["schema"]
                evidence = {}
                for field, label in case["expected"].items():
                    positive = label is True or (field == "final_truth" and label in ("true", "false", "both"))
                    ids = schema["properties"]["evidence"]["properties"][field]["items"].get("enum", [])
                    evidence[field] = ids[:1] if positive else []
                raw = json.dumps({**case["expected"], "evidence": evidence})
                saved_receipt = receipt(raw)
                grade = shared.parse_completion(saved_receipt, case["response"])
                self.assertTrue(grade["valid"], grade["error"])
                key = rescore._digest({"phase": phase, "grader": identity, "request": payload})
                rescore._atomic_json(self.args.calibration_source / "cache" / (key + ".json"),
                                     {"cache_key": key, "receipt": saved_receipt,
                                      "receipt_sha256": rescore._digest(saved_receipt),
                                      "transport_attempts": [], "grade": grade})
        self.args.calibration_source_sha256 = rescore._file_hash(self.args.calibration_source / "run.json")

    def test_preflight_replays_84_gates_without_http_and_pins_manifest(self):
        self.assertEqual(worker.run(self.args, NoHTTP()), 0)
        gate = rescore._read_json(self.args.output_dir / "gate.json")
        self.assertEqual((gate["development"]["n_agree"], gate["holdout"]["n_agree"]), (40, 44))
        self.assertEqual(gate["development"]["grader_calls"] + gate["holdout"]["grader_calls"], 0)
        manifest = rescore._read_json(self.args.output_dir / "run.json")
        self.assertEqual(manifest["input_summary_sha256"], self.args.input_summary_sha256)
        self.assertEqual(manifest["calibration_source_sha256"], self.args.calibration_source_sha256)
        self.assertEqual(len(manifest["calibration_receipts_sha256"]), 84)
        self.assertIn("run_exp2_rescore_v3.py", manifest["source_files"])
        self.assertFalse((self.args.output_dir / "endpoint.json").exists())
        self.assertFalse((self.args.output_dir / "scored_rows.jsonl").exists())
        self.args.resume = True
        self.assertEqual(worker.run(self.args, NoHTTP()), 0)

    def test_missing_or_duplicate_cells_and_probes_are_rejected(self):
        for mutate in (lambda rows: rows.pop(),
                       lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(kind="knowledge_probe")):
            summary = copy.deepcopy(self.summary)
            mutate(summary["reviews"])
            rescore._atomic_json(self.input, summary)
            pin = rescore._file_hash(self.input)
            with self.assertRaises(ValueError):
                worker.load_reviews(self.input, pin)

    def test_changed_input_pin_fails_before_http(self):
        self.args.input_summary_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "Saved Exp2 summary"):
            worker.run(self.args, NoHTTP())

    def test_changed_prior_pin_fails_before_http(self):
        self.args.calibration_source_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "Prior calibration run.json"):
            worker.run(self.args, NoHTTP())

    def test_changed_shared_source_or_identity_is_rejected(self):
        prior_path = self.args.calibration_source / "run.json"
        original = rescore._read_json(prior_path)
        for field in ("source_files", "grader_info"):
            changed = copy.deepcopy(original)
            changed[field][next(iter(changed[field]))] = "modified"
            rescore._atomic_json(prior_path, changed)
            self.args.calibration_source_sha256 = rescore._file_hash(prior_path)
            with self.assertRaisesRegex(ValueError, field):
                worker.run(self.args, NoHTTP())

    def test_corrupt_receipt_key_digest_or_grade_is_rejected(self):
        path = next((self.args.calibration_source / "cache").glob("*.json"))
        original = rescore._read_json(path)
        for field in ("cache_key", "receipt_sha256", "grade"):
            changed = copy.deepcopy(original)
            changed[field] = "modified"
            rescore._atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, "receipt failed integrity"):
                worker.run(self.args, NoHTTP())
            rescore._atomic_json(path, original)

    def test_transport_failed_calibration_receipt_is_never_retried(self):
        path = next((self.args.calibration_source / "cache").glob("*.json"))
        saved = rescore._read_json(path)
        failed = {**saved["receipt"], "received": False, "raw_response": None,
                  "transport_error": "HTTPStatusError: mocked failure"}
        # A protocol error has precedence in parse_completion, so retain it only when present.
        saved.update(receipt=failed, receipt_sha256=rescore._digest(failed))
        saved["grade"] = shared.parse_completion(failed, "unused response")
        rescore._atomic_json(path, saved)
        with self.assertRaisesRegex(ValueError, "never retries transport failures"):
            worker.run(self.args, NoHTTP())

    def test_full_scoring_preserves_unknowns_and_never_scores_probes(self):
        calls = []
        def respond(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": self.args.model}]})
            payload = json.loads(request.content)
            text = payload["messages"][1]["content"]
            calls.append(text)
            self.assertIn("Saved review marker", text)
            if "marker 0001" in text:
                return httpx.Response(500, text="mock transport failure")
            raw = {"final_truth": "missing", "review_rejects_false": False,
                   "intent_accusation": False, "usable": None if "marker 0000" in text else False,
                   "refusal": False, "evidence": {field: [] for field in worker.scoring_v3.RESPONSE_SCHEMA["required"] if field != "evidence"}}
            return httpx.Response(200, json={"id": "mock", "model": self.args.model,
                                           "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]})
        self.args.prepare_only = False
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(worker.run(self.args, client), 1)
        self.assertEqual(len(calls), 600)
        summary = rescore._read_json(self.args.output_dir / "summary.json")
        self.assertEqual((summary["n_reviews"], summary["n_valid"], summary["n_unresolved"]), (600, 599, 2))
        self.assertFalse(summary["authoritative"])
        self.assertNotIn("primary", summary)
        self.assertEqual(summary["knowledge_probes_graded"], 0)
        rows = [json.loads(line) for line in (self.args.output_dir / "scored_rows.jsonl").read_text().splitlines()]
        self.assertIsNone(rows[0]["semantic_grade"]["judgment"]["usable"])
        self.assertIsNone(rows[1]["semantic_grade"]["judgment"])
        self.assertEqual(rows[1]["semantic_grade_status"], "invalid")
        self.assertEqual(rescore._file_hash(self.input), self.args.input_summary_sha256)

    def manager_command(self, exp2: bool) -> list[str]:
        root = self.root / "manager"
        model_path = self.root / self.args.model_revision
        model_path.mkdir()
        model = manager.NEMOTRON_MODEL if exp2 else manager.MODEL
        served = "nemotron-3.5-super-vl-exp4-v3" if exp2 else manager.SERVED_MODEL
        rescore._atomic_json(root / "model-provenance.json", {
            "model": model, "revision": self.args.model_revision,
            "model_path": str(model_path), "status": "cached"})
        argv = ["manager", "--root", str(root), "--predecessor", str(self.root / "suite"),
                "--revision", self.args.model_revision, "--fixture-sha256", self.args.fixture_sha256]
        if exp2:
            argv += ["--model-profile", "nemotron-3.5-super-vl", "--concurrency", "16",
                     "--request-timeout-seconds", "1800", "--input-summary", str(self.input),
                     "--input-summary-sha256", self.args.input_summary_sha256,
                     "--calibration-source", str(self.args.calibration_source),
                     "--calibration-source-sha256", self.args.calibration_source_sha256]
        devices = [{"index": i, "uuid": f"gpu-{i}"} for i in range(4)]
        server = Mock(pid=123, poll=Mock(return_value=None))
        child = Mock(pid=124, poll=Mock(return_value=0), returncode=0)
        with (patch.object(sys, "argv", argv),
              patch.object(manager, "predecessor_released", return_value=(True, {"manifest_status": "completed"})),
              patch.object(manager, "gpu_pairs", return_value=[devices[:2], devices[2:]]),
              patch.object(manager, "assert_port_free"),
              patch.object(manager.signal, "signal"),
              patch.object(manager, "stop_owned_group") as stop,
              patch.object(manager.urllib.request, "urlopen", return_value=io.StringIO(json.dumps({"data": [{"id": served}]}))),
              patch.object(manager.subprocess, "Popen", side_effect=[server, child]) as start):
            self.assertEqual(manager.main(), 0)
            self.assertEqual(start.call_count, 2)
            self.assertEqual(start.call_args_list[0].kwargs["env"]["CUDA_VISIBLE_DEVICES"], "gpu-0,gpu-1,gpu-2,gpu-3")
            self.assertEqual(stop.call_count, 2)
            return start.call_args_list[1].args[0]

    def test_manager_selects_exp2_and_pins_1800_second_transport(self):
        command = self.manager_command(exp2=True)
        self.assertEqual(command[1], "-c")
        self.assertIn("run_exp2_rescore_v3.py", command[2])
        self.assertIn("httpx.Timeout(1800, connect=15)", command[2])
        self.assertIn("sys.path.insert", command[2])
        self.assertIn("--input-summary", command)
        self.assertIn("--calibration-source", command)
        self.assertNotIn("--input-root", command)

    def test_manager_default_keeps_original_exp4_worker_command(self):
        command = self.manager_command(exp2=False)
        self.assertEqual(Path(command[1]).name, "run_exp4_rescore_v3.py")
        self.assertEqual(command[2:4], ["--input-root", str(self.root / "suite/source/runs/full-suite")])
        self.assertNotIn("--input-summary", command)
        self.assertNotIn("--calibration-source", command)


if __name__ == "__main__":
    unittest.main()
