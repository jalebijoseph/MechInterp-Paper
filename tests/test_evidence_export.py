"""Standard-library regression checks for exact saved-evidence verification."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "evidence_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_evidence_export.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class EvidenceExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"groups": {"sample": {"source_root": "/remote/study", "live": False,
                         "kind": "test_saved_evidence", "capture_status": "terminal_snapshot",
                         "expected_target_traces": 1}}, "artifacts": []}
        self.add("sample/run/traces/00001.npz", b"PK\x03\x04saved binary trace", "trace")
        self.add_jsonl("sample/run/turns.jsonl", [{"trace_file": "traces/00001.npz"}])

    def add(self, path, data, kind="saved_evidence", source_path=None):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        row = {"path": path, "source_path": source_path or "/remote/study/" + path.split("/", 1)[1],
               "group": "sample", "kind": kind, "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest["artifacts"].append(row)
        return row

    def add_json(self, path, value):
        return self.add(path, json.dumps(value).encode())

    def add_jsonl(self, path, rows):
        return self.add(path, "".join(json.dumps(row) + "\n" for row in rows).encode())

    def replace_turns(self, rows):
        self.manifest["artifacts"] = [row for row in self.manifest["artifacts"]
                                      if not row["path"].endswith("turns.jsonl")]
        self.add_jsonl("sample/run/turns.jsonl", rows)

    def verify(self):
        totals = Counter(files=len(self.manifest["artifacts"]))
        for row in self.manifest["artifacts"]:
            totals["bytes"] += row["bytes"]
            totals["npz_files"] += row["path"].endswith(".npz")
        self.manifest["groups"]["sample"].update(totals)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return verifier.verify(path)

    def test_good_relative_and_absolute_source_references(self):
        result = self.verify()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["groups"]["sample"]["referenced_target_traces"], 1)
        self.replace_turns([{"trace_file": "/remote/study/run/traces/00001.npz"}])
        self.assertTrue(self.verify()["ok"])
        self.replace_turns([{"trace_file": "run/traces/00001.npz"}])
        self.assertTrue(self.verify()["ok"])

    def test_checksum_mismatch_and_missing_trace_are_rejected(self):
        trace = self.root / "sample/run/traces/00001.npz"
        trace.write_bytes(b"PK\x03\x04changed bytes")
        self.assertTrue(any("Checksum" in e for e in self.verify()["errors"]))
        trace.unlink()
        result = self.verify()
        self.assertFalse(result["ok"])
        self.assertTrue(any("Missing exported file" in e for e in result["errors"]))

    def test_duplicate_trace_reference_is_rejected(self):
        self.replace_turns([{"trace_file": "traces/00001.npz"}] * 2)
        self.assertTrue(any("Duplicate target trace" in e for e in self.verify()["errors"]))

    def test_duplicate_path_and_traversal_are_rejected(self):
        self.manifest["artifacts"].append(dict(self.manifest["artifacts"][0]))
        self.assertTrue(any("Duplicate artifact" in e for e in self.verify()["errors"]))
        self.manifest["artifacts"][-1]["path"] = "../outside.npz"
        self.assertTrue(any("escapes" in e for e in self.verify()["errors"]))

    def test_lfs_pointer_has_actionable_error(self):
        self.manifest["artifacts"] = [row for row in self.manifest["artifacts"]
                                      if not row["path"].endswith(".npz")]
        self.add("sample/run/traces/00001.npz", verifier.LFS_HEADER + b"\noid sha256:abc\n", "trace")
        errors = self.verify()["errors"]
        self.assertTrue(any("git lfs pull" in e for e in errors))

    def test_receipt_and_scientific_source_pins(self):
        receipt = {"received": True, "raw_response": "saved final"}
        self.add_json("sample/grading/cache/receipt.json", {
            "receipt": receipt, "receipt_sha256": verifier.digest(receipt)})
        source = self.add("sample/source/scripts/run_exp4_rescore_v3.py", b"# frozen runner\n",
                          "scientific_source")
        self.add_json("sample/grading/run.json", {
            "source_files": {"run_exp4_rescore_v3.py": source["sha256"]}})
        result = self.verify()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["receipt_digests_checked"], 1)
        self.assertEqual(result["scientific_pins_checked"], 1)
        self.manifest["artifacts"] = [row for row in self.manifest["artifacts"]
                                      if row["path"] != "sample/grading/cache/receipt.json"]
        self.add_json("sample/grading/cache/receipt.json", {"receipt": receipt, "receipt_sha256": "0" * 64})
        self.assertTrue(any("receipt digest" in e for e in self.verify()["errors"]))

    def test_live_group_cannot_claim_final_capture(self):
        self.manifest["groups"]["sample"].update(live=True, capture_status="completed")
        self.assertTrue(any("Live group" in e for e in self.verify()["errors"]))
        self.manifest["groups"]["sample"]["capture_status"] = "active_snapshot"
        result = self.verify()
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["notes"])

    def test_original_completion_requires_four_experiments_and_17_jobs(self):
        info = {"kind": "completed_original_target_suite",
                "expected_target_traces": 5776, "expected_smoke_traces": 26}
        original = {"status": "completed", "experiments": {
            f"exp{i}": {"status": "completed"} for i in range(1, 5)},
            "jobs": [{"status": "completed"}] * 17}
        documents = {"sample/source/runs/full-suite/manifest.json": original}
        errors = []
        verifier.completion_checks("sample", info, documents, errors)
        self.assertEqual(errors, [])
        original["jobs"] = original["jobs"][:-1]
        verifier.completion_checks("sample", info, documents, errors)
        self.assertTrue(errors)

    def test_corrected_target_completion_ignores_separate_grader_failure(self):
        info = {"kind": "completed_corrected_targets", "expected_target_traces": 1972}
        documents = {f"sample/results/{job}/followup-v2/validation.json": {
            "status": "completed", "n_turns": turns, "n_traces": traces,
            "validation_errors": []} for job, (turns, traces) in verifier.CORRECTED_COUNTS.items()}
        documents["sample/worker-status.json"] = {"status": "incomplete"}
        errors = []
        verifier.completion_checks("sample", info, documents, errors)
        self.assertEqual(errors, [])
        documents["sample/results/exp2/followup-v2/validation.json"]["validation_errors"] = ["missing"]
        verifier.completion_checks("sample", info, documents, errors)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
