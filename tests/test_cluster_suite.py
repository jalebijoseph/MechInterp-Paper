"""Scheduler/config tests run with stdlib alone and make no inference requests."""
import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cluster_suite.py"
SPEC = importlib.util.spec_from_file_location("cluster_suite", SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class SuiteTests(unittest.TestCase):
    def test_full_sizes_and_independent_benchmarks(self):
        jobs = {j["id"]: j for j in suite.build_jobs()}
        counts = {key: suite.planned_counts(j["config"]) for key, j in jobs.items()}
        self.assertEqual(len(jobs), 19)
        self.assertEqual(counts["exp0-core"]["requests"], 80)
        self.assertEqual(counts["exp0-hitom"]["requests"], 60)
        self.assertEqual(counts["exp1-full"]["turns"], 96)
        self.assertEqual(counts["exp2-core"]["reviews"], 360)
        self.assertEqual(counts["exp2-narcbench"]["reviews"], 0)
        self.assertEqual(counts["exp2-narcbench"]["narcbench_scenarios"], 20)
        self.assertEqual(counts["exp3-full"]["decisions"],
                         {"context": 48, "family": 432, "self_preference": 24, "bandit": 96})
        self.assertEqual(counts["exp3-full"]["plans"], 24)
        ablations = [j for j in jobs.values() if j["experiment"] == 4]
        self.assertEqual(len(ablations), 13)
        self.assertEqual(sum(counts[j["id"]]["reviews"] for j in ablations), 4680)
        for j in ablations:
            self.assertEqual(counts[j["id"]]["reviews"], 360)
            self.assertEqual(j["config"]["interventions"], [j["portion"]])
            self.assertNotIn("narcbench", j["config"])
        self.assertNotIn("narcbench", jobs["exp2-core"]["config"])
        for j in jobs.values():
            self.assertNotIn("qwen3-1.7b", json.dumps(j["config"]))
        sp = jobs["exp3-full"]["config"]
        self.assertIn("same-family", sp["self_preference_scope"])
        self.assertEqual(sp["self_preference"]["other"]["endpoint"], "qwen3-8b")

    def test_missing_hitom_is_incomplete(self):
        cfg = next(j["config"] for j in suite.build_jobs() if j["id"] == "exp0-hitom")
        validation = suite.validate_summary(cfg, {"rows": [{}], "items": [],
            "hitom_skipped_not_single_token": ["a"]}, {"hitom": {"ids": ["a"]}}, [])
        self.assertTrue(any("expected 60" in err for err in validation["errors"]))
        self.assertTrue(any("omitted 1" in err for err in validation["errors"]))

    def test_missing_primary_ablation_is_incomplete(self):
        cfg = suite.read_json(suite.CONFIGS / suite.CONFIG_FILES[4])
        result = suite.validate_summary(cfg, {"rows": [{}]}, {}, [])
        self.assertTrue(any("paired primary" in err for err in result["errors"]))

    def test_every_job_attempted_on_failures_and_resume_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, str(SCRIPT), "--output-dir", tmp, "--python", "/usr/bin/false"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            manifest = suite.read_json(Path(tmp) / "manifest.json")
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(len(manifest["jobs"]), 19)
            for job in manifest["jobs"]:
                self.assertEqual(job["status"], "failed")
                self.assertEqual(len(job["attempts"]), 1)
                self.assertEqual(job["attempts"][0]["exit_code"], 1)
                self.assertTrue(Path(job["attempts"][0]["log"]).is_file())
            subprocess.run(command, capture_output=True, text=True, timeout=30)
            resumed = suite.read_json(Path(tmp) / "manifest.json")
            self.assertTrue(all(len(j["attempts"]) == 1 for j in resumed["jobs"]))
            retried = subprocess.run(command + ["--retry-failed", "--ports", "18080", "18081"],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(retried.returncode, 1, retried.stderr)
            resumed = suite.read_json(Path(tmp) / "manifest.json")
            self.assertTrue(all(len(j["attempts"]) == 2 for j in resumed["jobs"]))

    def fragments(self):
        cfg = suite.read_json(suite.CONFIGS / suite.CONFIG_FILES[4])
        items = suite.bank("imposter_facts", "items")[:cfg["n_items"]]
        fragments = []
        for index, condition in enumerate(cfg["interventions"]):
            live = [{"condition": condition, "source": source, "claim": claim,
                     "item": item["id"], "claim_in_draft": True, "catch_score": index / 20,
                     "fluency": 1.0, "copies_peer": 0.0}
                    for source in ("incentive", "no_incentive", "document")
                    for claim in ("false", "true") for item in items]
            rows = [{"condition": condition, "source": source, "claim": claim,
                     "n": 60, "catch_score": index / 20, "fluency": 1.0}
                    for source in ("incentive", "no_incentive", "document") for claim in ("false", "true")]
            fragments.append((condition, {"live": live, "rows": rows,
                "contrasts": {condition: {"synthetic": True}}, "cluster_fragment": {"condition": condition}}))
        return cfg, fragments

    @staticmethod
    def primary(live, condition, control, fluency_drop):
        units = lambda cond: {(r["source"], r["item"]): r for r in live
                             if r["condition"] == cond and r["claim"] == "false" and r["claim_in_draft"]}
        left, right = units(condition), units(control)
        paired = set(left) & set(right)
        return {"condition": condition, "control": control, "n_pairs": len(paired),
                "catch_score": {"diff": sum(left[u]["catch_score"] - right[u]["catch_score"] for u in paired) / len(paired)}}

    def test_all_fragments_combine_with_180_pairs_and_baseline_deltas(self):
        cfg, fragments = self.fragments()
        result = suite.combine_exp4(fragments, cfg, self.primary)
        self.assertEqual(len(result["live"]), 4680)
        self.assertEqual(len(result["rows"]), 78)
        self.assertEqual(len(result["contrasts"]), 13)
        self.assertEqual(len(result["deltas_vs_none"]), 72)
        self.assertEqual(result["primary"]["n_pairs"], 180)
        self.assertAlmostEqual(result["primary"]["catch_score"]["diff"], -0.05)
        self.assertAlmostEqual(result["deltas_vs_none"][0]["d_catch_score"], 0.05)

    def test_aggregation_rejects_missing_duplicate_and_unpaired_data(self):
        cfg, fragments = self.fragments()
        with self.assertRaisesRegex(ValueError, "all 13"):
            suite.combine_exp4(fragments[:-1], cfg, self.primary)
        broken = copy.deepcopy(fragments)
        broken[1][1]["live"][0] = broken[1][1]["live"][1]
        with self.assertRaisesRegex(ValueError, "exactly one review"):
            suite.combine_exp4(broken, cfg, self.primary)
        with self.assertRaisesRegex(ValueError, "expected 180 pairs"):
            suite.combine_exp4(fragments, cfg, lambda *args: {"n_pairs": 179})

    def test_completed_fragments_alone_do_not_complete_experiment(self):
        manifest = {"jobs": suite.build_jobs()}
        for job in manifest["jobs"]:
            job["status"] = "completed"
        suite.summarize(manifest)
        self.assertEqual(manifest["experiments"]["exp4"]["status"], "incomplete")
        self.assertEqual(manifest["experiments"]["exp3"]["status"], "completed")

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "yaml", "pydantic", "httpx")),
                         "original statistics integration needs the project runtime dependencies")
    def test_saved_aggregate_uses_original_primary_statistics(self):
        _, fragments = self.fragments()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"jobs": suite.build_jobs()}
            for job in manifest["jobs"]:
                if job["experiment"] != 4:
                    continue
                summary = dict(fragments)[job["portion"]]
                path = Path(tmp) / job["id"] / "summary.json"
                suite.atomic_json(path, summary)
                job["status"] = "completed"
                job["attempts"] = [{"summary": str(path),
                    "summary_sha256": suite.hashlib.sha256(path.read_bytes()).hexdigest()}]
            suite.aggregate_exp4(manifest, Path(tmp))
            aggregate = manifest["exp4_aggregate"]
            self.assertEqual(aggregate["status"], "completed", aggregate.get("error"))
            saved = suite.read_json(aggregate["summary"])
            self.assertEqual(saved["primary"]["n_pairs"], 180)
            self.assertAlmostEqual(saved["primary"]["catch_score"]["diff"], -0.05)
            self.assertLess(saved["primary"]["catch_score"]["p"], 0.001)
            self.assertTrue(Path(tmp, "exp4-aggregate", "summary.md").is_file())
            suite.summarize(manifest)
            self.assertEqual(manifest["experiments"]["exp4"]["status"], "completed")
            Path(aggregate["summary"]).unlink()
            suite.summarize(manifest)
            self.assertEqual(manifest["experiments"]["exp4"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
