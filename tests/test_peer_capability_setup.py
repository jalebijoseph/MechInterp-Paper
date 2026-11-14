"""Offline setup checks; unittest avoids the model/tokenizer pytest fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from exp5_test_utils import temporary_directory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jpeers.experiments import peer_capability_setup as setup


class PeerCapabilitySetupTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/exp5_setup.json").read_text(encoding="utf-8"))

    def test_protocol_validation(self):
        config = self.config()
        setup.validate_config(config)
        mutations = [
            ("schema_version", True),
            ("stage", "evaluation"),
            ("delegation_choices", ["A", "B", "abstain"]),
            ("model", {"id": "another/model", "revision": "main"}),
            ("storage_budget_bytes", 1073741825),
            ("storage_budget_bytes", True),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(config)
                changed[key] = value
                with self.assertRaises(ValueError):
                    setup.validate_config(changed)
        for key, value in [("enable_thinking", True), ("sampling_policy", "greedy")]:
            changed = copy.deepcopy(config)
            changed["inference"][key] = value
            with self.assertRaises(ValueError):
                setup.validate_config(changed)
        for count in [0, 1, 3, 66, True]:
            changed = copy.deepcopy(config)
            changed["fixtures"]["cases_per_domain"] = count
            with self.assertRaises(ValueError):
                setup.validate_config(changed)

    def test_private_input_schedules_and_world_truth(self):
        fixtures = setup.build_fixtures(self.config())
        self.assertEqual(len(fixtures), 72)
        self.assertEqual(len({row["id"] for row in fixtures}), 72)
        groups = {}
        for row in fixtures:
            groups.setdefault((row["family_id"], row["domain"]), []).append(row)
            world = row["analysis"]["world"]
            self.assertEqual(row["analysis"]["true_answer"], world["Alice"])
            for peer, facts in row["private_facts"].items():
                self.assertEqual(len(facts), 2)
                self.assertEqual(len({fact["subject"] for fact in facts}), 2)
                condition = row["analysis"]["conditions"][peer]
                for fact in facts:
                    if condition != "incorrect" or fact["subject"] != "Alice":
                        self.assertEqual(fact["value"], world[fact["subject"]])
        for rows in groups.values():
            self.assertEqual(len(rows), 4)
            for peer in ["A", "B"]:
                condition = rows[0]["analysis"]["conditions"][peer]
                supplied = correct = 0
                for row in rows:
                    target = next((f for f in row["private_facts"][peer]
                                   if f["subject"] == "Alice"), None)
                    supplied += target is not None
                    correct += target is not None and target["value"] == row["analysis"]["true_answer"]
                self.assertEqual((supplied, correct),
                                 {"correct": (4, 4), "incomplete": (2, 2),
                                  "incorrect": (4, 2)}[condition])

    def test_reference_assignments_and_public_information(self):
        config = self.config()
        fixtures = setup.build_fixtures(config)
        scenes = {}
        for row in fixtures:
            scene_id = row["analysis"]["scene_id"]
            if scene_id in scenes:
                self.assertEqual(row["analysis"]["world"], scenes[scene_id])
            scenes[scene_id] = row["analysis"]["world"]
        self.assertEqual(len(scenes), 8)
        for comparison in ["correct_vs_incomplete", "correct_vs_incorrect"]:
            assignments = {tuple(row["analysis"]["reference_by_domain"][d]
                                 for d in config["domains"])
                           for row in fixtures if row["analysis"]["comparison"] == comparison}
            self.assertEqual(assignments, {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")})
        row = fixtures[0]
        rendered = setup.render_prompts(row, config)
        changed = copy.deepcopy(row)
        changed["analysis"] = {"true_answer": "TOP_SECRET", "reference_by_domain": "TOP_SECRET"}
        changed["private_facts"]["A"][0]["value"] = "PRIVATE_ONLY"
        other = setup.render_prompts(changed, config)
        self.assertEqual(rendered["orchestrator"], other["orchestrator"])
        self.assertNotEqual(rendered["peers"]["A"], other["peers"]["A"])
        self.assertEqual(rendered["peers"]["B"], other["peers"]["B"])
        public_text = json.dumps(other["orchestrator"])
        for secret in ["TOP_SECRET", "PRIVATE_ONLY", "reference_by_domain", "correct_vs_"]:
            self.assertNotIn(secret, public_text)

    def test_objective_checker(self):
        for row in setup.build_fixtures(self.config()):
            truth = row["analysis"]["true_answer"]
            self.assertTrue(setup.grade_answer(row, f" {truth.upper()} "))
            for answer in ["unknown", "", f"The answer is {truth}", None]:
                self.assertFalse(setup.grade_answer(row, answer))
            for facts in row["private_facts"].values():
                for fact in facts:
                    if fact["subject"] == "Alice" and fact["value"] != truth:
                        self.assertFalse(setup.grade_answer(row, fact["value"]))

    def test_export_is_deterministic_offline_and_not_execution_ready(self):
        with temporary_directory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            with mock.patch.object(socket, "socket", side_effect=AssertionError("network call")), \
                 mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "jlens": None}):
                report = setup.prepare_setup(ROOT / "configs/exp5_setup.json", first)
                setup.prepare_setup(ROOT / "configs/exp5_setup.json", second)
            self.assertEqual({p.name for p in first.iterdir()},
                             {"config.json", "fixtures.jsonl", "manifest.json"})
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["ready_for_execution"])
            self.assertEqual(manifest["model_calls"], 0)
            self.assertTrue(manifest["pending_execution_gates"])
            for name, receipt in manifest["files"].items():
                data = (first / name).read_bytes()
                self.assertEqual(receipt["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(receipt["bytes"], len(data))
            self.assertEqual(report["setup_bytes"], sum(p.stat().st_size for p in first.iterdir()))
            self.assertLessEqual(report["setup_bytes"], 1073741824)
            self.assertEqual(report["storage_scope"], "setup_bundle_only")
            self.assertIsNone(report["full_study_bytes"])
            self.assertFalse(report["ready_for_execution"])
            rows = [json.loads(line) for line in (first / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 72)
            self.assertEqual(rows[0]["prompts"], setup.render_prompts(rows[0], self.config()))

    def test_export_preserves_existing_outputs_and_checks_budget(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            existing = folder / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                setup.prepare_setup(ROOT / "configs/exp5_setup.json", existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(existing.iterdir()), [sentinel])
            config = self.config()
            config["storage_budget_bytes"] = 1
            small = folder / "small.json"
            small.write_text(json.dumps(config), encoding="utf-8")
            output = folder / "too-large"
            with self.assertRaisesRegex(ValueError, "budget|storage"):
                setup.prepare_setup(small, output)
            self.assertFalse(output.exists())

    def test_cli_prepares_only_setup_artifacts(self):
        with temporary_directory() as temp:
            output = Path(temp) / "bundle"
            command = [sys.executable, str(ROOT / "scripts/prepare_exp5.py"), "--config",
                       str(ROOT / "configs/exp5_setup.json"), "--out", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, cwd=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["fixture_count"], 72)
            self.assertEqual(report["model_calls"], 0)
            self.assertFalse(report["ready_for_execution"])
            rejected = subprocess.run(command + ["--run"], capture_output=True, text=True, cwd=temp)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("unrecognized arguments", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
