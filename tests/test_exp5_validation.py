"""Statistical procedure preflight, independent of any language model."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jpeers.exp5.validation import noninferiority_preflight


class ValidationTests(unittest.TestCase):
    def config(self):
        cfg = json.loads((ROOT / "configs/exp5_study.json").read_text(encoding="utf-8"))
        cfg["evaluation"]["families"] = 4
        cfg["statistics"]["ni_preflight"] = {"replications": 100, "permutations": 128}
        return cfg

    def test_actual_seeded_null_calibration_and_provenance(self):
        cfg = self.config()
        a, b = noninferiority_preflight(cfg), noninferiority_preflight(cfg)
        self.assertEqual(a, b)
        self.assertTrue(a["passed"])
        self.assertEqual(a["model_calls"], 0)
        self.assertEqual(len(a["distributions"]), 2)
        self.assertTrue(all(r["replications"] == 100 for r in a["distributions"]))
        self.assertIn("symmetric", " ".join(a["assumptions"]))

    def test_insufficient_calibration_and_work_limits_fail_closed(self):
        cfg = self.config()
        cfg["statistics"]["ni_preflight"]["replications"] = 3
        self.assertFalse(noninferiority_preflight(cfg)["passed"])
        cfg = self.config()
        cfg["statistics"]["ni_preflight"]["max_sign_products"] = 1
        result = noninferiority_preflight(cfg)
        self.assertFalse(result["passed"])
        self.assertEqual(result["distributions"], [])

    def test_invalid_preflight_parameters_are_rejected(self):
        cfg = self.config()
        cfg["statistics"]["ni_preflight"]["replications"] = 0
        with self.assertRaises(ValueError):
            noninferiority_preflight(cfg)


if __name__ == "__main__":
    unittest.main()
