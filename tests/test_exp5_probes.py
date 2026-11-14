"""Numerical contract checks; no checkpoint, network, or model dependency."""
import itertools
import json
from pathlib import Path
import sys
from exp5_test_utils import temporary_directory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jpeers.exp5.probes import (conditional_bases, fit_readout, load_artifact,
                                predict_readout, save_artifact, select_readout)


def panels(families=4):
    rows = []
    for family in range(families):
        for i, bits in enumerate(itertools.product((.2, 1.), repeat=4)):
            y = np.array(bits)
            rows.append(dict(family=family, id=f"{family}-{i}", profile=i,
                             controlled=True, delay=512, target=y,
                             site="neutral_receipt", match_id=f"matched-{family}",
                             mask=np.ones(4, bool), captures={0: y * 3 + 8,
                             1: np.ones(4)}, split="development"))
    return rows


class ReadoutTests(unittest.TestCase):
    def test_linear_preprocessing_masks_and_unobserved_heads(self):
        rows = panels(1)
        for row in rows:
            row["mask"][3] = False
            row["target"][3] = float("nan")
        art = fit_readout(rows, ridge=1e-6, layer=0)
        out = predict_readout(art, np.array([8.6, 11, 8.6, 11]))
        np.testing.assert_allclose(out["prediction"][:3], [.2, 1, .2], atol=1e-5)
        self.assertTrue(np.isnan(out["prediction"][3]))
        np.testing.assert_allclose(art["mean"], [9.8] * 4)
        self.assertEqual(art["counts"].tolist(), [16, 16, 16, 0])

    def test_grouped_selection_recovers_signal_and_leaves_failures_failed(self):
        artifact, report = select_readout(panels(), [0, 1], [1e-6, 1])
        self.assertEqual(artifact["layer"], 0)
        self.assertGreaterEqual(report["balanced_accuracy"], .99)
        self.assertTrue(report["gates"]["readout_accuracy"])
        self.assertTrue(report["gates"]["identity_blind_improvement"])
        self.assertFalse(report["gates"]["behavioral_validity"])
        for fold in report["folds"]:
            self.assertNotIn(fold["test_family"], fold["train_families"])
        _, fail = select_readout(panels(), [1], [1])
        self.assertFalse(fail["gates"]["readout_accuracy"])

    def test_preprocessing_is_fit_on_training_families_only(self):
        rows = panels(3)
        for row in rows:
            if row["family"] == 2:
                row["captures"][0] = row["captures"][0] + 1000
        _, report = select_readout(rows, [0], [1e-6])
        fold = next(f for f in report["folds"] if f["test_family"] == 2)
        np.testing.assert_allclose(fold["mean"], [9.8] * 4)

    def test_frozen_readout_cannot_mix_capture_sites(self):
        rows = panels(2)
        rows[0]["site"] = "pre_choice"
        with self.assertRaises(ValueError):
            fit_readout(rows, ridge=.001, layer=0)

    def test_eval_and_one_family_cannot_select_development_instrument(self):
        with self.assertRaises(ValueError):
            select_readout(panels(1), [0], [1])
        rows = panels(2)
        rows[0]["split"] = "evaluation"
        with self.assertRaises(ValueError):
            select_readout(rows, [0], [1])

    def test_anchor_midpoints_do_not_enter_binary_controlled_scores(self):
        rows = panels(3)
        for f in range(3):
            rows.append(dict(family=f, profile="anchor:0:high", delay=512,
                             target=[1,.6,.2,.2], mask=[True]*4,
                             captures={0: np.array([0,0,0,0])}, controlled=True))
        artifact, report = select_readout(rows, [0], [1e-6])
        self.assertEqual(artifact["provenance"]["n_records"], 48)
        self.assertEqual(report["balanced_accuracy"], 1.)
        self.assertTrue(report["gates"]["attribution_binding"])

    def test_conditional_contrasts_do_not_cross_unmatched_episode_ids(self):
        rows = [dict(family=0, delay=512, site="C2", match_id="one", target=[.2]*4,
                     mask=[True]*4, captures={0: np.zeros(4)}),
                dict(family=0, delay=512, site="C2", match_id="two", target=[1,.2,.2,.2],
                     mask=[True]*4, captures={0: np.ones(4)})]
        self.assertEqual(conditional_bases(rows, layer=0, rank=1)[0].shape, (4,0))

    def test_missing_behavioral_results_cannot_pass_selection_gates(self):
        rows = panels(3)
        rows[0].update(valid=True, routing_correct=True)
        _, report = select_readout(rows, [0], [1e-6])
        self.assertFalse(report["gates"]["behavioral_validity"])
        self.assertFalse(report["gates"]["evidence_supported_routing"])
        self.assertEqual(report["behavioral_coverage"]["valid"]["n_assigned"], 48)
        self.assertEqual(report["behavioral_coverage"]["valid"]["n_reported"], 1)
        for row in rows:
            row.update(valid=True, routing_correct=True)
        _, complete = select_readout(rows, [0], [1e-6])
        self.assertTrue(complete["gates"]["behavioral_validity"])
        self.assertTrue(complete["gates"]["evidence_supported_routing"])

    def test_routing_gate_excludes_only_explicitly_nonapplicable_ties(self):
        rows = panels(3)
        for row in rows:
            unequal = row["target"][0] != row["target"][2]
            row.update(valid=True, routing_applicable=bool(unequal), routing_correct=bool(unequal))
        _, report = select_readout(rows, [0], [1e-6])
        routing = report["behavioral_coverage"]["routing_correct"]
        self.assertEqual(routing["rate"], 1.)
        self.assertEqual(routing["n_assigned"], 48)
        self.assertEqual(routing["n_applicable"], 24)
        self.assertEqual(routing["n_nonapplicable"], 24)
        self.assertTrue(report["gates"]["evidence_supported_routing"])
        self.assertEqual(report["behavioral_coverage"]["valid"]["n_applicable"], 48)
        for row in rows:
            row.pop("routing_applicable")
        _, missing = select_readout(rows, [0], [1e-6])
        self.assertFalse(missing["gates"]["evidence_supported_routing"])
        self.assertEqual(missing["behavioral_coverage"]["routing_correct"]["n_applicable"], 48)
        for row in rows:
            row.update(valid=False, routing_applicable=False)
        _, ties = select_readout(rows, [0], [1e-6])
        self.assertIsNone(ties["behavioral_coverage"]["routing_correct"]["rate"])
        self.assertFalse(ties["gates"]["evidence_supported_routing"])
        self.assertFalse(ties["gates"]["behavioral_validity"])

    def test_conditional_bases_require_development_and_matching_provenance(self):
        for split in ("evaluation", "eval", "test"):
            with self.subTest(split=split):
                rows = panels(2)
                rows[0]["split"] = split
                with self.assertRaises(ValueError):
                    conditional_bases(rows, layer=0, rank=1)
        for field in ("site", "match_id"):
            for value in (None, "", "  ", 0):
                with self.subTest(field=field, value=value):
                    rows = panels(2)
                    for row in rows:
                        if value is None:
                            row.pop(field)
                        else:
                            row[field] = value
                    bases = conditional_bases(rows, layer=0, rank=1)
                    self.assertTrue(all(b.shape == (4, 0) for b in bases.values()))

    def test_bounded_artifact_payload_round_trip_preserves_capture_site(self):
        from jpeers.exp5.probes import artifact_payloads
        artifact = fit_readout(panels(2), ridge=.001, layer=0)
        self.assertEqual(artifact["site"], "neutral_receipt")
        payloads = artifact_payloads(artifact, stem="frozen")
        self.assertEqual(set(payloads), {"frozen.json", "frozen.npz"})
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with temporary_directory(dir=root) as tmp:
            for name, payload in payloads.items():
                self.assertIsInstance(payload, bytes)
                (Path(tmp) / name).write_bytes(payload)
            restored = load_artifact(Path(tmp) / "frozen.json")
            self.assertEqual(restored["site"], "neutral_receipt")
            np.testing.assert_array_equal(restored["coef"], artifact["coef"])
            self.assertEqual(restored["provenance"], artifact["provenance"])
            save_artifact(Path(tmp) / "saved.json", artifact)
            self.assertEqual((Path(tmp) / "saved.json").read_bytes(), payloads["frozen.json"])

    def test_no_pickle_round_trip_preserves_geometry_and_provenance(self):
        art = fit_readout(panels(2), ridge=.001, layer=0)
        art["bases"] = {0: np.eye(4)[:, :1]}
        art["validation"] = {"passed": False, "reason": "synthetic"}
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with temporary_directory(dir=root) as tmp:
            path = Path(tmp) / "readout.json"
            save_artifact(path, art)
            restored = load_artifact(path)
            np.testing.assert_array_equal(restored["bases"][0], art["bases"][0])
            self.assertEqual(restored["provenance"], art["provenance"])
            np.testing.assert_array_equal(predict_readout(restored, np.ones(4))["raw"],
                                          predict_readout(art, np.ones(4))["raw"])
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                self.assertTrue(all(a.dtype != object for a in arrays.values()))
            meta = json.loads(path.read_text())
            meta["version"] = 999
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_artifact(path)

    def test_conditional_bases_are_single_cell_encoding_contrasts(self):
        rows = panels(2)
        bases = conditional_bases(rows, layer=0, rank=2)
        for cell in range(4):
            self.assertEqual(bases[cell].shape, (4, 1))
            expected = np.eye(4)[:, cell]
            np.testing.assert_allclose(bases[cell] @ bases[cell].T,
                                      np.outer(expected, expected), atol=1e-10)
        unrelated = [dict(family=0, delay=512, target=[.2]*4,
                          mask=[True]*4, captures={0: np.zeros(4)}),
                     dict(family=1, delay=512, target=[1,.2,.2,.2],
                          mask=[True]*4, captures={0: np.ones(4)})]
        self.assertEqual(conditional_bases(unrelated, layer=0, rank=1)[0].shape, (4, 0))


if __name__ == "__main__":
    unittest.main()
