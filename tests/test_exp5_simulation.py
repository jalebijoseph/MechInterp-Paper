"""Software-only simulator contracts; no checkpoint, Torch, or network required."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SimulationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.backend = importlib.import_module("jpeers.exp5.simulation").SimulationBackend()
        except ModuleNotFoundError:
            self.fail("SimulationBackend is missing")

    def append(self, tokens, event):
        return self.backend.append_message(tokens, "user", json.dumps(event))

    def history(self):
        tokens, _ = self.append([], {"event": "roster", "roster": {"A": "Cedar", "B": "Maple"}})
        for peer, domain, outcomes in (
            ("A", "possession", [True, False, False, False]),
            ("A", "location", [True, True, True, False]),
            ("B", "possession", [True, True, True, True]),
            ("B", "location", [False, False, False, False]),
        ):
            for correct in outcomes:
                tokens, _ = self.append(tokens, {"event": "peer_feedback", "peer": peer,
                    "domain": domain, "response": '{"answer":"ball"}', "correct": correct})
        tokens, _ = self.backend.append_message(tokens, "assistant", "Recorded.", close=False)
        return tokens

    def generate(self, tokens, event, *, seed=17, sampler=None, edits=(), masks=(), cap=256):
        tokens = self.backend.finish_assistant(tokens)
        tokens, _ = self.append(tokens, event)
        tokens, _ = self.backend.append_message(tokens, "assistant", "", close=False)
        result = self.backend.generate(tokens, sampler=sampler or {"temperature": 0}, seed=seed,
                                       max_new_tokens=cap, edits=edits, masks=masks)
        return result

    def test_append_retains_native_bytes_and_returns_exact_intervals(self):
        b = self.backend
        original = [1, 255, 2]
        tokens, span = b.append_message(original, "user", "scene: café")
        self.assertEqual(original, [1, 255, 2])
        self.assertEqual(tokens[:3], original)
        self.assertEqual(b.decode(tokens[slice(*span)]), "scene: café")
        self.assertEqual(b.encode_text("<|im_end|>"), [257])
        opened, receipt = b.append_message(tokens, "assistant", "Recorded.", close=False)
        self.assertEqual(b.decode(opened[slice(*receipt)]), "Recorded.")
        finished = b.finish_assistant(opened)
        self.assertEqual(finished[:len(opened)], opened)
        self.assertTrue(b.decode(finished).endswith("Recorded.<|im_end|>\n"))
        self.assertEqual(b.finish_assistant(finished), finished)
        self.assertEqual(b.finish_assistant(finished[:-1]), finished)

    def test_feedback_encoding_is_causal_and_unknown_cells_remain_explicit(self):
        b = self.backend
        tokens, _ = self.append([], {"event": "task", "domain": "location", "question": "Where?"})
        tokens, _ = self.append(tokens, {"event": "peer_feedback", "peer": "B",
            "response": '{"answer":"garden"}', "correct": False})
        old = len(tokens) - 1
        tokens, _ = b.append_message(tokens, "assistant", "Recorded.", close=False)
        observed = b.observe(tokens, {63: [old, len(tokens) - 1], 31: [len(tokens) - 1]})
        expected = [.5, .5, .5, 0., 0., 0., 0., 1.]
        np.testing.assert_allclose(observed["captures"][63][old], expected)
        np.testing.assert_allclose(observed["captures"][63][len(tokens) - 1], expected)
        np.testing.assert_allclose(observed["captures"][31][len(tokens) - 1], np.array(expected) / 2)
        before = b.observe(tokens, {63: [0]})["captures"][63][0]
        np.testing.assert_allclose(before, [.5, .5, .5, .5, 0, 0, 0, 0])
        self.assertEqual(observed["logits"].shape, (259,))
        self.assertEqual(observed["captures"][63][old].dtype, np.float32)

    def test_source_uses_only_queried_subject_own_facts(self):
        event = {"event": "peer_task", "domain": "possession", "question": "What?",
            "query_subject": "Dana", "options": ["ball", "chair"],
            "private_facts": [{"subject": "Evan", "value": "ball"}],
            "analysis": {"true_answer": "chair", "world": {"Dana": "chair"}}}
        tokens, _ = self.append([], event)
        tokens, _ = self.backend.append_message(tokens, "assistant", "", close=False)
        absent = self.backend.generate(tokens, sampler={}, seed=2, max_new_tokens=64)
        self.assertEqual(json.loads(absent["text"]), {"answer": "unknown"})
        event["private_facts"].append({"subject": "Dana", "value": "ball"})
        found = self.generate([], event)
        self.assertEqual(json.loads(found["text"]), {"answer": "ball"})

    def test_diagnostics_use_observed_rates_and_roster_bindings(self):
        tokens = self.history()
        for domain, choice in (("possession", "B"), ("location", "A")):
            result = self.generate(tokens, {"event": "delegation_probe", "domain": domain})
            self.assertEqual(json.loads(result["text"]), {"choice": choice})
            self.assertIn(result["token_ids"][-1], self.backend.stop_ids)
            self.assertFalse(result["hit_cap"])
        result = self.generate(tokens, {"event": "capability_probe"})
        self.assertEqual(json.loads(result["text"]), {"A": {"possession": .25, "location": .75},
                                                       "B": {"possession": 1., "location": 0.}})
        result = self.generate(tokens, {"event": "identity_probe", "peer_name": "Maple"})
        self.assertEqual(json.loads(result["text"]), {"choice": "B"})

    def test_prior_checkpoint_patch_changes_one_later_decision(self):
        b = self.backend
        tokens = self.history()
        position, layer = len(tokens) - 1, 31
        donor = b.observe(tokens, {layer: [position]})["captures"][layer][position].copy()
        donor[2] = 0
        edit = {"layer": layer, "position": position, "kind": "patch",
                "basis": np.eye(8, dtype=np.float32)[:, 2:3], "donor": donor,
                "strength": 1., "seed": 3, "tag": "B-possession"}
        delayed = tokens + b.encode_text(" neutral context" * 20)
        result = self.generate(delayed, {"event": "delegation_probe", "domain": "possession"}, edits=[edit])
        self.assertEqual(json.loads(result["text"]), {"choice": "A"})
        result = self.generate(delayed, {"event": "delegation_probe", "domain": "location"}, edits=[edit])
        self.assertEqual(json.loads(result["text"]), {"choice": "A"})
        clean = b.observe(delayed, {layer: [position - 1, position], 63: [len(delayed) - 1]})
        changed = b.observe(delayed, {layer: [position - 1, position], 63: [len(delayed) - 1]}, edits=[edit])
        np.testing.assert_allclose(clean["captures"][layer][position - 1], changed["captures"][layer][position - 1])
        np.testing.assert_allclose(changed["captures"][63][len(delayed) - 1], [.25, .75, 0., 0., 4, 4, 4, 4])

    def test_delete_and_random_patch_match_realized_geometry(self):
        b = self.backend
        tokens = self.history()
        position = len(tokens) - 1
        clean = b.observe(tokens, {31: [position]})["captures"][31][position]
        base = {"layer": 31, "position": position, "basis": np.eye(8, dtype=np.float32)[:, :2],
                "strength": .5, "seed": 13, "tag": "geometry", "donor": np.ones(8, dtype=np.float32)}
        for kind in ("delete", "patch"):
            target = b.observe(tokens, {31: [position]}, edits=[dict(base, kind=kind)])
            random = b.observe(tokens, {31: [position]}, edits=[dict(base, kind="random_" + kind)])
            d1 = target["captures"][31][position] - clean
            d2 = random["captures"][31][position] - clean
            self.assertAlmostEqual(float(np.linalg.norm(d1)), float(np.linalg.norm(d2)), places=5)
            self.assertAlmostEqual(float(d1 @ clean), float(d2 @ clean), places=5)
            receipt = random["receipt"]["edits"][0]
            for key in ("requested_norm", "requested_inner_product", "realized_norm", "realized_inner_product",
                        "reference_realized_norm", "reference_realized_inner_product", "norm_matching_error",
                        "inner_product_matching_error", "rank", "dtype"):
                self.assertIn(key, receipt)

    def test_mask_filters_synthetic_evidence_without_claiming_attention(self):
        b = self.backend
        tokens, span = self.append([], {"event": "feedback", "peer": "A", "domain": "possession",
            "response": '{"answer":"ball"}', "correct": True})
        start = len(tokens)
        tokens, _ = b.append_message(tokens, "assistant", "Recorded.", close=False)
        mask = {"start_query": start, "key_spans": [span], "tag": "source"}
        clean = b.observe(tokens, {63: [len(tokens) - 1]})
        masked = b.observe(tokens, {63: [len(tokens) - 1]}, masks=[mask])
        self.assertEqual(float(clean["captures"][63][len(tokens) - 1][0]), 1.)
        self.assertEqual(float(masked["captures"][63][len(tokens) - 1][0]), .5)
        self.assertFalse(masked["receipt"]["actual_source_attention_verification"])
        self.assertEqual(masked["receipt"]["masks"][0]["tag"], "source")

    def test_sampling_is_local_and_caps_are_reported_honestly(self):
        tokens, _ = self.append([], {"event": "task", "domain": "possession"})
        tokens, _ = self.backend.append_message(tokens, "assistant", "", close=False)
        state = np.random.get_state()
        first = self.backend.generate(tokens, sampler={"temperature": .7}, seed=93, max_new_tokens=64)
        again = self.backend.generate(tokens, sampler={"temperature": .7}, seed=93, max_new_tokens=64)
        self.assertEqual(first["token_ids"], again["token_ids"])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        capped = self.backend.generate(tokens, sampler={}, seed=93, max_new_tokens=1)
        self.assertTrue(capped["hit_cap"])
        self.assertEqual(len(capped["token_ids"]), 1)
        self.assertEqual(capped["text"], "{")
        self.assertEqual(first["receipt"]["generated_tokens"], len(first["token_ids"]))

    def test_bad_tokens_sites_geometry_sampler_and_context_are_rejected(self):
        b = self.backend
        for tokens in ([], [True], [-1], [259]):
            with self.assertRaises(ValueError):
                b.observe(tokens, {})
        for capture in ({64: [0]}, {0: [1]}, {-1: [0]}):
            with self.assertRaises(ValueError):
                b.observe([1], capture)
        with self.assertRaises(ValueError):
            b.generate([1], sampler={"temperature": float("nan")}, seed=1, max_new_tokens=4)
        with self.assertRaises(ValueError):
            b.generate([1], sampler={}, seed=1, max_new_tokens=0)
        edit = {"layer": 0, "position": 0, "kind": "delete", "basis": np.ones((8, 1))}
        with self.assertRaises(ValueError):
            b.observe([1], {}, edits=[edit])
        with self.assertRaisesRegex(ValueError, "generated-position"):
            b.generate([1], sampler={}, seed=1, max_new_tokens=4,
                       edits=[dict(edit, position=1, basis=np.eye(8, dtype=np.float32)[:, :1])])
        with self.assertRaisesRegex(ValueError, "during generation"):
            b.generate([1, 2], sampler={}, seed=1, max_new_tokens=4,
                       masks=[{"start_query": 2, "key_spans": [(0, 1)]}])
        b.max_context = 2
        with self.assertRaises(ValueError):
            b.generate([1], sampler={}, seed=1, max_new_tokens=2)

    def test_runtime_can_validate_software_but_never_target_model_readiness(self):
        b = self.backend
        info, runtime = b.info(), b.validate_runtime()
        self.assertEqual(info["n_layers"], 64)
        self.assertEqual(info["hidden_size"], 8)
        self.assertTrue(info["synthetic"])
        self.assertFalse(info["target_gate"])
        self.assertTrue(runtime["passed"], runtime)
        self.assertTrue(runtime["synthetic"])
        self.assertFalse(runtime["target_gate"])
        self.assertFalse(runtime["actual_source_attention_verification"])
        self.assertFalse(runtime["checks"]["blocked_attention"]["supported"])
        self.assertNotEqual(runtime["checks"]["blocked_attention"]["passed"], True)
        self.assertEqual(runtime["fingerprint"], b.info()["fingerprint"])


if __name__ == "__main__":
    unittest.main()
