"""Runner integration checks use only the explicitly synthetic local backend."""
import copy
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jpeers.exp5.protocol import load_protocol
from jpeers.exp5.simulation import SimulationBackend
from jpeers.exp5 import runner


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def config(self):
        return runner.smoke_config(load_protocol(ROOT / "configs/exp5_study.json"))

    def destination(self, name):
        return ROOT / "runs" / ("exp5-test-" + name + "-" + uuid.uuid4().hex)

    def rows(self, directory):
        return [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]

    def test_preflight_is_read_only_and_requires_real_execution_budget(self):
        config = self.config()
        result = runner.preflight(config)
        self.assertFalse(result["ready_for_execution"])
        self.assertIn("execution_budget", result["checks"])
        self.assertFalse(result["checks"]["execution_budget"])

    def test_complete_simulation_handoff_remains_synthetic_and_detects_tampering(self):
        config = self.config()
        development = self.destination("development")
        manifest = runner.develop(config, development, SimulationBackend(), execute=True)
        self.assertEqual("complete", manifest["status"])
        self.assertFalse(manifest["ready_for_execution"])
        self.assertEqual(48, len([r for r in self.rows(development) if r["type"] == "turn"]))
        frozen = self.destination("frozen")
        with self.assertRaises(ValueError):
            runner.freeze(development, self.destination("forbidden"))
        runner.freeze(development, frozen, simulation=True)
        frozen_meta = json.loads((frozen / "frozen.json").read_text())
        self.assertFalse(frozen_meta["ready_for_execution"])
        self.assertFalse(frozen_meta["gates"]["runtime_validation_passed"])
        evaluation = self.destination("evaluation")
        evaluated = runner.evaluate(config, frozen, evaluation, SimulationBackend(), execute=True, simulation=True)
        self.assertEqual("complete", evaluated["status"])
        summary = runner.report(evaluation)
        self.assertTrue(summary["synthetic"])
        self.assertEqual(["P1", "P2", "P3", "P4", "P5", "P6", "P7"], summary["primary_registry"])
        self.assertTrue(all(not p["target_model_support"] for p in summary["primaries"].values()))
        self.assertEqual(2, summary["n_families"])
        self.assertLess(manifest["total_bytes"] + evaluated["total_bytes"] + (frozen / "manifest.json").stat().st_size, 1073741824)
        (evaluation / "records.jsonl").write_text("{}\n")
        with self.assertRaises(ValueError):
            runner.report(evaluation, self.destination("bad-report"))

    def test_budget_stop_is_retained_as_incomplete(self):
        config = self.config()
        config["limits"]["max_model_calls"] = 3
        destination = self.destination("budget")
        manifest = runner.develop(config, destination, SimulationBackend(), execute=True)
        self.assertEqual("incomplete", manifest["status"])
        self.assertEqual(3, manifest["model_calls"])
        self.assertTrue(any(r["type"] == "run_status" and r["incomplete"] for r in self.rows(destination)))
        with self.assertRaises(ValueError):
            runner.freeze(destination, self.destination("bad-freeze"), simulation=True)

    def test_unexecuted_development_never_creates_a_run(self):
        destination = self.destination("dry")
        result = runner.develop(self.config(), destination, None)
        self.assertFalse(result["ready_for_execution"])
        self.assertFalse(destination.exists())

    def engine(self, config, backend, name):
        artifact = dict(layer=0, mean=np.zeros(8), scale=np.ones(8), coef=np.eye(8)[:, :4]*64,
            intercept=np.zeros(4), counts=np.ones(4), bases={"delete": {}, "restore": {}},
            causal_settings=dict(layer=0, later_layer=2, rank=1, strength=1),
            identity_blind=dict(mean=np.zeros(4), scale=np.ones(4), coef=np.zeros((4, 4)),
                                intercept=np.ones(4)*.5, counts=np.ones(4)))
        return runner.Engine(config, backend, runner._store(config, self.destination(name), "evaluation"),
                             artifact, backend.validate_runtime())

    def test_diagnostic_calls_do_not_change_main_choices_or_exposures(self):
        config = self.config()
        config["evaluation"]["families"] = 1
        config["diagnostics"].update(conditions=["I0"], routes=["R0"], delays=[4])
        plain = self.engine(config, SimulationBackend(), "plain")
        measured = self.engine(config, SimulationBackend(), "measured")
        runner._run_sessions(plain, "evaluation", False)
        runner._run_sessions(measured, "evaluation", True)
        fields = ("session", "round", "peer", "choice_output", "observed_successes", "observed_exposures", "observed_unknown")
        turns = lambda engine: [{k:r[k] for k in fields} for r in engine.records if r["type"] == "turn"]
        self.assertEqual(turns(plain), turns(measured))
        self.assertGreater(measured.calls, plain.calls)
        causal = [r for r in measured.records if r["type"] == "causal"]
        self.assertTrue(causal)
        for row in causal:
            self.assertTrue(row["route_match_valid"])
            self.assertEqual(row["geometry"]["source_count"], row["geometry"]["control_count"])
            self.assertTrue(row["geometry"]["source_control_disjoint"])
        for turn in [r for r in measured.records if r["type"] == "turn"]:
            raw = measured.store.read_history(turn["history"])
            prompt = measured.backend.decode(raw)
            for hidden in ('"analysis"', '"true_answer"', '"world"', '"conditions"', '"reference_by_domain"', '"private_facts"'):
                self.assertNotIn(hidden, prompt)

    def test_unknown_peer_answer_is_an_observed_unrewarded_exposure(self):
        class UnknownBackend(SimulationBackend):
            def generate(self, tokens, **kwargs):
                result = super().generate(tokens, **kwargs)
                if self._events(tokens)[-1]["value"]["event"] == "peer_task":
                    text = '{"answer":"unknown"}'
                    result.update(text=text, token_ids=self.encode_text(text)+[257], hit_cap=False)
                return result
        config = self.config()
        config["evaluation"]["families"] = 1
        engine = self.engine(config, UnknownBackend(), "unknown")
        runner._run_sessions(engine, "evaluation", False)
        turns = [r for r in engine.records if r["type"] == "turn"]
        self.assertEqual(12, len(turns))
        self.assertTrue(all(r["unknown"] and r["reward"] == 0 and r["valid"] for r in turns))
        for turn in turns:
            self.assertEqual(turn["round"], sum(turn["observed_exposures"]))
            self.assertEqual(turn["observed_exposures"], turn["observed_unknown"])
            self.assertEqual([0, 0, 0, 0], turn["observed_successes"])

    def test_invalid_choice_never_reprompts_or_creates_source_exposure(self):
        class InvalidChoiceBackend(SimulationBackend):
            def generate(self, tokens, **kwargs):
                result = super().generate(tokens, **kwargs)
                events = self._events(tokens)
                if events[-1]["value"]["event"] != "peer_task":
                    result.update(text='{"choice":"C"}', token_ids=self.encode_text('{"choice":"C"}')+[257], hit_cap=False)
                return result
        config = self.config()
        config["evaluation"]["families"] = 1
        engine = self.engine(config, InvalidChoiceBackend(), "invalid")
        runner._run_sessions(engine, "evaluation", False)
        turns = [r for r in engine.records if r["type"] == "turn"]
        self.assertEqual(12, len(turns))
        self.assertTrue(all(not r["valid"] and r["reward"] == 0 for r in turns))
        self.assertTrue(all(r["observed_exposures"] == [0, 0, 0, 0] for r in turns))
        self.assertTrue(all(r["peer_call_id"] is None for r in turns))
        self.assertEqual(60, engine.calls)  # 24 fresh audits + 12 choices + 24 readouts.

    def test_terminal_checkpoint_uses_a_fresh_independent_task(self):
        config = self.config()
        config["evaluation"]["families"] = 1
        engine = self.engine(config, SimulationBackend(), "terminal")
        calls = []
        def checkpoint(engine, context, task, target_cell):
            calls.append((context, task))
        with patch.object(runner, "run_checkpoint", checkpoint):
            runner._run_sessions(engine, "evaluation", True)
        self.assertEqual(3, len(calls))
        for context, task in calls:
            self.assertTrue(context["terminal_continuation"])
            self.assertTrue(task["id"].startswith("audit:"))
            main_ids = {event["task"]["id"] for event in context["events"] if "task" in event}
            self.assertNotIn(task["id"], main_ids)

    def test_incomplete_report_retains_unstarted_families_and_blocks_claims(self):
        config = self.config()
        destination = self.destination("incomplete-report")
        store = runner._store(config, destination, "evaluation")
        store.append(dict(type="turn", family="0", session="correct_vs_incomplete", split="evaluation",
                          round=1, phase="initial", peer="A", domain="possession", valid=True, correct=True))
        runner._finish(store, "incomplete", synthetic=False, gates={key:True for key in
            ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed", "causal_discovery_passed", "ni_preflight_passed")})
        summary = runner.report(destination)
        self.assertEqual(["1"], summary["family_inventory"]["unstarted_families"])
        self.assertEqual(["0", "1"], summary["family_inventory"]["incomplete_families"])
        self.assertTrue(all(p["p"] == 1 and not p["target_model_support"] for p in summary["primaries"].values()))
        self.assertTrue(all(not p["validation_gates"]["execution_complete"] for p in summary["primaries"].values()))

    def candidate_rows(self):
        records = []
        for family in range(8):
            for route in ("R0", "R2"):
                for index in range(8):
                    records.append(dict(type="causal", family=str(family), session="anchor", route=route,
                        condition=f"I{index}", valid=True, choice="B" if index == 4 else "A",
                        correct=index not in (1, 5), donor_choice="B", status="available",
                        literal_correct=True, other_domain_correct=True, non_target_error=0,
                        control_match_valid=True, route_match_valid=True, source_attention_valid=False,
                        synthetic=False))
        return records

    def test_causal_candidate_cannot_pass_without_actual_source_attention(self):
        result = runner._candidate_score(self.candidate_rows(), self.config())
        self.assertFalse(result["passed"])
        self.assertFalse(result["source_attention_passed"])

    def test_candidate_cannot_pool_a_failed_route_with_a_successful_route(self):
        rows = self.candidate_rows()
        for row in rows:
            row["source_attention_valid"] = True
            if row["route"] == "R0":
                row.update(choice="A", correct=True)
        result = runner._candidate_score(rows, self.config())
        self.assertFalse(result["passed"])

    def test_preservation_is_required_for_each_condition_and_route(self):
        rows = self.candidate_rows()
        for row in rows:
            row["source_attention_valid"] = True
            if row["condition"] == "I1":
                row["non_target_error"] = .2
        result = runner._candidate_score(rows, self.config())
        self.assertFalse(result["passed"])

    def test_preflight_projects_work_and_storage_before_execution(self):
        result = runner.preflight(self.config(), backend_info=SimulationBackend().info(),
                                  output_dir=self.destination("forecast"))
        self.assertEqual(4, result["planned_inventory"]["families"])
        self.assertEqual(48, result["planned_inventory"]["main_turns"])
        self.assertGreater(result["forecast"]["model_calls_nominal"], 48)
        self.assertGreater(result["forecast"]["estimated_new_bytes"], 0)
        self.assertGreaterEqual(result["forecast"]["upper_bound_bytes"], result["forecast"]["estimated_new_bytes"])
        self.assertGreater(result["physical_free_bytes"], 0)

    def test_preflight_checks_combined_development_and_evaluation_storage(self):
        config = load_protocol(ROOT / "configs/exp5_study.json")
        result = runner.preflight(config)
        evaluation = runner.preflight(config, phase="evaluation")
        self.assertEqual(result["forecast"]["estimated_new_bytes"] + evaluation["forecast"]["estimated_new_bytes"],
                         result["study_forecast"]["estimated_minimum_bytes"])
        self.assertFalse(result["checks"]["planned_study_storage_within_allowance"])
        self.assertEqual(0, result["model_calls"])

    def test_evaluation_forecast_includes_verified_ancestor_bytes(self):
        config = self.config()
        ancestor_bytes = config["storage"]["total_bytes"]-1000
        with patch.object(runner, "_verify", return_value=(None, None, None, {"total_bytes": ancestor_bytes})) as verify:
            result = runner.preflight(config, phase="evaluation", ancestor_dir="frozen-handoff")
        verify.assert_called_once()
        self.assertEqual(ancestor_bytes, result["study_forecast"]["retained_ancestor_bytes"])
        self.assertFalse(result["checks"]["planned_study_storage_within_allowance"])

    def test_holdout_decoder_accuracy_cannot_replace_actual_routing_validity(self):
        config = self.config()
        engine = self.engine(config, SimulationBackend(), "holdout-invalid")
        rates = [.9, .3, .3, .9]
        rows = [dict(family=str(family), delay=4, captures={0:np.r_[np.array(rates)/64, np.zeros(4)]},
                     target=rates, mask=[True]*4, profile="binary:1001", valid=False,
                     routing_applicable=True, routing_correct=False) for family in (2, 3)]
        result = runner._heldout_validation(engine, rows, 2)
        self.assertFalse(result["passed"])

    def online_gate(self, attribution):
        records = [dict(type="readout", family=str(family), session=runner.COMPARISONS[0],
                        target=[.9], prediction=[.9], identity_blind_prediction=[.5], mask=[True])
                   for family in (2, 3)]
        engine = SimpleNamespace(records=records + attribution, config=self.config(), runtime_validation={})
        return runner._gates(engine, {"gates": {"accuracy": True}}, {"passed": True},
                             {"passed": True}, 2, {"passed": True})["online_transfer_passed"]

    def attribution_rows(self):
        return [dict(diagnostic="attribution", family=str(family), session=runner.COMPARISONS[1],
                     attribution_applicable=True, status="available", attribution_error_improvement=.1)
                for family in (2, 3)]

    def test_online_gate_cannot_use_controlled_anchor_attribution(self):
        rows = self.attribution_rows()
        for row in rows:
            row["session"] = "calibration:anchor"
        self.assertFalse(self.online_gate(rows))

    def test_online_gate_requires_applicable_evidence_for_every_heldout_family(self):
        self.assertFalse(self.online_gate(self.attribution_rows()[:1]))

    def test_online_gate_rejects_failed_or_nonfinite_applicable_attribution(self):
        for invalid in (dict(status="unavailable"), dict(attribution_error_improvement=float("nan")),
                        dict(attribution_error_improvement=float("inf"))):
            with self.subTest(invalid=invalid):
                rows = self.attribution_rows()
                rows[0].update(invalid)
                self.assertFalse(self.online_gate(rows))

    def test_online_gate_exempts_explicitly_unobserved_or_tied_attribution(self):
        rows = self.attribution_rows()
        rows.append(dict(diagnostic="attribution", family="2", session=runner.COMPARISONS[0],
                         attribution_applicable=False, status="not_applicable"))
        self.assertTrue(self.online_gate(rows))

    def development_fixture(self):
        config = self.config()
        engine = self.engine(config, SimulationBackend(), "freeze-fixture")
        engine.store.metadata["phase"] = "development"
        engine.artifact.update(version=1, kind="exp5-linear-readout")
        for name, payload in runner.artifact_payloads(engine.artifact).items():
            engine.store.write("shared", name, payload)
        engine.store.json("shared", "runtime.json", engine.runtime_validation)
        engine.store.json("shared", "backend.json", engine.backend.info())
        engine.store.json("shared", "validation.json", dict(gates={key:False for key in
            ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed", "causal_discovery_passed", "ni_preflight_passed")}))
        engine.store.json("shared", "noninferiority-preflight.json", dict(passed=False))
        runner._finish(engine.store, "complete", synthetic=True)
        return config, engine.store.root

    def test_freeze_can_finalize_sample_size_without_refitting(self):
        config, development = self.development_fixture()
        frozen = self.destination("sample-size")
        runner.freeze(development, frozen, simulation=True, evaluation_families=3)
        frozen_config = json.loads((frozen / "config.json").read_text())
        self.assertEqual(3, frozen_config["evaluation"]["families"])
        self.assertEqual(2, config["evaluation"]["families"])
        self.assertEqual((development / "probe.npz").read_bytes(), (frozen / "probe.npz").read_bytes())
        result = runner.evaluate(frozen_config, frozen, self.destination("sample-size-eval"), None, simulation=True)
        self.assertEqual(3, result["planned_inventory"]["families"])
        with self.assertRaises(ValueError):
            runner.evaluate(config, frozen, self.destination("old-size"), None, simulation=True)

    def test_real_execution_refuses_insufficient_storage_before_inference(self):
        class RealMetadataBackend(SimulationBackend):
            def info(self):
                result = super().info()
                result.update(synthetic=False, model_id="Qwen/Qwen3-32B",
                              revision="9216db5781bf21249d130ec9da846c4624c16137")
                return result
            def validate_runtime(self):
                raise AssertionError("Inference must not start")
        config = self.config()
        config.pop("software_test_protocol")
        config["limits"]["max_model_calls"] = 1
        config["storage"] = dict(total_bytes=100000, raw_development_bytes=20000,
            text_bytes=20000, shared_bytes=20000, compact_bytes=20000, manifest_reserve_bytes=10000)
        with self.assertRaises(ValueError):
            runner.develop(config, self.destination("storage-refusal"), RealMetadataBackend(), execute=True)


if __name__ == "__main__":
    unittest.main()
