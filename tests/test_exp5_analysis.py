"""Family-level inference and conservative primary-registry checks."""
import json
from pathlib import Path
import sys
from exp5_test_utils import temporary_directory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jpeers.exp5.analysis import family_test, holm, noninferiority_test, summarize_records, write_report, report_payloads


VALIDATED = dict(runtime_validation_passed=True, probe_validation_passed=True,
                 online_transfer_passed=True, causal_discovery_passed=True,
                 ni_preflight_passed=True, n_bootstrap=100)


def real_registry_records():
    """Eight independent families with every required matched primary measure."""
    rows = []
    for family in range(8):
        rows.extend([dict(type="readout", family=family, target=[1]*4, prediction=[1]*4,
                          mask=[True]*4, identity_blind_prediction=[.5]*4,
                          attribution_error_improvement=.1),
                     dict(type="update", family=family, q_initial=.2, q_reversal=1.,
                          q_confirming=.2, h_initial=.2, h_reversal=1., h_confirming=.2)])
        for peer in ("A", "B"):
            rows.append(dict(type="audit", family=family, session="s", phase="stationary",
                             domain="location", peer=peer, successes=int(peer == "B"), exposures=1))
        for i, peer in enumerate(("A", "B")):
            rows.append(dict(type="turn", family=family, session="s", round=i,
                             phase="stationary", domain="location", peer=peer, valid=True))
        for route in ("R0", "R2", "R2_control"):
            for condition in ("I0", "I1", "I2", "I3", "I4", "I5", "I6"):
                rows.append(dict(type="causal", family=family, fork_id="x", route=route,
                                 delay=512,
                                 condition=condition, valid=True, correct=condition not in ("I1", "I5"),
                                 choice="B" if condition == "I4" else "A", donor_choice="B",
                                 control_match_valid=True, route_match_valid=True, source_attention_valid=True))
    return rows


class FamilyTests(unittest.TestCase):
    def test_exact_sign_flip_uses_families_and_conservative_ties(self):
        result = family_test([1, 1, 1], n_bootstrap=1000)
        self.assertEqual(result["p"], .125)
        self.assertEqual(result["n_families"], 3)
        self.assertEqual(result["ci"], [1., 1.])
        self.assertEqual(result["assignments"], 8)
        self.assertEqual(family_test([0]*3, n_bootstrap=1000)["p"], 1.)
        self.assertEqual(family_test([], n_bootstrap=1000)["status"], "unavailable")

    def test_batched_default_bootstrap_and_exact_twenty_family_boundary(self):
        result = family_test([1.]*20)
        self.assertEqual(result["assignments"], 1048576)
        self.assertEqual(result["bootstrap_resamples"], 100000)
        self.assertEqual(result["p"], 1 / 1048576)

    def test_seeded_monte_carlo_and_separate_noninferiority(self):
        first = family_test([.1]*21, seed=4, n_permutations=400, n_bootstrap=800)
        second = family_test([.1]*21, seed=4, n_permutations=400, n_bootstrap=800)
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "monte_carlo_sign_flip")
        self.assertLess(first["p"], .01)
        ni = noninferiority_test([0]*8, margin=.05, n_bootstrap=1000)
        self.assertLess(ni["p"], .01)
        self.assertEqual(ni["null_boundary"], -.05)
        boundary = noninferiority_test([-.05]*8, margin=.05, n_bootstrap=1000)
        self.assertEqual(boundary["p"], 1.)

    def test_holm_keeps_all_seven_and_stepdown(self):
        result = holm({"P1": .001, "P2": .009})
        self.assertEqual(set(result), {f"P{i}" for i in range(1, 8)})
        self.assertTrue(result["P1"]["reject"])
        self.assertFalse(result["P2"]["reject"])
        self.assertEqual(result["P7"]["p"], 1.)
        self.assertAlmostEqual(result["P2"]["adjusted_p"], .054)

    def test_missing_primaries_and_synthetic_are_explicit_and_reportable(self):
        records = [dict(type="turn", family=1, session="a", round=0,
                        synthetic=True, split="evaluation", valid=False, peer=None,
                        correct=False, reward=0, domain="possession", phase="stationary",
                        observed_successes=[1,0,0,0], observed_exposures=[1,0,0,0])]
        summary = summarize_records(records, {"analysis": {"n_bootstrap": 500}})
        self.assertEqual(len(summary["primaries"]), 7)
        self.assertTrue(summary["synthetic"])
        self.assertFalse(summary["target_model_evidence"])
        self.assertTrue(summary["incomplete"])
        self.assertEqual(summary["allocation"]["unconditional"]["invalid"], 1.)
        self.assertEqual(summary["coverage"]["unobserved_cells"], 3)
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with temporary_directory(dir=root) as tmp:
            write_report(Path(tmp), summary)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertTrue(report["synthetic"])
            self.assertIn("Synthetic", (Path(tmp) / "report.md").read_text())
            self.assertTrue(report["trajectories"])
            payloads = report_payloads(summary)
            self.assertEqual(payloads["report.json"], (Path(tmp) / "report.json").read_bytes())
            self.assertEqual(payloads["report.md"], (Path(tmp) / "report.md").read_bytes())

    def test_conjunction_cannot_hide_failed_component_or_missing_r0(self):
        rows = []
        for f in range(4):
            for cond, correct in (("I0", True), ("I1", False), ("I2", True),
                                  ("I3", True), ("I5", True)):
                rows.append(dict(type="causal", family=f, fork_id="x", route="R2",
                                 delay=512,
                                 condition=cond, valid=True, correct=correct,
                                 choice="A", donor_choice="B", split="evaluation"))
        summary = summarize_records(rows, {"analysis": {"n_bootstrap": 500}})
        self.assertEqual(summary["primaries"]["P5"]["p"], 1.)
        p7 = summary["primaries"]["P7"]
        self.assertEqual(p7["components"]["rescue_vs_random"]["estimate"], 0.)
        self.assertEqual(p7["p"], 1.)

    def test_update_requires_absolute_improvement_and_branch_separation(self):
        rows = [dict(type="update", family=f, split="evaluation", q_initial=.2,
                     q_reversal=1., q_confirming=.2, h_initial=.2,
                     h_reversal=.1, h_confirming=0.) for f in range(8)]
        p3 = summarize_records(rows, {"analysis": {"n_bootstrap": 500}})["primaries"]["P3"]
        self.assertLess(p3["components"]["absolute_improvement"]["estimate"], 0)
        self.assertGreater(p3["components"]["branch_separation"]["estimate"], 0)
        self.assertEqual(p3["p"], 1.)

    def test_p2_needs_independent_audits_and_uses_no_observed_quality_labels(self):
        turns = [dict(type="turn", family=f, session="s", split="evaluation", round=i,
                      peer="A" if i == 0 else "B", domain="possession", phase="stationary",
                      valid=True, correct=True, observed_successes=[9,0,1,0],
                      observed_exposures=[10,0,10,0]) for f in range(8) for i in range(2)]
        config = {"analysis": {"n_bootstrap": 500, "early_rounds": [0], "late_rounds": [1]}}
        missing = summarize_records(turns, config)["primaries"]["P2"]
        self.assertEqual(missing["status"], "unavailable")
        audits = [dict(type="audit", family=f, session="s", phase="stationary", domain="possession",
                       peer=p, valid=True, correct=p == "B", split="evaluation")
                  for f in range(8) for p in ("A", "B")]
        p2 = summarize_records(turns + audits, config)["primaries"]["P2"]
        self.assertEqual(p2["components"]["late_minus_early"]["estimate"], 1.)
        self.assertEqual(p2["components"]["late_above_random"]["estimate"], .5)

    def test_repeated_turns_do_not_increase_independent_family_count(self):
        rows = [dict(type="readout", family=f, split="evaluation", target=[1]*4,
                     prediction=[1]*4, mask=[True]*4, identity_blind_prediction=[.5]*4,
                     attribution_error_improvement=.1) for f in range(3) for _ in range(20)]
        p1 = summarize_records(rows, {"analysis": VALIDATED})["primaries"]["P1"]
        self.assertEqual(p1["components"]["prediction_improvement"]["n_families"], 3)
        self.assertEqual(p1["p"], .125)

    def test_aggregated_independent_audit_counts_are_used(self):
        rows = []
        for f in range(3):
            rows.extend([dict(type="audit", family=f, peer="A", domain="location", phase="stationary", successes=9, exposures=10),
                         dict(type="audit", family=f, peer="B", domain="location", phase="stationary", successes=2, exposures=10)])
            for i, p in enumerate(("B", "A")):
                rows.append(dict(type="turn", family=f, round=i, phase="stationary", domain="location", peer=p, valid=True))
        result = summarize_records(rows, {"analysis": {"early_rounds": [0], "late_rounds": [1], "n_bootstrap": 100}})
        self.assertEqual(result["primaries"]["P2"]["components"]["late_minus_early"]["estimate"], 1.)
        self.assertEqual(result["audits"][0]["n"], 10.)

    def test_missing_family_primary_observations_do_not_select_survivors(self):
        rows = [dict(type="readout", family=f, target=[1]*4, prediction=[1]*4,
                     mask=[True]*4, identity_blind_prediction=[.5]*4,
                     attribution_error_improvement=.1) for f in range(3)]
        rows.append(dict(type="turn", family=3, round=0, valid=False))
        result = summarize_records(rows, {"analysis": {"n_bootstrap": 100}})
        self.assertEqual(result["primaries"]["P1"]["status"], "unavailable")

    def test_failed_realized_control_or_route_geometry_keeps_primaries_unavailable(self):
        rows = []
        for family in range(8):
            for route in ("R0", "R2", "R2_control"):
                for cond in ("I0", "I1", "I2", "I3", "I4", "I5", "I6"):
                    rows.append(dict(type="causal", family=family, fork_id="x", route=route,
                                     delay=512,
                                     condition=cond, valid=True, correct=cond != "I1",
                                     choice="B" if cond == "I4" else "A", donor_choice="B",
                                     control_match_valid=not (family == 0 and cond in ("I2", "I5", "I6")),
                                     route_match_valid=False if family == 0 else True))
        result = summarize_records(rows, {"analysis": {"n_bootstrap": 100, "ni_preflight_passed": True}})
        for key in ("P4", "P5", "P6", "P7"):
            self.assertEqual(result["primaries"][key]["p"], 1.)
            self.assertEqual(result["primaries"][key]["status"], "unavailable")
            self.assertTrue(result["primaries"][key]["gate_failures"])

    def test_missing_causal_receipts_are_not_equivalent_to_passing_receipts(self):
        for field in ("control_match_valid", "route_match_valid", "source_attention_valid"):
            for state in ("missing", None, False, "true", 1):
                with self.subTest(receipt=field, state=state):
                    rows = real_registry_records()
                    for row in rows:
                        if row["type"] == "causal":
                            if state == "missing":
                                row.pop(field)
                            else:
                                row[field] = state
                    result = summarize_records(rows, {"analysis": VALIDATED})
                    for key in ("P4", "P5", "P6", "P7"):
                        self.assertEqual(result["primaries"][key]["status"], "unavailable")
                        self.assertEqual(result["primaries"][key]["p"], 1.)
                        self.assertIn(field, result["primaries"][key]["gate_failures"])

    def test_explicit_validation_gates_apply_to_the_registered_primaries(self):
        records = real_registry_records()
        baseline = summarize_records(records, {"analysis": VALIDATED})
        self.assertTrue(all(p["target_model_support"] for p in baseline["primaries"].values()))
        applicability = {"runtime_validation_passed": {"P1","P2","P3","P4","P5","P6","P7"},
                         "probe_validation_passed": {"P1","P3","P5","P6","P7"},
                         "online_transfer_passed": {"P1","P3","P5","P6","P7"},
                         "causal_discovery_passed": {"P5","P6","P7"},
                         "ni_preflight_passed": {"P4"}}
        for field, affected in applicability.items():
            for state in ("missing", False, "true"):
                with self.subTest(gate=field, state=state):
                    settings = dict(VALIDATED)
                    if state == "missing":
                        settings.pop(field)
                    else:
                        settings[field] = state
                    result = summarize_records(records, {"analysis": settings})
                    for key, primary in result["primaries"].items():
                        if key in affected:
                            self.assertEqual(primary["status"], "unavailable")
                            self.assertEqual(primary["p"], 1.)
                            self.assertFalse(primary["target_model_support"])
                            self.assertFalse(primary["validation_gates"][field])
                            self.assertIn(field, primary["gate_failures"])
                        else:
                            self.assertEqual(primary["status"], "available")
        synthetic = [dict(r, synthetic=True) for r in records]
        result = summarize_records(synthetic, {"analysis": VALIDATED})
        self.assertFalse(any(p["target_model_support"] for p in result["primaries"].values()))

    def test_quality_audits_match_session_without_cross_condition_borrowing(self):
        rows = []
        for family in range(3):
            for session, better in (("incomplete", "A"), ("incorrect", "B")):
                for peer in ("A", "B"):
                    rows.append(dict(type="audit", family=family, session=session,
                                     phase="stationary", domain="location", peer=peer,
                                     successes=9 if peer == better else 1, exposures=10))
                for i, peer in enumerate(("B" if better == "A" else "A", better)):
                    rows.append(dict(type="turn", family=family, session=session, round=i,
                                     phase="stationary", domain="location", peer=peer, valid=True))
        result = summarize_records(rows, {"analysis": VALIDATED})
        self.assertEqual(result["primaries"]["P2"]["components"]["late_minus_early"]["estimate"], 1.)
        self.assertEqual(len(result["audits"]), 12)
        mismatched = [r for r in rows if not (r["type"] == "audit" and r["session"] == "incorrect")]
        result = summarize_records(mismatched, {"analysis": VALIDATED})
        self.assertEqual(result["primaries"]["P2"]["status"], "unavailable")

    def test_learning_counts_use_session_phase_order_and_explicit_lists_override(self):
        rows = []
        for family in range(3):
            for session, offset in (("one", 100), ("two", 1000)):
                for phase, phase_offset in (("initial", 0), ("reversed", 100), ("equal", 200)):
                    for peer in ("A", "B"):
                        rows.append(dict(type="audit", family=family, session=session, phase=phase,
                                         domain="location", peer=peer, successes=int(peer == "B"), exposures=1))
                    peers = ("A", "A", "B", "B", "B", "B", "A", "B") if phase == "initial" else ("B", "B", "B", "A")
                    for i, peer in enumerate(peers):
                        rows.append(dict(type="turn", family=family, session=session, phase=phase,
                                         round=offset + phase_offset + i, domain="location", peer=peer,
                                         stationary=True, valid=True))
        settings = dict(VALIDATED, early_count=3, late_count=1)
        counted = summarize_records(rows, {"analysis": settings})["primaries"]["P2"]
        self.assertAlmostEqual(counted["components"]["late_minus_early"]["estimate"], 2 / 3)
        self.assertEqual(counted["components"]["late_above_random"]["estimate"], .5)
        explicit = dict(settings, early_rounds=[100, 1000], late_rounds=[106, 1006])
        listed = summarize_records(rows, {"analysis": explicit})["primaries"]["P2"]
        self.assertEqual(listed["components"]["late_minus_early"]["estimate"], 0.)
        default = summarize_records(rows, {"analysis": VALIDATED})["primaries"]["P2"]
        self.assertEqual(default["components"]["late_minus_early"]["estimate"], .5)

    def test_retention_events_are_displayed_without_entering_p1_readouts(self):
        rows = real_registry_records()
        for family in range(8):
            rows.append(dict(type="retention", family=family, session="s", round=10,
                             delay=2048, prediction=[.2]*4, raw=[.2]*4,
                             target=[1]*4, mask=[True]*4, synthetic=False))
        result = summarize_records(rows, {"analysis": VALIDATED})
        self.assertEqual(len(result["retention"]), 16)
        self.assertEqual(sum(r["delay"] == 2048 for r in result["retention"]), 8)
        p1 = result["primaries"]["P1"]
        self.assertEqual(p1["status"], "available")
        self.assertEqual(p1["components"]["prediction_improvement"]["estimate"], .5)

    def test_only_registered_delay_forks_enter_causal_primaries(self):
        rows = real_registry_records()
        for family in range(8):
            for delay in (0, 128):
                for route in ("R0", "R2", "R2_control"):
                    rows.append(dict(type="causal", family=family, delay=delay, fork_id=f"secondary-{delay}",
                                     route=route, condition="I0", valid=False, correct=False,
                                     control_match_valid=False, route_match_valid=False, source_attention_valid=False))
                    rows.append(dict(type="retention", family=family, delay=delay, route=route))
        result = summarize_records(rows, {"analysis": VALIDATED})
        for key in ("P4", "P5", "P6", "P7"):
            self.assertEqual(result["primaries"][key]["status"], "available")
        self.assertEqual(len(result["retention"]), 56)
        self.assertEqual(result["specificity"]["I0"]["valid"]["n_assigned"], 72)
        missing = [dict(r) for r in rows]
        next(r for r in missing if r.get("family") == 0 and r.get("route") == "R2"
             and r.get("condition") == "I3" and r.get("delay") == 512).pop("delay")
        self.assertEqual(summarize_records(missing, {"analysis": VALIDATED})["primaries"]["P7"]["status"], "unavailable")
        shifted = [dict(r, delay=1024) if r.get("type") == "causal" and r.get("delay") == 512 else dict(r) for r in rows]
        fallback = summarize_records(shifted, {"analysis": VALIDATED, "diagnostics": {"primary_delay": 1024}})
        self.assertEqual(fallback["primaries"]["P5"]["status"], "available")
        override = summarize_records(shifted, {"analysis": dict(VALIDATED, primary_delay=512), "diagnostics": {"primary_delay": 1024}})
        self.assertEqual(override["primaries"]["P5"]["status"], "unavailable")

    def test_missing_assigned_attribution_diagnostics_cannot_select_survivors(self):
        for value in ("missing", None, float("nan"), float("inf")):
            with self.subTest(value=value):
                rows = real_registry_records()
                for row in rows:
                    row.pop("attribution_error_improvement", None)
                for family in range(8):
                    rows.extend([dict(type="diagnostic", diagnostic="attribution", family=family,
                                      attribution_error_improvement=.1),
                                 dict(type="diagnostic", diagnostic="attribution", family=family,
                                      attribution_error_improvement=.2)])
                failed = rows[-1]
                if value == "missing":
                    failed.pop("attribution_error_improvement")
                else:
                    failed["attribution_error_improvement"] = value
                p1 = summarize_records(rows, {"analysis": VALIDATED})["primaries"]["P1"]
                self.assertEqual(p1["status"], "unavailable")
                self.assertEqual(p1["p"], 1.)
                self.assertEqual(p1["components"]["attribution_control"]["missing_measurements"], 1)
                self.assertEqual(p1["components"]["prediction_improvement"]["status"], "available")
                failed["attribution_applicable"] = False
                p1 = summarize_records(rows, {"analysis": VALIDATED})["primaries"]["P1"]
                self.assertEqual(p1["status"], "available")
        rows = real_registry_records()
        rows[0]["attribution_error_improvement"] = None
        self.assertEqual(summarize_records(rows, {"analysis": VALIDATED})["primaries"]["P1"]["status"], "unavailable")

    def test_expected_family_inventory_keeps_absent_families_and_bounds(self):
        rows = real_registry_records()
        for config in ({"analysis": dict(VALIDATED, expected_families=list(range(9)))},
                       {"analysis": VALIDATED, "evaluation": {"families": 96}}):
            with self.subTest(config=config):
                result = summarize_records(rows, config)
                for primary in result["primaries"].values():
                    self.assertEqual(primary["status"], "unavailable")
                    self.assertEqual(primary["p"], 1.)
                    self.assertFalse(primary["target_model_support"])
                    component = next(iter(primary["components"].values()))
                    self.assertEqual(component["unresolved_effect_bounds"], [-1., 1.])
                    self.assertTrue(component["family_effects"])
        # A count never licenses inventing numeric IDs for absent families.
        named = [dict(r, family=f"source-{r['family']}") for r in rows]
        result = summarize_records(named, {"analysis": VALIDATED, "evaluation": {"families": 9}})
        self.assertEqual(result["family_inventory"]["observed"], [f"source-{i}" for i in range(8)])
        self.assertEqual(len(result["family_inventory"]["missing"]), 1)
        exact = summarize_records(rows, {"analysis": dict(VALIDATED, expected_families=list(range(8))), "evaluation": {"families": 96}})
        self.assertTrue(all(p["target_model_support"] for p in exact["primaries"].values()))

    def test_execution_incomplete_invalidates_support_without_hiding_components(self):
        for status in (dict(status="incomplete"), dict(status="error"), dict(uncompleted=True), dict(completed=False)):
            with self.subTest(status=status):
                rows = real_registry_records() + [dict(type="run_status", **status)]
                result = summarize_records(rows, {"analysis": VALIDATED})
                self.assertTrue(result["execution_incomplete"])
                for primary in result["primaries"].values():
                    self.assertEqual(primary["status"], "unavailable")
                    self.assertEqual(primary["p"], 1.)
                    self.assertFalse(primary["target_model_support"])
                    self.assertTrue(all(c["status"] == "available" for c in primary["components"].values()))
        rows = real_registry_records() + [dict(type="run_status", status="incomplete")]
        result = summarize_records(rows, {"analysis": VALIDATED, "evaluation": {"families": 96}})
        self.assertFalse(any(p["target_model_support"] for p in result["primaries"].values()))

    def test_assay_unavailability_alone_does_not_block_behavioral_learning(self):
        rows = real_registry_records()
        rows = [r for r in rows if r["type"] != "causal"]
        rows.extend([dict(type="causal", family=f, delay=512, fork_id="unavailable", route="R2", condition="I0",
                          status="unavailable", reason="no applicable donor", valid=False) for f in range(8)])
        rows.append(dict(type="diagnostic", diagnostic="attribution", family=0,
                         status="unavailable", attribution_applicable=False))
        result = summarize_records(rows, {"analysis": VALIDATED})
        self.assertFalse(result["execution_incomplete"])
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["primaries"]["P2"]["status"], "available")
        self.assertTrue(result["primaries"]["P2"]["target_model_support"])


if __name__ == "__main__":
    unittest.main()
