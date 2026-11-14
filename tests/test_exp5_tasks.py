"""Standalone world-task checks; no model imports, downloads, or temporary files."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jpeers.exp5 import tasks


def task_config():
    return {
        "tasks": {
            "seed": 20261111,
            "rounds": 40,
            "reversal_round": None,
            "peer_names": {
                "development": ["Cedar", "Maple", "River", "Flint"],
                "evaluation": ["Birch", "Stone", "Willow", "Brook"],
            },
            "subjects": {
                "development": ["Alice", "Bob", "Casey"],
                "evaluation": ["Dana", "Evan", "Finley"],
            },
            "objects": ["ball", "chair"],
            "locations": ["garden", "kitchen"],
        },
        "development": {
            "observations_per_cell": 10,
            "high_rate": 0.9,
            "low_rate": 0.3,
            "anchor_rate": 0.6,
        },
        "audits": {"cases_per_domain": 20},
    }


class SessionTaskTests(unittest.TestCase):
    def test_domain_schedule_is_balanced_exogenous_and_worlds_are_matched(self):
        config = task_config()
        base = tasks.session_tasks(config, "development", 3, "equal_information")
        self.assertEqual(len(base), 40)
        self.assertEqual([r["domain"] for r in base].count("possession"), 20)
        self.assertEqual([r["domain"] for r in base].count("location"), 20)
        self.assertEqual(len({r["id"] for r in base}), 40)
        for comparison in ("correct_vs_incomplete", "correct_vs_incorrect"):
            rows = tasks.session_tasks(config, "development", 3, comparison)
            for left, right in zip(base, rows):
                for key in ("domain", "question", "query_subject", "options"):
                    self.assertEqual(left[key], right[key])
                for key in ("world", "true_answer", "reference_by_domain", "phase"):
                    self.assertEqual(left["analysis"][key], right["analysis"][key])
        self.assertEqual(base, tasks.session_tasks(config, "development", 3, "equal_information"))

    def test_each_scene_has_independent_hidden_facts_and_exact_packet_schedules(self):
        config = task_config()
        rows = tasks.session_tasks(config, "development", 9, "correct_vs_incomplete")
        self.assertGreater(len({json.dumps(r["analysis"]["world"], sort_keys=True) for r in rows}), 5)
        for comparison in ("equal_information", "correct_vs_incomplete", "correct_vs_incorrect"):
            rows = tasks.session_tasks(config, "development", 9, comparison)
            for domain in ("possession", "location"):
                subset = [r for r in rows if r["domain"] == domain]
                for peer in ("A", "B"):
                    counts = [0, 0]
                    condition = subset[0]["analysis"]["conditions"][peer]
                    for row in subset:
                        world = row["analysis"]["world"]
                        self.assertEqual(row["analysis"]["true_answer"], world[row["query_subject"]])
                        packet = row["private_facts"][peer]
                        self.assertEqual(len(packet), 2)
                        self.assertEqual(len({f["subject"] for f in packet}), 2)
                        for fact in packet:
                            if fact["subject"] == row["query_subject"]:
                                counts[0] += 1
                                counts[1] += fact["value"] == world[fact["subject"]]
                            else:
                                self.assertEqual(fact["value"], world[fact["subject"]])
                        if condition != "incorrect":
                            self.assertTrue(all(f["value"] == world[f["subject"]] for f in packet))
                    self.assertEqual(tuple(counts), {
                        "correct": (20, 20), "incomplete": (10, 10), "incorrect": (20, 10),
                    }[condition])

    def test_equal_information_packets_are_identical(self):
        for row in tasks.session_tasks(task_config(), "development", 1, "equal_information"):
            self.assertEqual(row["private_facts"]["A"], row["private_facts"]["B"])

    def test_roster_identity_order_and_reference_assignments_vary_independently(self):
        config = task_config()
        assignments, presentations = set(), set()
        for family in range(64):
            roster = tasks.roster(config, "development", family)
            self.assertEqual(set(roster), {"A", "B"})
            self.assertEqual(len(set(roster.values())), 2)
            rows = tasks.session_tasks(config, "development", family, "correct_vs_incomplete")
            reference = rows[0]["analysis"]["reference_by_domain"]
            assignments.add((reference["possession"], reference["location"]))
            self.assertTrue(all(r["analysis"]["reference_by_domain"] == reference for r in rows))
            presentations.add((tuple(roster), tuple(sorted(roster.items()))))
        self.assertEqual(assignments, {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")})
        self.assertGreaterEqual(len(presentations), 8)

    def test_split_names_subjects_and_template_families_do_not_overlap(self):
        config = task_config()
        names, questions, subjects = {}, {}, {}
        for split in ("development", "evaluation"):
            names[split], questions[split], subjects[split] = set(), set(), set()
            for family in range(4):
                names[split].update(tasks.roster(config, split, family).values())
                for row in tasks.session_tasks(config, split, family, "equal_information"):
                    subjects[split].update(row["analysis"]["world"])
                    text = row["question"]
                    for subject in config["tasks"]["subjects"][split]:
                        text = text.replace(subject, "SUBJECT")
                    questions[split].add(text)
                    self.assertIn("fresh", text.lower())
        self.assertFalse(names["development"] & names["evaluation"])
        self.assertFalse(subjects["development"] & subjects["evaluation"])
        self.assertFalse(questions["development"] & questions["evaluation"])
        self.assertGreaterEqual(len(questions["development"]), 4)
        self.assertGreaterEqual(len(questions["evaluation"]), 4)

    def test_reversal_changes_conditions_at_boundary_and_keeps_identities(self):
        config = task_config()
        config["tasks"]["reversal_round"] = 20
        before = tasks.roster(config, "development", 2)
        rows = tasks.session_tasks(config, "development", 2, "correct_vs_incorrect")
        self.assertEqual([r["analysis"]["phase"] for r in rows], ["initial"] * 20 + ["reversed"] * 20)
        first = rows[0]["analysis"]["reference_by_domain"]
        for index, row in enumerate(rows):
            reference = row["analysis"]["reference_by_domain"]
            for domain in ("possession", "location"):
                self.assertEqual(reference[domain], first[domain] if index < 20 else {"A": "B", "B": "A"}[first[domain]])
        self.assertEqual(before, tasks.roster(config, "development", 2))
        for phase, domain in itertools.product(("initial", "reversed"), ("possession", "location")):
            subset = [r for r in rows if r["analysis"]["phase"] == phase and r["domain"] == domain]
            self.assertEqual(len(subset), 10)
            for peer in ("A", "B"):
                if subset[0]["analysis"]["conditions"][peer] == "incorrect":
                    correct_inputs = sum(any(f["subject"] == r["query_subject"] and f["value"] == r["analysis"]["true_answer"] for f in r["private_facts"][peer]) for r in subset)
                    self.assertEqual(correct_inputs, 5)

    def test_audits_are_balanced_independent_and_cover_each_reversal_phase(self):
        config = task_config()
        config["tasks"]["reversal_round"] = 20
        main = tasks.session_tasks(config, "development", 0, "correct_vs_incomplete")
        audits = tasks.session_tasks(config, "development", 0, "correct_vs_incomplete", audit=True)
        self.assertEqual(len(audits), 80)
        self.assertFalse({r["id"] for r in main} & {r["id"] for r in audits})
        for phase, domain in itertools.product(("initial", "reversed"), ("possession", "location")):
            self.assertEqual(sum(r["analysis"]["phase"] == phase and r["domain"] == domain for r in audits), 20)
            main_worlds = [r["analysis"]["world"] for r in main if r["analysis"]["phase"] == phase and r["domain"] == domain]
            audit_worlds = [r["analysis"]["world"] for r in audits if r["analysis"]["phase"] == phase and r["domain"] == domain]
            self.assertNotEqual(main_worlds, audit_worlds[:len(main_worlds)])
        changed = copy.deepcopy(config)
        changed["tasks"]["rounds"] = 80
        self.assertEqual(audits, tasks.session_tasks(changed, "development", 0, "correct_vs_incomplete", audit=True))
        changed = copy.deepcopy(config)
        changed["audits"]["cases_per_domain"] = 40
        tasks.session_tasks(changed, "development", 0, "correct_vs_incomplete", audit=True)
        self.assertEqual(main, tasks.session_tasks(changed, "development", 0, "correct_vs_incomplete"))

    def test_builders_do_not_consume_global_random_state(self):
        config = task_config()
        before = random.getstate()
        tasks.session_tasks(config, "development", 1, "correct_vs_incomplete")
        tasks.session_tasks(config, "development", 1, "correct_vs_incomplete", audit=True)
        tasks.calibration_tasks(config, 1, [0.9, 0.3, 0.3, 0.9])
        self.assertEqual(before, random.getstate())


class CalibrationTaskTests(unittest.TestCase):
    def test_all_binary_profiles_and_eight_single_cell_anchors(self):
        profiles = tasks.calibration_profiles(task_config())
        self.assertEqual(len(profiles), 24)
        self.assertEqual(len({p["id"] for p in profiles}), 24)
        core = [p["rates"] for p in profiles if set(p["rates"]) <= {0.3, 0.9}]
        self.assertEqual({tuple(p) for p in core}, set(itertools.product((0.3, 0.9), repeat=4)))
        anchors = [p["rates"] for p in profiles if 0.6 in p["rates"]]
        self.assertEqual(len(anchors), 8)
        for target in range(4):
            competitor = (target + 2) % 4
            pairs = [(left, right) for left in anchors for right in anchors if left[target] == 0.3 and right[target] == 0.9 and all(left[c] == right[c] for c in range(4) if c != target)]
            self.assertTrue(any(left[competitor] == 0.6 for left, _ in pairs))

    def test_scripted_answers_have_real_checker_consistent_success_rates(self):
        config = task_config()
        order = (("A", "possession"), ("A", "location"), ("B", "possession"), ("B", "location"))
        for profile in tasks.calibration_profiles(config):
            rows = tasks.calibration_tasks(config, 4, profile["rates"])
            self.assertEqual(len(rows), 40)
            self.assertEqual(len({r["id"] for r in rows}), 40)
            for index, cell in enumerate(order):
                subset = [r for r in rows if (r["peer"], r["domain"]) == cell]
                self.assertEqual(len(subset), 10)
                self.assertEqual(sum(r["correct"] for r in subset), round(profile["rates"][index] * 10))
            for row in rows:
                result = tasks.grade_peer(row, row["response"])
                self.assertTrue(result["valid"])
                self.assertFalse(result["unknown"])
                self.assertEqual(row["correct"], result["correct"])
                self.assertEqual(json.loads(row["response"])["answer"] == row["analysis"]["true_answer"], row["correct"])

    def test_counterfactual_profiles_keep_worlds_order_and_nuisance_settings_matched(self):
        config = task_config()
        low = tasks.calibration_tasks(config, 5, [0.3, 0.9, 0.3, 0.9])
        high = tasks.calibration_tasks(config, 5, [0.9, 0.9, 0.3, 0.9])
        for left, right in zip(low, high):
            for key in ("id", "domain", "question", "query_subject", "options", "peer", "private_facts"):
                self.assertEqual(left[key], right[key])
            self.assertEqual(left["analysis"], right["analysis"])
            if (left["peer"], left["domain"]) != ("A", "possession"):
                self.assertEqual(left["response"], right["response"])
        cells = [(r["peer"], r["domain"]) for r in low]
        self.assertNotEqual(cells[:10], [("A", "possession")] * 10)
        for cell in (("A", "possession"), ("A", "location"), ("B", "possession"), ("B", "location")):
            low_last = [r["correct"] for r in low if (r["peer"], r["domain"]) == cell][-1]
            high_last = [r["correct"] for r in high if (r["peer"], r["domain"]) == cell][-1]
            self.assertEqual(low_last, high_last)


class TaskBoundaryTests(unittest.TestCase):
    def row(self):
        return tasks.session_tasks(task_config(), "development", 0, "correct_vs_incorrect")[0]

    def test_public_and_each_peer_only_receive_their_allowlisted_information(self):
        row = self.row()
        visible = tasks.public_task(row)
        self.assertEqual(set(visible), {"event", "domain", "question", "query_subject", "options"})
        self.assertEqual(visible["event"], "task")
        row["analysis"]["secret_canary"] = "NEVER_RENDER_HIDDEN_ANALYSIS"
        for peer in ("A", "B"):
            messages = tasks.peer_messages(row, peer)
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            payload = json.loads(messages[1]["content"])
            self.assertEqual(set(payload), set(visible) | {"private_facts"})
            self.assertEqual(payload["event"], "peer_task")
            self.assertEqual(payload["private_facts"], row["private_facts"][peer])
            self.assertNotIn("NEVER_RENDER_HIDDEN_ANALYSIS", json.dumps(messages))
        visible["options"].append("polluted")
        self.assertNotIn("polluted", row["options"])

    def test_strict_peer_parser_keeps_unknown_and_invalid_distinct(self):
        row = self.row()
        truth = row["analysis"]["true_answer"]
        wrong = next(v for v in row["options"] if v != truth)
        for answer, expected in ((truth, (True, True, False)), (wrong, (True, False, False)), ("unknown", (True, False, True))):
            result = tasks.grade_peer(row, json.dumps({"answer": answer}))
            self.assertEqual((result["valid"], result["correct"], result["unknown"]), expected)
            self.assertEqual(result["answer"], answer)
        invalid = [None, "", "[]", "null", json.dumps({"answer": True}), json.dumps({"answer": truth, "reason": "x"}), f'{{"answer":"{truth}","answer":"{wrong}"}}', f'```json\n{{"answer":"{truth}"}}\n```', json.dumps({"answer": truth.upper()}), json.dumps({"answer": " " + truth}), '{"answer": NaN}', 'before ' + json.dumps({"answer": truth})]
        for text in invalid:
            with self.subTest(text=text):
                result = tasks.grade_peer(row, text)
                self.assertEqual(result, {"answer": None, "valid": False, "correct": False, "unknown": False})

    def test_counterfactual_feedback_changes_the_answer_and_truthful_checker_together(self):
        row = self.row()
        for correct in (True, False):
            feedback = json.loads(tasks.counterfactual_feedback(row, "A", correct))
            self.assertEqual(feedback["peer"], "A")
            self.assertEqual(feedback["correct"], correct)
            self.assertEqual(tasks.grade_peer(row, feedback["response"])["correct"], correct)

    def test_invalid_config_and_requests_fail_explicitly(self):
        config = task_config()
        for split, family, comparison in (("other", 0, "equal_information"), ("development", -1, "equal_information"), ("development", True, "equal_information"), ("development", 0, "other")):
            with self.assertRaises(ValueError):
                tasks.session_tasks(config, split, family, comparison)
        for profile in ([0.9], [0.9, 0.3, 0.3, float("nan")], [0.9, 0.3, 0.3, 1.1], [0.95, 0.3, 0.3, 0.9]):
            with self.assertRaises(ValueError):
                tasks.calibration_tasks(config, 0, profile)
        for key, value in (("rounds", 6), ("reversal_round", 6), ("objects", ["ball"]), ("objects", ["ball", "ball"])):
            changed = copy.deepcopy(config)
            changed["tasks"][key] = value
            with self.assertRaises(ValueError):
                tasks.session_tasks(changed, "development", 0, "correct_vs_incomplete")
        changed = copy.deepcopy(config)
        changed["tasks"]["subjects"]["evaluation"][0] = "Alice"
        with self.assertRaises(ValueError):
            tasks.session_tasks(changed, "development", 0, "equal_information")
        with self.assertRaises(ValueError):
            tasks.peer_messages(self.row(), "C")

    def test_malformed_calibration_and_audit_sections_raise_value_error(self):
        for section in ("development", "audits"):
            for value in (None, [], "invalid"):
                with self.subTest(section=section, value=value):
                    config = task_config()
                    config[section] = value
                    with self.assertRaises(ValueError):
                        if section == "development":
                            tasks.calibration_profiles(config)
                        else:
                            tasks.session_tasks(config, "development", 0, "equal_information", audit=True)


if __name__ == "__main__":
    unittest.main()
