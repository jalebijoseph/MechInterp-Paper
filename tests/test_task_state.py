"""CPU checks for Exp. 8 (task-state tracking): scripts, labels, parsers and analysis."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from jpeers.benchmarks import load_task_bank
from jpeers.harness import RunWriter
from jpeers.schemas import GenerateResponse, TokenSpan


def module():
    from jpeers.followup import task_state
    return task_state


def bank(n=2):
    b = load_task_bank("task_state")
    return {**b, "projects": b["projects"][:n]}


class FakeEndpoint:
    """Answers every probe correctly from the cell it can see; the trace favours the
    first roster name so name priors are present and must cancel."""

    name = "cpu-fixture"

    def __init__(self, answers):
        self.answers = answers  # full probe messages (JSON) -> reply
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        text = self.answers[json.dumps(req.messages)]
        watch = req.record.watch
        trace = None
        positions = [0, 1, 2]
        if req.record.enabled:
            ranks = np.full((3, 2, len(watch)), 4)
            ranks[:, :, 0] = 1  # literal first name always ranks better
            trace = {"positions": positions,
                     "topk_ids": np.zeros((3, 2, 1), dtype=int).tolist(),
                     "topk_logits": np.ones((3, 2, 1)).tolist(),
                     "watch_ranks": ranks.tolist(),
                     "watch_ranks_logit": ranks.tolist(),
                     "output_ranks": ranks[:, 0, :].tolist()}
        spans = [TokenSpan(label=s.label, start=0, end=2) for s in req.spans]
        return GenerateResponse(model="cpu", text=text, prompt_len=3, n_generated=1,
                                tokens=["a", "b", "c", "d"], spans=spans, layers=[10, 11],
                                watch=watch, watch_ids=[[i] for i in range(len(watch))],
                                vocab={0: "word"}, trace_encoding="json", trace=trace,
                                interventions_applied=0, timings={})


def correct_answers(cells, controls):
    m = module()
    answers = {}
    keys = {"owner": "owner", "content": "codes", "provenance": "submitted_by",
            "access": "seen_by", "status": "status", "action": "next_action"}
    for cell, include_events, outcomes in [(c, True, m.probes_for(c)) for c in cells] + \
            [(c, False, ("owner", "content")) for c in controls]:
        messages, _ = m.transcript_messages(cell, include_events=include_events)
        for outcome in outcomes:
            prompt = m.probe_messages(messages, m.probe_question(outcome, cell))
            answers[json.dumps(prompt)] = json.dumps({keys[outcome]: m.truth(outcome, cell)})
    return answers


class DesignTests(unittest.TestCase):
    def test_cells_cross_conditions_swaps_and_delays(self):
        m = module()
        cells = m.cells(bank())
        self.assertEqual(len(cells), 2 * 8 * 2 * 2)
        for cell in cells:
            self.assertNotEqual(cell["first"], cell["second"])
            self.assertEqual(sorted(cell["roster"]), sorted([cell["first"], cell["second"]]))
        pairs = {}
        for cell in cells:
            pairs.setdefault((cell["project"], cell["condition"], cell["delay"]), []).append(cell)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["roster"], pair[1]["roster"])  # literal order fixed
            self.assertNotEqual(pair[0]["labels"]["owner"], pair[1]["labels"]["owner"])

    def test_dissociation_conditions_separate_owner_from_contributor(self):
        m = module()
        by = {c["condition"]: c for c in m.cells(bank(1), delays=("none",))
              if not c["role_swap"]}
        re, cf = by["reassigned"]["labels"], by["conflicting"]["labels"]
        self.assertEqual(re["owner"], by["reassigned"]["second"])
        self.assertEqual(re["submitter"], by["reassigned"]["first"])
        self.assertEqual(re["seen_by"], sorted([by["reassigned"]["first"],
                                                by["reassigned"]["second"]]))
        self.assertEqual(cf["owner"], by["conflicting"]["first"])
        self.assertEqual(cf["submitter"], by["conflicting"]["second"])
        self.assertIsNone(by["document"]["labels"]["submitter"])
        self.assertEqual(len(cf["codes"]), 2)

    def test_codes_appear_only_where_the_script_puts_them(self):
        m = module()
        for cell in m.cells(bank()):
            text = json.dumps(m.transcript_messages(cell)[0])
            for code in cell["labels"]["codes"]:
                self.assertIn(code, text)
            if not cell["labels"]["codes"]:
                target = m.contribution_code(0, cell["project"], cell["target"]["id"], "first")
                self.assertNotIn(target, text)
            if cell["condition"] == "document":
                doc = next(e for e in cell["events"] if e["kind"] == "document")
                for name in cell["roster"]:
                    self.assertNotIn(name, doc["text"])

    def test_delay_only_appends_neutral_filler(self):
        m = module()
        cells = {(c["project"], c["condition"], c["role_swap"], c["delay"]): c
                 for c in m.cells(bank())}
        for (p, cond, swap, delay), cell in cells.items():
            if delay != "filler":
                continue
            base = cells[(p, cond, swap, "none")]["events"]
            self.assertEqual(cell["events"][:len(base)], base)
            extra = cell["events"][len(base):]
            self.assertEqual(len(extra), 3)
            for e in extra:
                self.assertEqual(e["kind"], "system")
                for name in cell["roster"]:
                    self.assertNotIn(name, e["text"])

    def test_messages_alternate_and_end_with_the_last_event(self):
        m = module()
        for cell in m.cells(bank()):
            messages, end_text = m.transcript_messages(cell)
            roles = [msg["role"] for msg in messages]
            self.assertEqual(roles[0], "system")
            self.assertTrue(all(a != b for a, b in itertools.pairwise(roles[1:])))
            self.assertTrue(messages[-1]["content"].endswith(end_text))

    def test_bank_subtasks_name_no_peer_state_or_action(self):
        m = module()
        from jpeers.followup.peer_selection import NAME_PAIRS
        names = {n.lower() for pair in NAME_PAIRS for n in pair}
        banned = names | set(m.STATUSES) | set(m.ACTIONS) | {"blocked", "verified", "done"}
        for project in load_task_bank("task_state")["projects"]:
            for sub in project["subtasks"]:
                words = set(f"{sub['label']} {sub['description']}".lower().split())
                self.assertFalse(words & banned, f"{project['id']}/{sub['id']}: {words & banned}")


class ParserTests(unittest.TestCase):
    def test_strict_parsers(self):
        m = module()
        roster = ["Alice", "Bob"]
        cases = [
            ("owner", '{"owner": "Alice"}', True, "Alice"),
            ("owner", '{"owner": null}', True, None),
            ("owner", '{"owner": "alice"}', False, None),
            ("owner", '{"owner": "Alice", "why": "x"}', False, None),
            ("owner", "Alice", False, None),
            ("content", '{"codes": ["B2", "A1"]}', True, ["A1", "B2"]),
            ("content", '{"codes": ["A1", "A1"]}', False, None),
            ("content", '{"codes": "A1"}', False, None),
            ("provenance", '{"submitted_by": null}', True, None),
            ("access", '{"seen_by": ["Bob", "Alice"]}', True, ["Alice", "Bob"]),
            ("access", '{"seen_by": ["Orchestrator"]}', False, None),
            ("status", '{"status": "blocked"}', True, "blocked"),
            ("status", '{"status": "stuck"}', False, None),
            ("action", '{"next_action": "reassign"}', True, "reassign"),
            ("action", '{"action": "reassign"}', False, None),
        ]
        for outcome, text, valid, value in cases:
            with self.subTest(outcome=outcome, text=text):
                got = m.parse_probe(outcome, text, roster)
                self.assertEqual((got["valid"], got["value"]), (valid, value))


class AnalysisTests(unittest.TestCase):
    def test_role_swap_average_cancels_name_prior(self):
        m = module()
        rows = []
        for swap, owner, (a, b) in [(False, "Alice", (0.9, 0.3)), (True, "Bob", (0.6, 0.4))]:
            rows.append({"project": "p", "condition": "completed", "delay": "none",
                         "role_swap": swap, "first": owner,
                         "second": "Bob" if owner == "Alice" else "Alice",
                         "labels": {"owner": owner, "submitter": owner},
                         "scores": {"pre_query": {"Alice": {"jlens": a}, "Bob": {"jlens": b}}}})
        # Alice prior +0.2 in both; owner margins 0.6 and -0.2 -> mean 0.2 is the
        # owner effect with the prior cancelled ((0.6 - 0.2) + (-0.2 + 0.2)) / 2.
        effects = m.swap_effects(rows)
        self.assertAlmostEqual(effects[0]["effect"], 0.2)

    def test_incomplete_swap_pair_is_an_error(self):
        m = module()
        row = {"project": "p", "condition": "c", "delay": "none", "role_swap": False}
        with self.assertRaises(ValueError):
            m.swap_effects([row])


class RunTests(unittest.TestCase):
    def test_full_protocol_runs_counts_and_scores(self):
        m = module()
        b = bank(2)
        cells = m.cells(b)
        controls = []
        for c in m.cells(b, conditions=("assigned_silent",), delays=("none",)):
            if not c["role_swap"]:
                controls.append({**c, "labels": {**c["labels"], "owner": None, "codes": []}})
        answers = correct_answers(cells, controls)
        # Every probe prompt is distinct, so no two cells can share an answer.
        self.assertEqual(len(answers), sum(len(m.probes_for(c)) for c in cells) + 2 * len(controls))
        endpoint = FakeEndpoint(answers)
        pool = SimpleNamespace(get=lambda name: endpoint)
        with tempfile.TemporaryDirectory() as tmp:
            writer = RunWriter(tmp, "task_state", "test")
            from unittest.mock import patch
            with patch.object(m, "load_task_bank", return_value=b):
                result = m.run_task_state({"orchestrator": "x", "n_projects": 2}, pool, writer)
            turns = (writer.dir / "turns.jsonl").read_text().splitlines()
        self.assertEqual(result["counts"], m.expected_counts(2))
        self.assertEqual(len(turns), result["counts"]["probes"] + result["counts"]["no_peer_probes"])
        self.assertTrue(all(p["correct"] for p in result["probes"]))
        self.assertEqual(result["summary"]["no_peer"], {"owner": 1.0, "content": 1.0})
        # The fixture always favours the literal first name: a pure name prior, no
        # owner signal. Swap-averaging must report zero effect.
        stats = result["summary"]["jspace"]["filler/pre_query/jlens"]["stats"]
        self.assertAlmostEqual(stats["diff"], 0.0)
        self.assertEqual(stats["n"], 2)

    def test_cli_entry_point_writes_summary(self):
        from unittest.mock import patch

        from jpeers.experiments import task_state as entry
        m = module()
        b = bank(1)
        cells = m.cells(b)
        controls = [{**c, "labels": {**c["labels"], "owner": None, "codes": []}}
                    for c in m.cells(b, conditions=("assigned_silent",), delays=("none",))
                    if not c["role_swap"]]
        endpoint = FakeEndpoint(correct_answers(cells, controls))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(entry, "make_pool", return_value=SimpleNamespace(get=lambda n: endpoint)), \
                patch.object(m, "load_task_bank", return_value=b):
            summary = entry.run({"orchestrator": "x", "out_dir": tmp, "n_projects": 1,
                                 "run_id": "cli"})
            run_dir = summary["run_dir"]
            from pathlib import Path
            self.assertTrue((Path(run_dir) / "summary.json").is_file())
            self.assertIn("acc_owner", (Path(run_dir) / "summary.md").read_text())

    def test_expected_counts_for_the_default_config(self):
        m = module()
        self.assertEqual(m.expected_counts(16), {"cells": 512, "probes": 2624,
                                                 "recorded": 512, "no_peer_probes": 32})


if __name__ == "__main__":
    unittest.main()
