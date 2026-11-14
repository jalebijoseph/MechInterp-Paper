"""Checkpoint forks use real token simulation; no model or network is loaded."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jpeers.exp5.protocol import CELL_ORDER, derived_seed
from jpeers.exp5.simulation import SimulationBackend
from jpeers.exp5.storage import BudgetExceeded
from jpeers.exp5.tasks import grade_peer, peer_messages


class Engine:
    def __init__(self):
        self.backend = SimulationBackend()
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/exp5_study.json").read_text())
        self.config["diagnostics"].update(delays=[0, 8], primary_delay=8)
        self.config["inference"]["max_orchestrator_tokens"] = 256
        self.artifact = dict(layer=63, mean=np.zeros(8), scale=np.ones(8),
                             coef=np.eye(8)[:, :4], intercept=np.zeros(4), counts=np.ones(4),
                             bases={"delete": {i: np.eye(8)[:, i:i+1] for i in range(4)},
                                    "restore": {i: np.eye(8)[:, i:i+1] for i in range(4)}},
                             causal_settings=dict(layer=31, later_layer=33, rank=1, strength=1))
        self.runtime_validation = self.backend.validate_runtime()
        self.records, self.streams, self.peer_requests = [], [], []
        self.store = type("Store", (), {"history": lambda _, tokens: {"tokens": len(tokens)}})()

    def generate(self, tokens, *, stream, role="orchestrator", masks=(), edits=(), max_new_tokens=None):
        self.streams.append(stream)
        out = self.backend.generate(tokens, sampler={"temperature": 0}, seed=derived_seed(9, *stream),
                                    max_new_tokens=max_new_tokens or 256, masks=masks, edits=edits)
        return dict(out, call_id=f"call:{len(self.streams)}")

    def observe(self, tokens, capture, *, stream, masks=(), edits=()):
        self.streams.append(stream)
        return self.backend.observe(tokens, capture, masks=masks, edits=edits)

    def record(self, record):
        self.records.append(copy.deepcopy(record))

    def peer_reply(self, task, peer, *, stream):
        self.peer_requests.append(copy.deepcopy(task))
        tokens = []
        for msg in peer_messages(task, peer):
            tokens, _ = self.backend.append_message(tokens, msg["role"], msg["content"])
        tokens, _ = self.backend.append_message(tokens, "assistant", "", close=False)
        out = self.generate(tokens, stream=stream, role="peer")
        return dict(out, **grade_peer(task, out["text"]))

    def replay(self, events):
        tokens, source, control = [], [], []
        checkpoint = None
        for event in events:
            start = len(tokens)
            if event["kind"] == "message":
                tokens, _ = self.backend.append_message(tokens, event["role"], event["content"],
                                                        close=event.get("close", True))
                if event.get("purpose") == "receipt":
                    checkpoint = len(tokens) - 1
            elif event["kind"] == "tokens":
                tokens += event["tokens"]
            else:
                tokens = self.backend.finish_assistant(tokens)
            if event.get("source"):
                source.append((start, len(tokens)))
            if event.get("control"):
                control.append((start, len(tokens)))
        return dict(tokens=tokens, source_spans=source, control_spans=control, checkpoint_position=checkpoint)


def fixture(engine):
    task = dict(id="probe-task", domain="possession", question="What does Dana hold?", query_subject="Dana",
                options=["ball", "chair"], private_facts={p: [{"subject": "Evan", "value": "chair"},
                    {"subject": "Dana", "value": "ball"}] for p in ("A", "B")},
                analysis={"true_answer": "ball", "world": {"Dana": "ball", "Evan": "chair"}})
    events = [dict(kind="message", role="user", content=json.dumps({"event": "roster", "roster": {
        "A": "Cedar", "B": "Maple"}}), source=False, purpose="roster")]
    successes = [0] * 4
    for cell, outcomes in enumerate(([True, True, True, False], [True, True, False, False],
                                     [True, False, False, False], [True, False, False, False])):
        peer, domain = CELL_ORDER[cell]
        for i, correct in enumerate(outcomes):
            source_task = copy.deepcopy(task)
            source_task.update(id=f"{cell}:{i}", domain=domain)
            response = json.dumps({"answer": "ball" if correct else "chair"})
            feedback = dict(kind="message", role="user", content=json.dumps({"event": "peer_feedback",
                "peer": peer, "domain": domain, "response": response, "correct": correct}),
                source=True, purpose="feedback", peer=peer, task=source_task, correct=correct)
            padded, _ = engine.backend.append_message([], "user", feedback["content"])
            events.append(dict(kind="tokens", tokens=engine.backend.encode_text(" ")*len(padded),
                               source=False, control=True, purpose="control"))
            events.append(feedback)
            successes[cell] += correct
    events.append(dict(kind="message", role="assistant", content="Recorded.", close=False,
                       source=False, purpose="receipt"))
    replay = engine.replay(events)
    return dict(replay, family="family-1", session="session-1", split="evaluation", round=16,
                events=events, successes=successes, exposures=[4]*4,
                audit_quality={("A", "possession"): .75, ("B", "possession"): .25,
                               ("A", "location"): .5, ("B", "location"): .25}), task


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.diag = importlib.import_module("jpeers.exp5.diagnostics")
        except ModuleNotFoundError:
            self.fail("checkpoint diagnostics module is missing")

    def test_routes_start_at_question_or_after_checkpoint_and_control_counts_matter(self):
        context = dict(source_spans=[(2, 5), (7, 9)], control_spans=[(10, 15)], checkpoint_position=20)
        self.assertEqual(self.diag.route_masks(context, "R0", 40), [])
        self.assertEqual(self.diag.route_masks(context, "R1", 40)[0]["start_query"], 40)
        self.assertEqual(self.diag.route_masks(context, "R2", 40)[0]["start_query"], 21)
        self.assertEqual(self.diag.route_masks(context, "R2_control", 40)[0]["key_spans"], [(10,15)])
        self.assertTrue(self.diag.control_geometry_valid([(2,5),(7,9)], [(10,15)], 20, 10))
        self.assertFalse(self.diag.control_geometry_valid([(2,5),(7,9)], [(10,14)], 20, 10))
        self.assertFalse(self.diag.control_geometry_valid([(2,5),(7,9)], [(10,15)], 20, 1))

    def test_donor_changes_only_one_cell_with_actual_checker_consistent_new_replies(self):
        engine = Engine()
        context, _ = fixture(engine)
        before = copy.deepcopy(context)
        donor = self.diag.build_donor(engine, context, 0)
        self.assertEqual(context, before)
        self.assertTrue(donor["available"], donor)
        self.assertEqual(donor["donor_choice"], "B")
        self.assertEqual(donor["successes"], [0,2,1,1])
        self.assertGreater(len(engine.peer_requests), 0)
        for old, new in zip(context["events"], donor["events"]):
            if old != new:
                self.assertEqual((new["peer"],new["task"]["domain"]), ("A","possession"))
                payload = json.loads(new["content"])
                self.assertEqual(grade_peer(new["task"], payload["response"])["correct"], payload["correct"])

    def test_checkpoint_retains_inventory_truthful_simulation_gates_and_main_history(self):
        engine = Engine()
        context, task = fixture(engine)
        before = copy.deepcopy(context)
        out = self.diag.run_checkpoint(engine, context, task, 0,
                                      routes=["R0","R2","R2_control"], delays=[0,8])
        self.assertEqual(context, before)
        causal = [r for r in engine.records if r["type"] == "causal"]
        self.assertEqual(len(causal), 20)  # 8x2 at primary + other route/delays I0.
        self.assertEqual({r["condition"] for r in causal}, {f"I{i}" for i in range(8)})
        self.assertTrue(all(r["source_attention_valid"] is False for r in causal))
        self.assertTrue(all(r["route_match_valid"] is True for r in causal))
        self.assertTrue(all(type(r["route_match_valid"]) is bool and type(r["control_match_valid"]) is bool for r in causal))
        self.assertTrue(all(s[0] == "diagnostic" for s in engine.streams))
        update = next(r for r in engine.records if r["type"] == "update")
        self.assertEqual(update["q_initial"], .75)
        self.assertEqual(update["q_reversal"], .375)
        self.assertEqual(update["q_confirming"], .875)
        binding = next(r for r in engine.records if r.get("diagnostic") == "attribution")
        self.assertGreater(binding["attribution_error_improvement"], 0)
        self.assertEqual(out["target_cell"], 0)

    def test_failed_dtype_matching_and_zero_other_domain_edit_remain_inconclusive(self):
        engine = Engine()
        context, task = fixture(engine)
        generate = engine.generate
        def mismatched(*args, **kwargs):
            out = generate(*args, **kwargs)
            for edit in out["receipt"]["edits"]:
                if edit["kind"].startswith("random_"):
                    edit["norm_matching_error"] = 1
            return out
        engine.generate = mismatched
        engine.artifact["bases"]["restore"][1] = np.eye(8)[:, 4:5]
        self.diag.run_checkpoint(engine, context, task, 0, conditions=["I2","I7"], routes=["R0"], delays=[8])
        rows = [r for r in engine.records if r["type"] == "causal"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["status"] == "inconclusive" for r in rows))
        self.assertTrue(all(r["control_match_valid"] is False for r in rows))
        self.assertEqual(rows[1]["receipt"]["edits"][0]["realized_norm"], 0)

    def test_native_choice_terminators_survive_matched_attribution_swap(self):
        engine = Engine()
        context, task = fixture(engine)
        for peer in ("A", "B"):
            tokens = engine.backend.encode_text(json.dumps({"choice": peer}))
            tokens += engine.backend.encode_text("<|im_end|>\n")
            context["events"].insert(-1, dict(kind="tokens", tokens=tokens,
                source=True, control=False, purpose="choice", peer=peer, task=copy.deepcopy(task)))
        captured = []
        replay = engine.replay
        def tracks(events):
            captured.append(copy.deepcopy(events))
            return replay(events)
        engine.replay = tracks
        self.diag.run_checkpoint(engine, context, task, 0, conditions=["I0"], routes=["R0"], delays=[8])
        swapped = next(events for events in captured if [e.get("peer") for e in events
            if e.get("purpose") == "choice"] == ["B", "A"])
        choices = [e for e in swapped if e.get("purpose") == "choice"]
        self.assertTrue(engine.backend.decode(choices[0]["tokens"]).startswith('{"choice": "B"}'))
        self.assertTrue(all(engine.backend.decode(e["tokens"]).endswith("<|im_end|>\n") for e in choices))

    def test_missing_quality_reference_and_context_failure_keep_registered_cases(self):
        engine = Engine()
        context, task = fixture(engine)
        context["audit_quality"] = {}
        engine.backend.max_context = len(context["tokens"]) - 1
        self.diag.run_checkpoint(engine, context, task, 0, conditions=["I0"], routes=["R0","R2"], delays=[8])
        rows = [r for r in engine.records if r["type"] == "causal"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["reference_choice"] is None for r in rows))
        self.assertTrue(all(r["status"] == "unavailable" for r in rows))
        retention = [r for r in engine.records if r["type"] == "retention"]
        self.assertEqual(len(retention), 2)
        self.assertTrue(all(r["status"] == "unavailable" for r in retention))
        self.assertEqual(len([r for r in engine.records if r["type"] == "specificity"]), 2)

    def test_receipted_route_mismatch_cannot_pass_the_route_gate(self):
        engine = Engine()
        context, task = fixture(engine)
        generate = engine.generate
        def loses_mask(*args, **kwargs):
            out = generate(*args, **kwargs)
            out["receipt"]["masks"] = []
            return out
        engine.generate = loses_mask
        self.diag.run_checkpoint(engine, context, task, 0, conditions=["I0"], routes=["R2"], delays=[8])
        row = next(r for r in engine.records if r["type"] == "causal")
        self.assertFalse(row["route_match_valid"])
        self.assertEqual(row["status"], "inconclusive")

    def test_attribution_unknown_and_tied_evidence_are_explicitly_not_applicable(self):
        for exposures, successes in (([0,4,4,4], [0,2,1,1]), ([4]*4, [1,2,1,1])):
            with self.subTest(exposures=exposures, successes=successes):
                engine = Engine()
                context, _ = fixture(engine)
                context.update(exposures=exposures, successes=successes)
                rows = []
                self.diag._attribution(engine, context, "possession", rows.append)
                self.assertEqual(len(rows), 1)
                self.assertIs(rows[0]["attribution_applicable"], False)
                self.assertEqual(rows[0]["status"], "not_applicable")
                self.assertNotIn("attribution_error_improvement", rows[0])

    def test_attribution_failures_remain_applicable_unavailable_and_budgets_propagate(self):
        for failing_operation in ("replay", "observe", "tokens", "missing_feedback"):
            with self.subTest(failing_operation=failing_operation):
                engine = Engine()
                context, task = fixture(engine)
                def fails(*args, **kwargs):
                    raise ValueError("assigned diagnostic failed")
                if failing_operation in ("replay", "observe"):
                    setattr(engine, failing_operation, fails)
                elif failing_operation == "tokens":
                    context["events"].insert(-1, dict(kind="tokens", tokens=engine.backend.encode_text('bad choice'),
                        source=True, purpose="choice", peer="A", task=task))
                else:
                    context["events"] = [e for e in context["events"] if e.get("purpose") != "feedback"]
                rows = []
                self.diag._attribution(engine, context, "possession", rows.append)
                self.assertEqual(len(rows), 1)
                self.assertIs(rows[0]["attribution_applicable"], True)
                self.assertEqual(rows[0]["status"], "unavailable")
                self.assertNotIn("attribution_error_improvement", rows[0])
        engine = Engine()
        context, _ = fixture(engine)
        def exhausted(*args, **kwargs):
            raise BudgetExceeded("test budget")
        engine.replay = exhausted
        with self.assertRaises(BudgetExceeded):
            self.diag._attribution(engine, context, "possession", lambda _: None)

    def test_failed_donor_replay_is_an_unavailable_assigned_donor(self):
        engine = Engine()
        context, _ = fixture(engine)
        def fails(*args, **kwargs):
            raise ValueError("donor replay failed")
        engine.replay = fails
        donor = self.diag.build_donor(engine, context, 0)
        self.assertFalse(donor["available"])
        self.assertIn("donor replay failed", donor["reason"])

    def test_impossible_live_donor_is_recorded_without_fabricating_outcomes(self):
        engine = Engine()
        context, task = fixture(engine)
        def refuses(task, peer, *, stream):
            return dict(text='{"answer":"unknown"}', valid=True, correct=False, unknown=True, call_id="refusal")
        engine.peer_reply = refuses
        donor = self.diag.build_donor(engine, context, 2)
        self.assertFalse(donor["available"])
        self.diag.run_checkpoint(engine, context, task, 2, conditions=["I0","I4","I6","I7"], routes=["R0"], delays=[8])
        rows = [r for r in engine.records if r["type"] == "causal"]
        self.assertEqual(len(rows),4)
        self.assertEqual(next(r for r in rows if r["condition"] == "I4")["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
