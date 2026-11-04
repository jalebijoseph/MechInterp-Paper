"""CPU-only checks for the audited peer-selection protocol."""
from __future__ import annotations

import importlib.util
import random
from types import SimpleNamespace
import unittest

from jpeers.benchmarks import load_task_bank


def subject():
    from jpeers.followup import peer_selection
    return peer_selection


class ParserTests(unittest.TestCase):
    def test_name_choice_requires_exactly_one_roster_name(self):
        # A permissive substring/first-name parser would accept the rejected examples.
        self.assertIsNotNone(importlib.util.find_spec('jpeers.followup.peer_selection'))
        cases = [
            ('Kai', 'Kai'), (' **Kai**. ', 'Kai'), ('`Rosa`', 'Rosa'),
            ('***Rosa.***', 'Rosa'), ('```\nKai\n```', 'Kai'),
            ('Kai or Rosa', None), ('Kai. Rosa', None), ('Kairos', None),
            ('Assign to: Kai', None), ('Kai..', None), ('Kai because he is best', None), ('kai', None),
            ('', None), ('**Kai** and **Rosa**', None),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(subject().parse_name_choice(text, ('Kai', 'Rosa')), want)

    def test_plan_parser_requires_one_strict_json_enum(self):
        self.assertIsNotNone(importlib.util.find_spec('jpeers.followup.peer_selection'))
        cases = [
            ('{"choice":"A"}', 'A'), (' {"choice": "B"} ', 'B'),
            ('{"choice":"tie"}', 'tie'), ('A', None),
            ('A is better than B', None), ('{"choice":"a"}', None),
            ('{"choice":"A","reason":"B is worse"}', None),
            ('{"choice":"A","choice":"B"}', None),
            ('```json\n{"choice":"A"}\n```', None),
            ('{"choice":["A","B"]}', None),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(subject().parse_plan_choice(text), want)


class ContextDesignTests(unittest.TestCase):
    def test_role_swap_holds_literal_roster_order_and_covers_all_cells(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'context_cells'))
        cells = mod.context_cells(load_task_bank('peer_selection'))
        self.assertEqual(len(cells), 192)
        groups = {}
        for cell in cells:
            key = (cell['task']['id'], tuple(cell['names']), cell['reverse_roster'])
            groups.setdefault(key, []).append(cell)
            messages = mod.context_messages(cell, load_task_bank('peer_selection'))
            self.assertEqual([m['role'] for m in messages], ['system', 'user'])
            self.assertIn(cell['task']['text'], messages[-1]['content'])
        self.assertEqual(len(groups), 96)
        for pair in groups.values():
            self.assertEqual(pair[0]['roster'], pair[1]['roster'])
            self.assertNotEqual(pair[0]['dev_name'], pair[1]['dev_name'])
            self.assertNotEqual(pair[0]['right_name'], pair[1]['right_name'])

    def test_role_swap_difference_cancels_name_priors_before_task_averaging(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'context_task_effects'))
        rows = []
        for task, kind, values in [('t1', 'dev', [(0.8, 0.2), (0.4, 0.9)]),
                                   ('t2', 'research', [(0.1, 0.3), (0.9, 0.3)])]:
            for swap, (a, b) in enumerate(values):
                rows.append({'task': task, 'task_kind': kind, 'names': ['Kai', 'Rosa'],
                             'reverse_roster': False, 'role_swap': bool(swap),
                             'scores': {'task_last': {'Kai': {'jlens': a},
                                                      'Rosa': {'jlens': b}}}})
        effects = mod.context_task_effects(rows)
        self.assertEqual(len(effects), 2)
        self.assertAlmostEqual(effects[0]['effect'], 1.1)
        self.assertAlmostEqual(effects[1]['effect'], 0.8)
        self.assertEqual(effects[0]['n_role_pairs'], 1)


class BanditProtocolTests(unittest.TestCase):
    def test_probability_mapping_uses_seed_roster_order_from_original_bank(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'bandit_probabilities'))
        roster, probabilities = mod.bandit_probabilities(['Kai', 'Rosa'], 2, [0.5, 0.8])
        self.assertEqual(roster, ['Kai', 'Rosa'])
        self.assertEqual(probabilities, {'Kai': 0.5, 'Rosa': 0.8})
        roster, probabilities = mod.bandit_probabilities(['Kai', 'Rosa'], 3, [0.5, 0.8])
        self.assertEqual(roster, ['Rosa', 'Kai'])
        self.assertEqual(probabilities, {'Rosa': 0.5, 'Kai': 0.8})

    def test_task_and_feedback_rng_sequence_matches_original_protocol(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'bandit_schedule'))
        reference = random.Random(7)
        expected = [(reference.randrange(12), reference.random()) for _ in range(40)]
        self.assertEqual(mod.bandit_schedule(7, 12, 40), expected)

    def test_session_metrics_separate_staying_after_feedback_and_learning(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'bandit_session_metrics'))
        rows = [
            {'choice': 'Kai', 'executed_choice': 'Kai', 'accepted': True,
             'parsed': True, 'regret': 0.3, 'best_arm': False},
            {'choice': 'Kai', 'executed_choice': 'Kai', 'accepted': False,
             'parsed': True, 'regret': 0.3, 'best_arm': False},
            {'choice': 'Rosa', 'executed_choice': 'Rosa', 'accepted': True,
             'parsed': True, 'regret': 0.0, 'best_arm': True},
            {'choice': 'Rosa', 'executed_choice': 'Rosa', 'accepted': False,
             'parsed': True, 'regret': 0.0, 'best_arm': True},
        ]
        metrics = mod.bandit_session_metrics(rows)
        self.assertAlmostEqual(metrics['entropy_bits'], 1.0)
        self.assertAlmostEqual(metrics['total_regret'], 0.6)
        self.assertEqual(metrics['early_best_arm_rate'], 0.0)
        self.assertEqual(metrics['late_best_arm_rate'], 1.0)
        self.assertEqual(metrics['stay_after_acceptance'], {'n': 2, 'rate': 1.0})
        self.assertEqual(metrics['stay_after_rejection'], {'n': 1, 'rate': 0.0})
        rows[1]['parsed'] = False
        rows[1]['choice'] = None
        metrics = mod.bandit_session_metrics(rows)
        self.assertEqual(metrics['stay_after_acceptance'], {'n': 1, 'rate': 1.0})
        self.assertEqual(metrics['stay_after_rejection'], {'n': 0, 'rate': None})


class OrchestrationTests(unittest.TestCase):
    class Endpoint:
        def __init__(self, name, family):
            self.name, self.family, self.requests = name, family, []

        def info(self):
            return {'family': self.family, 'model': self.name}

        def generate(self, req):
            from jpeers.schemas import GenerateResponse
            self.requests.append(req)
            system = req.messages[0]['content']
            if 'two-sentence plan' in system:
                text = 'Inspect the task. Complete and verify the work.'
            elif 'Plan A:' in req.messages[-1]['content']:
                text = '{"choice":"tie"}'
            else:
                all_names = [n for pair in subject().NAME_PAIRS for n in pair]
                hits = [(system.find(n), n) for n in all_names if n in system]
                text = min(hits)[1]
            watch = req.record.watch
            trace = None
            if req.record.enabled:
                trace = {'positions': [5, 9], 'topk_ids': [[[0], [0]], [[0], [0]]],
                         'topk_logits': [[[0.0], [0.0]], [[0.0], [0.0]]],
                         'watch_ranks': [[[1] * len(watch)] * 2] * 2,
                         'watch_ranks_logit': [[[2] * len(watch)] * 2] * 2,
                         'output_ranks': [[3] * len(watch)] * 2}
            return GenerateResponse(model=self.name, text=text, prompt_len=10,
                                    n_generated=4, tokens=['token'] * 14,
                                    spans=[{'label': q.label, 'start': 2, 'end': 6}
                                           for q in req.spans],
                                    layers=[1, 2], watch=watch, watch_ids=[[i] for i in range(len(watch))],
                                    vocab={0: 'token'}, trace_encoding='json', trace=trace,
                                    interventions_applied=0, timings={})

    class Writer:
        def __init__(self):
            self.records = []

        def write(self, rec, trace):
            rec.index = len(self.records)
            self.records.append((rec, trace))
            return rec

    def make_pool(self, other_family='olmo'):
        endpoints = {'qwen3-32b': self.Endpoint('qwen3-32b', 'qwen'),
                     'olmo2-32b': self.Endpoint('olmo2-32b', other_family)}
        return SimpleNamespace(get=endpoints.__getitem__), endpoints

    def config(self):
        return {'orchestrator': 'qwen3-32b', 'orchestrator_family': 'qwen',
                'self_preference': {'own': {'endpoint': 'qwen3-32b'},
                                    'other': {'endpoint': 'olmo2-32b'}}}

    def test_full_protocol_counts_ties_actual_sources_and_bounded_recording(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'run_peer_selection'))
        pool, endpoints = self.make_pool()
        writer = self.Writer()
        result = mod.run_peer_selection(self.config(), pool, writer)
        self.assertEqual(result['counts'], {'context': 192, 'family': 432,
                                          'self_preference': 24, 'plan_controls': 12,
                                          'bandit': 640, 'plan_generations': 24})
        self.assertEqual(len(result['rows']), 1300)
        self.assertEqual(result['summary']['context']['primary']['n'], 12)
        self.assertEqual(len(result['summary']['bandit']['sessions']), 16)
        self.assertEqual(result['summary']['self_preference']['ties'], 24)
        self.assertEqual(result['summary']['plan_controls']['ties'], 12)
        for row in result['rows']:
            if row['variant'] in ('self_preference', 'plan_controls'):
                self.assertEqual(row['source_families'], {'own': 'qwen', 'other': 'olmo'})
                self.assertIsNone(row['chose_own'])
                self.assertNotIn('jspace_prefers_own', row)
            if row['variant'] == 'family':
                self.assertTrue(row['hypothetical'])
                self.assertEqual(row['own_size_b'], 32)
        requests = [req for endpoint in endpoints.values() for req in endpoint.requests]
        recorded = [req for req in requests if req.record.enabled]
        plans = [req for req in requests if not req.record.enabled]
        self.assertEqual(len(recorded), 1300)
        self.assertEqual(len(plans), 24)
        self.assertTrue(all(req.assistant_prefill == '' for req in requests))
        self.assertTrue(all(not any(a['role'] == b['role'] == 'user'
                                   for a, b in zip(req.messages, req.messages[1:]))
                            for req in requests))
        self.assertTrue(all(req.temperature == 0 for req in requests))
        self.assertTrue(all(req.max_new_tokens == 128 for req in plans))
        for req in requests:
            if 'Plan A:' in req.messages[-1]['content']:
                self.assertEqual(req.max_new_tokens, 32)
                self.assertNotIn('qwen', str(req.messages).lower())
                self.assertNotIn('olmo', str(req.messages).lower())
            elif req.record.enabled:
                self.assertEqual(req.max_new_tokens, 16)
        for row in result['rows']:
            if row['variant'] == 'bandit':
                self.assertEqual(row['accepted'],
                                 row['feedback_uniform'] < row['probabilities'][row['executed_choice']])
        self.assertTrue(all(req.record.positions.scope == 'spans' for req in recorded
                            if any(q.label == 'task' for q in req.spans)))
        self.assertTrue(all(len(req.spans) <= 1 for req in recorded))

    def test_same_family_peer_is_rejected_before_plan_generation(self):
        mod = subject()
        self.assertTrue(hasattr(mod, 'run_peer_selection'))
        pool, endpoints = self.make_pool(other_family='qwen')
        cfg = {**self.config(), 'variants': ['self_preference']}
        with self.assertRaisesRegex(ValueError, 'different.*famil'):
            mod.run_peer_selection(cfg, pool, self.Writer())
        self.assertFalse(any(e.requests for e in endpoints.values()))


class PlanAggregationTests(unittest.TestCase):
    def test_valid_tie_then_preference_counts_as_order_sensitive(self):
        rows = [{'task': 't1', 'own_first': True, 'parsed': True, 'choice': 'tie',
                 'chose_own': None, 'preference': 0.0},
                {'task': 't1', 'own_first': False, 'parsed': True, 'choice': 'B',
                 'chose_own': True, 'preference': 1.0}]
        summary = subject()._plan_summary(rows)
        self.assertEqual(summary['order_sensitive_pairs'], 1)
        self.assertTrue(summary['tasks'][0]['order_sensitive'])
        rows[1].update(choice='tie', chose_own=None, preference=0.0)
        summary = subject()._plan_summary(rows)
        self.assertEqual(summary['order_sensitive_pairs'], 0)
        self.assertFalse(summary['tasks'][0]['order_sensitive'])

    def test_order_position_counts_and_invalid_task_pairs_are_explicit(self):
        rows = [
            {'task': 't1', 'own_first': True, 'parsed': True, 'choice': 'A',
             'chose_own': True, 'preference': 1.0},
            {'task': 't1', 'own_first': False, 'parsed': True, 'choice': 'B',
             'chose_own': True, 'preference': 1.0},
            {'task': 't2', 'own_first': True, 'parsed': True, 'choice': 'A',
             'chose_own': True, 'preference': 1.0},
            {'task': 't2', 'own_first': False, 'parsed': True, 'choice': 'A',
             'chose_own': False, 'preference': -1.0},
            {'task': 't3', 'own_first': True, 'parsed': True, 'choice': 'tie',
             'chose_own': None, 'preference': 0.0},
            {'task': 't3', 'own_first': False, 'parsed': False, 'choice': None,
             'chose_own': None, 'preference': None},
        ]
        summary = subject()._plan_summary(rows)
        self.assertEqual(summary['primary']['n'], 2)
        self.assertEqual(summary['primary']['n_total'], 3)
        self.assertEqual(summary['primary']['n_missing'], 1)
        self.assertEqual(summary['primary']['diff'], 0.5)
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['ties'], 1)
        self.assertEqual(summary['incomplete_task_pairs'], 1)
        self.assertEqual(summary['order_sensitive_pairs'], 1)
        self.assertIn('by_order', summary)
        self.assertEqual(summary['by_order']['own_first']['position_counts'],
                         {'A': 2, 'B': 0, 'tie': 1})
        self.assertEqual(summary['by_order']['other_first']['position_counts'],
                         {'A': 1, 'B': 1, 'tie': 0})


if __name__ == '__main__':
    unittest.main()
