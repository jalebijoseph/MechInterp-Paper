"""Follow-up site-selection regressions; tensor recorder checks need only CPU torch."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import importlib.util
import io
import json
from unittest.mock import patch
import urllib.error
import sys
import tempfile
import unittest

import numpy as np

from jpeers.followup.common import paired_stats, site_scores
from jpeers.followup.experiments12 import _site
from jpeers.schemas import GenerateRequest, PositionSelector, RecordSpec
from jpeers.trace import JSpaceTrace

try:
    import torch
except ImportError:
    torch = None


class SiteSelectionTests(unittest.TestCase):
    def trace(self):
        # Three recorded sites: draft body, final prompt position, generated text.
        return JSpaceTrace(
            positions=np.array([3, 9, 11]), topk_ids=np.zeros((3, 2, 1), int),
            topk_logits=np.zeros((3, 2, 1)),
            watch_ranks=np.array([[[0, 0], [3, 0]], [[9, 0], [1, 0]],
                                  [[99, 0], [99, 0]]]),
            watch_ranks_logit=np.array([[[0, 0], [0, 0]], [[1, 0], [3, 0]],
                                        [[99, 0], [99, 0]]]),
            output_ranks=np.array([[0, 0], [4, 0], [99, 0]]),
            layers=[24, 25], watch=['target', 'other'], prompt_len=10,
            spans={'draft': [(3, 10)]})

    def test_single_site_mask_applies_to_each_lens_and_output_baseline(self):
        scores = site_scores(self.trace(), ['target'], [False, True, False])
        self.assertAlmostEqual(scores['jlens'], 0.3)
        self.assertAlmostEqual(scores['logit'], 0.375)
        self.assertAlmostEqual(scores['output'], 0.2)
        self.assertEqual(scores['by_layer'], {'24': {'jlens': 0.1, 'logit': 0.5},
                                              '25': {'jlens': 0.5, 'logit': 0.25}})

    def test_prompt_span_and_generated_observables_use_disjoint_sites(self):
        reading = _site(self.trace(), ['target'], 'draft')
        writing = _site(self.trace(), ['target'], generated=True)
        self.assertAlmostEqual(reading['jlens'], 0.4625)
        self.assertAlmostEqual(writing['jlens'], 0.01)
        self.assertAlmostEqual(reading['output'], 0.6)
        self.assertAlmostEqual(writing['output'], 0.01)

    def test_missing_watch_word_and_invalid_selected_rank_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            site_scores(self.trace(), ['absent'], [False, True, False])
        trace = self.trace()
        trace.watch_ranks[1, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'rank'):
            site_scores(trace, ['target'], [False, True, False])

    def test_cluster_missingness_and_empty_samples_remain_visible(self):
        result = paired_stats([None, float('nan'), float('inf'), 2.0])
        self.assertEqual((result['n'], result['n_total'], result['n_missing']), (1, 4, 3))
        self.assertEqual(result['diff'], 2.0)
        self.assertEqual(result['p_exact'], 1.0)
        empty = paired_stats([None])
        self.assertEqual((empty['n'], empty['n_total'], empty['n_missing']), (0, 1, 1))
        self.assertIsNone(empty['diff'])
        self.assertIsNone(empty['p'])


@unittest.skipUnless(torch is not None, 'CPU torch is available in the remote test environment')
class RuntimeMaskTests(unittest.TestCase):
    def run_state(self):
        from jpeers.runtime import _Run
        run = _Run.__new__(_Run)
        run.prompt_len = 10
        run.spans = {'task': [(2, 4)], 'unselected_history': [(0, 2)]}
        return run

    def selected(self, selector, start=0, stop=12):
        pos = torch.arange(start, stop)
        mask = self.run_state()._mask(selector, pos)
        return pos[mask].tolist()

    def test_span_recording_keeps_last_prompt_site_only_when_opted_in(self):
        legacy = PositionSelector(scope='spans', labels=['task'])
        followup = PositionSelector(scope='spans', labels=['task'], include_last=True)
        self.assertEqual(self.selected(legacy), [2, 3])
        self.assertEqual(self.selected(followup), [2, 3, 9])
        self.assertEqual(self.selected(followup, start=8), [9])

    def test_last_prompt_site_can_be_recorded_without_any_span_or_prefill(self):
        selector = PositionSelector(scope='spans', labels=[], include_last=True)
        self.assertEqual(self.selected(selector), [9])
        selector.include_last = False
        self.assertEqual(self.selected(selector), [])

    def test_generated_union_and_skip_first_preserve_absolute_positions(self):
        selector = PositionSelector(scope='spans', labels=['task'], include_generated=True,
                                    include_last=True, skip_first=3)
        self.assertEqual(self.selected(selector), [3, 9, 10, 11])
        self.assertEqual(self.selected(selector, start=8), [9, 10, 11])
        selector = PositionSelector(scope='generated', include_last=True)
        self.assertEqual(self.selected(selector), [9, 10, 11])

    def test_lens_and_output_recorders_retain_identical_span_and_last_sites(self):
        from jpeers.runtime import _Run
        logits = torch.zeros(12, 4)
        logits[2] = torch.tensor([0., 1., -1., -1.])
        logits[3] = torch.tensor([0., 2., 3., -1.])
        logits[9] = torch.tensor([0., 1., 2., 3.])
        model = SimpleNamespace(
            band=lambda _: [1], word_token_ids=lambda _: [0],
            vocab_strings=['target', 'one', 'two', 'three'], vocab_size=4,
            lens_logits=lambda layer, hidden, use_jacobian=True: logits[hidden[:, 0].long()])
        selector = PositionSelector(scope='spans', labels=['task'], include_last=True)
        req = GenerateRequest(prompt='fixture', max_new_tokens=0,
                              record=RecordSpec(watch=['target'], top_k=1,
                                                positions=selector, word_like_only=False))
        run = _Run(model, req, 10, {'task': [(2, 4)]})
        positions = torch.arange(12)
        run._record(1, positions[:, None].float(), positions)
        run.note_model_logits(logits)
        arrays = run.finish()
        self.assertEqual(arrays['positions'].tolist(), [2, 3, 9])
        self.assertEqual(arrays['watch_ranks'][:, 0, 0].tolist(), [1, 2, 3])
        self.assertEqual(arrays['watch_ranks_logit'][:, 0, 0].tolist(), [1, 2, 3])
        self.assertEqual(arrays['output_ranks'][:, 0].tolist(), [1, 2, 3])


class WorkerValidationTests(unittest.TestCase):
    def validator(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from followup_worker import validate_run
        return validate_run

    def test_narcbench_validates_40_reader_passes_from_20_source_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'traces').mkdir()
            turns = []
            for index in range(40):
                trace = f'traces/{index:05d}.npz'
                (root / trace).write_bytes(b'nonempty recorded fixture')
                turns.append({'trace_file': trace})
            summary = {'runs': [{}] * 40, 'rows': [{}] * 120,
                       'counts': {'source_runs': 20, 'matched_scenarios': 10,
                                  'teacher_forced_passes': 40, 'speaker_rows': 120}}
            self.assertEqual(self.validator()('narcbench', summary, turns, root), [])

    def test_reused_trace_reference_cannot_pass_the_recorded_turn_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'traces').mkdir()
            (root / 'traces/shared.npz').write_bytes(b'one existing trace')
            turns = [{'trace_file': 'traces/shared.npz'} for _ in range(32)]
            turns += [{'trace_file': None} for _ in range(64)]
            summary = {'reviews': [{}] * 32, 'frozen_notes': [{}] * 16,
                       'transfer_probes': [{}] * 32, 'no_peer_probes': [{}] * 16}
            self.assertTrue(self.validator()('exp1', summary, turns, root))


class ManagerReadinessTests(unittest.TestCase):
    def test_permanent_loader_error_in_503_health_body_is_visible_to_manager(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location('followup_review_controller',
                                                     scripts / 'queue_followup.py')
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        payload = {'status': 'error', 'detail': 'model initialization failed'}
        error = urllib.error.HTTPError('http://127.0.0.1:18211/health', 503,
                                       'Service unavailable', {},
                                       io.BytesIO(json.dumps(payload).encode()))
        with patch.object(controller.urllib.request, 'urlopen', side_effect=error):
            result = controller.get_json('http://127.0.0.1:18211/health')
        self.assertEqual(result, payload)


if __name__ == '__main__':
    unittest.main()
