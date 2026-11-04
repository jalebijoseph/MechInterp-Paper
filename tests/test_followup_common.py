"""Regression checks for fair pooling, strict parsing and follow-up recording."""
import importlib.util
import unittest

import numpy as np

from jpeers.schemas import PositionSelector
from jpeers.trace import JSpaceTrace


class CommonTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('jpeers.followup.common'))
        from jpeers.followup import common
        return common

    def test_primary_score_averages_layers_instead_of_maximizing_them(self):
        common = self.module()
        ranks = np.array([[[0, 99], [99, 0]], [[0, 99], [99, 0]]])
        trace = JSpaceTrace(positions=np.array([1, 2]), topk_ids=np.zeros((2, 2, 1), int),
                           topk_logits=np.zeros((2, 2, 1)), watch_ranks=ranks,
                           watch_ranks_logit=ranks, output_ranks=np.array([[1, 2], [1, 2]]),
                           layers=[24, 25], watch=['first', 'second'])
        score = common.site_scores(trace, ['first'], np.ones(2, bool))
        self.assertAlmostEqual(score['jlens'], .505)
        self.assertAlmostEqual(score['output'], .5)
        self.assertEqual(common.site_scores(trace, ['first'], np.zeros(2, bool))['jlens'], None)

    def test_exact_cluster_test_and_strict_json_parser(self):
        common = self.module()
        result = common.paired_stats([1., 1., 1., 1.])
        self.assertEqual(result['n'], 4)
        self.assertAlmostEqual(result['p_exact'], .125)
        self.assertEqual(common.parse_json_object('```json\n{"choice":"tie"}\n```'), {'choice': 'tie'})
        for response in ['A or B', '{"choice":"A"} {"choice":"B"}', '["A"]',
                         '{"choice":"A","choice":"B"}']:
            with self.assertRaises(ValueError):
                common.parse_json_object(response)

    def test_last_prompt_recording_is_explicit_and_opt_in(self):
        selected = PositionSelector(scope='spans', labels=['task'], include_last=True)
        self.assertTrue(getattr(selected, 'include_last', False))
        self.assertFalse(PositionSelector().include_last)


if __name__ == '__main__':
    unittest.main()
