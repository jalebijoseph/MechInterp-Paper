"""Reproduce the compact audit from saved CPU audit data using only stdlib."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/exp0_audit/compact_measurement.py'
AUDIT = ROOT / 'results/exp0-audit-20261104'


class MeasurementSummaryTests(unittest.TestCase):
    def run_summary(self, output, *args):
        result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), *args],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(output.read_text())

    def test_reproduces_saved_report_and_independent_known_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_summary(Path(tmp) / 'compact.json')
        expected = json.loads((AUDIT / 'measurement_summary.json').read_text())
        self.assertEqual(summary, expected)
        self.assertEqual(sum(r['integrity_checks']['traces'] for r in summary.values()), 280)
        self.assertEqual(sum(r['integrity_checks']['watch_words_checked']
                             for r in summary.values()), 1554)
        for model, wins in [('8b', 19), ('32b', 16)]:
            row = summary[model]
            self.assertEqual(row['integrity_errors'], [])
            self.assertEqual(row['core_story_end']['all']['ties'], 0)
            self.assertEqual(row['core_story_end']['controls']['n'], 40)
            self.assertEqual(row['core_story_end']['controls']['wins'], wins)
            self.assertEqual(row['hitom']['first_token_letter'], 60)
            self.assertEqual(row['hitom']['target_noun_collisions'], 41)

    def test_explicit_input_preserves_integrity_errors_and_rank_boundary(self):
        detail = json.loads((AUDIT / 'measurement_trace_results.json').read_text())
        detail = {'8b': copy.deepcopy(detail['8b'])}
        detail['8b']['errors'] = [['fb000', 'example_integrity_error']]
        hitom = [r for r in detail['8b']['items'] if r['framing'].startswith('hitom')]
        for row in hitom:
            row['generated_best_jlens_rank'] = 10
            row['answer_jlens_rank'] = 9
        hitom[0]['generated_best_jlens_rank'] = 9
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'detailed.json'
            source.write_text(json.dumps(detail))
            summary = self.run_summary(Path(tmp) / 'compact.json', '--input', str(source))['8b']
        self.assertEqual(summary['integrity_errors'], [['fb000', 'example_integrity_error']])
        self.assertEqual(summary['hitom']['later_best_jlens_hit10'], 1)
        self.assertEqual(summary['hitom']['answer_start_jlens_hit10'], 60)

    def test_output_path_is_required(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--output', result.stderr)


if __name__ == '__main__':
    unittest.main()
