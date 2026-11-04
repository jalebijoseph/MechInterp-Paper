"""Build the compact measurement audit from saved detailed JSON, using only stdlib.

Example: python scripts/exp0_audit/compact_measurement.py --output /tmp/measurement_summary.json
The output path is explicit so reproducing the report leaves saved results untouched.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'results/exp0-audit-20261104/measurement_trace_results.json'


def core_metadata():
    """Recover template and control labels from the unchanged benchmark generator."""
    spec = importlib.util.spec_from_file_location(
        'compact_audit_benchmarks', ROOT / 'src/jpeers/benchmarks.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {(framing, item.id): item
            for framing in ('people', 'agents')
            for item in module.false_belief_items(20, framing=framing, seed=0)}


def site_counts(rows, site):
    counts = Counter(row['sites'][site]['outcome'] for row in rows)
    return {'wins': counts['win'], 'ties': counts['tie'], 'losses': counts['loss']}


def compact_measurement(detailed):
    metadata = core_metadata()
    result = {}
    for model, audit in detailed.items():
        core = [row for row in audit['items'] if row['framing'] in ('people', 'agents')]
        hitom = [row for row in audit['items'] if row['framing'].startswith('hitom-')]

        def selected(*, control=None, framing=None, template=None):
            return [row for row in core
                    if (control is None or metadata[row['framing'], row['item']].control == control)
                    and (framing is None or row['framing'] == framing)
                    and (template is None or metadata[row['framing'], row['item']].kind == template)]

        story = {}
        for name, rows in [('all', core), ('false_belief', selected(control=False)),
                           ('controls', selected(control=True))]:
            counts = site_counts(rows, 'story_end')
            story[name] = {'n': len(rows), **counts,
                           'win_rate': counts['wins'] / len(rows) if rows else 0.0}
        templates = []
        for framing in ('people', 'agents'):
            for template in ('transfer', 'contents'):
                for control in (False, True):
                    rows = selected(control=control, framing=framing, template=template)
                    templates.append({'framing': framing, 'template': template, 'control': control,
                                      'n': len(rows), 'correct': sum(row['correct'] for row in rows),
                                      **{site: site_counts(rows, site)
                                         for site in ('story_end', 'question', 'answer')}})
        result[model] = {
            'integrity_checks': audit['checks'],
            'integrity_errors': audit['errors'],
            'core_story_end': story,
            'per_template': templates,
            'question_pooling': {
                'span_vs_last_token_flips': sum(
                    (row['sites']['question']['outcome'] == 'win')
                    != row['question_last_token_beats'] for row in core),
                'n': len(core)},
            'hitom': {
                'n': len(hitom),
                'correct': sum(row['correct'] for row in hitom),
                'first_token_letter': sum(bool(row['generated_tokens']) and bool(
                    re.fullmatch(r'[A-Za-z]', row['generated_tokens'][0].strip())) for row in hitom),
                'matching_target_token_forms': sum(
                    row['generated_contains_watched_target_form'] for row in hitom),
                'target_noun_collisions': sum(
                    row['n_choices_sharing_target_noun'] > 1 for row in hitom),
                # Ranks are zero-based: top ten means 0 through 9, not 10.
                'answer_start_jlens_hit10': sum(row['answer_jlens_rank'] < 10 for row in hitom),
                'later_best_jlens_hit10': sum(row['generated_best_jlens_rank'] < 10 for row in hitom),
                'correct_later_jlens_hit10': sum(
                    row['correct'] and row['generated_best_jlens_rank'] < 10 for row in hitom)}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error('--output must differ from --input')
    summary = compact_measurement(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print('Wrote', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
