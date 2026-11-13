"""Reproduce the audit's behavioral comparisons from saved original and diagnostic data."""
import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results-root', type=Path, default=Path(__file__).resolve().parents[2] / 'results')
args = ap.parse_args()
RUNS = args.results_root.resolve()
AUDIT = RUNS / 'exp0-audit-20261104'
SOURCE = Path(__file__).resolve().parents[2] / 'src/jpeers/benchmarks.py'
spec = importlib.util.spec_from_file_location('audit_benchmarks', SOURCE)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
items = {(f, it.id): it for f in ['people', 'agents']
         for it in benchmark.false_belief_items(20, framing=f, seed=0)}
originals = {
    '8b': RUNS / 'qwen3-8b-20261104/jobs/exp0-core/attempt-001/results/tom_validation/exp0-core',
    '32b': RUNS / 'qwen3-32b-exp0-20261104/tom_validation/qwen3-32b-exp0-attempt-2-1793812722733295233',
}

def words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def first_answer_correct(row):
    ws = words(row['answer'])
    return bool(ws) and ws[0] == row['expected']

result = {'purpose': 'Post-hoc setup audit; does not change original results or preregistered gate',
          'scoring': {'original_check': 'Independent whole-word target presence compared with saved correct flags',
                      'diagnostic': 'First alphabetic answer word, ignoring capitalization and Markdown',
                      'non_single_answers': 'Reported explicitly; all manually checked to begin with an answer word'},
          'models': {}, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
for model, directory in originals.items():
    summary = json.loads((directory / 'summary.json').read_text())
    baseline = {(r['framing'], r['item']): r for r in summary['items']
                if r['framing'] in ['people', 'agents']}
    turns = [json.loads(s) for s in (directory / 'turns.jsonl').read_text().splitlines()]
    turns = [r for r in turns if not r['extra']['item'].startswith('hitom-')]
    prompt_errors, score_errors, target_errors = [], [], []
    for turn in turns:
        key = (turn['tag'].split('/')[-1], turn['extra']['item'])
        it = items[key]
        user = it.story + '\n\nQuestion: ' + it.question
        if turn['messages'][1]['content'] != user:
            prompt_errors.append(key)
        row = baseline[key]
        if (it.belief_answer in words(row['answer'])) != row['correct']:
            score_errors.append(key)
        if it.control and (it.belief_answer != it.true_answer or it.decoy == it.true_answer):
            target_errors.append(key)
    data = json.loads((AUDIT / f'full-{model}.json').read_text())
    direct = json.loads((AUDIT / f'direct-{model}.json').read_text())
    all_rows = data['rows'] + direct['rows']
    replay = [r for r in data['rows'] if r['variant'] == 'original']
    mismatches = [dict(framing=r['framing'], item=r['item']) for r in replay
                  if baseline[(r['framing'], r['item'])]['answer'] != r['answer']]
    grouped = []
    for variant in data['metadata']['variants'] + direct['metadata']['variants']:
        for framing in ['people', 'agents']:
            for kind in ['transfer', 'contents']:
                rs = [r for r in all_rows if r['variant'] == variant
                      and r['framing'] == framing and r['kind'] == kind]
                fb = [r for r in rs if not r['control']]
                controls = [r for r in rs if r['control']]
                pairs = collections.defaultdict(list)
                for r in rs:
                    pairs[r['item'].removesuffix('c')].append(first_answer_correct(r))
                grouped.append({'variant': variant, 'framing': framing, 'kind': kind,
                                'n_false_belief': len(fb), 'false_belief_correct': sum(map(first_answer_correct, fb)),
                                'n_controls': len(controls), 'controls_correct': sum(map(first_answer_correct, controls)),
                                'pairs_both_correct': sum(all(v) for v in pairs.values()),
                                'unique_control_user_prompts': len({r['messages'][1]['content'] for r in controls})})
    result['models'][model] = {
        'checks': {'original_core_turns': len(turns), 'prompt_errors': prompt_errors,
                   'whole_word_score_errors': score_errors, 'control_target_errors': target_errors,
                   'reverse_order_native_replay_n': len(replay), 'replay_string_mismatches': mismatches},
        'diagnostic_metadata': data['metadata'], 'direct_belief_metadata': direct['metadata'], 'groups': grouped,
        'non_single_word_outputs': [{k: r[k] for k in ['variant', 'framing', 'item', 'expected', 'answer']}
                                   for r in all_rows if len(words(r['answer'])) != 1],
        'diagnostic_data_sha256': hashlib.sha256((AUDIT / f'full-{model}.json').read_bytes()).hexdigest(),
        'direct_belief_data_sha256': hashlib.sha256((AUDIT / f'direct-{model}.json').read_bytes()).hexdigest(),
    }
    assert len(turns) == 80 and len(replay) == 80 and len(data['rows']) == 400
    assert len(direct['rows']) == 160
    assert not prompt_errors and not score_errors and not target_errors and not mismatches
small = json.loads((AUDIT / 'diagnostic-results.json').read_text())
base = {(r['framing'], r['item']): r['answer'] for r in small['rows'] if r['variant'] == 'original_native'}
result['instrumentation_check_8b'] = {}
for variant in ['original_runtime_record_False', 'original_runtime_record_True']:
    rs = [r for r in small['rows'] if r['variant'] == variant]
    n_equal = sum(r['answer'] == base[(r['framing'], r['item'])] for r in rs)
    result['instrumentation_check_8b'][variant] = {'n': len(rs), 'exact_native_matches': n_equal}
    assert n_equal == len(rs) == 16
result['measurement_audit'] = json.loads((AUDIT / 'measurement_summary.json').read_text())
for model in ['8b', '32b']:
    assert not result['measurement_audit'][model]['integrity_errors']
(AUDIT / 'audit_summary.json').write_text(json.dumps(result, indent=2))
for model, data in result['models'].items():
    print(model, 'integrity checks:', data['checks'])
    for row in data['groups']:
        if row['variant'] in ['original', 'no_answer_prefill', 'world_question', 'direct_belief']:
            print(row['variant'], row['framing'], row['kind'],
                  'FB', row['false_belief_correct'], '/', row['n_false_belief'],
                  'control', row['controls_correct'], '/', row['n_controls'],
                  'both', row['pairs_both_correct'])
print('Wrote', AUDIT / 'audit_summary.json')
