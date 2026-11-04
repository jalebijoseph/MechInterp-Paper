"""Read saved Exp0 traces on CPU; do not load any model weights or change runs."""
import argparse
import collections
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from jpeers.benchmarks import false_belief_items, load_hitom
from jpeers.experiments.tom_validation import _best, _beats_all, _site_masks, hitom_nouns
from jpeers.registry import get_model_spec
from jpeers.runtime import LensedModel, _locate_spans, whole_word_ids
from jpeers.schemas import GenerateRequest, SpanQuery
from jpeers.trace import JSpaceTrace

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results-root', type=Path, default=Path(__file__).resolve().parents[2] / 'results')
args = ap.parse_args()
ROOT = args.results_root.resolve()
core = {(framing, it.id): it for framing in ('people', 'agents')
        for it in false_belief_items(20, framing=framing, seed=0)}
hitom = {it.id: it for it in load_hitom(orders=(0, 1, 2, 3), max_items=60)}
runs = {
    '8b': [ROOT/'qwen3-8b-20261104/jobs/exp0-core/attempt-001/results',
           ROOT/'qwen3-8b-20261104/jobs/exp0-hitom/attempt-001/results'],
    '32b': [ROOT/'qwen3-32b-exp0-20261104'],
}
result = {}
revisions = {'8b': 'b968826d9c46dd6066d109eabc6255188de91218',
             '32b': '9216db5781bf21249d130ec9da846c4624c16137'}
for model_name, roots in runs.items():
    spec = get_model_spec('qwen3-' + model_name)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=revisions[model_name],
                                               local_files_only=True)
    facade = object.__new__(LensedModel)
    facade.spec, facade.tokenizer = spec, tokenizer
    facade.lm = SimpleNamespace(input_device=torch.device('cpu'))
    checks = collections.Counter()
    errors, examples, item_details = [], [], []
    grouped = collections.defaultdict(collections.Counter)
    for root in roots:
        for summary_path in root.rglob('summary.json'):
            summary = json.loads(summary_path.read_text())
            if 'items' not in summary:
                continue
            directory = summary_path.parent
            summaries = {(r['framing'], r['item']): r for r in summary['items']}
            turn_path = next(directory.glob('*.jsonl'))
            turns = [json.loads(line) for line in turn_path.read_text().splitlines()]
            for turn in turns:
                trace = JSpaceTrace.load(str(directory/turn['trace_file']))
                item_id = turn['extra']['item']
                is_hitom = item_id.startswith('hitom-')
                framing = 'hitom-o' + str(hitom[item_id].order) if is_hitom else turn['tag'].split('/')[-1]
                item = hitom[item_id] if is_hitom else core[(framing, item_id)]
                saved = summaries[(framing, item_id)]
                target = item.answer.split('_')[-1] if is_hitom else item.belief_answer
                competitors = ([n for n in hitom_nouns(item.choices) if n != target]
                               if is_hitom else [item.decoy])
                label = f'{framing}/' + ('false' if saved['false_belief'] else 'control')
                checks['traces'] += 1
                if len(trace.tokens) != turn['prompt_len'] + turn['n_generated']:
                    errors.append([item_id, 'token_count'])
                end = turn['prompt_len'] + max(turn['n_generated'] - 1, 0)
                if np.any(trace.positions >= end) or np.any(np.diff(trace.positions) <= 0):
                    errors.append([item_id, 'trace_positions'])
                for word, ids in zip(trace.watch, trace.watch_ids):
                    checks['watch_words_checked'] += 1
                    if whole_word_ids(tokenizer, word) != ids:
                        errors.append([item_id, 'watch_ids', word])
                req = GenerateRequest(messages=turn['messages'], assistant_prefill='Answer:',
                    spans=[SpanQuery(label='story_tail', text=item.story[-40:]),
                           SpanQuery(label='question', text=item.question),
                           SpanQuery(label='answer', text='Answer:')])
                rendered, add_special = facade.render(req)
                ids, offsets = facade.encode(rendered, add_special)
                expected_tokens = tokenizer.batch_decode([[int(i)] for i in ids[0]])
                if expected_tokens != trace.tokens[:trace.prompt_len]:
                    errors.append([item_id, 'prompt_tokenization'])
                expected_spans = _locate_spans(rendered, offsets, req, trace.prompt_len)
                if expected_spans != trace.spans:
                    errors.append([item_id, 'spans', expected_spans, trace.spans])
                last = trace.mask((trace.prompt_len - 1, trace.prompt_len))
                if int(last.sum()) != 1:
                    errors.append([item_id, 'last_mask'])
                sites = _site_masks(trace, last)
                if _best(trace, target, last) != saved['belief_rank']:
                    errors.append([item_id, 'belief_rank_reproduction'])
                for site, mask in sites.items():
                    if _beats_all(trace, target, competitors, mask, 'jlens') != saved['beats_' + site]:
                        errors.append([item_id, 'beats_reproduction', site])
                for lens in ('logit', 'output'):
                    if _beats_all(trace, target, competitors, last, lens) != saved['beats_answer_' + lens]:
                        errors.append([item_id, 'baseline_reproduction', lens])
                for row_index, pos in enumerate(trace.positions):
                    # Prompt tokens are teacher-forced and need not be argmax.
                    # Only sampled continuation tokens validate greedy output alignment.
                    if pos < trace.prompt_len - 1 or pos + 1 >= len(trace.tokens):
                        continue
                    next_token = trace.tokens[pos + 1]
                    for wi, word in enumerate(trace.watch):
                        forms = {trace.vocab[tid] for tid in trace.watch_ids[wi]}
                        if next_token in forms:
                            checks['actual_next_token_matches_watched_form'] += 1
                            if int(trace.output_ranks[row_index, wi]) != 0:
                                errors.append([item_id, 'actual_next_token_output_rank', int(pos), word,
                                               next_token, int(trace.output_ranks[row_index, wi])])
                detail = {'item': item_id, 'framing': framing, 'correct': saved['correct'],
                          'target': target, 'competitors': competitors, 'answer': turn['text'],
                          'generated_tokens': trace.tokens[trace.prompt_len:],
                          'target_ids': trace.watch_ids[trace.watch_index(target)],
                          'target_forms': [trace.vocab[i] for i in trace.watch_ids[trace.watch_index(target)]],
                          'sites': {}}
                for site, mask in sites.items():
                    target_rank = _best(trace, target, mask)
                    competitor_rank = min(_best(trace, word, mask) for word in competitors)
                    outcome = 'win' if target_rank < competitor_rank else 'tie' if target_rank == competitor_rank else 'loss'
                    key = label + '/' + site
                    grouped[key]['n'] += 1
                    grouped[key][outcome] += 1
                    grouped[key]['zero_ties'] += int(target_rank == competitor_rank == 0)
                    detail['sites'][site] = {'target_rank': target_rank, 'competitor_rank': competitor_rank,
                                             'outcome': outcome, 'positions': trace.positions[mask].tolist(),
                                             'tokens': [trace.tokens[p] for p in trace.positions[mask]]}
                    if site == 'question':
                        end_question = trace.spans['question'][-1][1]
                        qlast = trace.mask((end_question - 1, end_question))
                        last_win = _beats_all(trace, target, competitors, qlast, 'jlens')
                        pooled_win = outcome == 'win'
                        grouped[key]['entire_question_vs_last_token_flips'] += int(last_win != pooled_win)
                        detail['question_last_token_beats'] = last_win
                target_index = trace.watch_index(target)
                detail['answer_output_rank'] = int(trace.output_ranks[last, target_index][0])
                detail['answer_logit_rank'] = _best(trace, target, last, 'logit')
                detail['answer_jlens_rank'] = _best(trace, target, last, 'jlens')
                gmask = trace.mask(generated=True)
                detail['generated_best_jlens_rank'] = _best(trace, target, gmask)
                detail['generated_best_output_rank'] = _best(trace, target, gmask, 'output')
                if is_hitom:
                    choices = re.findall(r'[a-z]+(?:_[a-z]+)+', item.choices)
                    detail['correct_choice'] = item.answer
                    detail['n_choices'] = len(choices)
                    detail['n_distinct_choice_nouns'] = len(hitom_nouns(item.choices))
                    detail['n_choices_sharing_target_noun'] = sum(c.split('_')[-1] == target for c in choices)
                    detail['generated_contains_watched_target_form'] = any(
                        token in detail['target_forms'] for token in detail['generated_tokens'])
                    detail['generated_position_readouts'] = [
                        {'position': int(p), 'token': trace.tokens[p],
                         'next_token': trace.tokens[p + 1] if p + 1 < len(trace.tokens) else None,
                         'jlens_best_rank': int(trace.watch_ranks[ri, :, target_index].min()),
                         'best_layer': trace.layers[int(trace.watch_ranks[ri, :, target_index].argmin())],
                         'output_rank': int(trace.output_ranks[ri, target_index])}
                        for ri, p in enumerate(trace.positions) if p >= trace.prompt_len]
                if not is_hitom:
                    a = len(trace.watch_ids[target_index]); b = len(trace.watch_ids[trace.watch_index(competitors[0])])
                    grouped[label]['variant_count_equal' if a == b else 'target_more_variants' if a > b else 'target_fewer_variants'] += 1
                item_details.append(detail)
    result[model_name] = {'checks': dict(checks), 'errors': errors,
                          'groups': {k: dict(v) for k, v in grouped.items()}, 'items': item_details}
print(json.dumps(result, indent=2))
