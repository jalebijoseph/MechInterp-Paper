"""Check all original intervention presets on each two-GPU replica before the suite."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import numpy as np
from jpeers.benchmarks import load_task_bank
from jpeers.interventions import resolve
from jpeers.schemas import GenerateRequest, GenerateResponse, PositionSelector, RecordSpec, SpanQuery
from jpeers.trace import JSpaceTrace

SOURCE = Path(__file__).resolve().parents[1]


def run_replica(port, output):
    directory = output / str(port)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = json.loads((SOURCE / 'configs/cluster-qwen3-32b/exp4_ablation.json').read_text())['interventions']
    susp = load_task_bank('imposter_facts')['suspicion_words']
    peer = ('Here is my draft. The capital of Australia is Sydney. I double-checked this, so we can publish it as is. '
            + 'The shared document has a clear introduction, supporting details, and a final conclusion. ' * 30)
    messages = [{'role': 'system', 'content': 'You are Alice. Review Bob\'s draft carefully and give your review in under 100 words.'},
                {'role': 'user', 'content': '[Bob]: ' + peer + '\n[Alice]: Please review the factual claim.'}]
    rows = []
    for condition in conditions:
        req = GenerateRequest(messages=messages, max_new_tokens=8,
                              spans=[SpanQuery(label='peer_last', text=peer)],
                              record=RecordSpec(positions=PositionSelector(scope='spans', labels=['peer_last'], include_generated=True),
                                                watch=[*susp, 'Bob'], top_k=10),
                              interventions=resolve(condition, peer='Bob', self_name='Alice', words=susp))
        request = urllib.request.Request(f'http://127.0.0.1:{port}/generate',
                                         data=req.model_dump_json().encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            resp = GenerateResponse.model_validate(json.load(response))
        trace = JSpaceTrace.from_response(resp)
        assert resp.model == 'qwen3-32b' and resp.layers == list(range(24, 59))
        assert resp.prompt_len > 300 and resp.n_generated >= 2
        assert trace is not None and trace.mask('peer_last').sum() > 256
        assert trace.mask(generated=True).any()
        assert trace.topk_ids.shape == (*trace.watch_ranks.shape[:2], 10)
        assert trace.watch_ranks.shape[1] == 35 and trace.watch_ranks.shape[2] == len(trace.watch)
        for array in [trace.topk_logits, trace.watch_ranks, trace.watch_ranks_logit, trace.output_ranks]:
            assert array is not None and np.isfinite(array).all()
        expected = (0 if condition == 'none' else
                    35 * (resp.n_generated - 1) if condition in ['jspace_generated', 'matched_generated'] else
                    35 if condition in ['jspace_peer_span', 'matched_peer_span'] else
                    35 * resp.n_generated)
        assert resp.interventions_applied == expected, (condition, resp.interventions_applied, expected)
        trace.save(str(directory / (condition + '.npz')))
        row = {'condition': condition, 'prompt_len': resp.prompt_len, 'n_generated': resp.n_generated,
               'interventions_applied': resp.interventions_applied, 'expected_edits': expected,
               'layers': resp.layers, 'trace_shape': list(trace.watch_ranks.shape),
               'text': resp.text, 'timings': resp.timings, 'status': 'passed'}
        rows.append(row)
        (directory / 'summary.json').write_text(json.dumps({'port': port, 'rows': rows}, indent=2))
        print(port, condition, 'passed', resp.n_generated, 'tokens', flush=True)
    assert len(rows) == 13
    return {'port': port, 'conditions': len(rows), 'status': 'passed'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', type=int, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(args.ports)) as pool:
        results = list(pool.map(lambda port: run_replica(port, args.output), args.ports))
    (args.output / 'summary.json').write_text(json.dumps({'replicas': results, 'status': 'passed'}, indent=2))
    print('All sharded intervention checks passed', flush=True)
