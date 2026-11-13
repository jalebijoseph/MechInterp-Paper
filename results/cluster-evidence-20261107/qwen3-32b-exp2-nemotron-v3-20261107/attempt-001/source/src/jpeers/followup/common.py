"""Recording and statistics shared by the audited follow-up protocols."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import re

import numpy as np

from jpeers.harness import TurnRecord
from jpeers.schemas import GenerateRequest, PositionSelector, RecordSpec
from jpeers.trace import JSpaceTrace


def parse_json_object(text: str) -> dict:
    body = text.strip()
    if body.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', body, re.IGNORECASE)
        if not match:
            raise ValueError('Expected one complete JSON code fence')
        body = match.group(1)
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj
    result = json.loads(body, object_pairs_hook=unique_object)
    if not isinstance(result, dict):
        raise ValueError('Expected one JSON object')
    return result


def request(endpoint, writer, *, messages, tag, extra, watch=(), spans=(),
            max_new_tokens=192, record=True, prefill='', temperature=0., seed=0):
    messages = json.loads(json.dumps(messages))
    spans = list(spans)
    req = GenerateRequest(
        messages=messages, assistant_prefill=prefill,
        max_new_tokens=max_new_tokens, temperature=temperature, seed=seed, spans=spans,
        record=RecordSpec(enabled=record, watch=list(dict.fromkeys(watch)), top_k=10,
                          positions=PositionSelector(scope='spans',
                              labels=list(dict.fromkeys(s.label for s in spans)),
                              include_generated=True, include_last=True)))
    response = endpoint.generate(req)
    trace = JSpaceTrace.from_response(response)
    if record and trace is None:
        raise ValueError('Recorded follow-up request returned no trace')
    rec = TurnRecord(index=0, agent=extra.get('agent', 'subject'), endpoint=endpoint.name,
                     model=response.model, text=(prefill + response.text).strip(),
                     prompt_len=response.prompt_len, n_generated=response.n_generated,
                     timings=response.timings, interventions=[], trace_file=None, tag=tag,
                     messages=messages, extra={**extra, 'assistant_prefill': prefill,
                         'max_new_tokens': max_new_tokens, 'temperature': temperature,
                         'seed': seed, 'protocol': 'audited-followup-v2'})
    if writer is not None:
        writer.write(rec, trace)
    return rec, trace


def _reciprocal_mean(array):
    if array is None or not np.size(array):
        return None
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Invalid rank array')
    return float(np.mean(1. / (array + 1.)))


def site_scores(trace, words, mask):
    words = list(dict.fromkeys(words))
    missing = [word for word in words if word not in trace.watch]
    if missing:
        raise ValueError(f'Watch words missing from trace: {missing}')
    selected = [trace.watch.index(word) for word in words]
    mask = np.asarray(mask, dtype=bool)
    out = {'jlens': None, 'logit': None, 'output': None, 'by_layer': {}}
    if not selected or not mask.any():
        return out
    for lens, array in [('jlens', trace.watch_ranks), ('logit', trace.watch_ranks_logit)]:
        if array is None:
            continue
        # Best word at each position/layer, then average reciprocal ranks. Each
        # layer contributes equally; there is no best-of-band selection.
        ranks = np.asarray(array)[mask][:, :, selected].min(axis=2)
        out[lens] = _reciprocal_mean(ranks)
        for li, layer in enumerate(trace.layers):
            out['by_layer'].setdefault(str(layer), {})[lens] = _reciprocal_mean(ranks[:, li])
    if trace.output_ranks is not None:
        out['output'] = _reciprocal_mean(np.asarray(trace.output_ranks)[mask][:, selected].min(axis=1))
    return out


def paired_stats(diffs, *, permutations=50_000, seed=0):
    supplied = list(diffs)
    values = np.asarray([v for v in supplied if v is not None and np.isfinite(v)], dtype=float)
    n = len(values)
    result = {'n': n, 'n_total': len(supplied), 'n_missing': len(supplied) - n,
              'diff': float(values.mean()) if n else None, 'p': None,
              'p_exact': None, 'method': 'no_valid_pairs'}
    if not n:
        return result
    observed = abs(float(values.mean()))
    if n <= 20:
        hits = total = 0
        for signs in itertools.product((-1., 1.), repeat=n):
            hits += abs(float(np.dot(values, signs) / n)) >= observed - 1e-12
            total += 1
        result.update(p=hits / total, p_exact=hits / total, method='exact_cluster_sign_flip')
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1., 1.]), size=(permutations, n))
        hits = int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-12).sum())
        result.update(p=(hits + 1) / (permutations + 1),
                      method='monte_carlo_cluster_sign_flip', permutations=permutations, seed=seed)
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)
