"""Focused assay checks use a deterministic synthetic endpoint, never a model download."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jpeers.followup import handoff as h


def test_factorial_inventory_balances_all_nuisances_and_has_equal_label_cells():
    tasks = h.make_episodes()
    dev = [x for x in tasks if x.split == 'development']
    heldout = [x for x in tasks if x.split == 'heldout']
    assert (len(dev), len(heldout)) == (256, 768)
    assert {x.template_id for x in dev}.isdisjoint(x.template_id for x in heldout)
    assert len({h.digest(x.messages()) for x in tasks}) == 1024
    for split in (dev, heldout):
        groups = {}
        for x in split:
            groups.setdefault((x.template_id, x.source, x.names, x.order), []).append(x)
        for group in groups.values():
            assert Counter((x.own, x.incoming) for x in group) == Counter(
                {('red', 'red'): 1, ('red', 'blue'): 1, ('blue', 'red'): 1, ('blue', 'blue'): 1})
        for x in split:
            prompt = x.messages()[-1]['content']
            assert prompt.count(h.SUFFIX) == 1
            assert prompt.endswith(h.SUFFIX)
            assert prompt.index(h.SUFFIX) > prompt.index(x.own_note())
            assert prompt.index(h.SUFFIX) > prompt.index(x.incoming_note())


def test_counterfactual_preserves_all_but_incoming_and_includes_equal_cells():
    rows = h.make_episodes()[:32]
    by_id = {x.key: x for x in rows}
    for x in rows:
        donor = h.opposite_donor(x, rows)
        assert donor.key in by_id
        assert donor.own == x.own and donor.incoming != x.incoming
        assert (donor.template_id, donor.source, donor.names, donor.order) == (
            x.template_id, x.source, x.names, x.order)
        assert x.messages()[-1]['content'].replace(x.incoming_note(), 'INCOMING') == (
            donor.messages()[-1]['content'].replace(donor.incoming_note(), 'INCOMING'))


@pytest.mark.parametrize('text,own,incoming,output_format,raw_strict', [
    (' {"own":"red","incoming":"blue"}\n', 'red', 'blue', 'bare_json', True),
    ('{"own":"blue","incoming":null}', 'blue', None, 'bare_json', True),
    ('```json\n{"own":"red","incoming":"blue"}\n```', 'red', 'blue', 'fenced_json', False),
    ('```json\n{"own":"blue","incoming":null}\n```', 'blue', None, 'fenced_json', False),
    ('```\n{"own":"blue","incoming":"red"}\n```', 'blue', 'red', 'fenced_json', False),
    ('```\n{"own":"red","incoming":null}\n```', 'red', None, 'fenced_json', False),
    (' \n```json\r\n{"own":"red","incoming":"red"}\r\n```\n',
     'red', 'red', 'fenced_json', False),
])
def test_json_v2_scores_one_whole_bare_or_fenced_object(text, own, incoming, output_format, raw_strict):
    score = h.score_output(text, own=own, incoming=incoming, n_generated=20, cap=48)
    assert score['known'] and score['format_valid'] and score['joint_correct']
    assert (score['predicted_own'], score['predicted_incoming']) == (own, incoming)
    assert score['json_scoring_version'] == 'json-object-v2'
    assert score['output_format'] == output_format
    assert score['raw_strict_format_valid'] is raw_strict


@pytest.mark.parametrize('wrap', ['{}', '```json\n{}\n```', '```\n{}\n```'])
@pytest.mark.parametrize('payload', [
    '{"own":"red","incoming":"blue","x":0}',
    '{"own":"red","own":"red","incoming":"blue"}',
    '{"own":"red","incoming":"blue","incoming":null}',
    '{"own":"red","incoming":"green"}',
    '{"own":"green","incoming":"blue"}',
    '{"own":null,"incoming":"blue"}',
    '{"own":"red","incoming":false}',
    '{"own":1,"incoming":"blue"}',
    '{"own":"red","incoming":[]}',
    '{"own":{},"incoming":"blue"}',
    '{"own":"red"}', '[]',
    '{"own":"red","incoming":"blue"',
    '{"own":"red","incoming":"blue",}',
    '{"own":"red","incoming":"blue"} {"own":"red","incoming":"blue"}',
])
def test_json_v2_counts_invalid_schema_or_json_as_failure(wrap, payload):
    text = wrap.replace('{}', payload)
    s = h.score_output(text, own='red', incoming='blue', n_generated=12, cap=48)
    assert not s['format_valid'] and not s['own_correct'] and not s['incoming_correct']
    assert s['known']


@pytest.mark.parametrize('text', [
    'Answer: {"own":"red","incoming":"blue"}',
    '{"own":"red","incoming":"blue"} Done.',
    'Answer:\n```json\n{"own":"red","incoming":"blue"}\n```',
    '```json\n{"own":"red","incoming":"blue"}\n```\nDone.',
    '```python\n{"own":"red","incoming":"blue"}\n```',
    '```json\n{"own":"red","incoming":"blue"}',
    '```json {"own":"red","incoming":"blue"}```',
    '```json\n{"own":"red","incoming":"blue"}\n```\n```json\n{"own":"red","incoming":"blue"}\n```',
    '```json\n{"own":"red","incoming":"blue"}\n```\n{"own":"red","incoming":"blue"}',
])
def test_json_v2_rejects_prose_multiple_answers_and_incomplete_or_unknown_fences(text):
    score = h.score_output(text, own='red', incoming='blue', n_generated=20, cap=48)
    assert score['known'] and not score['format_valid'] and not score['joint_correct']


@pytest.mark.parametrize('wrap', ['{}', '```json\n{}\n```', '```\n{}\n```'])
def test_json_v2_keeps_label_and_null_correctness_separate_from_format(wrap):
    score = h.score_output(wrap.replace('{}', '{"own":"blue","incoming":null}'),
                           own='red', incoming='blue', n_generated=20, cap=48)
    assert score['format_valid']
    assert not score['own_correct'] and not score['incoming_correct'] and not score['joint_correct']


@pytest.mark.parametrize('wrap,raw_strict', [('{}', True), ('```json\n{}\n```', False),
                                           ('```\n{}\n```', False)])
@pytest.mark.parametrize('n_generated', [None, 48, 49])
def test_json_v2_never_accepts_complete_looking_cap_hits(wrap, raw_strict, n_generated):
    score = h.score_output(wrap.replace('{}', '{"own":"red","incoming":null}'),
                           own='red', incoming=None, n_generated=n_generated, cap=48)
    assert score['cap_hit'] and not score['format_valid']
    assert not score['own_correct'] and not score['incoming_correct'] and not score['joint_correct']
    assert score['raw_strict_format_valid'] is raw_strict


def test_cap_hits_failure_and_missing_is_unknown():
    good = '{"own":"red","incoming":"blue"}'
    assert h.score_output(good, own='red', incoming='blue', n_generated=10, cap=48)['joint_correct']
    cap = h.score_output(good, own='red', incoming='blue', n_generated=48, cap=48)
    assert cap['cap_hit'] and not cap['incoming_correct']
    absent = h.score_output(None, own='red', incoming='blue', n_generated=None, cap=48)
    assert not absent['known'] and absent['incoming_correct'] is None
    assert absent['json_scoring_version'] == 'json-object-v2'
    assert absent['output_format'] is None and absent['raw_strict_format_valid'] is None
    assert h.score_output('{"own":"red","incoming":null}', own='red', incoming=None,
                          n_generated=10, cap=48)['joint_correct']


def test_conditional_decoder_template_folds_control_own_and_nuisance():
    rows = [x for x in h.make_episodes() if x.split == 'development']
    # Coordinate zero carries incoming, one carries own; a J span containing both
    # must isolate the former under the full crossing.
    states = np.array([[1 if x.incoming == 'red' else -1,
                        1 if x.own == 'red' else -1, (x.order * 2 - 1)] for x in rows], float)
    fit = h.fit_component(rows, states, np.eye(3)[:2], ridge=1e-6)
    assert fit['incoming_cv'] == 1
    assert fit['own_cv'] == 1
    assert fit['incoming_basis_own_cv'] <= .6
    assert fit['incoming_basis_nuisance_cv'] <= .6
    assert np.asarray(fit['basis']).shape == (1, 3)
    assert np.allclose(np.abs(fit['basis'][0]), [1, 0, 0], atol=1e-5)
    assert all(set(f['train_templates']).isdisjoint(f['test_templates']) for f in fit['folds'])
    contaminated = h.fit_component(rows, states[:, [1, 2]], np.eye(2), ridge=1e-6)
    assert contaminated['incoming_cv'] < .8


def test_site_preflight_requires_one_prompt_token_at_neutral_suffix_endpoint():
    trace = SimpleNamespace(positions=np.array([9]), prompt_len=12, tokens=['x'] * 9 + ['.', 'a', 'b'],
        spans={'causal_site': [(9, 10)], 'neutral_suffix': [(6, 10)],
               'own_payload': [(1, 3)], 'incoming_payload': [(3, 6)]},
        watch=['red', 'blue'], watch_ids=[[5], [6]],
        capture_metadata={'direction_watch_ids': [5, 6]})
    assert h.validate_capture_site(trace)['position'] == 9
    trace.spans['causal_site'] = [(8, 10)]
    with pytest.raises(ValueError, match='single'):
        h.validate_capture_site(trace)


def test_resume_rejects_request_change_and_corrupt_receipt(tmp_path):
    store = h.ReceiptStore(tmp_path, {'protocol': 'test'})
    req = {'messages': [{'role': 'user', 'content': 'a'}]}
    store.save('one', req, {'model': 'toy', 'text': 'ok'}, {'incoming_correct': True})
    assert store.load('one', req)['response']['text'] == 'ok'
    with pytest.raises(ValueError, match='request'):
        store.load('one', {'messages': []})
    path = store.path('one')
    data = json.loads(path.read_text())
    data['response']['text'] = 'edited'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='integrity'):
        store.load('one', req)
    with pytest.raises(ValueError, match='provenance'):
        h.ReceiptStore(tmp_path, {'protocol': 'other'})


def test_receipt_integer_vocab_keys_round_trip_and_tampering_still_fails(tmp_path):
    store = h.ReceiptStore(tmp_path, {'protocol': 'test'})
    request = {'messages': [{'role': 'user', 'content': 'a'}]}
    response = {'model': 'toy', 'text': 'ok', 'vocab': {2: 'two', 10: 'ten'}}
    store.save('integer-vocab', request, response, {'incoming_correct': True})
    receipt = store.load('integer-vocab', request)
    assert receipt['response'] == {
        'model': 'toy', 'text': 'ok', 'vocab': {'2': 'two', '10': 'ten'}}
    assert response['vocab'] == {2: 'two', 10: 'ten'}
    path = store.path('integer-vocab')
    tampered = json.loads(path.read_text())
    tampered['response']['vocab']['10'] = 'edited'
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='integrity'):
        store.load('integer-vocab', request)


def test_failed_discovery_prevents_any_heldout_request(tmp_path):
    h.atomic_json(tmp_path / 'status.json', {'status': 'discovery_failed', 'reason': 'no decoder'})
    with pytest.raises(ValueError, match='discovery'):
        h.load_frozen(tmp_path, {'version': 'test'})


def test_freeze_is_hashed_immutable_and_bound_to_provenance(tmp_path):
    provenance = {'version': 'test'}
    value = {'layer': 40, 'dose': .25, 'basis': [[1., 0.]], 'rescue_basis': [[0., 1.]]}
    frozen = h.freeze_selection(tmp_path, provenance, value)
    assert h.load_frozen(tmp_path, provenance) == frozen
    with pytest.raises(ValueError, match='frozen'):
        h.freeze_selection(tmp_path, provenance, {**value, 'dose': .5})
    with pytest.raises(ValueError, match='provenance'):
        h.load_frozen(tmp_path, {'version': 'changed'})


def test_primary_summary_keeps_missing_and_malformed_denominators():
    rows = h.make_episodes()[:32]
    results = {}
    for x in rows:
        for condition in h.CONDITIONS:
            results[(x.key, condition)] = h.score_output('{"own":"red","incoming":"red"}',
                own=x.own, incoming=x.incoming, n_generated=10, cap=48)
    results[(rows[0].key, 'delete')] = h.score_output(None, own='red', incoming='red',
                                                     n_generated=None, cap=48)
    summary = h.summarize(rows, results, bootstrap_samples=100)
    deletion = summary['conditions']['delete']
    assert deletion['n_total'] == 32 and deletion['n_missing'] == 1
    assert deletion['incoming_bounds'][1] - deletion['incoming_bounds'][0] == pytest.approx(1/32)
    assert summary['contrasts']['targeted_minus_random']['n_missing'] == 1
    assert len(summary['joint_outcomes']['delete']) >= 1


def test_condition_edits_are_local_ordered_and_wrong_donor_changes_only_coordinates():
    selection = {'layer': 40, 'rescue_layer': 42, 'dose': .25,
                 'basis': [[1., 0., 0.]], 'rescue_basis': [[0., 1., 0.]], 'random_seed': 20261107}
    clean = {'coefficients': [2.], 'sha256': 'clean'}
    wrong = {'coefficients': [-2.], 'sha256': 'wrong'}
    edits = h.condition_edits('wrong_donor', selection, clean, wrong)
    assert [e.kind for e in edits] == ['fixed_delete', 'fixed_patch']
    assert [e.layers for e in edits] == [[40], [42]]
    assert edits[0].deletion_fraction == .25
    assert edits[1].donor_coefficients == [-2.]
    assert edits[0].positions.span_last and not edits[0].positions.include_generated
    random = h.condition_edits('random_restore', selection, clean, wrong)
    assert random[1].kind == 'fixed_random_restore' and random[1].donor_coefficients == [2.]


def test_geometry_gate_requires_realized_matched_edits_and_rejects_zero_or_bad_match():
    event = {'kind': 'fixed_delete', 'layer': 40, 'position': 9, 'edit_norm': 2.,
             'h_dot_delta': -4., 'hidden_norm_before': 10., 'hidden_norm_after': 9.8,
             'desired_edit_norm': 2., 'desired_h_dot_delta': -4.,
             'desired_hidden_norm_after': 9.8, 'quantization_error_norm': 0.}
    assert h.geometry_pair([event], [{**event, 'kind': 'fixed_random_matched'}], .05)['passed']
    assert not h.geometry_pair([event], [{**event, 'edit_norm': 0.}], .05)['passed']
    assert not h.geometry_pair([event], [{**event, 'hidden_norm_after': 10.}], .05)['passed']


class SyntheticEndpoint:
    """Known factorial coordinates and deterministic localized edit receipts."""
    name = 'synthetic'

    def __init__(self, *, signal=True):
        self.signal = signal
        self.requests = []

    def info(self):
        return {'hf_id': h.DEFAULT_CONFIG['model'], 'model': 'synthetic', 'n_layers': 64, 'd_model': 4}

    def generate(self, req):
        import re
        from jpeers.schemas import GenerateResponse
        from jpeers.trace import encode_arrays
        self.requests.append(req.model_dump())
        text = req.messages[-1]['content']
        own = re.search(r'own label = (red|blue)', text).group(1)
        match = re.search(r'incoming label = (red|blue)', text)
        incoming = match.group(1) if match else None
        expected = None if 'own-only extraction' in text else incoming
        prediction = expected
        edits = []
        for i, iv in enumerate(req.interventions):
            if iv.kind == 'fixed_delete':
                prediction = 'blue' if incoming == 'red' else 'red'
            elif iv.kind == 'fixed_patch':
                prediction = 'red' if iv.donor_coefficients[0] > 0 else 'blue'
            norm = iv.deletion_fraction if i == 0 else .5
            dot = -.25 if i == 0 else .2
            before, after = (2., 1.9) if i == 0 else (1.9, 2.)
            edits.append({'intervention_index': i, 'kind': iv.kind, 'layer': iv.layers[0],
                'position': 9, 'site': 'block_output', 'edit_norm': norm, 'h_dot_delta': dot,
                'hidden_norm_before': before, 'hidden_norm_after': after,
                'desired_edit_norm': norm, 'desired_h_dot_delta': dot,
                'desired_hidden_norm_after': after, 'quantization_error_norm': 0.})
        layers = req.record.layers
        arrays = {'positions': np.array([9]), 'topk_ids': np.zeros((1, len(layers), 1), int),
                  'topk_logits': np.zeros((1, len(layers), 1)),
                  'watch_ranks': np.zeros((1, len(layers), 2), int)}
        metadata = {'residual_site': 'block_output', 'when': 'post',
                    'direction_token_rule': 'bare', 'label_tokenization': {
                        'red': {'bare_ids': [5], 'primary_id': 15, 'direction_id': 5},
                        'blue': {'bare_ids': [6], 'primary_id': 16, 'direction_id': 6}}}
        if req.record.capture_residuals:
            state = [((1 if incoming == 'red' else -1) if self.signal else 0),
                     1 if own == 'red' else -1, 0., 1.]
            arrays['residuals'] = np.tile(state, (1, len(layers), 1)).astype(float)
            directions = np.tile(np.array([[1., 1., 0., 0.], [1., -1., 0., 0.]]) / np.sqrt(2),
                                 (len(layers), 1, 1))
            arrays.update(directions_jacobian=directions, directions_logit=directions,
                          direction_token_ids=np.array([5, 6]))
        return GenerateResponse(model='synthetic', text=json.dumps({'own': own, 'incoming': prediction}),
            prompt_len=12, n_generated=10, tokens=['x'] * 9 + ['.', 'a', 'b'],
            spans=[{'label': k, 'start': a, 'end': b} for k, a, b in
                   [('causal_site', 9, 10), ('neutral_suffix', 6, 10),
                    ('own_payload', 1, 3), ('incoming_payload', 3, 6)]],
            layers=layers, watch=['red', 'blue'], watch_ids=[[15, 5], [16, 6]], vocab={5: 'red', 6: 'blue'},
            trace_encoding='json', trace=encode_arrays(arrays, 'json'),
            interventions_applied=len(edits), timings={}, capture_metadata=metadata, causal_edits=edits)


def test_fenced_score_is_versioned_in_provenance_and_keeps_raw_response(tmp_path):
    endpoint = SyntheticEndpoint()
    generate = endpoint.generate
    def fenced(req):
        response = generate(req)
        response.text = f'```json\n{response.text}\n```'
        return response
    endpoint.generate = fenced
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    receipt = runner.query(h.make_controls()[0], 'baseline', layers=[40, 42], tag='format')
    score = receipt['score']
    assert score['joint_correct'] and score['output_format'] == 'fenced_json'
    assert score['raw_strict_format_valid'] is False
    assert runner.provenance['json_scoring_version'] == score['json_scoring_version']
    assert runner.provenance['output_format_protocol'] == (
        'One entire JSON object, bare or in one whole triple-backtick json/unlabeled fence; '
        'exact own/incoming keys and red/blue labels (incoming may be null); '
        'duplicate keys, prose, multiple answers, malformed JSON, and cap hits fail. '
        'raw_strict_format_valid reports bare-object schema validity before the cap check.')
    saved_provenance = json.loads((runner.root / 'provenance.json').read_text())
    assert saved_provenance['json_scoring_version'] == 'json-object-v2'
    turn = json.loads((runner.root / receipt['turn_directory'] / 'turns.jsonl').read_text())
    assert turn['text'] == receipt['response']['text'] == '```json\n{"own": "red", "incoming": null}\n```'
    assert turn['extra']['score'] == score
    old_provenance = dict(saved_provenance)
    old_provenance.pop('json_scoring_version')
    old_provenance.pop('output_format_protocol')
    h.atomic_json(runner.root / 'provenance.json', old_provenance)
    with pytest.raises(ValueError, match='provenance mismatch'):
        h.HandoffRunner(endpoint, runner.root, h.DEFAULT_CONFIG, resume=True)


def test_real_request_and_turn_record_preserve_edits_and_capture_ids(tmp_path):
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    row = h.make_episodes()[0]
    selection = {'layer': 40, 'rescue_layer': 42, 'dose': .25,
                 'basis': [[1., 0., 0., 0.]], 'rescue_basis': [[1., 0., 0., 0.]],
                 'random_seed': 20261107}
    baseline = runner.query(row, 'baseline', layers=[40, 42], tag='test')
    assert h.validate_capture_site(runner.trace(baseline))['direction_token_ids'] == [5, 6]
    receipt = runner.query(row, 'delete', layers=[40, 42], selection=selection, tag='test')
    turn = json.loads((runner.root / receipt['turn_directory'] / 'turns.jsonl').read_text())
    assert turn['interventions'] == endpoint.requests[-1]['interventions']
    assert turn['extra']['causal_edits'] == receipt['response']['causal_edits']
    assert turn['extra']['request_sha256'] == receipt['request_sha256']
    before = len(endpoint.requests)
    assert runner.query(row, 'delete', layers=[40, 42], selection=selection, tag='test') == receipt
    assert len(endpoint.requests) == before


def test_synthetic_discovery_failure_never_opens_heldout(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'DEVELOPMENT_TEMPLATES', h.DEVELOPMENT_TEMPLATES[:2])
    monkeypatch.setattr(h, 'HELDOUT_TEMPLATES', h.HELDOUT_TEMPLATES[:2])
    endpoint = SyntheticEndpoint(signal=False)
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    outcome = runner.run('all')
    assert outcome['status'] == 'discovery_failed'
    assert not (runner.root / 'frozen.json').exists()
    assert not any('heldout' in p.name for p in (runner.root / 'receipts').iterdir())
    assert all(not any(t in q['messages'][-1]['content'] for t in h.HELDOUT_TEMPLATES)
               for q in endpoint.requests)
    assert json.loads((runner.root / 'status.json').read_text())['status'] == 'discovery_failed'


def test_synthetic_stages_freeze_before_heldout_and_resume_without_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'DEVELOPMENT_TEMPLATES', h.DEVELOPMENT_TEMPLATES[:2])
    monkeypatch.setattr(h, 'HELDOUT_TEMPLATES', h.HELDOUT_TEMPLATES[:2])
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    frozen = runner.run('discovery')
    assert frozen['selection']['dose'] == .25
    assert (runner.root / 'frozen.json').exists()
    assert all(not any(t in q['messages'][-1]['content'] for t in h.HELDOUT_TEMPLATES)
               for q in endpoint.requests)
    requests_after_discovery = len(endpoint.requests)
    summary = runner.run('heldout')
    assert summary['n_templates'] == 2 and summary['n_contexts'] == 64
    assert len(endpoint.requests) - requests_after_discovery == 64 * 6
    assert summary['contrasts']['targeted_minus_random']['diff'] == -1
    assert summary['contrasts']['clean_rescue_minus_delete']['diff'] == 1
    assert json.loads((runner.root / 'status.json').read_text())['status'] == 'completed'
    before = len(endpoint.requests)
    resumed = h.HandoffRunner(endpoint, runner.root, h.DEFAULT_CONFIG, resume=True)
    assert resumed.run('all') == summary
    assert len(endpoint.requests) == before
    with pytest.raises(ValueError, match='resume'):
        h.HandoffRunner(endpoint, runner.root, h.DEFAULT_CONFIG)


def test_own_only_control_has_no_system_instruction_conflict():
    row = next(x for x in h.make_controls() if x.control == 'own_only')
    system = row.messages()[0]['content']
    user = row.messages()[1]['content']
    assert 'own-only extraction' in system and 'incoming to null' in system
    assert 'own-only extraction' in user and 'incoming to null' in user
    assert row.incoming is not None and row.expected_incoming is None


def test_frozen_discovery_receipt_is_bound_to_final_report(tmp_path):
    report = {'status': 'discovery_passed', 'gates': {'baseline': 1.}}
    h.atomic_json(tmp_path / 'discovery.json', report)
    selection = {'discovery_summary_sha256': h.digest(report)}
    h.freeze_selection(tmp_path, {'version': 'test'}, selection)
    assert h.load_frozen(tmp_path, {'version': 'test'})['selection'] == selection
    h.atomic_json(tmp_path / 'discovery.json', {**report, 'status': 'discovery_running'})
    with pytest.raises(ValueError, match='discovery report'):
        h.load_frozen(tmp_path, {'version': 'test'})


def test_heldout_partial_baseline_failure_retains_known_receipts_and_unknown_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'DEVELOPMENT_TEMPLATES', h.DEVELOPMENT_TEMPLATES[:2])
    monkeypatch.setattr(h, 'HELDOUT_TEMPLATES', h.HELDOUT_TEMPLATES[:2])
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    runner.run('discovery')
    generate = endpoint.generate
    calls = [0]
    def interrupted(req):
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError('synthetic interruption')
        return generate(req)
    endpoint.generate = interrupted
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        runner.run('heldout')
    summary = json.loads((runner.root / 'summary.json').read_text())
    assert summary['conditions']['baseline']['n_missing'] == 61
    assert summary['conditions']['delete']['n_missing'] == 64
    endpoint.generate = generate
    resumed = h.HandoffRunner(endpoint, runner.root, h.DEFAULT_CONFIG, resume=True)
    assert resumed.run('heldout')['conditions']['baseline']['n_missing'] == 0


def test_durable_response_receipt_precedes_turn_projection_crash(tmp_path, monkeypatch):
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    row = h.make_episodes()[0]
    write = h.RunWriter.write
    def interrupted(*args, **kwargs):
        raise RuntimeError('turn projection interrupted')
    monkeypatch.setattr(h.RunWriter, 'write', interrupted)
    with pytest.raises(RuntimeError, match='projection interrupted'):
        runner.query(row, 'baseline', layers=[40, 42], tag='crash')
    assert len(endpoint.requests) == 1
    monkeypatch.setattr(h.RunWriter, 'write', write)
    receipt = runner.query(row, 'baseline', layers=[40, 42], tag='crash')
    assert len(endpoint.requests) == 1
    assert (runner.root / receipt['turn_directory'] / 'turns.jsonl').exists()


def test_output_lock_rejects_simultaneous_writer(tmp_path):
    with h.output_lock(tmp_path / 'run'):
        with pytest.raises(RuntimeError, match='already running'):
            with h.output_lock(tmp_path / 'run'):
                pass
    with h.output_lock(tmp_path / 'run'):
        pass


def test_source_provenance_includes_statistic_and_serialization_implementations():
    sources = h.make_provenance(h.DEFAULT_CONFIG, {})['source_sha256']
    assert 'src/jpeers/followup/common.py' in sources
    assert 'src/jpeers/harness.py' in sources


def test_rejected_capture_receipt_is_preserved_without_regeneration(tmp_path):
    endpoint = SyntheticEndpoint()
    generate = endpoint.generate
    def bad_capture(req):
        response = generate(req)
        response.capture_metadata = {}
        return response
    endpoint.generate = bad_capture
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    row = h.make_episodes()[0]
    for _ in range(2):
        with pytest.raises(ValueError, match='direction IDs'):
            runner.query(row, 'baseline', layers=[40, 42], tag='bad')
    assert len(endpoint.requests) == 1
    assert len(list((runner.root / 'receipts').glob('*.json'))) == 1


def test_progress_counts_durable_receipts_and_resume_cache_hits(tmp_path):
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    row = h.make_episodes()[0]
    runner.query(row, 'baseline', layers=[40, 42], tag='progress')
    first = json.loads((runner.root / 'progress.json').read_text())
    assert first['receipt_count'] == 1 and first['session_new_calls'] == 1
    runner.query(row, 'baseline', layers=[40, 42], tag='progress')
    second = json.loads((runner.root / 'progress.json').read_text())
    assert second['receipt_count'] == 1 and second['session_cache_hits'] == 1
    assert second['condition'] == 'baseline' and second['template_id'] == row.template_id
    resumed = h.HandoffRunner(endpoint, runner.root, h.DEFAULT_CONFIG, resume=True)
    resumed.query(row, 'baseline', layers=[40, 42], tag='progress')
    last = json.loads((runner.root / 'progress.json').read_text())
    assert last['receipt_count'] == 1 and last['session_new_calls'] == 0
    assert last['session_cache_hits'] == 1


def test_partial_turn_json_rebuilds_from_receipt_without_new_model_call(tmp_path, monkeypatch):
    endpoint = SyntheticEndpoint()
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    row = h.make_episodes()[0]
    write = h.RunWriter.write
    def interrupted(writer, *args, **kwargs):
        (writer.dir / 'turns.jsonl').write_text('{"index":')
        raise RuntimeError('partial turn projection')
    monkeypatch.setattr(h.RunWriter, 'write', interrupted)
    with pytest.raises(RuntimeError, match='partial turn'):
        runner.query(row, 'baseline', layers=[40, 42], tag='partial')
    monkeypatch.setattr(h.RunWriter, 'write', write)
    receipt = runner.query(row, 'baseline', layers=[40, 42], tag='partial')
    assert len(endpoint.requests) == 1
    directory = runner.root / receipt['turn_directory']
    assert json.loads((directory / 'turns.jsonl').read_text())['text'] == receipt['response']['text']
    assert len(list(directory.glob('turns.incomplete-*.jsonl'))) == 1


def test_nondeterministic_clean_capture_stops_discovery_before_interventions(tmp_path):
    endpoint = SyntheticEndpoint()
    generate = endpoint.generate
    def unstable(req):
        response = generate(req)
        if len(endpoint.requests) == len(h.make_controls()) + 1:
            response.trace['residuals'][0][0][0] += .1
        return response
    endpoint.generate = unstable
    runner = h.HandoffRunner(endpoint, tmp_path / 'run', h.DEFAULT_CONFIG)
    result = runner.run('all')
    assert result['status'] == 'discovery_failed'
    assert 'replay' in result['reason']
    assert not (runner.root / 'frozen.json').exists()
    assert all(not req['interventions'] for req in endpoint.requests)
