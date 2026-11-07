"""A bounded, staged incoming-information mediation assay.

Discovery uses development templates only. No held-out generation is requested
until a successful setting, task inventory, and analysis are frozen on disk.
The assay identifies incoming content; source specificity is tested separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from jpeers.harness import RunWriter, TurnRecord
from jpeers.schemas import GenerateRequest, InterventionSpec, PositionSelector, RecordSpec, SpanQuery
from jpeers.trace import JSpaceTrace
from .common import atomic_json, paired_stats

PROTOCOL = 'incoming-mediation-v1'
SUFFIX = 'End of notes.'
LABELS = ('red', 'blue')
NAMES = (('Avery', 'Blake'), ('Casey', 'Drew'))
CONDITIONS = ('baseline', 'delete', 'random_delete', 'clean_rescue', 'wrong_donor', 'random_restore')
# Authored for this assay, 2026-11-07. No original experiment prompts are reused.
DEVELOPMENT_TEMPLATES = (
    'Complete the two-field transfer ledger using the notes below.',
    'Copy the two independent labels from these records into the requested fields.',
    'Prepare a compact record of the private label and the incoming label.',
    'Read this handoff and extract each label without changing it.',
    'Fill the recipient record from the two labeled entries below.',
    'Check the private entry and the incoming entry, then report their labels.',
    'Transcribe the independent labels into their corresponding JSON fields.',
    'Consolidate these notes into a two-field label record.',
)
HELDOUT_TEMPLATES = (
    'Resolve the two requested entries by reading the attached notes.',
    'Populate a receipt with the recipient label and the supplied incoming label.',
    'Use the notes to complete a concise two-entry register.',
    'Record what each note states in the matching output field.',
    'Retrieve both labels and return the requested JSON record.',
    'Build the final label summary from this pair of notes.',
    'Consult the records and preserve the distinction between own and incoming.',
    'Extract the private value and the received value for this exchange.',
    'Complete the label intake form from the information that follows.',
    'Translate these records into the specified two-field response.',
    'Read the transfer record and copy the label for each category.',
    'Capture the recipient entry and the incoming entry in one response.',
    'Assemble the requested label pair using the written notes.',
    'Report the two independently assigned labels stated in this packet.',
    'Audit the entries by copying each stated label to the proper field.',
    'Prepare the label receipt after reading the records below.',
    'Summarize this packet as an own label and an incoming label.',
    'Take the private label from its record and the incoming label from its note.',
    'Return the two label entries associated with this transfer.',
    'Produce the final two-field record for these independent notes.',
    'Carry the labels from the source notes into the output record.',
    'Look up the requested labels in this exchange packet.',
    'Use this packet to fill both slots of the transfer summary.',
    'Compile a minimal JSON receipt containing the recorded labels.',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def output_lock(root):
    """One CLI writer per output tree, including initialization and resumption."""
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with open(root.parent / f'.{root.name}.causal-handoff.lock', 'a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A handoff worker is already running for this output') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class Episode:
    split: str
    template_id: str
    instruction: str
    own: str
    incoming: str | None
    source: str
    names: tuple[str, str]
    order: int
    control: str = ''

    @property
    def key(self):
        return digest(asdict(self))[:24]

    @property
    def expected_incoming(self):
        return None if self.control == 'own_only' else self.incoming

    def own_note(self):
        return f'Private record for {self.names[0]}: own label = {self.own}.'

    def incoming_note(self):
        if self.incoming is None:
            return 'No incoming note is available.'
        return f'Incoming note: incoming label = {self.incoming}.'

    def messages(self):
        source = (f'Named peer {self.names[1]} supplies the incoming note.' if self.source == 'peer'
                  else 'An anonymous document supplies the incoming note.')
        instruction = self.instruction
        if self.control == 'own_only':
            instruction += ' This is an own-only extraction: ignore the incoming label and set incoming to null.'
        notes = [self.own_note(), self.incoming_note()]
        if self.order:
            notes.reverse()
        content = (f'{instruction}\nRecipient: {self.names[0]}. Counterpart label: {self.names[1]}.\n'
                   f'{source}\n' + '\n'.join(notes) + '\n' + SUFFIX)
        return [{'role': 'system', 'content':
                 'Return exactly one JSON object with exactly the keys "own" and "incoming". '
                 'Use only the label strings "red" or "blue". When incoming information is '
                 'unavailable, use JSON null for incoming. In an own-only extraction, set incoming to null '
                 'even when an incoming note is present. Otherwise copy each available label exactly. '
                 'Never infer one label from the other. No Markdown or explanation.'},
                {'role': 'user', 'content': content}]


def make_episodes():
    rows = []
    for split, templates in [('development', DEVELOPMENT_TEMPLATES), ('heldout', HELDOUT_TEMPLATES)]:
        for ti, instruction in enumerate(templates):
            for own, incoming, source, names, order in itertools.product(
                    LABELS, LABELS, ('peer', 'document'), NAMES, (0, 1)):
                rows.append(Episode(split, f'{split}-{ti:02d}', instruction, own, incoming,
                                    source, names, order))
    return rows


def make_controls():
    # Two labels × two framings × two name pairs × two orders for each gate.
    base = make_episodes()[0]
    return [replace(base, template_id=f'control-{kind}', own=own,
                    incoming=None if kind == 'no_handoff' else LABELS[1 - LABELS.index(own)],
                    source=source, names=names, order=order, control=kind)
            for kind, own, source, names, order in itertools.product(
                ('no_handoff', 'own_only'), LABELS, ('peer', 'document'), NAMES, (0, 1))]


def opposite_donor(episode, rows):
    matches = [x for x in rows if (x.template_id, x.source, x.names, x.order, x.own, x.control) ==
               (episode.template_id, episode.source, episode.names, episode.order,
                episode.own, episode.control) and x.incoming != episode.incoming]
    if len(matches) != 1:
        raise ValueError('Counterfactual donor must be unique and differ only in incoming label')
    return matches[0]


def score_output(text, *, own, incoming, n_generated, cap):
    result = {'known': text is not None, 'format_valid': None, 'own_correct': None,
              'incoming_correct': None, 'joint_correct': None, 'cap_hit': False,
              'predicted_own': None, 'predicted_incoming': None}
    if text is None:
        return result
    result.update(format_valid=False, own_correct=False, incoming_correct=False, joint_correct=False,
                  cap_hit=n_generated is None or n_generated >= cap)
    try:
        def unique(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError('duplicate key')
                obj[k] = v
            return obj
        data = json.loads(text.strip(), object_pairs_hook=unique)
        valid = (isinstance(data, dict) and set(data) == {'own', 'incoming'} and
                 data['own'] in LABELS and data['incoming'] in (*LABELS, None))
        if not valid:
            return result
    except (ValueError, TypeError):
        return result
    result.update(predicted_own=data['own'], predicted_incoming=data['incoming'])
    if result['cap_hit']:
        return result
    result.update(format_valid=True, own_correct=data['own'] == own,
                  incoming_correct=data['incoming'] == incoming)
    result['joint_correct'] = result['own_correct'] and result['incoming_correct']
    return result


def _signed(values):
    return np.array([1. if x == LABELS[0] else -1. for x in values])


def _nuisance(rows):
    return np.array([[1., 1. if r.own == LABELS[0] else -1.,
                      1. if r.source == 'peer' else -1.,
                      1. if r.names == NAMES[0] else -1., r.order * 2. - 1.] for r in rows])


def _ridge(a, b, ridge):
    return np.linalg.solve(a.T @ a + np.eye(a.shape[1]) * ridge, a.T @ b)


def _row_span(rows):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError('Direction span must be a finite row matrix')
    _, s, vh = np.linalg.svd(rows, full_matrices=False)
    if not len(s) or s[0] <= 1e-12:
        return np.empty((0, rows.shape[1]))
    return vh[s > s[0] * 1e-7][:2]


def _learn(states, rows, span, ridge):
    z = _nuisance(rows)
    y = _signed([r.incoming for r in rows])
    # Low dimensional regression within the declared lens span; full contrast uses
    # the same rank-one factorial construction, not a more flexible classifier.
    coefficients = _ridge(np.column_stack([z, y]), states, ridge)
    contrast = coefficients[-1]
    if span is not None:
        contrast = (contrast @ span.T) @ span
    norm = np.linalg.norm(contrast)
    basis = contrast / norm if norm > 1e-9 else np.zeros(states.shape[1])
    return basis, coefficients[:-1]


def _scalar_prediction(train_x, train_y, test_x, ridge):
    fit = _ridge(np.column_stack([np.ones(len(train_x)), train_x]), train_y, ridge)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ fit >= 0


def fit_component(rows, states, directions=None, *, ridge=1e-6):
    """Template-held-out incoming decoder and raw-coordinate leakage checks.

    Every learned contrast and nuisance adjustment is refit inside each fold.
    Own/nuisance leakage uses the raw incoming coordinate without regressing out
    the true validation own value. Own decoding is fitted independently.
    """
    states = np.asarray(states, dtype=float)
    if len(rows) != len(states) or not np.isfinite(states).all():
        raise ValueError('Residual inventory or finiteness mismatch')
    span = None if directions is None else _row_span(directions)
    incoming_y = _signed([r.incoming for r in rows])
    own_y = _signed([r.own for r in rows])
    z = _nuisance(rows)
    incoming_predictions, own_predictions, own_leakage = [], [], []
    nuisance_correct = [[], [], []]
    folds = []
    templates = sorted({r.template_id for r in rows})
    if len(templates) < 2:
        raise ValueError('At least two development templates are needed for CV')
    for template in templates:
        train = np.array([r.template_id != template for r in rows])
        test = ~train
        train_rows = [r for r, selected in zip(rows, train) if selected]
        b, nuisance_fit = _learn(states[train], train_rows, span, ridge)
        raw_train, raw_test = states[train] @ b, states[test] @ b
        incoming_predictions.extend((_scalar_prediction(
            (states[train] - z[train] @ nuisance_fit) @ b, incoming_y[train],
            (states[test] - z[test] @ nuisance_fit) @ b, ridge) == (incoming_y[test] > 0)).tolist())
        own_leakage.extend((_scalar_prediction(raw_train, own_y[train], raw_test, ridge) ==
                            (own_y[test] > 0)).tolist())
        for ni in range(3):
            nuisance_correct[ni].extend((_scalar_prediction(raw_train, z[train, ni + 2],
                raw_test, ridge) == (z[test, ni + 2] > 0)).tolist())
        # Separate own contrast, controlling incoming and nuisance on the train fold.
        own_design = np.column_stack([z[train][:, [0, 2, 3, 4]], incoming_y[train], own_y[train]])
        own_b = _ridge(own_design, states[train], ridge)[-1]
        if span is not None:
            own_b = (own_b @ span.T) @ span
        own_predictions.extend((_scalar_prediction(states[train] @ own_b, own_y[train],
            states[test] @ own_b, ridge) == (own_y[test] > 0)).tolist())
        folds.append({'train_templates': [t for t in templates if t != template],
                      'test_templates': [template], 'n_test': int(test.sum()),
                      'basis_sha256': digest(b.tolist())})
    basis, _ = _learn(states, rows, span, ridge)
    return {'basis': [basis.tolist()] if np.linalg.norm(basis) > .5 else [],
            'rank': int(np.linalg.norm(basis) > .5),
            'incoming_cv': float(np.mean(incoming_predictions)),
            'own_cv': float(np.mean(own_predictions)),
            'incoming_basis_own_cv': float(np.mean(own_leakage)),
            'incoming_basis_nuisance_cv': float(max(np.mean(x) for x in nuisance_correct)),
            'span_rank': None if span is None else len(span), 'folds': folds,
            'capacity': 'one conditional factorial contrast; template-held-out folds'}


def validate_capture_site(trace):
    spans = trace.spans
    site = spans.get('causal_site', [])
    suffix = spans.get('neutral_suffix', [])
    if len(site) != 1 or site[0][1] - site[0][0] != 1:
        raise ValueError('Causal site must be one single prompt token')
    if len(suffix) != 1 or site[0][1] != suffix[0][1]:
        raise ValueError('Causal site must end the unique neutral suffix')
    position = site[0][0]
    if position >= trace.prompt_len or list(trace.positions) != [position]:
        raise ValueError('Capture must contain exactly the selected prompt site')
    if trace.tokens[position].strip() != '.':
        raise ValueError('Causal suffix endpoint must be the bare punctuation token')
    for label in ('own_payload', 'incoming_payload'):
        if len(spans.get(label, [])) != 1 or spans[label][0][1] > suffix[0][0]:
            raise ValueError('Neutral suffix must follow both payload notes')
    if trace.watch != list(LABELS) or any(len(ids) < 1 for ids in trace.watch_ids):
        raise ValueError('Balanced label watch IDs unavailable')
    metadata = getattr(trace, 'capture_metadata', {})
    captured = getattr(trace, 'direction_token_ids', None)
    if captured is None:
        captured = metadata.get('direction_watch_ids')
    expected_ids = [metadata.get('label_tokenization', {}).get(word, {}).get('direction_id', ids[0])
                    for word, ids in zip(trace.watch, trace.watch_ids)]
    if captured is None or list(captured) != expected_ids:
        raise ValueError('Captured direction IDs must match primary whole-token label IDs')
    if len(set(captured)) != 2:
        raise ValueError('Balanced labels must have different primary token IDs')
    return {'position': int(position), 'direction_token_ids': list(map(int, captured)),
            'suffix_span': list(suffix[0])}


class ReceiptStore:
    """Atomic exact-request receipts; no permissive cross-run cache reuse."""
    def __init__(self, root, provenance):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root / 'provenance.json'
        if identity.exists():
            if json.loads(identity.read_text()) != provenance:
                raise ValueError('Run provenance mismatch; use a fresh output directory')
        else:
            atomic_json(identity, provenance)
        (self.root / 'receipts').mkdir(exist_ok=True)

    def path(self, key):
        return self.root / 'receipts' / f'{digest(key)}.json'

    def load(self, key, request):
        path = self.path(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        integrity = value.pop('integrity_sha256', None)
        if integrity != digest(value):
            raise ValueError('Receipt integrity mismatch')
        if value['request_sha256'] != digest(request) or value['request'] != request:
            raise ValueError('Cached request mismatch')
        return value

    def save(self, key, request, response, score, **extra):
        value = {'request': request, 'request_sha256': digest(request), 'response': response,
                 'score': score, **extra}
        value['integrity_sha256'] = digest(value)
        atomic_json(self.path(key), value)


def freeze_selection(root, provenance, selection):
    root = Path(root)
    path = root / 'frozen.json'
    value = {'provenance_sha256': digest(provenance), 'selection': selection,
             'analysis': {'unit': 'template', 'sign_flips': 50000, 'sign_flip_seed': 0,
                          'bootstrap_samples': 100000, 'bootstrap_seed': 20261107,
                          'holm_family': ['targeted_minus_random', 'clean_rescue_minus_delete',
                                          'source_interaction'],
                          'specificity_margin': .05, 'primary': 'all cells; malformed and cap fail',
                          'missing': 'unknown retained with worst-case bounds',
                          'claim_scope': 'exploratory incoming-information mediation'}}
    value['freeze_sha256'] = digest(value)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Cannot change frozen selection in an existing run')
    atomic_json(path, value)
    atomic_json(root / 'status.json', {'status': 'discovery_passed',
                                      'freeze_sha256': value['freeze_sha256']})
    return value


def load_frozen(root, provenance):
    root = Path(root)
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text()).get('status') if status_path.exists() else None
    if status not in ('discovery_passed', 'heldout_running', 'completed', 'failed'):
        raise ValueError(f'Held-out stage requires successful discovery; status={status}')
    value = json.loads((root / 'frozen.json').read_text())
    check = dict(value)
    claimed = check.pop('freeze_sha256', None)
    if claimed != digest(check):
        raise ValueError('Frozen selection integrity mismatch')
    if value['provenance_sha256'] != digest(provenance):
        raise ValueError('Frozen provenance mismatch')
    selection = value['selection']
    report_hash = selection.get('discovery_summary_sha256')
    if report_hash:
        report_path = root / 'discovery.json'
        if not report_path.exists() or digest(json.loads(report_path.read_text())) != report_hash:
            raise ValueError('Frozen discovery report integrity mismatch')
    for basis_key in ('basis', 'rescue_basis'):
        claimed_basis = selection.get(f'{basis_key}_sha256')
        if claimed_basis and claimed_basis != digest(selection.get(basis_key)):
            raise ValueError('Frozen basis integrity mismatch')
    return value


def _site_selector():
    return PositionSelector(scope='spans', labels=['neutral_suffix'], span_last=True,
                            include_generated=False, include_last=False)


def condition_edits(condition, selection, clean=None, wrong=None):
    if condition == 'baseline':
        return []
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown causal condition {condition}')
    deletion = InterventionSpec(kind='fixed_random_matched' if condition == 'random_delete'
                               else 'fixed_delete', layers=[selection['layer']],
                               positions=_site_selector(), basis=selection['basis'],
                               deletion_fraction=selection['dose'], seed=selection['random_seed'])
    if condition in ('delete', 'random_delete'):
        return [deletion]
    donor = wrong if condition == 'wrong_donor' else clean
    if not donor:
        raise ValueError('A downstream patch requires an independently captured clean donor')
    patch = InterventionSpec(kind='fixed_random_restore' if condition == 'random_restore'
                            else 'fixed_patch', layers=[selection['rescue_layer']],
                            positions=_site_selector(), basis=selection['rescue_basis'],
                            donor_coefficients=donor['coefficients'], seed=selection['random_seed'])
    return [deletion, patch]


def geometry_pair(target, random, tolerance):
    """Check actual, quantized geometry, not only idealized FP32 construction."""
    problems = []
    if len(target) != 1 or len(random) != 1:
        return {'passed': False, 'problems': ['expected one matched event per condition']}
    a, b = target[0], random[0]
    needed = ('layer', 'position', 'edit_norm', 'h_dot_delta', 'hidden_norm_before',
              'hidden_norm_after', 'desired_edit_norm', 'desired_h_dot_delta',
              'desired_hidden_norm_after', 'quantization_error_norm')
    if any(k not in e or not np.isfinite(e[k]) for e in (a, b) for k in needed):
        return {'passed': False, 'problems': ['missing or nonfinite geometry metadata']}
    if (a['layer'], a['position']) != (b['layer'], b['position']):
        problems.append('site/layer mismatch')
    norm_scale = max(a['desired_edit_norm'], 1e-12)
    dot_scale = max(norm_scale * a['hidden_norm_before'], 1e-12)
    if a['edit_norm'] <= 0 or b['edit_norm'] <= 0:
        problems.append('zero realized edit')
    discrepancies = {
        'edit_norm_relative': abs(a['edit_norm'] - b['edit_norm']) / norm_scale,
        'dot_relative': abs(a['h_dot_delta'] - b['h_dot_delta']) / dot_scale,
        'post_norm_relative_to_edit': abs(a['hidden_norm_after'] - b['hidden_norm_after']) / norm_scale,
        'pre_state_norm_relative_to_edit': abs(a['hidden_norm_before'] - b['hidden_norm_before']) / norm_scale,
    }
    for ei, e in enumerate((a, b)):
        discrepancies[f'{ei}_intended_norm_error'] = abs(e['edit_norm'] - e['desired_edit_norm']) / norm_scale
        discrepancies[f'{ei}_intended_dot_error'] = abs(e['h_dot_delta'] - e['desired_h_dot_delta']) / dot_scale
        discrepancies[f'{ei}_quantization_error'] = e['quantization_error_norm'] / norm_scale
    problems.extend(k for k, v in discrepancies.items() if v > tolerance)
    return {'passed': not problems, 'problems': problems, 'tolerance': tolerance,
            'discrepancies': discrepancies, 'target': a, 'random': b}


def _bootstrap(values, samples):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(20261107)
    estimates = np.mean(values[rng.integers(0, len(values), (samples, len(values)))], axis=1)
    return list(map(float, np.quantile(estimates, [.025, .975])))


def _metric_bounds(scores, metric):
    values = [s.get(metric) if s is not None else None for s in scores]
    known_sum = sum(float(x) for x in values if x is not None)
    missing = sum(x is None for x in values)
    return [(known_sum / len(values)), (known_sum + missing) / len(values)]


def summarize(rows, results, *, bootstrap_samples=100000):
    templates = sorted({x.template_id for x in rows})
    empty = score_output(None, own='red', incoming='red', n_generated=None, cap=1)
    def get(r, c):
        return results.get((r.key, c), empty)
    report = {'n_contexts': len(rows), 'n_templates': len(templates), 'conditions': {},
              'joint_outcomes': {}, 'contrasts': {}, 'specificity': {},
              'scope': 'exploratory; template clustering; no claim of peer specificity from equal source effects'}
    for c in CONDITIONS:
        scores = [get(r, c) for r in rows]
        report['conditions'][c] = {'n_total': len(rows),
            'n_missing': sum(not s['known'] for s in scores),
            'cap_hits': sum(bool(s['cap_hit']) for s in scores),
            **{f'{label}_bounds': _metric_bounds(scores, metric) for label, metric in
               [('incoming', 'incoming_correct'), ('own', 'own_correct'), ('format', 'format_valid')]}}
        joint = {}
        for s in scores:
            key = ('unknown' if not s['known'] else
                   f"incoming={int(s['incoming_correct'])},own={int(s['own_correct'])},format={int(s['format_valid'])}")
            joint[key] = joint.get(key, 0) + 1
        report['joint_outcomes'][c] = joint
    def contrast(a, b, metric='incoming_correct', interaction=False):
        effects, bounds = [], []
        for template in templates:
            chosen = [r for r in rows if r.template_id == template]
            def cell_group(group):
                av = [get(r, a).get(metric) for r in group]
                bv = [get(r, b).get(metric) for r in group]
                low = np.mean([(0. if x is None else float(x)) - (1. if y is None else float(y))
                               for x, y in zip(av, bv)])
                high = np.mean([(1. if x is None else float(x)) - (0. if y is None else float(y))
                                for x, y in zip(av, bv)])
                effect = None if any(x is None for x in av + bv) else float(low)
                return effect, [float(low), float(high)]
            if interaction:
                peer, pb = cell_group([r for r in chosen if r.source == 'peer'])
                document, db = cell_group([r for r in chosen if r.source == 'document'])
                effect = None if peer is None or document is None else peer - document
                bound = [pb[0] - db[1], pb[1] - db[0]]
            else:
                effect, bound = cell_group(chosen)
            effects.append(effect)
            bounds.append(bound)
        stat = paired_stats(effects, permutations=50000, seed=0)
        stat.update(ci95_cluster_bootstrap=_bootstrap(effects, bootstrap_samples),
                    effect_bounds=np.mean(bounds, axis=0).tolist(),
                    template_effects=dict(zip(templates, effects)))
        return stat
    report['contrasts']['targeted_minus_random'] = contrast('delete', 'random_delete')
    report['contrasts']['clean_rescue_minus_delete'] = contrast('clean_rescue', 'delete')
    report['contrasts']['source_interaction'] = contrast('delete', 'random_delete', interaction=True)
    family = report['contrasts']
    ordered = sorted([(k, v['p']) for k, v in family.items() if v['p'] is not None], key=lambda kv: kv[1])
    previous = 0.
    for i, (key, p) in enumerate(ordered):
        previous = max(previous, min(1., p * (len(family) - i)))
        family[key]['p_holm'] = previous
    for condition in CONDITIONS[1:]:
        report['specificity'][condition] = {}
        for metric in ('own_correct', 'format_valid'):
            stats = contrast(condition, 'baseline', metric)
            ci = stats['ci95_cluster_bootstrap']
            stats['noninferiority_margin'] = .05
            stats['exploratory_ci_above_margin'] = bool(ci is not None and ci[0] > -.05 and
                                                       stats['n_missing'] == 0)
            report['specificity'][condition][metric] = stats
    # A wrong donor must selectively move incoming toward that donor, not merely
    # make the recipient wrong or damage the own field/JSON.
    report['wrong_donor_content'] = {
        'n_total': len(rows),
        'toward_opposite_incoming': sum(bool(get(r, 'wrong_donor')['format_valid']) and
            get(r, 'wrong_donor')['predicted_incoming'] == LABELS[1 - LABELS.index(r.incoming)]
            for r in rows),
        'own_preserved': sum(get(r, 'wrong_donor')['own_correct'] is True for r in rows),
        'n_missing': sum(not get(r, 'wrong_donor')['known'] for r in rows)}
    return report


DEFAULT_CONFIG = {
    'protocol': PROTOCOL, 'model': 'Qwen/Qwen3-32B',
    'revision': '9216db5781bf21249d130ec9da846c4624c16137',
    'lens_repository': 'neuronpedia/jacobian-lens',
    'lens_revision': 'e166deb5ad540932100ef803c8c9bcc2a5453694',
    'lens_file': 'qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt',
    'deletion_layers': [40, 48, 52], 'rescue_offset': 2, 'doses': [.25, .5],
    'max_new_tokens': 48, 'seed': 20261107, 'random_seed': 20261107,
    'baseline_accuracy_min': .95, 'decoder_cv_min': .8, 'leakage_cv_max': .6,
    'incoming_effect_min': .05, 'specificity_margin': .05,
    'geometry_relative_tolerance': .05, 'ridge': 1e-6,
    'analysis': 'exploratory; no confirmatory power claim; 24 held-out template clusters',
}


def _validate_config(config):
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError('Config must contain exactly the declared bounded-assay fields')
    if config['protocol'] != PROTOCOL:
        raise ValueError('Unknown handoff protocol')
    if config['deletion_layers'] != [40, 48, 52] or config['rescue_offset'] != 2:
        raise ValueError('This runner is restricted to the declared 32B layer search')
    if config['doses'] != [.25, .5] or config['specificity_margin'] != .05:
        raise ValueError('This runner requires the frozen bounded dose search and margin')
    for key in ('baseline_accuracy_min', 'decoder_cv_min', 'leakage_cv_max',
                'incoming_effect_min', 'geometry_relative_tolerance'):
        if not 0 < config[key] <= 1:
            raise ValueError(f'Invalid gate {key}')
    if config['max_new_tokens'] < 16 or config['max_new_tokens'] > 64:
        raise ValueError('Short JSON completion cap must be between 16 and 64')


def make_provenance(config, endpoint_info):
    rows = make_episodes()
    controls = make_controls()
    source_root = Path(__file__).resolve().parents[3]
    sources = ['src/jpeers/followup/handoff.py', 'src/jpeers/followup/common.py',
               'src/jpeers/harness.py', 'src/jpeers/client.py', 'src/jpeers/schemas.py',
               'src/jpeers/runtime.py', 'src/jpeers/trace.py', 'src/jpeers/interventions.py',
               'scripts/run_causal_handoff.py']
    hashes = {p: file_digest(source_root / p) for p in sources if (source_root / p).exists()}
    inventory = [{'episode': asdict(r), 'messages': r.messages(), 'key': r.key} for r in rows + controls]
    return {'protocol': PROTOCOL, 'config': config, 'config_sha256': digest(config),
            'endpoint_info': endpoint_info, 'source_sha256': hashes,
            'inventory_sha256': digest(inventory), 'templates_sha256': digest(
                [DEVELOPMENT_TEMPLATES, HELDOUT_TEMPLATES]),
            'template_provenance': 'New English instructions authored for this assay 2026-11-07',
            'splits': {'development': 8, 'heldout': 24},
            'counts': {'development': 256, 'heldout': 768, 'heldout_generations': 4608,
                       'assay_controls': len(controls), 'engineering_clean_replay': 1},
            'model_revision_verification': 'Expected revision recorded; endpoint info alone does not attest weight hashes'}


class HandoffRunner:
    def __init__(self, endpoint, output_dir, config, *, resume=False):
        self.endpoint = endpoint
        self.root = Path(output_dir)
        self.config = dict(config)
        _validate_config(self.config)
        info = endpoint.info()
        if info.get('hf_id', info.get('model')) != config['model']:
            raise ValueError('Endpoint model identity differs from the configured target')
        required = config['deletion_layers'] + [l + config['rescue_offset'] for l in config['deletion_layers']]
        if info.get('n_layers', 0) <= max(required):
            raise ValueError('Endpoint does not contain all declared deletion/rescue layers')
        self.provenance = make_provenance(self.config, info)
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise ValueError('Output already contains artifacts; pass --resume or use a fresh directory')
        self.store = ReceiptStore(self.root, self.provenance)
        self.rows = make_episodes()
        inventory = {'provenance': self.provenance,
                     'episodes': [{'key': x.key, **asdict(x), 'messages_sha256': digest(x.messages())}
                                  for x in self.rows + make_controls()]}
        inventory_path = self.root / 'task_manifest.json'
        if inventory_path.exists() and json.loads(inventory_path.read_text()) != json.loads(json.dumps(inventory)):
            raise ValueError('Task manifest changed on resume')
        atomic_json(inventory_path, inventory)
        self._stage = 'initialized'
        self._counts = {'receipt_count': len(list((self.root / 'receipts').glob('*.json'))),
                        'session_new_calls': 0, 'session_cache_hits': 0}

    def _progress(self, episode, condition, tag, state):
        atomic_json(self.root / 'progress.json', {**self._counts, 'stage': self._stage,
                    'state': state, 'tag': tag, 'condition': condition,
                    'template_id': episode.template_id, 'episode_key': episode.key,
                    'split': episode.split, 'updated_at': datetime.now(timezone.utc).isoformat()})

    def _status(self, status, **fields):
        atomic_json(self.root / 'status.json', {'status': status, 'protocol': PROTOCOL,
                    'provenance_sha256': digest(self.provenance), **fields})

    def _request(self, episode, *, layers, edits=(), capture=False):
        return GenerateRequest(messages=episode.messages(), max_new_tokens=self.config['max_new_tokens'],
            temperature=0., seed=self.config['seed'], interventions=list(edits),
            spans=[SpanQuery(label='neutral_suffix', text=SUFFIX),
                   SpanQuery(label='causal_site', text='.', occurrence='last'),
                   SpanQuery(label='own_payload', text=episode.own_note()),
                   SpanQuery(label='incoming_payload', text=episode.incoming_note())],
            record=RecordSpec(enabled=True, layers=sorted(layers), watch=list(LABELS), top_k=1,
                positions=_site_selector(), capture_residuals=capture,
                capture_directions=['jacobian', 'logit'] if capture else [],
                direction_token_form='bare', when='post', max_positions=1))

    def query(self, episode, condition, *, layers, selection=None, clean=None, wrong=None, tag=''):
        edits = condition_edits(condition, selection, clean, wrong) if selection else []
        req = self._request(episode, layers=layers, edits=edits, capture=condition == 'baseline')
        payload = req.model_dump()
        key = f'{tag}:{episode.key}:{condition}'
        cached = self.store.load(key, payload)
        if cached is None:
            self._counts['session_new_calls'] += 1
            self._progress(episode, condition, tag, 'generating')
            try:
                response = self.endpoint.generate(req)
            except Exception:
                self._progress(episode, condition, tag, 'request_failed')
                raise
            body = response.model_dump()
            score = score_output(response.text, own=episode.own, incoming=episode.expected_incoming,
                                 n_generated=response.n_generated, cap=req.max_new_tokens)
            # Raw response is durable before validation or the derived TurnRecord.
            # A projection crash cannot silently trigger another model generation.
            self.store.save(key, payload, body, score,
                turn_directory=str(Path('turns') / 'requests' / digest(key)),
                donor_metadata={'clean': clean, 'wrong': wrong})
            self._counts['receipt_count'] += 1
            self._progress(episode, condition, tag, 'response_durable')
            cached = self.store.load(key, payload)
        else:
            self._counts['session_cache_hits'] += 1
            self._progress(episode, condition, tag, 'cache_hit')
        body = cached['response']
        trace = JSpaceTrace.from_response(body)
        if trace is None:
            raise ValueError('Recorded causal request returned no trace')
        if condition == 'baseline':
            self._validate_capture(trace)
        events = body.get('causal_edits', [])
        if len(events) != len(edits):
            raise ValueError('Actual causal edit receipts do not match the requested edits')
        for index, (event, edit) in enumerate(zip(events, edits)):
            if (event.get('intervention_index'), event.get('kind'), event.get('layer')) != (
                    index, edit.kind, edit.layers[0]):
                raise ValueError('Actual edit order/kind/layer differs from request')
            suffix_end = trace.spans['neutral_suffix'][0][1] - 1
            if event.get('position') != suffix_end or event.get('site') != 'block_output':
                raise ValueError('Actual causal edit site differs from frozen prompt endpoint')
        self._materialize_turn(key, episode, condition, cached, trace)
        self._progress(episode, condition, tag, 'ready')
        return cached

    def _materialize_turn(self, key, episode, condition, receipt, trace):
        payload, body = receipt['request'], receipt['response']
        target = self.root / receipt['turn_directory']
        turns = target / 'turns.jsonl'
        if turns.exists():
            text = turns.read_text()
            try:
                saved = json.loads(text)
            except json.JSONDecodeError:
                # The raw receipt is authoritative. Preserve a hard-kill partial
                # derived view by hash, then reconstruct without another request.
                partial_hash = hashlib.sha256(text.encode()).hexdigest()
                turns.rename(target / f'turns.incomplete-{partial_hash}.jsonl')
            else:
                extra = saved.get('extra', {})
                if (extra.get('request_sha256') != receipt['request_sha256'] or
                        saved.get('interventions') != payload['interventions'] or
                        saved.get('messages') != payload['messages'] or
                        saved.get('text') != body['text'] or
                        extra.get('causal_edits') != body.get('causal_edits', []) or
                        extra.get('capture_metadata') != body.get('capture_metadata', {}) or
                        extra.get('donors') != receipt['donor_metadata'] or
                        extra.get('score') != receipt['score']):
                    raise ValueError('Saved turn projection differs from the durable response receipt')
                return
        writer = RunWriter(self.root / 'turns', 'requests', digest(key), config={
            'request_sha256': receipt['request_sha256'],
            'provenance_sha256': digest(self.provenance)})
        record = TurnRecord(index=0, agent='recipient', endpoint=self.endpoint.name,
            model=body['model'], text=body['text'], prompt_len=body['prompt_len'],
            n_generated=body['n_generated'], timings=body['timings'],
            interventions=payload['interventions'], trace_file=None, tag=key,
            messages=payload['messages'], extra={'protocol': PROTOCOL, 'episode': asdict(episode),
                'condition': condition, 'request_sha256': receipt['request_sha256'],
                'capture_metadata': body.get('capture_metadata', {}),
                'causal_edits': body.get('causal_edits', []),
                'donors': receipt['donor_metadata'], 'score': receipt['score']})
        writer.write(record, trace)

    def _validate_capture(self, trace):
        validate_capture_site(trace)
        metadata = getattr(trace, 'capture_metadata', {})
        if metadata.get('residual_site') != 'block_output' or metadata.get('when') != 'post':
            raise ValueError('Residual capture convention does not match the declared block-output site')
        tokens = metadata.get('label_tokenization', {})
        ids = list(np.asarray(trace.direction_token_ids, dtype=int))
        for word, tid in zip(LABELS, ids):
            entry = tokens.get(word, {})
            if entry.get('bare_ids') != [int(tid)] or entry.get('direction_id') != int(tid):
                raise ValueError('Each balanced bare label must tokenize as its captured single direction token')
        residuals = getattr(trace, 'residuals', None)
        if residuals is None or np.asarray(residuals).shape[:2] != (1, len(trace.layers)):
            raise ValueError('Missing localized residual captures')
        for lens in ('jacobian', 'logit'):
            array = getattr(trace, f'directions_{lens}', None)
            if array is None or np.asarray(array).shape != (len(trace.layers), 2, residuals.shape[-1]):
                raise ValueError('Missing exact label-direction captures')
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite label-direction captures')

    @staticmethod
    def trace(receipt):
        return JSpaceTrace.from_response(receipt['response'])

    def donor(self, episode, baseline, selection):
        trace = self.trace(baseline)
        state = np.asarray(trace.residuals)[0, trace.layers.index(selection['rescue_layer'])]
        coefficients = np.asarray(selection['rescue_basis']) @ state
        value = {'episode_key': episode.key, 'messages_sha256': digest(episode.messages()),
                 'request_sha256': baseline['request_sha256'],
                 'response_sha256': digest(baseline['response']),
                 'rescue_basis_sha256': digest(selection['rescue_basis']),
                 'layer': selection['rescue_layer'], 'position': int(trace.positions[0]),
                 'prompt_len': trace.prompt_len, 'coefficients': coefficients.tolist()}
        value['sha256'] = digest(value)
        return value

    def _rate(self, receipts, metric):
        return float(np.mean([x['score'][metric] is True for x in receipts]))

    def _fail_discovery(self, reason, report):
        report.update(status='discovery_failed', reason=reason)
        atomic_json(self.root / 'discovery.json', report)
        self._status('discovery_failed', reason=reason)
        return report

    def discovery(self):
        self._stage = 'discovery'
        if (self.root / 'frozen.json').exists():
            return load_frozen(self.root, self.provenance)
        self._status('discovery_running')
        rows = [x for x in self.rows if x.split == 'development']
        layers = sorted(set(self.config['deletion_layers'] +
                            [l + self.config['rescue_offset'] for l in self.config['deletion_layers']]))
        report = {'status': 'discovery_running', 'gates': {}, 'decoders': {}, 'settings': [],
                  'selection_rule': 'weakest dose first, then declared deletion-layer order; no held-out responses'}
        controls = [self.query(x, 'baseline', layers=layers, tag='control') for x in make_controls()]
        report['gates']['controls'] = {kind: self._rate(
            [r for x, r in zip(make_controls(), controls) if x.control == kind], 'joint_correct')
            for kind in ('no_handoff', 'own_only')}
        if min(report['gates']['controls'].values()) < self.config['baseline_accuracy_min']:
            return self._fail_discovery('No-handoff abstention or own-only extraction gate failed', report)
        replay = self.query(make_controls()[0], 'baseline', layers=layers, tag='engineering-clean-replay')
        first_trace, replay_trace = self.trace(controls[0]), self.trace(replay)
        replay_passed = (controls[0]['request_sha256'] == replay['request_sha256'] and
            controls[0]['response']['text'] == replay['response']['text'] and
            controls[0]['response']['n_generated'] == replay['response']['n_generated'] and
            all(np.allclose(getattr(first_trace, key), getattr(replay_trace, key), rtol=1e-5, atol=1e-5)
                for key in ('residuals', 'directions_jacobian', 'directions_logit', 'direction_token_ids')))
        report['gates']['deterministic_clean_replay'] = {'passed': bool(replay_passed),
            'request_sha256': replay['request_sha256'], 'array_atol': 1e-5, 'array_rtol': 1e-5,
            'first_response_sha256': digest(controls[0]['response']),
            'replay_response_sha256': digest(replay['response'])}
        if not replay_passed:
            return self._fail_discovery('Deterministic clean capture replay gate failed', report)
        baseline = {x.key: self.query(x, 'baseline', layers=layers, tag='dev') for x in rows}
        base_rates = {metric: self._rate(baseline.values(), metric)
                      for metric in ('incoming_correct', 'own_correct', 'format_valid')}
        report['gates']['baseline'] = base_rates
        if min(base_rates.values()) < self.config['baseline_accuracy_min']:
            return self._fail_discovery('Fresh factorial task or strict JSON baseline failed', report)
        traces = [self.trace(baseline[x.key]) for x in rows]
        report['anchor_preflight'] = {'n_contexts': len(traces),
            'direction_token_ids': validate_capture_site(traces[0])['direction_token_ids'],
            'capture_metadata': traces[0].capture_metadata}
        fits = {}
        for layer in layers:
            states = np.stack([t.residuals[0, t.layers.index(layer)] for t in traces])
            fits[layer] = {}
            for lens in ('jacobian', 'logit', 'residual_contrast'):
                directions = None
                if lens != 'residual_contrast':
                    all_directions = np.stack([getattr(t, f'directions_{lens}')[t.layers.index(layer)]
                                               for t in traces])
                    # Lens vectors are fixed by the model/layer/token. Varying
                    # vectors would require fold-local span fitting, unsupported here.
                    if not np.allclose(all_directions, all_directions[0], atol=1e-5, rtol=1e-5):
                        return self._fail_discovery('Label direction span varies across episodes', report)
                    directions = all_directions[0]
                fits[layer][lens] = fit_component(rows, states, directions, ridge=self.config['ridge'])
            report['decoders'][str(layer)] = fits[layer]
        def stable(fit):
            return (fit['rank'] > 0 and fit['incoming_cv'] >= self.config['decoder_cv_min'] and
                    fit['incoming_basis_own_cv'] <= self.config['leakage_cv_max'] and
                    fit['incoming_basis_nuisance_cv'] <= self.config['leakage_cv_max'])
        atomic_json(self.root / 'discovery.json', report)
        for dose in self.config['doses']:
            for layer in self.config['deletion_layers']:
                rescue = layer + self.config['rescue_offset']
                earlier, later = fits[layer]['jacobian'], fits[rescue]['jacobian']
                if not stable(earlier) or not stable(later):
                    continue
                selection = {'layer': layer, 'rescue_layer': rescue, 'dose': dose,
                    'basis': earlier['basis'], 'rescue_basis': later['basis'],
                    'basis_sha256': digest(earlier['basis']),
                    'rescue_basis_sha256': digest(later['basis']),
                    'random_seed': self.config['random_seed'], 'random_direction_replicates': 1, 'site': 'neutral_suffix endpoint',
                    'site_text': SUFFIX, 'site_convention': 'block_output; one prompt punctuation token',
                    'direction_token_form': 'bare',
                    'direction_token_ids': report['anchor_preflight']['direction_token_ids'],
                    'donor_rule': 'same template/source/own/names/order; opposite incoming only for wrong donor',
                    'split_sha256': self.provenance['inventory_sha256'],
                    'geometry_relative_tolerance': self.config['geometry_relative_tolerance']}
                tag = f'dev-l{layer}-dose{dose}'
                candidate = {'layer': layer, 'dose': dose}
                deletion = [self.query(x, 'delete', layers=[layer, rescue], selection=selection, tag=tag) for x in rows]
                random = [self.query(x, 'random_delete', layers=[layer, rescue], selection=selection, tag=tag) for x in rows]
                geometry = [geometry_pair(a['response']['causal_edits'], b['response']['causal_edits'],
                             self.config['geometry_relative_tolerance']) for a, b in zip(deletion, random)]
                candidate['deletion_geometry'] = geometry
                candidate['incoming_target_minus_random'] = (self._rate(deletion, 'incoming_correct') -
                                                              self._rate(random, 'incoming_correct'))
                candidate['rates'] = {c: {m: self._rate(receipts, m) for m in base_rates}
                    for c, receipts in [('delete', deletion), ('random_delete', random)]}
                candidate['specificity_passed'] = all(candidate['rates'][c][m] >= base_rates[m] -
                    self.config['specificity_margin'] for c in candidate['rates'] for m in ('own_correct', 'format_valid'))
                report['settings'].append(candidate)
                atomic_json(self.root / 'discovery.json', report)
                if not all(g['passed'] for g in geometry):
                    return self._fail_discovery('Realized deletion/control geometry gate failed', report)
                if (not candidate['specificity_passed'] or candidate['incoming_target_minus_random'] >
                        -self.config['incoming_effect_min']):
                    continue
                donors = {x.key: self.donor(x, baseline[x.key], selection) for x in rows}
                rescues = {c: [self.query(x, c, layers=[layer, rescue], selection=selection,
                    clean=donors[x.key], wrong=donors[opposite_donor(x, rows).key], tag=tag)
                    for x in rows] for c in ('clean_rescue', 'wrong_donor', 'random_restore')}
                restore_geometry = [geometry_pair([a['response']['causal_edits'][-1]],
                    [b['response']['causal_edits'][-1]], self.config['geometry_relative_tolerance'])
                    for a, b in zip(rescues['clean_rescue'], rescues['random_restore'])]
                candidate['restoration_geometry'] = restore_geometry
                candidate['rescue_rates'] = {c: {m: self._rate(receipts, m) for m in base_rates}
                                             for c, receipts in rescues.items()}
                candidate['clean_rescue_improvement'] = (candidate['rescue_rates']['clean_rescue']['incoming_correct'] -
                                                          candidate['rates']['delete']['incoming_correct'])
                if not all(g['passed'] for g in restore_geometry):
                    return self._fail_discovery('Realized restoration/control geometry gate failed', report)
                if any(candidate['rescue_rates'][c][m] < base_rates[m] - self.config['specificity_margin']
                       for c in rescues for m in ('own_correct', 'format_valid')):
                    continue
                if candidate['clean_rescue_improvement'] < self.config['incoming_effect_min']:
                    continue
                # Counterfactual donors must really have identical token counts.
                if any(donors[x.key]['prompt_len'] != donors[opposite_donor(x, rows).key]['prompt_len'] for x in rows):
                    return self._fail_discovery('Counterfactual donor token lengths differ', report)
                report['status'] = 'discovery_passed'
                selection['discovery_summary_sha256'] = digest(report)
                selection['decoder_comparison'] = {str(l): {lens: {k: fit[k] for k in
                    ('incoming_cv', 'own_cv', 'incoming_basis_own_cv', 'span_rank', 'capacity')}
                    for lens, fit in fits[l].items()} for l in (layer, rescue)}
                selection['j_specific_claim'] = 'Not established by this feature selection; capacity-matched readout comparisons are descriptive'
                report['status'] = 'discovery_passed'
                atomic_json(self.root / 'discovery.json', report)
                return freeze_selection(self.root, self.provenance, selection)
        return self._fail_discovery('No stable conditional incoming feature and selective development setting', report)

    def heldout(self):
        self._stage = 'heldout'
        frozen = load_frozen(self.root, self.provenance)
        selection = frozen['selection']
        self._status('heldout_running', freeze_sha256=frozen['freeze_sha256'])
        rows = [x for x in self.rows if x.split == 'heldout']
        layers = [selection['layer'], selection['rescue_layer']]
        results = {}
        try:
            baseline = {}
            for x in rows:
                baseline[x.key] = self.query(x, 'baseline', layers=layers, tag='heldout')
                results[(x.key, 'baseline')] = baseline[x.key]['score']
            donors = {x.key: self.donor(x, baseline[x.key], selection) for x in rows}
            for x in rows:
                other = opposite_donor(x, rows)
                if donors[x.key]['prompt_len'] != donors[other.key]['prompt_len']:
                    raise ValueError('Held-out counterfactual donor token lengths differ')
                actual = {}
                for condition in CONDITIONS[1:]:
                    receipt = self.query(x, condition, layers=layers, selection=selection,
                        clean=donors[x.key], wrong=donors[other.key], tag='heldout')
                    results[(x.key, condition)] = receipt['score']
                    actual[condition] = receipt['response']['causal_edits']
                deletion_geometry = geometry_pair(actual['delete'], actual['random_delete'],
                                                   self.config['geometry_relative_tolerance'])
                rescue_geometry = geometry_pair([actual['clean_rescue'][-1]], [actual['random_restore'][-1]],
                                                 self.config['geometry_relative_tolerance'])
                if not deletion_geometry['passed'] or not rescue_geometry['passed']:
                    raise ValueError('Held-out realized edit geometry failed; remaining cells retained unknown')
        finally:
            # Even an interrupted run retains every intended denominator. Resume
            # rehydrates the exact receipts rather than regenerating completed cells.
            atomic_json(self.root / 'summary.json', summarize(rows, results))
        self._status('completed', n_contexts=len(rows), n_conditions=len(CONDITIONS),
                     n_scored=len(results), freeze_sha256=frozen['freeze_sha256'])
        return json.loads((self.root / 'summary.json').read_text())

    def run(self, stage='all'):
        try:
            if stage not in ('discovery', 'heldout', 'all'):
                raise ValueError('Stage must be discovery, heldout, or all')
            if stage in ('discovery', 'all'):
                discovery = self.discovery()
                if discovery.get('status') == 'discovery_failed' or stage == 'discovery':
                    return discovery
            return self.heldout()
        except Exception as exc:
            self._status('failed', error=f'{type(exc).__name__}: {exc}')
            raise
