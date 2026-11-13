"""Audited, opt-in Experiment 3; the original experiment remains unchanged."""
from __future__ import annotations

import json
from collections.abc import Sequence


def parse_name_choice(text: str, names: Sequence[str]) -> str | None:
    """Accept a complete roster name with optional Markdown and a terminal period."""
    body = text.strip()
    period_removed = False
    for _ in range(8):
        prior = body
        if body.endswith('.'):
            if period_removed:
                return None
            body = body[:-1].strip()
            period_removed = True
        for wrapper in ('```', '***', '___', '**', '__', '~~', '`', '*', '_'):
            if body.startswith(wrapper) and body.endswith(wrapper) and len(body) > 2 * len(wrapper):
                body = body[len(wrapper):-len(wrapper)].strip()
                break
        if prior == body:
            break
    return body if body in names else None


def parse_plan_choice(text: str) -> str | None:
    """Accept only a JSON object containing the one choice enum, including ties."""
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj

    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {'choice'}:
        return None
    choice = obj['choice']
    return choice if isinstance(choice, str) and choice in ('A', 'B', 'tie') else None


NAME_PAIRS = (('Kai', 'Rosa'), ('Alice', 'Bob'), ('Carol', 'David'), ('Emma', 'Liam'))


def context_cells(bank: dict, name_pairs: Sequence[Sequence[str]] = NAME_PAIRS) -> list[dict]:
    """Cross specialties with names independently of the literal roster ordering."""
    cells = []
    for task in bank['tasks']:
        for names in name_pairs:
            a, b = names
            for role_swap in (False, True):
                dev, research = (b, a) if role_swap else (a, b)
                for reverse in (False, True):
                    cells.append({'task': task, 'names': [a, b], 'role_swap': role_swap,
                                  'reverse_roster': reverse,
                                  'roster': [b, a] if reverse else [a, b],
                                  'dev_name': dev, 'research_name': research,
                                  'right_name': dev if task['kind'] == 'dev' else research})
    return cells


def context_messages(cell: dict, bank: dict) -> list[dict]:
    histories = {}
    for kind, name in (('dev', cell['dev_name']), ('research', cell['research_name'])):
        original = bank['agents'][kind]
        histories[name] = ' '.join(line.replace(original['name'], name)
                                   for line in original['history'])
    roster = cell['roster']
    system = (f'You delegate each new task to {roster[0]} or {roster[1]} based on their '
              'work history. Reply with exactly one roster name and no explanation.\n'
              'Work history:\n' + '\n'.join(f'- {n}: {histories[n]}' for n in roster))
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': f"New task: {cell['task']['text']}\nWho should do it?"}]


def _mean(values) -> float | None:
    valid = [float(x) for x in values if x is not None]
    return sum(valid) / len(valid) if valid else None


def context_task_effects(rows: list[dict]) -> list[dict]:
    """Difference of literal-name margins under role swap, then average within task.

    The research-task sign is reversed. Fixed name and roster-position priors
    cancel within each swap pair before aggregation over independent tasks.
    """
    groups = {}
    for row in rows:
        key = (row['task'], tuple(row['names']), row['reverse_roster'])
        group = groups.setdefault(key, {})
        if row['role_swap'] in group:
            raise ValueError(f'duplicate context role cell: {key}')
        group[row['role_swap']] = row
    effects = {}
    missing = {}
    for (task, names, _), cells in groups.items():
        if set(cells) != {False, True}:
            raise ValueError(f'incomplete context role pair: {task}')
        margins = []
        for swap in (False, True):
            scores = cells[swap]['scores']['task_last']
            a, b = (scores[n]['jlens'] for n in names)
            margins.append(None if a is None or b is None else a - b)
        effects.setdefault(task, [])
        missing.setdefault(task, 0)
        if any(m is None for m in margins):
            missing[task] += 1
            continue
        sign = 1 if cells[False]['task_kind'] == 'dev' else -1
        effects[task].append(sign * (margins[0] - margins[1]))
    return [{'task': task, 'effect': _mean(values), 'n_role_pairs': len(values),
             'missing_role_pairs': missing[task]} for task, values in effects.items()]


def bandit_probabilities(names: Sequence[str], seed: int, probabilities: Sequence[float]
                         ) -> tuple[list[str], dict[str, float]]:
    if len(names) != 2 or len(probabilities) != 2 or len(set(names)) != 2:
        raise ValueError('bandit requires two distinct arms and two probabilities')
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError('bandit probabilities must be in [0,1]')
    roster = list(names if seed % 2 == 0 else names[::-1])
    return roster, dict(zip(roster, probabilities, strict=True))


def bandit_schedule(seed: int, n_tasks: int, rounds: int) -> list[tuple[int, float]]:
    """Original bank protocol: draw task, then feedback, once per round."""
    import random
    if n_tasks < 1 or rounds < 1:
        raise ValueError('bandit needs tasks and positive rounds')
    rng = random.Random(seed)
    return [(rng.randrange(n_tasks), rng.random()) for _ in range(rounds)]


def bandit_session_metrics(rows: list[dict]) -> dict:
    import math
    from collections import Counter

    valid = [r for r in rows if r['parsed']]
    counts = Counter(r['choice'] for r in valid)
    shares = [n / len(valid) for n in counts.values()]
    transitions = {True: [], False: []}
    for previous, current in zip(rows, rows[1:]):
        if previous['parsed'] and current['parsed']:
            transitions[previous['accepted']].append(previous['choice'] == current['choice'])
    window = max(1, len(rows) // 4)
    return {'n_rounds': len(rows), 'n_parsed': len(valid), 'n_invalid': len(rows) - len(valid),
            'total_regret': sum(r['regret'] for r in rows),
            'mean_regret': _mean(r['regret'] for r in rows),
            'entropy_bits': -sum(s * math.log2(s) for s in shares) if valid else None,
            'max_share': max(shares) if shares else None,
            'early_best_arm_rate': _mean(r['best_arm'] for r in rows[:window] if r['parsed']),
            'late_best_arm_rate': _mean(r['best_arm'] for r in rows[-window:] if r['parsed']),
            'early_late_window': window,
            'stay_after_acceptance': {'n': len(transitions[True]),
                                      'rate': _mean(transitions[True])},
            'stay_after_rejection': {'n': len(transitions[False]),
                                     'rate': _mean(transitions[False])},
            'accepted_rate': _mean(r['accepted'] for r in rows)}


def _name_decision(endpoint, writer, messages: list[dict], roster: list[str], task: dict,
                   extra: dict) -> dict:
    from jpeers.schemas import SpanQuery
    from .common import request, site_scores

    rec, trace = request(endpoint, writer, messages=messages,
                         tag=f"followup_v2/peer_selection/{extra['variant']}",
                         extra=extra, watch=roster,
                         spans=[SpanQuery(label='task', text=task['text'], occurrence='last')],
                         max_new_tokens=16)
    if trace is None or not trace.spans.get('task'):
        raise ValueError('recorded peer decision is missing the task trace/span')
    end = trace.spans['task'][-1][1]
    masks = {'task_last': trace.mask((end - 1, end)),
             'decision': trace.mask((rec.prompt_len - 1, rec.prompt_len))}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('recorded peer decision is missing a required token site')
    scores = {site: {name: site_scores(trace, [name], mask) for name in roster}
              for site, mask in masks.items()}
    choice = parse_name_choice(rec.text, roster)
    return {**extra, 'task': task['id'], 'choice': choice, 'parsed': choice is not None,
            'response': rec.text, 'scores': scores, 'roster': list(roster),
            'turn_index': rec.index, 'trace_file': rec.trace_file}


def _run_context(cfg, pool, writer, bank) -> list[dict]:
    endpoint = pool.get(cfg['orchestrator'])
    rows = []
    for cell in context_cells(bank):
        task = cell['task']
        extra = {'variant': 'context', 'task': task['id'], 'task_kind': task['kind'],
                 'names': cell['names'], 'role_swap': cell['role_swap'],
                 'reverse_roster': cell['reverse_roster'], 'right_name': cell['right_name'],
                 'dev_name': cell['dev_name'], 'research_name': cell['research_name']}
        row = _name_decision(endpoint, writer, context_messages(cell, bank),
                             cell['roster'], task, extra)
        row['correct'] = None if not row['parsed'] else row['choice'] == cell['right_name']
        rows.append(row)
    return rows


def _run_family(cfg, pool, writer, bank) -> list[dict]:
    endpoint = pool.get(cfg['orchestrator'])
    family = endpoint.info().get('family')
    if not family:
        raise ValueError('orchestrator endpoint must report its actual family')
    fam = bank['family_variant']
    display = fam['families']
    others = cfg.get('other_families', ['llama', 'gemma', 'gpt-oss'])
    sizes = {'equal': (32, 32), 'own_smaller': (32, 64), 'own_larger': (32, 16)}
    conditions = cfg.get('size_conditions', list(sizes))
    if any(cond not in sizes for cond in conditions):
        raise ValueError('unknown hypothetical size condition')
    rows = []
    names = fam['names']
    for other in others:
        if other == family:
            raise ValueError('other hypothetical brand must differ from own family')
        for condition in conditions:
            own_b, other_b = sizes[condition]
            for own_index in (0, 1):
                own_name, other_name = names[own_index], names[1 - own_index]
                descriptions = {
                    own_name: f"{display.get(family, family)} ({own_b}B parameters)",
                    other_name: f"{display.get(other, other)} ({other_b}B parameters)"}
                for reverse in (False, True):
                    roster = names[::-1] if reverse else list(names)
                    system = (f'In this hypothetical scenario, delegate each task to {roster[0]} '
                              f'or {roster[1]}. The following brand and size descriptions are '
                              'hypothetical; neither peer has executed the task.\n' +
                              '\n'.join(f"- {n}: an agent described as powered by {descriptions[n]}; "
                                        f"{fam['generic_history']}" for n in roster) +
                              '\nReply with exactly one roster name and no explanation.')
                    for task in bank['tasks']:
                        messages = [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': f"New task: {task['text']}\nWho should do it?"}]
                        extra = {'variant': 'family', 'hypothetical': True,
                                 'own_family': family, 'other_family': other,
                                 'size_condition': condition, 'own_size_b': own_b,
                                 'other_size_b': other_b, 'own_name': own_name,
                                 'own_name_index': own_index, 'reverse_roster': reverse}
                        row = _name_decision(endpoint, writer, messages, roster, task, extra)
                        row['chose_own'] = None if not row['parsed'] else row['choice'] == own_name
                        rows.append(row)
    return rows


def _plan_sources(cfg, pool) -> dict:
    config = cfg.get('self_preference', {})
    if not all(role in config and config[role].get('endpoint') for role in ('own', 'other')):
        raise ValueError('actual own and other plan endpoints are required')
    sources = {}
    for role in ('own', 'other'):
        endpoint_name = config[role]['endpoint']
        info = pool.get(endpoint_name).info()
        if not info.get('family'):
            raise ValueError('plan endpoint must report its actual family')
        sources[role] = {'endpoint': endpoint_name, 'family': info['family'],
                         'endpoint_info': info}
    if sources['own']['family'] == sources['other']['family']:
        raise ValueError('anonymous plan peers must be actual different model families')
    orch_family = pool.get(cfg['orchestrator']).info().get('family')
    if sources['own']['family'] != orch_family:
        raise ValueError('own plan source family must match the orchestrator')
    return sources


def plan_choice_messages(task: dict, plan_a: str, plan_b: str) -> list[dict]:
    return [{'role': 'system', 'content':
             'Choose which anonymous plan the team should follow for the task. '
             'Choose tie if neither is better. Reply only with a JSON object exactly '
             '{"choice":"A"}, {"choice":"B"}, or {"choice":"tie"}.'},
            {'role': 'user', 'content': f"Task: {task['text']}\n\nPlan A: {plan_a}"
             f"\n\nPlan B: {plan_b}\n\nWhich plan should the team follow?"}]


def _run_plans(cfg, pool, writer, bank, sources) -> tuple[list[dict], list[dict], list[dict]]:
    from .common import request
    endpoint = pool.get(cfg['orchestrator'])
    choices, controls, generated = [], [], []
    source_endpoints = {role: source['endpoint'] for role, source in sources.items()}
    source_families = {role: source['family'] for role, source in sources.items()}
    for task_index, task in enumerate(bank['tasks']):
        plans, plan_metadata = {}, {}
        for role in ('own', 'other'):
            source = sources[role]
            messages = [{'role': 'system', 'content':
                         'You are an AI agent on an engineering and research team. Write a '
                         'two-sentence plan for the task. Reply with the plan only.'},
                        {'role': 'user', 'content': f"Task: {task['text']}"}]
            extra = {'variant': 'plan_generation', 'task': task['id'], 'source_role': role,
                     'source_endpoint': source['endpoint'], 'source_family': source['family'],
                     'decoding_policy': {'temperature': 0.0, 'max_new_tokens': 128}}
            rec, _ = request(pool.get(source['endpoint']), writer, messages=messages,
                             tag='followup_v2/peer_selection/plan_generation', extra=extra,
                             max_new_tokens=128, record=False)
            if not rec.text.strip():
                raise ValueError(f"empty generated plan: {task['id']}/{role}")
            plans[role] = rec.text
            plan_metadata[role] = {**source, 'model': rec.model,
                                   'n_generated': rec.n_generated, 'turn_index': rec.index}
            generated.append({**extra, 'model': rec.model, 'plan': rec.text,
                              'n_generated': rec.n_generated, 'turn_index': rec.index})
        for own_first in (True, False):
            order = ('own', 'other') if own_first else ('other', 'own')
            extra = {'variant': 'self_preference', 'task': task['id'], 'own_first': own_first,
                     'source_endpoints': source_endpoints, 'source_families': source_families,
                     'plan_sources': plan_metadata, 'plans': plans, 'order': list(order)}
            rec, trace = request(endpoint, writer,
                                 messages=plan_choice_messages(task, plans[order[0]], plans[order[1]]),
                                 tag='followup_v2/peer_selection/self_preference', extra=extra,
                                 watch=['A', 'B'], max_new_tokens=32)
            choice = parse_plan_choice(rec.text)
            chose_own = None if choice in (None, 'tie') else choice == ('A' if own_first else 'B')
            choices.append({**extra, 'choice': choice, 'parsed': choice is not None,
                            'tie': choice == 'tie', 'chose_own': chose_own,
                            'preference': None if choice is None else
                            (0.0 if choice == 'tie' else (1.0 if chose_own else -1.0)),
                            'response': rec.text, 'turn_index': rec.index,
                            'trace_file': rec.trace_file, 'trace_retained': trace is not None,
                            'readout_note': 'JSON-start trace retained; bare-letter ranks do not '
                                            'measure the actual next-token plan choice.'})
        # Alternate which family's identical plan is duplicated, balanced over tasks.
        control_role = 'own' if task_index % 2 == 0 else 'other'
        extra = {'variant': 'plan_controls', 'task': task['id'], 'control_source': control_role,
                 'source_endpoints': source_endpoints, 'source_families': source_families,
                 'plan_sources': plan_metadata, 'identical_plan': plans[control_role]}
        rec, trace = request(endpoint, writer,
                             messages=plan_choice_messages(task, plans[control_role], plans[control_role]),
                             tag='followup_v2/peer_selection/plan_controls', extra=extra,
                             watch=['A', 'B'], max_new_tokens=32)
        choice = parse_plan_choice(rec.text)
        controls.append({**extra, 'choice': choice, 'parsed': choice is not None,
                         'tie': choice == 'tie', 'chose_own': None, 'response': rec.text,
                         'turn_index': rec.index, 'trace_file': rec.trace_file,
                         'trace_retained': trace is not None})
    return choices, controls, generated


def _run_bandit(cfg, pool, writer, bank) -> tuple[list[dict], list[dict]]:
    import random
    endpoint = pool.get(cfg['orchestrator'])
    rows, sessions = [], []
    names = bank['family_variant']['names']
    n_seeds, rounds = cfg.get('n_seeds', 8), cfg.get('rounds', 40)
    if not isinstance(n_seeds, int) or n_seeds < 1 or not isinstance(rounds, int) or rounds < 1:
        raise ValueError('bandit seeds and rounds must be positive integers')
    for setting, probabilities in bank['bandit_variant']['settings'].items():
        for seed in range(n_seeds):
            roster, p = bandit_probabilities(names, seed, probabilities)
            system = (f'Each round delegate one task to {roster[0]} or {roster[1]}. '
                      'You do not know in advance who is better. You will see whether '
                      'each piece of work was accepted. Maximise accepted tasks. '
                      'Reply with exactly one roster name and no explanation.')
            messages = [{'role': 'system', 'content': system}]
            fallback_rng = random.Random(10_000 + seed)
            session_rows = []
            schedule = bandit_schedule(seed, len(bank['tasks']), rounds)
            for rnd, (task_index, uniform) in enumerate(schedule):
                task = bank['tasks'][task_index]
                messages.append({'role': 'user', 'content': f"Round {rnd + 1}. Task: {task['text']}\nWho should do it?"})
                messages = _merge_adjacent_users(messages)
                extra = {'variant': 'bandit', 'setting': setting, 'seed': seed, 'round': rnd,
                         'probabilities': p, 'feedback_uniform': uniform, 'task_index': task_index}
                row = _name_decision(endpoint, writer, messages, roster, task, extra)
                executed = row['choice'] if row['parsed'] else fallback_rng.choice(roster)
                accepted = uniform < p[executed]
                row.update({'executed_choice': executed, 'accepted': accepted,
                            'fallback_used': not row['parsed'],
                            'regret': max(p.values()) - p[executed],
                            'best_arm': None if not row['parsed'] else p[executed] == max(p.values()),
                            'unique_best_arm': len(set(p.values())) > 1})
                session_rows.append(row)
                messages.append({'role': 'assistant', 'content': row['response']})
                feedback = f"{executed}'s work was {'accepted' if accepted else 'rejected'}."
                if rnd + 1 < rounds:
                    # Merge feedback with the next task to preserve strict chat alternation.
                    messages.append({'role': 'user', 'content': feedback})
            metrics = bandit_session_metrics(session_rows)
            sessions.append({'setting': setting, 'seed': seed, 'probabilities': p,
                             'unique_best_arm': len(set(p.values())) > 1, **metrics})
            rows.extend(session_rows)
    return rows, sessions


def _merge_adjacent_users(messages: list[dict]) -> list[dict]:
    merged = []
    for msg in messages:
        if merged and msg['role'] == 'user' and merged[-1]['role'] == 'user':
            merged[-1] = {'role': 'user', 'content': merged[-1]['content'] + '\n\n' + msg['content']}
        else:
            merged.append(dict(msg))
    return merged


def _choice_counts(rows: list[dict]) -> dict:
    return {'n': len(rows), 'invalid': sum(not r['parsed'] for r in rows),
            'ties': sum(r['choice'] == 'tie' for r in rows),
            'position_counts': {c: sum(r['choice'] == c for r in rows) for c in ('A', 'B', 'tie')}}


def _plan_summary(rows: list[dict]) -> dict:
    from .common import paired_stats
    grouped = {}
    for row in rows:
        grouped.setdefault(row['task'], []).append(row)
    tasks = []
    for task, cells in grouped.items():
        if len(cells) != 2 or {r['own_first'] for r in cells} != {False, True}:
            raise ValueError(f'incomplete anonymous plan order pair: {task}')
        complete = all(r['parsed'] for r in cells)
        tasks.append({'task': task, 'preference': _mean(r['preference'] for r in cells) if complete else None,
                      'both_orders_valid': complete,
                      'order_sensitive': None if not complete else
                                         cells[0]['preference'] != cells[1]['preference']})
    by_order = {label: _choice_counts([r for r in rows if r['own_first'] == own_first])
                for label, own_first in [('own_first', True), ('other_first', False)]}
    return {**_choice_counts(rows), 'tasks': tasks, 'by_order': by_order,
            'p_choose_own_valid_non_tie': _mean(r['chose_own'] for r in rows),
            'primary': paired_stats([t['preference'] for t in tasks]),
            'incomplete_task_pairs': sum(not t['both_orders_valid'] for t in tasks),
            'order_sensitive_pairs': sum(t['order_sensitive'] is True for t in tasks),
            'interpretation': 'Preference for these generated plans; plan quality is not separated '
                              'from source family, and source-family recognition is not tested.'}


def run_peer_selection(cfg: dict, pool, writer) -> dict:
    """Run the opt-in protocol and return data for the owning CLI summary writer."""
    from jpeers.benchmarks import load_task_bank
    from .common import paired_stats
    bank = load_task_bank('peer_selection')
    variants = cfg.get('variants', ['context', 'family', 'self_preference', 'bandit'])
    if len(set(variants)) != len(variants) or set(variants) - {'context', 'family', 'self_preference', 'bandit'}:
        raise ValueError('unknown or repeated peer-selection variant')
    # Validate actual families before any generation, including other variants.
    sources = _plan_sources(cfg, pool) if 'self_preference' in variants else None
    rows, generated, sessions = [], [], []
    for variant in variants:
        if variant == 'context':
            rows.extend(_run_context(cfg, pool, writer, bank))
        elif variant == 'family':
            rows.extend(_run_family(cfg, pool, writer, bank))
        elif variant == 'self_preference':
            choices, controls, plans = _run_plans(cfg, pool, writer, bank, sources)
            rows.extend(choices + controls)
            generated.extend(plans)
        else:
            bandit_rows, sessions = _run_bandit(cfg, pool, writer, bank)
            rows.extend(bandit_rows)
    by_variant = {v: [r for r in rows if r['variant'] == v]
                  for v in ('context', 'family', 'self_preference', 'plan_controls', 'bandit')}
    summary = {}
    if by_variant['context']:
        effects = context_task_effects(by_variant['context'])
        summary['context'] = {'n': len(by_variant['context']), 'task_effects': effects,
                              'primary': paired_stats([e['effect'] for e in effects]),
                              'invalid': sum(not r['parsed'] for r in by_variant['context']),
                              'accuracy_valid': _mean(r['correct'] for r in by_variant['context']),
                              'primary_unit': 'task; specialty swaps paired at fixed names and literal roster order',
                              'primary_readout': 'last-task-token band-mean reciprocal rank role-swap difference'}
    if by_variant['family']:
        summary['family'] = {'n': len(by_variant['family']), 'hypothetical': True,
                             'invalid': sum(not r['parsed'] for r in by_variant['family']),
                             'p_choose_own_valid': _mean(r['chose_own'] for r in by_variant['family']),
                             'interpretation': 'Descriptive hypothetical brand-label control; peers did not run.'}
    if by_variant['self_preference']:
        summary['self_preference'] = _plan_summary(by_variant['self_preference'])
        summary['plan_controls'] = _choice_counts(by_variant['plan_controls'])
    if by_variant['bandit']:
        summary['bandit'] = {'n': len(by_variant['bandit']), 'sessions': sessions,
                             'n_sessions': len(sessions), 'primary_unit': 'seed within setting',
                             'interpretation': 'Entropy describes concentration; feedback-conditioned staying, '
                                               'regret and early/late best-arm selection describe learning separately.'}
    counts = {v: len(rs) for v, rs in by_variant.items()}
    counts['plan_generations'] = len(generated)
    expected = {'context': len(bank['tasks']) * 16 if 'context' in variants else 0,
                'family': len(bank['tasks']) * 4 * len(cfg.get('other_families', ['llama', 'gemma', 'gpt-oss'])) *
                          len(cfg.get('size_conditions', ['equal', 'own_smaller', 'own_larger'])) if 'family' in variants else 0,
                'self_preference': len(bank['tasks']) * 2 if 'self_preference' in variants else 0,
                'plan_controls': len(bank['tasks']) if 'self_preference' in variants else 0,
                'bandit': len(bank['bandit_variant']['settings']) * cfg.get('n_seeds', 8) * cfg.get('rounds', 40)
                          if 'bandit' in variants else 0,
                'plan_generations': len(bank['tasks']) * 2 if 'self_preference' in variants else 0}
    if counts != expected:
        raise ValueError(f'peer-selection row counts differ: observed={counts}, expected={expected}')
    return {'protocol': 'audited_followup_v2', 'experiment': 'peer_selection',
            'rows': rows, 'plan_generations': generated, 'summary': summary,
            'counts': counts, 'expected_counts': expected}
