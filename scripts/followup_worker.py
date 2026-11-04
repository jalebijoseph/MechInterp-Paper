"""Run the frozen post-audit protocols against externally owned HTTP servers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import traceback

from jpeers.client import EndpointPool
from jpeers.followup.common import atomic_json
from jpeers.harness import RunWriter


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_run(job, summary, turns, directory):
    expected = {'exp1': (96, 32), 'exp2': (1080, 600),
                'narcbench': (40, 40), 'exp3': (1324, 1300)}[job]
    errors = []
    if len(turns) != expected[0]:
        errors.append(f'Expected {expected[0]} saved turns, got {len(turns)}')
    recorded = [r for r in turns if r.get('trace_file')]
    if len(recorded) != expected[1]:
        errors.append(f'Expected {expected[1]} traces, got {len(recorded)}')
    if len({r['trace_file'] for r in recorded}) != len(recorded):
        errors.append('Recorded turns reuse a trace file')
    missing = [r['trace_file'] for r in recorded if not (Path(directory)/r['trace_file']).is_file()]
    if missing:
        errors.append(f'Missing {len(missing)} trace files')
    empty = [r['trace_file'] for r in recorded if (Path(directory)/r['trace_file']).is_file()
             and (Path(directory)/r['trace_file']).stat().st_size == 0]
    if empty:
        errors.append(f'Empty {len(empty)} trace files')
    if job == 'exp1':
        sizes = {'reviews':32,'frozen_notes':16,'transfer_probes':32,'no_peer_probes':16}
    elif job == 'exp2':
        sizes = {'reviews':600,'knowledge_probes':480}
    elif job == 'narcbench':
        sizes = {'runs':40,'rows':120}
    else:
        sizes = {'rows':1300,'plan_generations':24}
        if summary.get('counts') != summary.get('expected_counts'):
            errors.append('Exp3 cell counts differ from protocol')
    for key, count in sizes.items():
        if len(summary.get(key, [])) != count:
            errors.append(f'Expected {count} {key}, got {len(summary.get(key, []))}')
    return errors


def holm_adjust(tests):
    """Keep the five-test family size even when some tests are unavailable."""
    results = {name:{'p':p,'p_holm':None} for name,p in tests.items()}
    ordered = sorted((p,name) for name,p in tests.items() if p is not None)
    previous = 0.
    for index,(p,name) in enumerate(ordered):
        previous = max(previous, min(1., p*(len(tests)-index)))
        results[name]['p_holm'] = previous
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--qwen-port', type=int, required=True)
    ap.add_argument('--grader-port', type=int, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    run_config = json.loads((root/'followup-config.json').read_text())
    pool = EndpointPool({name:{'backend':'http','base_url':f'http://127.0.0.1:{port}',
                               'timeout_s':3600,'retries':2}
                         for name,port in [('qwen',args.qwen_port),('olmo',args.grader_port)]})
    identities = {name:pool.get(name).info() for name in ('qwen','olmo')}
    target_identity=identities['qwen']
    if (target_identity.get('family') != 'qwen' or target_identity.get('n_layers') != 64
            or target_identity.get('hf_id')!='Qwen/Qwen3-32B' or target_identity.get('d_model')!=5120):
        raise RuntimeError('Wrong target model identity')
    if (identities['olmo'].get('revision') != run_config['olmo_revision']
            or identities['olmo'].get('family')!='olmo'
            or identities['olmo'].get('hf_id')!=run_config['olmo_model']
            or identities['olmo'].get('params_b')!=32):
        raise RuntimeError('Wrong independent grader revision')
    atomic_json(root/'endpoint-identities.json', identities)
    cfg = {'agents':{'research':'qwen','implementation':'qwen','reviewer':'qwen'},
           'orchestrator':'qwen','orchestrator_family':'qwen','seed':0,
           'narcbench':{'frozen_inputs':str(root/'narcbench-frozen.json')},
           'self_preference':{'own':{'endpoint':'qwen'},'other':{'endpoint':'olmo'}},
           'n_seeds':8,'rounds':40,'max_new_tokens':192}
    lock = threading.Lock()
    state = {'protocol':'audited-followup-v2','status':'running','started_at':now(),
             'exploratory':True,'original_exp0_gate':'failed; not changed by follow-up',
             'config':cfg,'identities':identities,'jobs':{}}
    atomic_json(root/'worker-status.json', state)

    def update(name, value):
        with lock:
            state['jobs'][name] = value
            atomic_json(root/'worker-status.json', state)

    def job(name, action):
        prior = state['jobs'].get(name,{})
        update(name, {'status':'running','started_at':now(),**prior})
        try:
            result = action()
            status = 'completed' if result.get('status','completed') in ('complete','completed') else 'incomplete'
            update(name, {'status':status,'finished_at':now(), 'summary':result})
            return result
        except Exception as exc:
            update(name, {'status':'failed','finished_at':now(),'error':repr(exc),
                          'traceback':traceback.format_exc()})
            return None

    def target_run(name, function):
        writer = RunWriter(root/'results', name, 'followup-v2', config=cfg)
        result = function(cfg, pool, writer)
        turns = [json.loads(line) for line in (writer.dir/'turns.jsonl').read_text().splitlines()]
        errors = validate_run(name, result, turns, writer.dir)
        atomic_json(writer.dir/'summary.json', result)
        audit = {'status':'completed' if not errors else 'incomplete',
                 'run_dir':str(writer.dir),'summary_path':str(writer.dir/'summary.json'),
                 'n_turns':len(turns),'n_traces':sum(bool(r.get('trace_file')) for r in turns),
                 'validation_errors':errors}
        atomic_json(writer.dir/'validation.json', audit)
        if errors:
            raise ValueError(errors)
        # Keep large raw summaries in their own artifacts, not duplicated in status.
        return audit

    def target_pipeline():
        from jpeers.followup.experiments12 import run_collab, run_imposter, run_narcbench
        from jpeers.followup.peer_selection import run_peer_selection
        for name,function in [('exp1',run_collab),('exp2',run_imposter),
                              ('narcbench',run_narcbench),('exp3',run_peer_selection)]:
            job(name, lambda n=name,f=function:target_run(n,f))

    def old_grading_pipeline():
        from jpeers.followup.rescore import run_calibration, run_rescore
        endpoint = pool.get('olmo')
        info = identities['olmo']
        calibration = job('calibration', lambda:run_calibration(endpoint,root/'scoring/calibration',info,
                                                             cache_dir=root/'scoring/cache'))
        if not calibration or not calibration.get('passed'):
            update('scoring',{'status':'blocked_by_calibration',
                              'reason':'No authoritative semantic scoring until fixed calibration gate passes'})
            return None
        inputs = Path(run_config['predecessor'])/'source/runs/full-suite'
        job('exp2_original_rescore',lambda:run_rescore(inputs,root/'scoring/exp2-original',endpoint,
                    info,calibration,kind='exp2',cache_dir=root/'scoring/cache'))
        job('exp4_rescore',lambda:run_rescore(inputs,root/'scoring/exp4-original',endpoint,
                    info,calibration,kind='exp4',cache_dir=root/'scoring/cache'))
        return calibration

    # Independent grading can proceed while the target executes; each endpoint
    # serializes its own inference. OLMo plan calls share the same pinned model.
    with ThreadPoolExecutor(max_workers=2) as executor:
        target = executor.submit(target_pipeline)
        grades = executor.submit(old_grading_pipeline)
        target.result()
        calibration = grades.result()
    exp2 = state['jobs'].get('exp2',{})
    if calibration and calibration.get('passed') and exp2.get('status') == 'completed':
        from jpeers.followup.rescore import grade_rows
        summary = json.loads(Path(exp2['summary']['summary_path']).read_text())
        job('exp2_v2_rescore',lambda:grade_rows(summary['reviews'],root/'scoring/exp2-v2',pool.get('olmo'),
                    identities['olmo'],calibration,kind='exp2_v2',cache_dir=root/'scoring/cache'))
    def saved(name):
        value=state['jobs'].get(name,{})
        if value.get('status')!='completed':
            return {}
        return json.loads(Path(value['summary']['summary_path']).read_text())
    exp1_summary,exp2_summary,exp3_summary=saved('exp1'),saved('exp2'),saved('exp3')
    exp4_summary=state['jobs'].get('exp4_rescore',{}).get('summary',{})
    tests={
        'exp1_role_framing':exp1_summary.get('primary',{}).get('p'),
        'exp2_intent_interaction':exp2_summary.get('primary',{}).get('p'),
        'exp3_context_roles':exp3_summary.get('summary',{}).get('context',{}).get('primary',{}).get('p'),
        'exp3_anonymous_plan_preference':exp3_summary.get('summary',{}).get('self_preference',{}).get('primary',{}).get('p'),
        'exp4_semantic_final_truth':(exp4_summary.get('primary',{}).get('paired_stats',{}).get('p')
                                    if exp4_summary.get('authoritative') else None)}
    atomic_json(root/'primary-family.json',{'protocol':'audited-followup-v2','exploratory':True,
        'method':'Holm over five frozen primary tests; missing tests retain family size5',
        'tests':holm_adjust(tests)})
    wanted = ('exp1','exp2','narcbench','exp3','calibration','exp2_original_rescore','exp4_rescore','exp2_v2_rescore')
    state['status'] = 'completed' if all(state['jobs'].get(n,{}).get('status') == 'completed' for n in wanted) else 'incomplete'
    state['finished_at'] = now()
    atomic_json(root/'worker-status.json', state)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
