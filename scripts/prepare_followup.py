"""Freeze inputs and provenance on CPU before launching any follow-up inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from jpeers.benchmarks import load_narcbench
from jpeers.followup.common import atomic_json
from jpeers.followup.experiments12 import NARC_ORIGINAL_SHA256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--predecessor',type=Path,required=True)
    ap.add_argument('--source-base-sha',required=True)
    ap.add_argument('--olmo-revision',required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    if (root/'manager-status.json').exists():
        raise RuntimeError('Preparation cannot change a queued or running protocol')
    scenarios=[asdict(s) for s in load_narcbench('qwen3_32b','core',20)]
    content_hash=hashlib.sha256(json.dumps(scenarios,sort_keys=True,default=str).encode()).hexdigest()
    if content_hash!=NARC_ORIGINAL_SHA256:
        raise ValueError(f'Frozen NARC input differs from original audited input: {content_hash}')
    atomic_json(root/'narcbench-frozen.json',{'scenarios':scenarios,'content_sha256':content_hash})
    source=root/'source'
    files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ['src','scripts','configs','tests','docs/superpowers']
           for p in sorted((source/folder).rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
    atomic_json(root/'followup-config.json',{
        'protocol':'audited-followup-v2','frozen_at':datetime.now(timezone.utc).isoformat(),
        'predecessor':str(args.predecessor.resolve()),'source_base_sha':args.source_base_sha,
        'source_file_sha256':files,'narc_content_sha256':content_hash,
        'qwen_model':'Qwen/Qwen3-32B','qwen_revision':'9216db5781bf21249d130ec9da846c4624c16137',
        'olmo_model':'allenai/OLMo-2-0325-32B-Instruct','olmo_revision':args.olmo_revision,
        'primary_family':['exp1_role_framing','exp2_intent_interaction','exp3_context_roles',
                          'exp3_anonymous_plan_preference','exp4_semantic_final_truth'],
        'multiplicity':'Holm over the five declared primary tests; missing tests retain family size5',
        'original_exp0_gate':'failed; follow-up is exploratory',
        'allowed_physical_gpus':[0,1,2,3],
        'expected_requests':{'exp1':96,'exp2':1080,'narcbench':40,'exp3':1324},
        'expected_traces':{'exp1':32,'exp2':600,'narcbench':40,'exp3':1300},
        'expected_semantic_reviews':{'exp2_original':360,'exp2_v2':600,'exp4_original':4680}})
    print(json.dumps({'narc_content_sha256':content_hash,'source_files':len(files)}))


if __name__=='__main__':
    main()
