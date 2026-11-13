"""Validate pinned tokenizers and scorer context limits without allocating GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt_length(tokenizer,messages,**kwargs):
    # Transformers5 may return a BatchEncoding from tokenize=True; len() then
    # counts mapping keys. Match each server's render-then-tokenize policy.
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
    ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
    if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('Expected a nonempty flat prompt token list')
    return len(ids)


def main():
    from transformers import AutoTokenizer
    from jpeers.benchmarks import load_task_bank
    from jpeers.followup.common import atomic_json
    from jpeers.followup.experiments12 import (STRICT_INTENT_WORDS,FACTUAL_ERROR_WORDS,
                                              load_frozen_narcbench,author_spans)
    from jpeers.followup.peer_selection import NAME_PAIRS,context_cells,context_messages
    from jpeers.followup.scoring import build_fact_judge_messages
    from jpeers.runtime import whole_word_ids,_check_no_collisions

    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    cfg=json.loads((root/'followup-config.json').read_text())
    predecessor=Path(cfg['predecessor'])
    qwen=json.loads((predecessor/'model-provenance.json').read_text())
    olmo=json.loads((root/'olmo-provenance.json').read_text())
    tok=AutoTokenizer.from_pretrained(qwen['model_path'],revision=qwen['model_revision'],local_files_only=True)
    judge_tok=AutoTokenizer.from_pretrained(olmo['model_path'],revision=olmo['revision'],local_files_only=True)
    watched={}
    for task in load_task_bank('collab_tasks')['tasks']:
        watched['collab/'+task['id']]=list(dict.fromkeys(task['research']['keywords']+task['implementation']['keywords']))
    facts=load_task_bank('imposter_facts')
    for item in facts['items']:
        watched['imposter/'+item['id']]=list(dict.fromkeys([*STRICT_INTENT_WORDS,*FACTUAL_ERROR_WORDS,
            *facts['suspicion_words'],*facts['agreement_words'],item['true'],item['false']]))
    for names in NAME_PAIRS:
        watched['names/'+str(names)]=list(names)
    bank=load_task_bank('peer_selection')
    watched['family']=list(bank['family_variant']['names'])
    watched['plans']=['A','B']
    watched['narc']=[*STRICT_INTENT_WORDS,*FACTUAL_ERROR_WORDS]
    vocab={i:tok.decode([i]) for ids in watched.values() for word in ids for i in whole_word_ids(tok,word)}
    missing=[]
    mappings={}
    for key,words in watched.items():
        ids=[whole_word_ids(tok,word) for word in words]
        missing.extend((key,word) for word,word_ids in zip(words,ids) if not word_ids)
        _check_no_collisions(words,ids,vocab)
        mappings[key]=dict(zip(words,ids))
    if missing:
        raise ValueError(f'Follow-up words lack whole-word tokens: {missing}')
    narc,content=load_frozen_narcbench({'frozen_inputs':str(root/'narcbench-frozen.json')})
    lengths=[]
    for scenario in narc:
        for p in scenario['prompts']:
            if p['phase']!='final' or p['agent_name'] not in scenario['agents'][2:]:
                continue
            author_spans(p['user_prompt'],scenario['public_messages'],
                         [n for n in scenario['agents'] if n!=p['agent_name']])
            messages=[{'role':'system','content':p['system_prompt']},{'role':'user','content':p['user_prompt']}]
            lengths.append(prompt_length(tok,messages,enable_thinking=False))
    if len(lengths)!=40 or max(lengths)>4096:
        raise ValueError('NARC prompt would exceed complete recording cap')
    context_lengths=[prompt_length(tok,context_messages(cell,bank),enable_thinking=False)
                     for cell in context_cells(bank)]
    fixture=json.loads((root/'source/src/jpeers/data/fact_grader_calibration_v2.json').read_text())
    grade_lengths=[prompt_length(judge_tok,build_fact_judge_messages(case['item'],case['response']))
                   for case in fixture['cases']]
    if not grade_lengths or max(grade_lengths)+384>4096:
        raise ValueError('Calibration would exceed independent grader context')
    atomic_json(root/'preflight.json',{'status':'passed','n_watch_lists':len(watched),
        'watch_token_ids':mappings,'narc_content_sha256':content,'narc_lengths':lengths,
        'context_lengths':context_lengths,'grader_calibration_lengths':grade_lengths,
        'grader_output_budget':384,'qwen_revision':qwen['model_revision'],'olmo_revision':olmo['revision']})
    print(json.dumps({'status':'passed','watch_lists':len(watched),'narc_max':max(lengths),
                      'context_max':max(context_lengths),'grader_calibration_max':max(grade_lengths)}))


if __name__=='__main__':
    main()
