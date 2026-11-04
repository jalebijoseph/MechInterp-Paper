"""Post-hoc prompt sensitivity diagnostic; does not alter Experiment 0 or its gate."""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import transformers
from jpeers.benchmarks import false_belief_items

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['8b', '32b'], required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--direct-belief', action='store_true')
args = ap.parse_args()
torch.set_num_threads(4)
model_id = 'Qwen/Qwen3-' + args.model.upper()
revision = {'8b': 'b968826d9c46dd6066d109eabc6255188de91218',
            '32b': '9216db5781bf21249d130ec9da846c4624c16137'}[args.model]
tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
kwargs = {'dtype': torch.bfloat16}
if args.model == '32b':
    kwargs.update(device_map='balanced', max_memory={0: '34GiB', 1: '34GiB'})
hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs).eval()
if args.model == '8b':
    hf = hf.to('cuda')
input_device = hf.get_input_embeddings().weight.device
placements = {str(p.device) for p in hf.parameters()}
if 'cpu' in placements:
    raise RuntimeError('Unexpected CPU offload: ' + repr(placements))
variants = ['original', 'no_answer_prefill', 'current_question', 'current_no_prefill', 'world_question']
if args.direct_belief:
    variants = ['direct_belief', 'direct_belief_no_prefill']
rows = []
started = time.time()
metadata = {'model': model_id, 'revision': revision, 'thinking': False,
            'temperature': 0, 'max_new_tokens': 24, 'seed': 0,
            'order': 'reverse original item order, including controls before false belief twins',
            'placements': sorted(placements), 'variants': variants,
            'purpose': 'Post-hoc diagnostic only; original experiment and gate unchanged'}

def save():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'metadata': metadata, 'rows': rows}, indent=2))

items = [it for f in ['people', 'agents'] for it in false_belief_items(20, framing=f)]
for variant in variants:
    for it in reversed(items):
        question = it.question
        if variant in ['current_question', 'current_no_prefill']:
            question = 'After all the events described, ' + question[:1].lower() + question[1:]
        if variant == 'world_question':
            question = ('Where is the file actually located now?' if it.framing == 'agents'
                        else 'Where is the object actually located now?') if it.kind == 'transfer' else 'What is actually inside?'
            question += ' Answer with one word.'
        if variant.startswith('direct_belief'):
            if it.kind == 'transfer':
                if it.framing == 'agents':
                    obj = re.search(r'to find the (\w+) file', it.question).group(1)
                    question = f'After all the events in the story, which folder does {it.protagonist} currently believe contains the {obj} file?'
                else:
                    obj = re.search(r'look for the (\w+) first', it.question).group(1)
                    question = f'After all the events in the story, where does {it.protagonist} currently believe the {obj} is located?'
            else:
                container = 'file' if it.framing == 'agents' else re.search(r'Here is a (\w+)', it.story).group(1)
                question = f'After all the events in the story, what does {it.protagonist} currently believe is inside the {container}?'
            question += ' Answer with one word.'
        messages = [{'role': 'system', 'content': 'Read the story and answer the question with a single word.'},
                    {'role': 'user', 'content': it.story + '\n\nQuestion: ' + question}]
        prefill = '' if variant in ['no_answer_prefill', 'current_no_prefill', 'direct_belief_no_prefill'] else 'Answer:'
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) + prefill
        ids = tok(rendered, add_special_tokens=False, return_tensors='pt').to(input_device)
        with torch.inference_mode():
            output = hf.generate(**ids, do_sample=False, max_new_tokens=24)
        generated = output[0, ids.input_ids.shape[1]:].tolist()
        answer = tok.decode(generated, skip_special_tokens=True)
        expected = it.true_answer if variant == 'world_question' else it.belief_answer
        matches = re.findall(r'\b[a-z]+\b', answer.lower())
        exact_word = len(matches) == 1 and matches[0] == expected
        rows.append({'variant': variant, 'framing': it.framing, 'item': it.id,
                     'kind': it.kind, 'control': it.control, 'expected': expected,
                     'decoy': it.decoy, 'answer': answer, 'exact_word_correct': exact_word,
                     'expected_present': expected in matches,
                     'generated_ids': generated, 'messages': messages, 'prefill': prefill})
        save()
    print('finished', variant, len(rows), 'seconds', round(time.time()-started), flush=True)
metadata['elapsed_seconds'] = time.time()-started
save()
print('DONE', len(rows), flush=True)
