"""Small diagnostic: compare the original runtime with native HF generation and prompt variants."""
import argparse,json,os,time
from pathlib import Path
import torch,transformers
from jpeers.benchmarks import false_belief_items
from jpeers.registry import get_model_spec
from jpeers.runtime import LensedModel
from jpeers.schemas import GenerateRequest,RecordSpec

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
torch.set_num_threads(4)
spec=get_model_spec('qwen3-8b')
tok=transformers.AutoTokenizer.from_pretrained(spec.hf_id,revision='b968826d9c46dd6066d109eabc6255188de91218')
hf=transformers.AutoModelForCausalLM.from_pretrained(spec.hf_id,revision='b968826d9c46dd6066d109eabc6255188de91218',dtype=torch.bfloat16).to('cuda').eval()
rows=[];args.out.parent.mkdir(parents=True,exist_ok=True)
def write():args.out.write_text(json.dumps({'model':spec.hf_id,'rows':rows},indent=2))
def native(messages,prefill='',max_tokens=24):
 text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill
 ids=tok(text,add_special_tokens=False,return_tensors='pt').to('cuda')
 with torch.inference_mode():out=hf.generate(**ids,do_sample=False,max_new_tokens=max_tokens)
 return tok.decode(out[0,ids.input_ids.shape[1]:],skip_special_tokens=True)
def prompt(it,variant):
 story=it.story;question=it.question
 if variant=='current_question':
  question='After all the events described, '+it.question[:1].lower()+it.question[1:]
 if variant=='explicit_observation' and it.control:
  story+=f' {it.protagonist} sees that it is now '+(f'in the {it.true_answer}.' if it.kind=='transfer' else f'filled with {it.true_answer}.')
 if variant=='world_question':
  question=('Where is the file actually located now?' if it.framing=='agents' else 'Where is the object actually located now?') if it.kind=='transfer' else 'What is actually inside?'
  question+=' Answer with one word.'
 if variant=='reason_then_answer':question+=' Explain briefly, then give your final answer.'
 system='Read the story and answer the question with a single word.' if variant!='reason_then_answer' else 'Read the story carefully. Explain briefly and then give a final answer.'
 return [{'role':'system','content':system},{'role':'user','content':story+'\n\nQuestion: '+question}]
# Four matched pairs per framing: two transfer and two contents, no cherry-picking by result.
items=[it for framing in ('people','agents') for it in false_belief_items(4,framing=framing)]
for variant in ['original_native','no_answer_prefill','current_question','explicit_observation','world_question','reason_then_answer']:
 for it in items:
  pref='Answer:' if variant in ['original_native','current_question','explicit_observation','world_question'] else ''
  answer=native(prompt(it,variant),pref,max_tokens=6 if variant=='original_native' else 80 if variant=='reason_then_answer' else 24)
  rows.append({'variant':variant,'framing':it.framing,'item':it.id,'kind':it.kind,'control':it.control,'expected':it.true_answer if variant=='world_question' else it.belief_answer,'answer':answer,'messages':prompt(it,variant),'prefill':pref})
  write()
 print('finished',variant,flush=True)
# Verify wrapping and optional observations on exactly the same native loaded model.
m=LensedModel(spec,hf_model=hf,tokenizer=tok,device='cuda',dtype='bf16')
for enabled in [False,True]:
 for it in items:
  req=GenerateRequest(messages=prompt(it,'original_native'),assistant_prefill='Answer:',max_new_tokens=6,record=RecordSpec(enabled=enabled,watch=[it.belief_answer,it.decoy]))
  answer=m.generate(req).text
  rows.append({'variant':'original_runtime_record_'+str(enabled),'framing':it.framing,'item':it.id,'kind':it.kind,'control':it.control,'expected':it.belief_answer,'answer':answer});write()
 print('finished runtime record',enabled,flush=True)
print('DONE',len(rows),flush=True)
