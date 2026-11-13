"""Compare native vLLM observations with the original HF experiment runtime."""
import json,os,subprocess,time
from pathlib import Path
import httpx,numpy as np
from jpeers.schemas import GenerateRequest,GenerateResponse,RecordSpec,InterventionSpec,PositionSelector
from jpeers.trace import JSpaceTrace
root=Path(__file__).resolve().parents[2]
reports=[]
request=GenerateRequest(messages=[{'role':'user','content':'Bob says the capital of France is Rome. Check his claim.'}],max_new_tokens=8,record=RecordSpec(watch=['Bob','Paris','Rome'],include_model_logits=True))
def call(req):
 t=time.time();r=httpx.post('http://127.0.0.1:18080/generate',json=req.model_dump(),timeout=600)
 if r.status_code!=200:raise RuntimeError(f'{r.status_code}: {r.text[:1000]}')
 result=GenerateResponse.model_validate(r.json())
 print('vLLM',repr(result.text),'elapsed',time.time()-t,flush=True)
 return result
zero=call(request.model_copy(update={'max_new_tokens':0}))
assert zero.n_generated==0 and zero.text==''
assert len(JSpaceTrace.from_response(zero).positions)==zero.prompt_len
base=call(request)
(root/'vllm-smoke.json').write_text(base.model_dump_json())
plain=call(request.model_copy(update={'record':RecordSpec(enabled=False)}))
assert base.tokens==plain.tokens,'Observation hooks changed greedy generation'
from jpeers.registry import get_model_spec
from jpeers.runtime import LensedModel
hf=LensedModel(get_model_spec('qwen3-8b'),device='cuda',dtype='bf16')
t=time.time(); reference=hf.generate(request); print('HF',repr(reference.text),'elapsed',time.time()-t,flush=True)
(root/'hf-smoke.json').write_text(reference.model_dump_json())
a,b=JSpaceTrace.from_response(base),JSpaceTrace.from_response(reference)
report={'greedy_token_match':base.tokens==reference.tokens,'positions_match':np.array_equal(a.positions,b.positions),'layer_match':a.layers==b.layers,'lens_top1_agreement':float((a.topk_ids[:,:,0]==b.topk_ids[:,:,0]).mean()),'watch_rank_mae':float(np.abs(a.watch_ranks.astype(float)-b.watch_ranks).mean()),'vllm_text':base.text,'hf_text':reference.text}
assert report['positions_match'] and report['layer_match']
assert report['greedy_token_match'], report
assert report['lens_top1_agreement']>.9,report
edited_req=request.model_copy(update={'interventions':[InterventionSpec(kind='ablate_topk',k=10,positions=PositionSelector(scope='generated'))]})
edited=call(edited_req)
assert edited.interventions_applied>0
assert not np.array_equal(JSpaceTrace.from_response(edited).topk_logits,a.topk_logits)
report['intervention_edits']=edited.interventions_applied
report['edited_text']=edited.text
(root/'parity-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
