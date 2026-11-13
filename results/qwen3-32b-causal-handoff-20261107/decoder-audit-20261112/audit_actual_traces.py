#!/usr/bin/env python3
"""Read-only CPU audit: exact original development traces, no model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import argparse, json, sys, hashlib, dataclasses
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);args=p.parse_args()
sys.path.insert(0,str(Path(args.source)/'src'))
from jpeers.followup import handoff as h
from jpeers.trace import JSpaceTrace
root=Path(args.run); rows=[r for r in h.make_episodes() if r.split=='development']; traces=[]; receipt_checks=[]
source_hash=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
provenance=json.loads((root/'provenance.json').read_text());assert provenance['source_sha256']['src/jpeers/followup/handoff.py']==source_hash
manifest=json.loads((root/'task_manifest.json').read_text());manifest_keys={r['key']:r for r in manifest['episodes']}
for r in rows:
 key=f'dev:{r.key}:baseline';receipt_path=root/'receipts'/f'{h.digest(key)}.json';rec=json.loads(receipt_path.read_text());claimed=rec.pop('integrity_sha256');assert claimed==h.digest(rec)
 assert rec['request_sha256']==h.digest(rec['request']);assert rec['request']['messages']==r.messages();assert manifest_keys[r.key]['messages_sha256']==h.digest(r.messages())
 trace_path=root/rec['turn_directory']/'traces/00000.npz';t=JSpaceTrace.load(str(trace_path));t2=JSpaceTrace.from_response(rec['response']);assert t.layers==[40,42,48,50,52,54]
 h.validate_capture_site(t)
 for k in ('residuals','directions_jacobian','directions_logit','direction_token_ids','positions'):assert np.array_equal(getattr(t,k),getattr(t2,k))
 for k in ('incoming_correct','own_correct','format_valid'):assert rec['score'][k] is True
 turn=json.loads((root/rec['turn_directory']/'turns.jsonl').read_text());assert turn['tag']==key;assert turn['extra']['episode']['own']==r.own;assert turn['extra']['episode']['incoming']==r.incoming;assert turn['extra']['episode']['source']==r.source;assert turn['messages']==r.messages()
 traces.append(t);receipt_checks.append({'episode_key':r.key,'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'position':int(t.positions[0]),'prompt_len':t.prompt_len})
assert len(traces)==256
z=h._nuisance(rows);y=h._signed([r.incoming for r in rows]);own=h._signed([r.own for r in rows]);tids=np.array([r.template_id for r in rows]);templates=sorted(set(tids));ridge=1e-6
allstates=np.stack([t.residuals[0] for t in traces]);dirs_j=traces[0].directions_jacobian;dirs_l=traces[0].directions_logit
for t in traces:assert np.allclose(t.directions_jacobian,dirs_j,atol=1e-5,rtol=1e-5);assert np.allclose(t.directions_logit,dirs_l,atol=1e-5,rtol=1e-5)
np.savez_compressed(args.dataset,states=allstates,jacobian=dirs_j,logit=dirs_l,layers=np.array(traces[0].layers),z=z,y=y,own=own,templates=tids,row_metadata=np.array(json.dumps([dataclasses.asdict(r) for r in rows])),source_sha256=np.array(source_hash))
print('Verified 256 development receipts, prompts, labels, source, NPZ equality and capture sites; wrote compact dataset',flush=True)
def scores(a,y,b,mode='original'):
 if mode=='original':
  fit=h._ridge(np.column_stack([np.ones(len(a)),a]),y,ridge);return np.column_stack([np.ones(len(b)),b])@fit
 mu=a.mean();ym=y.mean();ac=a-mu;sxx=ac@ac;sxy=ac@(y-ym);n=len(a)
 if mode=='center_exact':
  beta=(sxy+ridge*mu*y.sum()/(n+ridge))/(sxx+ridge+ridge*mu*mu*n/(n+ridge));center=(y.sum()+ridge*mu*beta)/(n+ridge)
 else:beta=sxy/(sxx+ridge);center=ym
 return center+beta*(b-mu)
def augment(a,b):
 return np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(a.shape[1])]),np.concatenate([b,np.zeros((a.shape[1],)+b.shape[1:])]),rcond=None)[0]
def gramcond(x):
 a=np.column_stack([np.ones(len(x)),x]);return float(np.linalg.cond(a.T@a+ridge*np.eye(2)))
def correctness(trainx,trainy,testx,testy,mode='center_exact'):return ((scores(trainx,trainy,testx,mode)>=0)==(testy>0)).tolist()
out={'run':str(root),'source_sha256':source_hash,'n_contexts':256,'folds':templates,'alignment_checks':{'passed':True,'n_receipts':256,'receipt_integrity_request_hash_prompt_manifest_turn_labels':True,'npz_receipt_arrays_exact':True,'fixed_directions':True,'capture_site_validation':True},'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),'scope':'development only; original 8 leave-template-out folds; fixed declared ridge; diagnostic fixed-alpha1 raw and nuisance-adjusted probes do not replace frozen assay; all centering uses training rows; no scaling, hyperparameter selection or heldout data','encoding_gram_condition':[],'decoders':{}}
for template in templates:
 tr=tids!=template;a=np.column_stack([z[tr],y[tr]]);out['encoding_gram_condition'].append(float(np.linalg.cond(a.T@a)))
saved=json.loads((root/'discovery.json').read_text())['decoders']
for li,layer in enumerate(traces[0].layers):
 states=allstates[:,li].astype(float);out['decoders'][str(layer)]={}
 for lens in ('jacobian','logit','residual_contrast'):
  dirs=None if lens=='residual_contrast' else (dirs_j if lens=='jacobian' else dirs_l)[li];span=None if dirs is None else h._row_span(dirs)
  original=h.fit_component(rows,states,dirs,ridge=ridge);old=saved[str(layer)][lens]
  report={'original':{k:v for k,v in original.items() if k not in ('basis','folds')},'saved_metric_max_abs_difference':max(abs(original[k]-old[k]) for k in ['incoming_cv','own_cv','incoming_basis_own_cv','incoming_basis_nuisance_cv']),'saved_full_basis_max_abs_difference':float(np.max(np.abs(np.asarray(original['basis'])-np.asarray(old['basis'])))),'saved_fold_hash_matches':sum(a['basis_sha256']==b['basis_sha256'] for a,b in zip(original['folds'],old['folds'])),'variants':{},'fold_diagnostics':[]}
  cols={mode:{k:[] for k in ('incoming','own','raw_own','raw_source','raw_names','raw_order','adjusted_own')} for mode in ('original','center_exact','center_unpenalized')}
  pc={variant:{k:[] for k in ('incoming','raw_own','raw_source','raw_names','raw_order')} for variant in ('raw_alpha1','adjusted_alpha1')}
  for template in templates:
   tr=tids!=template;te=~tr;trainrows=[r for r,m in zip(rows,tr) if m];b,nuis=h._learn(states[tr],trainrows,span,ridge);a=np.column_stack([z[tr],y[tr]]);cs=augment(a,states[tr]);cb=cs[-1]
   if span is not None:cb=(cb@span.T)@span
   bs=cb/np.linalg.norm(cb) if np.linalg.norm(cb)>1e-9 else np.zeros(states.shape[1]);rt=states[tr]@b;rv=states[te]@b;at=(states[tr]-z[tr]@nuis)@b;av=(states[te]-z[te]@nuis)@b
   ast=(states[tr]-z[tr]@cs[:-1])@bs;asv=(states[te]-z[te]@cs[:-1])@bs
   od=np.column_stack([z[tr][:,[0,2,3,4]],y[tr],own[tr]]);ob=h._ridge(od,states[tr],ridge)[-1]
   if span is not None:ob=(ob@span.T)@span
   ot=states[tr]@ob;ov=states[te]@ob
   diag={'template':template,'contrast_basis_stable_max_abs':float(abs(b-bs).max()),'augmented_stable_incoming_prediction_disagreements':int(np.count_nonzero((scores(at,y[tr],av)>=0)!=(scores(ast,y[tr],asv,'center_exact')>=0))),'raw_scalar_mean':float(rt.mean()),'raw_scalar_std':float(rt.std()),'adjusted_scalar_mean':float(at.mean()),'adjusted_scalar_std':float(at.std()),'own_scalar_mean':float(ot.mean()),'own_scalar_std':float(ot.std()),'raw_gram_condition':gramcond(rt),'adjusted_gram_condition':gramcond(at),'own_gram_condition':gramcond(ot)}
   for mode,col in cols.items():
    for key,tx,ty,vx,vy in [('incoming',at,y[tr],av,y[te]),('own',ot,own[tr],ov,own[te]),('raw_own',rt,own[tr],rv,own[te]),('raw_source',rt,z[tr,2],rv,z[te,2]),('raw_names',rt,z[tr,3],rv,z[te,3]),('raw_order',rt,z[tr,4],rv,z[te,4]),('adjusted_own',at,own[tr],av,own[te])]:col[key].extend(correctness(tx,ty,vx,vy,mode))
   for key,tx,ty,vx in [('incoming',at,y[tr],av),('own',ot,own[tr],ov),('raw_own',rt,own[tr],rv)]:
    s0=scores(tx,ty,vx);sx=scores(tx,ty,vx,'center_exact');diag[key+'_stable_score_max_abs']=float(abs(s0-sx).max());diag[key+'_stable_prediction_disagreements']=int(np.count_nonzero((s0>=0)!=(sx>=0)))
   for variant in pc:
    xx=states if variant=='raw_alpha1' else states-z@nuis
    if span is not None:xx=xx@span.T
    mu=xx[tr].mean(0);xt=xx[tr]-mu;xv=xx[te]-mu
    # One fixed diagnostic alpha=1 in original orthonormal coordinates; no variance rescaling.
    if xt.shape[1]<=len(xt):w=np.linalg.solve(xt.T@xt+np.eye(xt.shape[1]),xt.T@y[tr])
    else:w=xt.T@np.linalg.solve(xt@xt.T+np.eye(len(xt)),y[tr])
    pc[variant]['incoming'].extend(((xv@w>=0)==(y[te]>0)).tolist());bw=w if span is None else w@span;norm=np.linalg.norm(bw);bw=bw/norm if norm else bw;rawt=states[tr]@bw;rawv=states[te]@bw
    for key,lab in [('raw_own',own),('raw_source',z[:,2]),('raw_names',z[:,3]),('raw_order',z[:,4])]:pc[variant][key].extend(correctness(rawt,lab[tr],rawv,lab[te]))
   report['fold_diagnostics'].append(diag)
  for mode,col in cols.items():report['variants'][mode]={k:float(np.mean(v)) for k,v in col.items()}
  report['probes']={variant:{k:float(np.mean(v)) for k,v in col.items()} for variant,col in pc.items()}
  for v in report['probes'].values():v['raw_nuisance_max']=max(v[k] for k in ('raw_source','raw_names','raw_order'))
  report['stable_prediction_disagreements_total']=sum(d[k+'_stable_prediction_disagreements'] for d in report['fold_diagnostics'] for k in ('incoming','own','raw_own'))
  out['decoders'][str(layer)][lens]=report
  Path(args.out).write_text(json.dumps(out,indent=2))
  print(layer,lens,'orig',original['incoming_cv'],'stable_disagreements',report['stable_prediction_disagreements_total'],'probes',report['probes'],flush=True)
out['completed']=True;out['receipt_checks']=receipt_checks;Path(args.out).write_text(json.dumps(out,indent=2));print('COMPLETED',args.out,flush=True)
