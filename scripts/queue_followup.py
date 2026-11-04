"""Wait for the original suite, then own and clean up the two follow-up servers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from queue_exp0_32b import (assert_port_free, atomic_json, select_idle_gpus,
                           terminate_process_group)

TERMINAL = {'completed','failed','incomplete','interrupted','cancelled','canceled','error'}


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessor_released(root):
    root = Path(root)
    try:
        manager = json.loads((root/'manager-status.json').read_text())
        manifest = json.loads((root/'source/runs/full-suite/manifest.json').read_text())
    except (OSError, ValueError) as exc:
        return False, {'reason':'Prerequisite state unavailable','error':str(exc)}
    released = (manager.get('status') in TERMINAL and manifest.get('status') in TERMINAL
                and bool(manager.get('servers_stopped_at')))
    return released, {'manager_status':manager.get('status'),'manifest_status':manifest.get('status'),
                      'servers_stopped_at':manager.get('servers_stopped_at')}


def allowed_pairs(gpu_csv, process_csv):
    _, rows = select_idle_gpus(gpu_csv, process_csv)
    idle = sorted((r for r in rows if not r['has_compute_process'] and r['memory_used_mib']<256),
                  key=lambda r:r['index'])
    # Two BF16 32B models require all four allowed devices. Never borrow4–7.
    if [r['index'] for r in idle] != [0,1,2,3]:
        return []
    return [idle[:2],idle[2:]]


def gpu_pairs():
    def query(fields,kind):
        return subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}',
                                        '--format=csv,noheader,nounits'],text=True,timeout=30)
    return allowed_pairs(query('index,uuid,memory.used','gpu'),query('gpu_uuid,pid','compute-apps'))


def get_json(url):
    try:
        with urllib.request.urlopen(url,timeout=10) as result:
            return json.load(result)
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            raise
        # Both loading and permanent failure use503; preserve their health body
        # so the controller can stop an unsuccessful load immediately.
        with exc:
            return json.load(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--predecessor',type=Path,required=True)
    ap.add_argument('--base-port',type=int,default=18210)
    ap.add_argument('--wait-timeout',type=int,default=7*86400)
    ap.add_argument('--run-timeout',type=int,default=7*86400)
    args = ap.parse_args()
    root = args.root.resolve()
    source = root/'source'
    root.mkdir(parents=True,exist_ok=True)
    manager_lock = (root/'.followup-manager.lock').open('w')
    fcntl.flock(manager_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'worker-status.json').exists():
        raise RuntimeError('A follow-up attempt already exists; use a new attempt root to preserve results')
    state = {'status':'waiting','started_at':now(),'pid':os.getpid(),'servers':[],
             'protocol':'audited-followup-v2','predecessor':str(args.predecessor)}
    status_path = root/'manager-status.json'
    children,logs = [],[]
    worker = None

    def update(**values):
        state.update(values)
        atomic_json(status_path,state)

    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Received signal{signum}')

    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        config = json.loads((root/'followup-config.json').read_text())
        if config['predecessor'] != str(args.predecessor.resolve()):
            raise RuntimeError('Predecessor differs from the frozen follow-up config')
        if json.loads((root/'preflight.json').read_text()).get('status')!='passed':
            raise RuntimeError('CPU preflight has not passed')
        update()
        deadline = time.monotonic()+args.wait_timeout
        while True:
            released,prior = predecessor_released(args.predecessor)
            try:
                olmo = json.loads((root/'olmo-provenance.json').read_text())
            except (OSError,ValueError):
                olmo = {'status':'unavailable'}
            pairs = gpu_pairs() if released and olmo.get('status')=='completed' else []
            update(checked_at=now(),predecessor_state=prior,model_download=olmo.get('status'),
                   reason='Waiting for original server shutdown, pinned model download, and idle physical GPUs0–3')
            if pairs:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Follow-up prerequisite wait expired')
            time.sleep(20)
        qwen = json.loads((args.predecessor/'model-provenance.json').read_text())
        if qwen['model_revision']!='9216db5781bf21249d130ec9da846c4624c16137':
            raise RuntimeError('Unexpected Qwen model revision')
        for name,expected in config['source_file_sha256'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Frozen follow-up source changed: {name}')
        if olmo['revision'] != config['olmo_revision']:
            raise RuntimeError('Independent model revision differs from frozen config')
        for port in [args.base_port,args.base_port+1]:
            assert_port_free(port)
        base_env = dict(os.environ,HF_HOME=qwen['hf_home'],OMP_NUM_THREADS='4',
                        PYTHONPATH=str(source/'src'),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        replica = root/'qwen-server'
        replica.mkdir()
        atomic_json(replica/'model-provenance.json',qwen)
        commands = [
            [sys.executable,str(source/'scripts/serve_sharded.py'),'--root',str(replica),
             '--port',str(args.base_port)],
            [sys.executable,str(source/'scripts/serve_plain_hf.py'),'--model-id',olmo['model'],
             '--revision',olmo['revision'],'--model-path',olmo['model_path'],
             '--name','olmo2-32b','--family','olmo','--params-b','32',
             '--port',str(args.base_port+1),'--provenance-output',str(root/'olmo-server.json')]]
        for index,(pair,command) in enumerate(zip(pairs,commands)):
            env = dict(base_env,CUDA_VISIBLE_DEVICES=','.join(r['uuid'] for r in pair))
            if index==0:
                env.update(JPEERS_MODEL_PATH=qwen['model_path'],JPEERS_MODEL_REVISION=qwen['model_revision'],
                           JPEERS_LENS_PATH=qwen['lens_path'],JPEERS_MODEL='qwen3-32b')
            handle = (root/f'server-{index}.log').open('a');logs.append(handle)
            process = subprocess.Popen(command,cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT,
                                       start_new_session=True)
            children.append(process)
            state['servers'].append({'pid':process.pid,'port':args.base_port+index,
                'gpus':[r['index'] for r in pair],'gpu_uuids':[r['uuid'] for r in pair],'command':command})
            update(status='starting')
        deadline = time.monotonic()+3600
        while True:
            ready = []
            for process,server in zip(children,state['servers']):
                if process.poll() is not None:
                    raise RuntimeError(f'Owned server exited: {server["port"]}, code{process.returncode}')
                try:
                    health = get_json(f'http://127.0.0.1:{server["port"]}/health')
                    if health.get('status')=='error':
                        raise RuntimeError(f'Owned server startup failed:{health}')
                    if health.get('status')=='ok':
                        server['info']=get_json(f'http://127.0.0.1:{server["port"]}/info')
                        ready.append(server['port'])
                except urllib.error.HTTPError as exc:
                    if exc.code not in (502,503):
                        raise
                except (urllib.error.URLError,TimeoutError):
                    pass
            update(ready_ports=ready)
            if len(ready)==2:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Model startup expired')
            time.sleep(3)
        command = [sys.executable,str(source/'scripts/followup_worker.py'),'--root',str(root),
                   '--qwen-port',str(args.base_port),'--grader-port',str(args.base_port+1)]
        handle = (root/'worker.log').open('a');logs.append(handle)
        worker = subprocess.Popen(command,cwd=source,env=dict(base_env,CUDA_VISIBLE_DEVICES=''),
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        update(status='running',worker_pid=worker.pid,worker_started_at=now())
        deadline = time.monotonic()+args.run_timeout
        while worker.poll() is None:
            if any(p.poll() is not None for p in children):
                raise RuntimeError('Owned inference server exited during follow-up')
            if time.monotonic()>deadline:
                raise TimeoutError('Follow-up run exceeded its explicit timeout')
            time.sleep(5)
        result = json.loads((root/'worker-status.json').read_text())
        update(status='completed' if worker.returncode==0 and result['status']=='completed' else 'incomplete',
               worker_exit_code=worker.returncode)
    except BaseException as exc:
        update(status='failed',error=repr(exc))
    finally:
        terminate_process_group(worker)
        for process in reversed(children):
            terminate_process_group(process,grace=35)
        for entry,process in zip(state['servers'],children):
            entry['exit_code']=process.returncode
        update(finished_at=now(),servers_stopped_at=now())
        for handle in logs:
            handle.close()
    return 0 if state['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
