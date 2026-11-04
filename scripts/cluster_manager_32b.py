"""Own experiment servers and guarantee their shutdown after suite completion/failure."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
import fcntl
from queue_exp0_32b import select_idle_gpus, assert_port_free, terminate_process_group


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def select_gpu_pairs(gpu_csv, process_csv):
    _, inventory = select_idle_gpus(gpu_csv, process_csv)
    available = sorted((r for r in inventory if r['memory_used_mib'] < 256
                        and not r['has_compute_process']), key=lambda r: r['index'])
    return [available[i:i+2] for i in range(0, len(available)-1, 2)]


def server_environment(base, pair):
    if len(pair) != 2 or any(r['index'] not in range(4) or not r['uuid'].startswith('GPU-') for r in pair):
        raise ValueError('Only two UUID-pinned physical GPUs among 0-3 are allowed')
    return dict(base, CUDA_VISIBLE_DEVICES=','.join(r['uuid'] for r in pair),
                JPEERS_MODEL='qwen3-32b', JPEERS_DEVICE='auto', JPEERS_DTYPE='bf16')


def stop_owned(process, timeout=35):
    terminate_process_group(process, grace=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-port', type=int, default=18110)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--startup-timeout', type=int, default=2400)
    ap.add_argument('--suite-timeout', type=int, default=172800)
    args = ap.parse_args()
    root = args.root.resolve()
    source = root / 'source'
    root.mkdir(parents=True, exist_ok=True)
    lock=(root/'.manager.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'manager-status.json'
    state = {'started_at': now(), 'status': 'starting', 'pid': os.getpid(), 'servers': []}
    children = []
    logs = []
    state['source_sha'] = '1f97522907c33cdb557163769b599088f8e7936b'
    provenance=json.loads((root/'model-provenance.json').read_text())
    env = dict(os.environ, HF_HOME=provenance['hf_home'], OMP_NUM_THREADS='4',
               PYTHONPATH=str(source/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
               JPEERS_MODEL_PATH=provenance['model_path'],
               JPEERS_MODEL_REVISION=provenance['model_revision'],
               JPEERS_LENS_PATH=provenance['lens_path'],
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    suite = None
    smoke = None
    try:
        gpu_csv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        process_csv = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                              '--format=csv,noheader,nounits'], text=True)
        pairs=select_gpu_pairs(gpu_csv,process_csv)
        state['allocation_snapshot']={'gpus':gpu_csv,'processes':process_csv}
        if not pairs:
            raise RuntimeError('No idle pair among physical GPUs 0-3; refusing to share occupied GPUs')
        for i,pair in enumerate(pairs):
            port=args.base_port+i
            assert_port_free(port)
        for i,pair in enumerate(pairs):
            port = args.base_port+i
            replica_root=root/'replicas'/f'replica-{i:02d}'
            replica_root.mkdir(parents=True,exist_ok=True)
            write(replica_root/'model-provenance.json',provenance)
            log = (root/f'server-replica-{i:02d}.log').open('a')
            logs.append(log)
            command = [sys.executable, str(source/'scripts/serve_sharded.py'),
                       '--root', str(replica_root), '--port', str(port)]
            child = subprocess.Popen(command, cwd=source, env=server_environment(env,pair),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            state['servers'].append({'gpus':[r['index'] for r in pair],
                                     'gpu_uuids':[r['uuid'] for r in pair],
                                     'port':port,'pid':child.pid,'command':command})
            write(state_path,state)
        deadline = time.monotonic()+args.startup_timeout
        ready = set()
        while len(ready) != len(children):
            for child, entry in zip(children,state['servers']):
                if child.poll() is not None:
                    raise RuntimeError(f'server on GPU {entry["gpus"]} exited {child.returncode}')
                if entry['port'] in ready:
                    continue
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{entry["port"]}/health', timeout=3) as r:
                        body=json.load(r)
                    if body.get('status') == 'ok':
                        with urllib.request.urlopen(f'http://127.0.0.1:{entry["port"]}/info', timeout=3) as r:
                            info=json.load(r)
                        if info.get('model')!='qwen3-32b' or info.get('n_layers')!=64 or info.get('d_model')!=5120:
                            raise ValueError(f'Wrong endpoint identity: {info}')
                        entry['info']=info
                        ready.add(entry['port'])
                        entry['ready_at']=now()
                        write(state_path,state)
                except urllib.error.HTTPError as exc:
                    body=json.load(exc)
                    if body.get('status')=='error':
                        raise RuntimeError(f'Permanent model startup failure: {body}')
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
            if time.monotonic()>deadline:
                raise TimeoutError('server startup timeout')
            time.sleep(2)
        state['status']='smoke_testing'
        write(state_path,state)
        smoke_command=[sys.executable,str(source/'scripts/smoke_32b_suite.py'),'--ports',
                       *[str(s['port']) for s in state['servers']], '--output',str(root/'smoke')]
        log=(root/'smoke.log').open('a');logs.append(log)
        smoke=subprocess.Popen(smoke_command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['smoke_pid']=smoke.pid;write(state_path,state)
        if smoke.wait(timeout=3600)!=0:
            raise RuntimeError('Sharded intervention smoke checks failed; see smoke.log')
        state['smoke_completed_at']=now()
        cli=json.loads((root/'suite-command.json').read_text())
        port_at=cli.index('--ports')+1
        output_at=cli.index('--output-dir')
        cli[port_at:output_at]=[str(s['port']) for s in state['servers']]
        state['status']='running'
        state['suite_started_at']=now()
        state['suite_command']=cli
        write(state_path,state)
        log=(root/'suite.log').open('a'); logs.append(log)
        suite=subprocess.Popen(cli,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        state['suite_pid']=suite.pid; write(state_path,state)
        deadline=time.monotonic()+args.suite_timeout
        while suite.poll() is None:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('An owned inference server exited during the suite')
            if time.monotonic()>deadline:
                raise TimeoutError('Suite exceeded its explicit timeout')
            time.sleep(5)
        code=suite.returncode
        state['suite_exit_code']=code
        state['status']='completed' if code==0 else 'failed'
    except BaseException as exc:
        state['status']='failed'
        state['error']=repr(exc)
        if suite is not None:
            stop_owned(suite)
        if smoke is not None:
            stop_owned(smoke)
    finally:
        for child in reversed(children):
            stop_owned(child)
        for entry,child in zip(state['servers'],children):
            entry['exit_code']=child.returncode
        state['servers_stopped_at']=now()
        state['finished_at']=now()
        write(state_path,state)
        for log in logs:
            log.close()
    return 0 if state['status']=='completed' else 1

if __name__=='__main__':
    raise SystemExit(main())
