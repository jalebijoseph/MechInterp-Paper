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


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def stop_owned(process, timeout=35):
    """Only terminate the session created for this child, never an unrelated server."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def gpu_uuid_bindings(rows, gpu_ids):
    inventory = {}
    for line in rows.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(',')]
        inventory[int(index)] = (uuid, float(memory))
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('GPU IDs must be unique')
    bindings = {}
    for gpu in gpu_ids:
        uuid, memory = inventory[gpu]
        if not uuid.startswith('GPU-'):
            raise ValueError(f'Invalid GPU UUID for physical index {gpu}')
        if memory >= 256:
            raise RuntimeError(f'GPU {gpu} is occupied ({memory} MiB); refusing to share it')
        bindings[gpu] = uuid
    return bindings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--base-port', type=int, default=18080)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--startup-timeout', type=int, default=2400)
    ap.add_argument('--suite-timeout', type=int, default=172800)
    args = ap.parse_args()
    root = args.root.resolve()
    source = root / 'source'
    state_path = root / 'manager-status.json'
    state = {'started_at': now(), 'status': 'starting', 'pid': os.getpid(), 'servers': []}
    children = []
    logs = []
    state['source_sha'] = '1f97522907c33cdb557163769b599088f8e7936b'
    env = dict(os.environ, HF_HOME=str(root/'hf-cache'), OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    suite = None
    try:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        bindings = gpu_uuid_bindings(rows, gpu_ids)
        for i,gpu in enumerate(gpu_ids):
            port = args.base_port+i
            log = (root/f'server-gpu{gpu}.log').open('a')
            logs.append(log)
            command = [sys.executable, '-m', 'uvicorn', 'jpeers.server:app',
                       '--host', '127.0.0.1', '--port', str(port), '--workers', '1']
            child = subprocess.Popen(command, cwd=source,
                                     env=dict(env, CUDA_VISIBLE_DEVICES=bindings[gpu], JPEERS_MODEL='qwen3-8b',
                                              JPEERS_DEVICE='cuda', JPEERS_DTYPE='bf16'),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            state['servers'].append({'gpu':gpu,'gpu_uuid':bindings[gpu],
                                     'port':port,'pid':child.pid,'command':command})
            write(state_path,state)
        deadline = time.monotonic()+args.startup_timeout
        ready = set()
        while len(ready) != len(children):
            for child, entry in zip(children,state['servers']):
                if child.poll() is not None:
                    raise RuntimeError(f'server on GPU {entry["gpu"]} exited {child.returncode}')
                if entry['port'] in ready:
                    continue
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{entry["port"]}/health', timeout=3) as r:
                        body=json.load(r)
                    if body.get('status') == 'ok':
                        ready.add(entry['port'])
                        entry['ready_at']=now()
                        write(state_path,state)
                except Exception:
                    pass
            if time.monotonic()>deadline:
                raise TimeoutError('server startup timeout')
            time.sleep(2)
        state['status']='running'
        state['suite_started_at']=now()
        urls=[f'http://127.0.0.1:{s["port"]}' for s in state['servers']]
        command=[sys.executable,'scripts/run_cluster_suite.py','--help']
        # This is set after the runner's CLI has been validated in preflight.
        cli=json.loads((root/'suite-command.json').read_text())
        state['suite_command']=cli
        write(state_path,state)
        log=(root/'suite.log').open('a'); logs.append(log)
        suite=subprocess.Popen(cli,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        state['suite_pid']=suite.pid; write(state_path,state)
        code=suite.wait(timeout=args.suite_timeout)
        state['suite_exit_code']=code
        state['status']='completed' if code==0 else 'failed'
    except BaseException as exc:
        state['status']='failed'
        state['error']=repr(exc)
        if suite is not None:
            stop_owned(suite)
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
