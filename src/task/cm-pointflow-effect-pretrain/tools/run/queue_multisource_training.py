"""Wait for bounded owned conversions, freeze corpus, check two GPUs, then fit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    part = path.with_suffix('.json.part'); part.write_text(json.dumps(value, indent=2)+'\n'); part.replace(path)


def read_status(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--contactpose', type=Path, required=True)
    p.add_argument('--native-pid', type=int, required=True)
    p.add_argument('--contactpose-pid', type=int, required=True)
    a = p.parse_args()
    outputs = ROOT/'outputs/cm-pointflow-effect-pretrain'
    output = a.output.resolve()
    output.relative_to(outputs.resolve())
    if output.exists(): raise FileExistsError(output)
    for path, pid, script in [(a.native, a.native_pid, 'prepare_native_full.py'),
                              (a.contactpose, a.contactpose_pid, 'prepare_contactpose_wm30.py')]:
        path.resolve().relative_to(outputs.resolve())
        cmdline = Path('/proc')/str(pid)/'cmdline'
        complete = path/'processed/manifest.json'
        if not (cmdline.is_file() and script in cmdline.read_bytes().decode()) and not (
                complete.exists() and json.loads(complete.read_text()).get('status') == 'COMPLETED'):
            raise ValueError('not a live owned converter or completed input: '+script)
    output.mkdir()
    stats = outputs/'pointworld-temporal-wm24-20261007/norm_stats.json'
    parent = outputs/'pointworld-action-ddp-20261007/train-action/latest.pt'
    config = TASK/'configs/pointworld_multisource_ddp2_wm24.json'
    mixed = outputs/'mixed-wm30-20261007'
    fit = outputs/'pointworld-multisource-20261007'
    check = outputs/'pointworld-multisource-full-check-20261007'
    files = [Path(__file__), config, stats, parent,
             *sorted((TASK/'src/oakink_wm').glob('*.py')),
             *[TASK/'tools/run'/name for name in ('prepare_mixed_manifest.py',
                 'train_oakink2_pointworld_ddp.py', 'train_oakink2_pointworld_temporal.py',
                 'evaluate_oakink2_pointworld_temporal.py', 'launch_pointworld_action_ddp.py')],
             *[f for f in (ROOT/'third_party/PointWorld/ptv3').rglob('*') if f.suffix in ('.py', '.yaml')]]
    frozen = {str(f):digest(f) for f in files}
    manifest = dict(status='WAITING_FOR_DATA', pid=os.getpid(), gpus=[1, 2],
                    native_pid=a.native_pid, contactpose_pid=a.contactpose_pid,
                    sources=frozen, started_at=time.time(), wait_seconds=3600,
                    deadline=1791424717.7631629, parent_checkpoint=str(parent),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    state_path = output/'run_manifest.json'; write(state_path, manifest)
    started = time.monotonic(); stop = [False]; process = None
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_:stop.__setitem__(0, True))
    env = dict(os.environ, TMPDIR=str(ROOT/'tmp'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
               TRITON_CACHE_DIR=str(ROOT/'tmp/triton-multisource-20261007'))
    try:
        while True:
            ready = True
            for root, pid in [(a.native, a.native_pid), (a.contactpose, a.contactpose_pid)]:
                path = root/'processed/manifest.json'
                record = read_status(path)
                if record.get('status') in ('FAILED', 'TIMED_OUT'):
                    raise RuntimeError('converter failed: '+str(root)+': '+str(record.get('error')))
                completed = record.get('status') == 'COMPLETED' and record.get('training_allowed') is True
                if not completed and not (Path('/proc')/str(pid)/'cmdline').exists():
                    raise RuntimeError('converter exited without a ready corpus: '+str(root))
                ready &= completed
            if ready: break
            if stop[0] or time.monotonic()-started > 3600: raise TimeoutError('stopped or data wait deadline')
            time.sleep(20)
        for name, expected in frozen.items():
            if digest(name) != expected: raise RuntimeError('queued implementation/input drift: '+name)
        manifest['status']='FREEZING_CORPUS'; write(state_path, manifest)
        command = [sys.executable,str(TASK/'tools/run/prepare_mixed_manifest.py'),
                   '--oakink2',str(outputs/'oakink2-wm30-k24-20261006'), '--grab',str(a.native.resolve()),
                   '--arctic',str(a.native.resolve()), '--contactpose',str(a.contactpose.resolve()),
                   '--stats',str(stats), '--output',str(mixed)]
        with (output/'freeze.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        spec = importlib.util
        module_spec = spec.spec_from_file_location('owned_launcher',TASK/'tools/run/launch_pointworld_action_ddp.py')
        launcher = spec.module_from_spec(module_spec); module_spec.loader.exec_module(launcher)
        if launcher.gpu_processes([1, 2]): raise RuntimeError('assigned GPUs occupied before engineering check')
        manifest['status']='GPU_ENGINEERING_CHECK'; write(state_path, manifest)
        env['CUDA_VISIBLE_DEVICES']='1,2'
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                   str(TASK/'tools/run/train_oakink2_pointworld_ddp.py'), '--data',str(mixed), '--mixed-data',
                   '--config',str(config), '--stats',str(stats), '--arm','action','--init-weights',str(parent),
                   '--fused-hilbert','--steps','12','--engineering','--output',str(check)]
        with (output/'engineering.log').open('w') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest['child_pid']=process.pid; write(state_path, manifest)
            before = time.monotonic()
            while process.poll() is None:
                if stop[0] or time.monotonic()-before > 600: raise TimeoutError('engineering check stopped/deadline')
                if any(not launcher.belongs_to(pid,process.pid) for pid in launcher.gpu_processes([1, 2])):
                    raise RuntimeError('foreign GPU process during engineering check')
                time.sleep(10)
            if process.returncode: raise RuntimeError('mixed GPU check failed; inspect engineering.log')
        result = json.loads((check/'result.json').read_text())
        if result.get('status') != 'COMPLETED' or not result.get('rank_parameters_identical'):
            raise RuntimeError('mixed engineering check did not qualify')
        manifest['status']='LAUNCHING_TRAINING'; write(state_path, manifest)
        command = [sys.executable,str(TASK/'tools/run/launch_pointworld_action_ddp.py'),
                   '--data',str(mixed),'--mixed-data','--stats',str(stats),'--config',str(config),
                   '--init-weights',str(parent),'--gpus','1,2','--deadline',str(manifest['deadline']),
                   '--output',str(fit)]
        with (output/'launcher.log').open('w') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest.update(status='TRAINING',child_pid=process.pid,training_output=str(fit)); write(state_path,manifest)
            while process.poll() is None:
                if stop[0]:
                    os.kill(process.pid,signal.SIGUSR1)
                    raise KeyboardInterrupt('user stopped queued fit; native launcher requested checkpoint save')
                rows = subprocess.check_output(['nvidia-smi','-i','1,2','--query-gpu=index,memory.used,utilization.gpu',
                                                '--format=csv,noheader,nounits'],text=True)
                with (output/'gpu_usage.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),rows=rows.strip().splitlines()))+'\n')
                time.sleep(30)
            status=json.loads((fit/'group_status.json').read_text())
            manifest.update(status=status['status'],exit_code=process.returncode)
            if process.returncode: raise RuntimeError('mixed training launcher failed')
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error))
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
        raise
    finally:
        manifest['elapsed_s']=time.monotonic()-started; write(state_path,manifest)


if __name__ == '__main__': main()
