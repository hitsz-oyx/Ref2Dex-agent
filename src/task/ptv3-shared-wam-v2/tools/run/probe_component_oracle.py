"""Conditionally follow the paired action probe with bounded component and TRAIN diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    out, upstream = a.output.resolve(), a.upstream.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=True)
    if a.detach:
        command = [sys.executable, str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != '--detach']
        with (scratch/'driver.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, REF2DEX_RUN_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1'))
        print(json.dumps(dict(supervisor_pid=child.pid, output=str(out))))
        return
    initial = json.loads((upstream/'process.json').read_text())
    pid = initial['supervisor_pid']
    identity = process_identity(pid)
    if initial['status'] != 'RUNNING' or identity is None:
        raise ValueError('Attach only to a verified live upstream action probe')
    out.mkdir()
    started = time.monotonic()
    status = dict(status='WAITING', supervisor_pid=os.getpid(), upstream_pid=pid,
        upstream_process_start=identity, upstream=str(upstream),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TASK, text=True).strip(),
        code_root=str(TASK.parents[2]), gpus=[6,7])
    def write():
        part = out/'process.json.part'
        part.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        part.replace(out/'process.json')
    write()
    env = dict(os.environ, REF2DEX_RUN_ROOT=str(ROOT))
    card = TASK/'docs/experiments/probes/P-20261015-inspire-component-oracle.md'
    checkpoint = ROOT/'outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1/inputs/control-step7000.pt'
    legacy = TASK.parent/'ptv3-shared-wam'
    def run(name, gpu, sequence, controls, hybrid):
        target = out/name
        command = [sys.executable, str(legacy/'tools/run/launch_native.py'),
            '--model-version', 'v2', '--shared-actor', '--checkpoint', str(checkpoint),
            '--output', str(target), '--gpu', str(gpu), '--controls', str(controls),
            '--execution-steps', '1', '--seed', '148', '--sequence', sequence,
            '--experiment-card', str(card)]
        if hybrid:
            command.append('--diagnostic-hybrid')
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(legacy/'tools/audit/audit_native.py'),
                '--native', str(target/'native')], env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=120)
        result = json.loads((target/'native/result.json').read_text())
        return dict(sequence=sequence, diagnostic_hybrid=hybrid,
            teacher_positive=result['outcomes'][0]['maximum_held_frames'] >= 45, result=result)
    try:
        while True:
            current = json.loads((upstream/'process.json').read_text())
            if current['supervisor_pid'] != pid:
                raise RuntimeError('Upstream identity changed')
            if current['status'] == 'COMPLETED':
                break
            if current['status'] != 'RUNNING' or process_identity(pid) != identity:
                raise RuntimeError('Upstream stopped without successful comparison')
            if time.monotonic()-started > 7200:
                raise TimeoutError('Two-hour bounded upstream wait')
            time.sleep(30)
        comparison = json.loads((upstream/'comparison.json').read_text())
        assert comparison['status'] == 'COMPLETED' and comparison['match']['status'] == 'PASS'
        success = any(o['maximum_held_frames'] >= 45 for rows in comparison['native'].values()
                      for row in rows for o in row['result']['outcomes'][1:])
        if success:
            status.update(status='COMPLETED', skipped='Paired probe already produced student held45; prioritize that signal')
            return
        if shutil.disk_usage(ROOT).free < 20*1024**3:
            raise RuntimeError('Shared disk below20GiB floor')
        # The completed pipeline has exited its peers; tolerate brief CUDA context teardown only.
        for attempt in range(12):
            busy = [subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                    '--format=csv,noheader'], text=True).strip() for gpu in (6,7)]
            if not any(busy):
                break
            time.sleep(5)
        if any(busy):
            raise RuntimeError('Released GPUs occupied; no foreign process will be stopped')
        status.update(status='RUNNING', stage='component-smoke');write()
        run('component-smoke', 6, 's10_airplane_pass_1', 4, True)
        status.update(stage='parallel-component-and-TRAIN');write()
        with ThreadPoolExecutor(max_workers=2) as pool:
            component = pool.submit(run, 'component-airplane', 6, 's10_airplane_pass_1', 149, True)
            train = pool.submit(run, 'TRAIN-airplane', 7, 's1_airplane_pass_1', 195, False)
            results = dict(component=component.result(), TRAIN=train.result())
        (out/'comparison.json').write_text(json.dumps(dict(status='COMPLETED', results=results,
            scope='Teacher-assisted component attribution and one TRAIN qualification; not matched cross-hand validation'),
            indent=2, allow_nan=False)+'\n')
        status.update(status='COMPLETED')
    except BaseException as error:
        status.update(status='FAILED', error=repr(error))
        raise
    finally:
        status['elapsed_s'] = time.monotonic()-started
        write()


if __name__ == '__main__':
    main()
