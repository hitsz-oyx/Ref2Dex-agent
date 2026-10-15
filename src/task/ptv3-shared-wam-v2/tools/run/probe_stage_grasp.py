"""Run a bounded parallel checkpoint panel through the existing native harness."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'src/task/ptv3-shared-wam-v2'
LEGACY = ROOT / 'src/task/ptv3-shared-wam'
STEPS = [1000, 3000, 3500, 4000, 8000, 9000, 10000, 11000, 12000]
PANEL = [('airplane', 's10_airplane_pass_1', 149),
         ('mug', 's10_mug_pass_1', 172), ('sphere', 's4_spheresmall_pass_1', 141)]


def write(path, data):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def summarize(run, step, sequence):
    import numpy as np
    native = run / 'native'
    result = json.loads((native / 'result.json').read_text())
    manifest = json.loads((native / 'manifest.json').read_text())
    audit = json.loads((native / 'independent_audit.json').read_text())
    if not audit['pass']:
        raise ValueError('independent command/PD/clock audit failed')
    with np.load(native / 'trajectory.npz') as trajectory, np.load(manifest['native_input']['trajectory']) as source:
        count = len(trajectory['obj'])
        rows = []
        for i, item in enumerate(result['outcomes']):
            hand = np.linalg.norm(trajectory['hand'][:, i] - source['hand'][:count], axis=-1)
            obj = np.linalg.norm(trajectory['obj'][:, i, :3, 3] - source['reference_obj'][:count, :3, 3], axis=-1)
            rows.append(dict(item, step=step, sequence=sequence,
                hand_recorded_mean_mm=float(hand.mean()*1000),
                object_reference_mean_mm=float(obj.mean()*1000),
                lift_max_mm=float((trajectory['obj'][:, i, 2, 3]-trajectory['obj'][0,i,2,3]).max()*1000),
                held45=bool(item['maximum_held_frames'] >= 45),
                teacher_held45=bool(result['outcomes'][0]['maximum_held_frames'] >= 45),
                audit_pass=True, path=str(run.relative_to(ROOT))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--card', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--previous-output', type=Path, help='reuse only completed matching physical groups; never overwrite them')
    a = p.parse_args()
    out, archive = a.output.resolve(), a.archive.resolve()
    gpus = [int(x) for x in a.gpus.split(',')]
    if out.exists() or out.parent != ROOT / 'outputs/ptv3-shared-wam-v2' or len(gpus) != 3 or len(set(gpus)) != 3:
        raise ValueError('fresh output and three distinct GPUs required')
    for step in STEPS:
        if not (archive / ('finetune-step%d.pt' % step)).is_file():
            raise FileNotFoundError(step)
    for gpu in gpus:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('requested GPU occupied')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('insufficient disk headroom')
    out.mkdir()
    started = time.monotonic()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status = dict(status='RUNNING', pid=os.getpid(), run_id=out.name, git_commit=commit,
                  steps=STEPS, panel=PANEL, gpus=gpus, seed=148, execution_steps=4,
                  max_seconds=1800, max_bytes=2_000_000_000, completed=0)
    write(out/'process.json', status)

    def worker(gpu, panel):
        short, sequence, controls = panel
        reports = []
        for step in STEPS:
            if time.monotonic()-started > 1600 or shutil.disk_usage(ROOT).free < 20*2**30:
                raise RuntimeError('panel deadline/storage guard')
            if sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > 2_000_000_000:
                raise RuntimeError('panel artifact budget')
            run = out / ('step%d-%s' % (step, short))
            previous = a.previous_output.resolve() / run.name if a.previous_output else None
            if previous and (previous / 'group_status.json').is_file():
                prior = json.loads((previous / 'group_status.json').read_text())
                expected = archive / ('finetune-step%d.pt' % step)
                if (prior.get('status') == 'COMPLETED' and prior.get('checkpoint') == str(expected)
                        and prior.get('controls') == controls and prior.get('execution_steps') == 4
                        and prior.get('seed') == 148 and prior.get('native_input',{}).get('sequence') == sequence):
                    if not (previous / 'native/independent_audit.json').exists():
                        audit = subprocess.run([sys.executable,str(LEGACY/'tools/audit/audit_native.py'),
                            '--native',str(previous/'native')],cwd=ROOT,capture_output=True,text=True)
                        (out / (run.name+'-previous-audit.log')).write_text(audit.stdout+audit.stderr)
                        if audit.returncode:
                            raise RuntimeError('completed previous group still fails audit; diagnose before replay')
                    reports.append(dict(step=step,sequence=sequence,status='COMPLETED',
                        reused_from=str(previous),outcomes=summarize(previous,step,sequence)))
                    write(out/('%s-progress.json'%short),reports)
                    continue
            cmd = [sys.executable, str(LEGACY/'tools/run/launch_native.py'), '--model-version','v2',
                   '--checkpoint',str(archive/('finetune-step%d.pt'%step)), '--output',str(run),
                   '--gpu',str(gpu),'--controls',str(controls),'--execution-steps','4',
                   '--sequence',sequence,'--seed','148','--deadline-seconds','180',
                   '--experiment-card',str(a.card.resolve())]
            # Each child launcher owns/cleans its two peer process groups.
            with (out/('step%d-%s.log'%(step,short))).open('w') as log:
                completed = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
            if completed.returncode:
                reports.append(dict(step=step,sequence=sequence,status='FAILED',path=str(run),exit_code=completed.returncode))
                # Keep other independent checkpoints testable; never change physics or labels.
                write(out/('%s-progress.json'%short),reports)
                continue
            audit = subprocess.run([sys.executable,str(LEGACY/'tools/audit/audit_native.py'),
                '--native',str(run/'native')],cwd=ROOT,capture_output=True,text=True)
            (run/'audit.log').write_text(audit.stdout+audit.stderr)
            if audit.returncode:
                reports.append(dict(step=step,sequence=sequence,status='AUDIT_FAILED',path=str(run)))
            else:
                reports.append(dict(step=step,sequence=sequence,status='COMPLETED',outcomes=summarize(run,step,sequence)))
            write(out/('%s-progress.json'%short),reports)
        return reports

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker,gpu,panel) for gpu,panel in zip(gpus,PANEL)]
            reports = [row for future in futures for row in future.result()]
        rows = [row for report in reports for row in report.get('outcomes',[])]
        state = 'COMPLETED' if all(r['status']=='COMPLETED' for r in reports) else 'COMPLETED_WITH_FAILURES'
        write(out/'result.json',dict(status=state,runs=reports,outcomes=rows,
            scope='single-seed three-dev-sequence Probe; held45 is a grasp proxy, not task success'))
        status.update(status=state,completed=sum(r['status']=='COMPLETED' for r in reports),
                      elapsed_s=time.monotonic()-started)
        write(out/'process.json',status)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started)
        write(out/'process.json',status)
        raise


if __name__ == '__main__':
    main()
