"""Owned, immutable-input, single-GPU supervised scale/hand diagnostic."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in
from scripts.prepare_surface_prior_data import file_stat


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); assert ROOT in out.parents and not out.exists()
    index = Path('/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json')
    hashes = {str(index): sha(index)}
    for relative in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines(): hashes[str(ROOT/relative)] = sha(ROOT/relative)
    for relative in ('docs/experiments/probes/P-20261003-cm-scale-cross-hand.md','docs/archive/2026-10-04-research-governance/decisions/D-20261003-cm-scale-cross-hand.md'): hashes[str(ROOT/relative)] = sha(ROOT/relative)
    for relative in ('scripts/run_surface_prior_probe.py','scripts/prepare_surface_prior_data.py','scripts/fit_surface_motion_prior.py','scripts/audit_surface_motion_prior.py','src/task/CmResidual/surface_motion_prior.py'): hashes[str(ROOT/relative)] = sha(ROOT/relative)
    begin = time.monotonic(); out.mkdir()
    manifest = dict(experiment_id='P-20261003-cm-scale-cross-hand',run_id=out.name,run_status='RUNNING',pid=os.getpid(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,
                    phases=[],wall_limit_seconds=3600,storage_limit_bytes=1<<30,policy_training=False)
    def save():
        manifest['wall_seconds'] = time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def verify():
        for filename,digest in hashes.items():
            if sha(Path(filename)) != digest: raise ValueError('protected input drift '+filename)
        if (out/'data/provenance.json').exists():
            provenance = json.loads((out/'data/provenance.json').read_text())
            for filename,stat in provenance['source_stat'].items():
                if file_stat(Path(filename)) != stat: raise ValueError('external source changed '+filename)
    def execute(name, command, gpu_phase, phase_limit):
        verify(); gpu = None
        if gpu_phase:
            for index_number in (6,4,5,3,7):
                try: gpu = admission(index_number); break
                except RuntimeError: pass
            if gpu is None: raise RuntimeError('no idle GPU; no CPU substitution')
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(out/'cache'),TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase = dict(name=name,command=command,run_status='RUNNING',admission=gpu,execution_device='cuda' if gpu else 'cpu')
        manifest['phases'].append(phase); save()
        def spawned(child): phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>3600 or bytes_in(out)>1<<30: raise RuntimeError('Probe wall/storage budget')
            if gpu:
                rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields = [part.strip() for part in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')): raise RuntimeError('unknown contention; stop owned phase only')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True)
        try:
            run_owned_child(command,ROOT,env,out/(name+'.log'),guard,min(phase_limit,3600-(time.monotonic()-begin)),spawned)
            verify();phase['run_status']='COMPLETED'
        except BaseException as error: phase.update(run_status='FAILED',error=repr(error));raise
        finally: save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        execute('data',[PYTHON,'-u',str(ROOT/'scripts/prepare_surface_prior_data.py'),'--index',str(index),'--output',str(out/'data')],False,1800)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_surface_motion_prior.py'),'--data',str(out/'data'),'--output',str(out/'fit')],True,1200)
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_surface_motion_prior.py'),'--root',str(out)],False,1800)
        result = json.loads((out/'fit/results.json').read_text()); audit = json.loads((out/'audit.json').read_text())
        assert audit['run_status']=='COMPLETED' and audit['label']==result['label']
        verify();manifest.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,
                                 actual_optimizer_updates=result['actual_optimizer_updates'])
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(label=result['label'],gates=result['gates'],updates=result['actual_optimizer_updates'])),flush=True)
    except BaseException as error: manifest.update(run_status='FAILED',error=repr(error));raise
    finally: save()


if __name__ == '__main__': main()
