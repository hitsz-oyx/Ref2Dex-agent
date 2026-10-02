"""Fit only after three valid native panels; preserve all failed execution parents."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-index',type=int,required=True);a=p.parse_args();previous=a.previous.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    pm=json.loads((previous/'run_manifest.json').read_text());assert pm['run_status']=='FAILED' and pm['experiment_id']=='P-20261002-measured-geometry-barriers' and "GPU occupied" in pm['error'];assert all(x['run_status']=='COMPLETED' for x in pm['phases']) and pm['phases'][-1]['name']=='s580_audit'
    original=Path(pm['prior_failed_run']);first=json.loads((original/'run_manifest.json').read_text());assert first['run_status']=='FAILED';carry=pm['conservative_cumulative_wall_seconds'];prior_bytes=bytes_in(original)+bytes_in(previous);hashes=dict(pm['input_sha256'])
    for f in [previous/'run_manifest.json',ROOT/'scripts/finish_measured_geometry_barrier_fit.py',ROOT/'docs/decisions/D-20261002-geometry-fit-only-resume.md']:hashes[str(f)]=sha(f)
    for seed in (578,579,580):
        d=previous/f's{seed}';assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED'
        for f in d.iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():assert sha(Path(f))==h,f
    verify();gpu=admission(a.gpu_index);out.mkdir();begin=time.monotonic()
    for seed in (578,579,580):(out/f's{seed}').symlink_to((previous/f's{seed}').resolve(),target_is_directory=True)
    m={k:v for k,v in pm.items() if k not in ['run_id','run_status','pid','git_commit','phases','error','wall_seconds','input_sha256','gpu']};m.update(run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,gpu=gpu,phases=[dict(name='all_three_native_cohort_reuse',run_status='COMPLETED',source=str(previous),no_native_or_audit_repeat=True)],prior_failed_run=str(previous),prior_run_manifest_sha256=sha(previous/'run_manifest.json'))
    def save():m.update(wall_seconds=time.monotonic()-begin,conservative_cumulative_wall_seconds=carry+time.monotonic()-begin);(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');phase=dict(name='fit',run_status='RUNNING',admission=gpu);m['phases'].append(phase);save()
    def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
    def guard():
        if carry+time.monotonic()-begin>1200 or prior_bytes+bytes_in(out)>1<<30:raise RuntimeError('cumulative scientific probe budget')
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        for row in rows.splitlines():
            fields=[v.strip() for v in row.split(',')]
            if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; only owned child may stop')
    try:
        cmd=[PYTHON,'-u',str(ROOT/'scripts/fit_measured_geometry_barriers.py'),'--source',str(out),'--output',str(out/'fit')];phase['command']=cmd;save();print(json.dumps(dict(phase='fit',status='STARTED',gpu=gpu)),flush=True)
        run_owned_child(cmd,ROOT,env,out/'fit.log',guard,min(420,1200-carry),spawned);verify();r=json.loads((out/'fit/results.json').read_text());assert r['run_status']=='COMPLETED';phase['run_status']='COMPLETED';m.update(run_status='COMPLETED',label=r['label'],bytes=bytes_in(out),conservative_cumulative_bytes=prior_bytes+bytes_in(out),inputs_unchanged=True);print(json.dumps(dict(status='COMPLETED',label=r['label'])),flush=True)
    except BaseException as e:phase.update(run_status='FAILED',error=repr(e));m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
