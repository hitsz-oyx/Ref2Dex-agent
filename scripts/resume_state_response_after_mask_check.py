"""Reuse audited570 after a pre-optimizer stream-mask assertion correction."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();old=a.source.resolve();out=a.output.resolve();diag=a.diagnosis.resolve()
    assert ROOT in out.parents and not out.exists();m0=json.loads((old/'run_manifest.json').read_text());d=json.loads((diag/'results.json').read_text())
    assert m0['run_status']=='FAILED' and m0['phases'][-1]['name']=='fit' and d['zero_optimizer_steps'] and d['original_manifest_sha256']==sha(old/'run_manifest.json') and not list((old/'fit').iterdir())
    hashes=dict(m0['input_sha256']);f=ROOT/'scripts/fit_state_response_field.py';assert hashes[str(f)]==sha(diag/'failed_fit_source.py');hashes[str(diag/'failed_fit_source.py')]=hashes.pop(str(f))
    for path in [f,Path(__file__),old/'run_manifest.json',diag/'results.json',ROOT/'scripts/audit_state_response_field_result.py']:hashes[str(path.resolve())]=sha(path)
    for path in (old/'s570').iterdir():
        if path.is_file():hashes[str(path.resolve())]=sha(path)
    for path,h in hashes.items():assert sha(Path(path))==h,path
    assert json.loads((old/'s570/panel_audit.json').read_text())['run_status']=='COMPLETED'
    gpu=admission(4);out.mkdir();(out/'s570').symlink_to(old/'s570',target_is_directory=True);begin=time.monotonic();prior_seconds=m0['wall_seconds']+60;prior_bytes=bytes_in(old)+bytes_in(diag)
    m=dict(m0,run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,phases=[],gpu=gpu,source_failed_run=str(old),diagnosis=str(diag),reuse_audited_native570=True,no_optimizer_retry=True,prior_seconds=prior_seconds,prior_bytes=prior_bytes)
    def save():
        m['wall_seconds']=time.monotonic()-begin;m['combined_wall_seconds']=prior_seconds+m['wall_seconds'];(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    cmd=[PYTHON,'-u',str(f),'--source',m0['source_run'],'--test',str(out),'--output',str(out/'fit')];phase=dict(name='fit',command=cmd,run_status='RUNNING');m['phases'].append(phase);save()
    def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
    def guard():
        if prior_seconds+time.monotonic()-begin>3600 or prior_bytes+bytes_in(out)>1<<30:raise RuntimeError('combined original+resume budget')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        for line in apps.splitlines():
            fields=[q.strip() for q in line.split(',')]
            if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention; stop owned child only')
    try:
        run_owned_child(cmd,ROOT,env,out/'fit.log',guard,3600-prior_seconds,spawned)
        for path,h in hashes.items():assert sha(Path(path))==h,path
        result=json.loads((out/'fit/results.json').read_text());assert result['run_status']=='COMPLETED';phase['run_status']='COMPLETED';m.update(run_status='COMPLETED',label=result['label'],combined_bytes=prior_bytes+bytes_in(out))
    except BaseException as e:phase.update(run_status='FAILED',error=repr(e));m.update(run_status='FAILED',error=repr(e));raise
    finally:save()
    print(json.dumps(dict(run_status=m['run_status'],label=m.get('label'),combined_seconds=m['combined_wall_seconds'],combined_bytes=m.get('combined_bytes'))),flush=True)

if __name__=='__main__':main()
