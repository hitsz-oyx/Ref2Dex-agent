"""One fresh u20 panel and matched force-aware impulse forecasts, bounded and owned."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED'
    heads=source/'u20/policy_heads.pt';base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in ['scripts/run_force_aware_impulse_probe.py','scripts/audit_impulse_native_panel.py','scripts/fit_force_aware_impulse.py','src/task/CmResidual/force_aware_impulse.py','docs/experiments/probes/P-20261002-force-aware-impulse.md']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in [source/'run_manifest.json',heads,base,*[source/'s547'/name for name in ('initial.pt','trace.pt','results.json','panel_audit.json')]]:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('source drift '+f)
    verify();gpu=admission(4);out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261002-force-aware-impulse',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),input_sha256=hashes,gpu=gpu,phases=[],wall_limit_seconds=1800,storage_limit_bytes=1<<30,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={'575':dict(path=str(heads.resolve()),sha256=sha(heads),update=20)})
    def save():
        m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def execute(name,cmd,gpu_compute,timeout):
        verify();admitted=admission(4) if gpu_compute else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu_compute else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=admitted,run_status='RUNNING',device_reason='GPU simulation/fits/inference' if gpu_compute else 'independent NumPy physical/state/request/PD/full-mesh audit');m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>1800 or bytes_in(out)>1<<30:raise RuntimeError('whole probe budget exceeded')
            if gpu_compute:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for line in apps.splitlines():
                    fields=[q.strip() for q in line.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; only owned child may stop')
        print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
        try:
            run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,1800-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED',seconds=phase['wall_seconds'])),flush=True)
    cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/'scripts/run_continuous_critic_environment_v2.py');d=out/'s575'
    for flag,value in [('--run-dir',d),('--eval-seed',575),('--seed',575),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',20)]:cmd[cmd.index(flag)+1]=str(value)
    assert '--deterministic' not in cmd;save()
    try:
        execute('s575',cmd,True,300)
        execute('s575_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_impulse_native_panel.py'),'--directory',str(out),'--panel','575'],False,300)
        for f in d.iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_force_aware_impulse.py'),'--source',str(source),'--test',str(out),'--output',str(out/'fit')],True,1650)
        result=json.loads((out/'fit/results.json').read_text());assert result['run_status']=='COMPLETED';m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
