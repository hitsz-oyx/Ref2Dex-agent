"""Bounded one-panel fixed-policy sensitivity to corrected collision ownership."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in


def counts(directory):
    result=np.zeros((3,4),np.int64)
    rows=json.loads((directory/'rows.json').read_text())
    assert len(rows)==768
    totals=np.zeros_like(result)
    for row in rows:
        result[row['motion'],row['arm']]+=int(row['physical105'])
        totals[row['motion'],row['arm']]+=1
    assert np.all(totals==64)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists();begin=time.monotonic()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED' and old['experiment_id']=='P-20261003-budgeted-physical-critic'
    cmd=list(next(phase['command'] for phase in old['phases'] if phase['name']=='s655'))
    hashes={}
    for name,digest in old['input_sha256'].items():
        if '/data/assets/' in name or '/P-20261001-hold-plateau-reference-r1/' in name:
            assert sha(Path(name))==digest;hashes[name]=digest
    for flag in ('--policy-checkpoint','--continuous-checkpoint','--checkpoint','--cfg_env','--cfg_train','--references-manifest','--option-actors'):
        path=Path(cmd[cmd.index(flag)+1]).resolve();hashes[str(path)]=sha(path)
    for suffix in ('rows.json','panel_audit.json','results.json'):
        path=source/'s655'/suffix;hashes[str(path)]=sha(path)
    hashes[str(source/'run_manifest.json')]=sha(source/'run_manifest.json')
    for name in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():hashes[str(ROOT/name)]=sha(ROOT/name)
    for name in ('scripts/run_inspire_filter_impact.py','scripts/run_inspire_filter_environment.py','scripts/audit_inspire_filter_panel.py'):
        hashes[str(ROOT/name)]=sha(ROOT/name)
    for name in ('docs/decisions/D-20261003-inspire-shape-filter-impact.md','docs/experiments/probes/P-20261003-inspire-filter-impact.md'):hashes[str(ROOT/name)]=sha(ROOT/name)
    base=Path(old['base_checkpoint']);heads=Path(old['panel_checkpoints']['655']['path']);actors=source/'fit/actors.pt'
    assert sha(base)==old['policy_sha256'] and sha(heads)==old['panel_checkpoints']['655']['sha256'] and sha(actors)==old['panel_actor_sha256']['655']
    def verify():
        for name,digest in hashes.items():
            if sha(Path(name))!=digest:raise ValueError('protected input drift '+name)
    verify();out.mkdir();(out/'fit').mkdir();(out/'fit/actors.pt').symlink_to(actors)
    m=dict(experiment_id='P-20261003-inspire-filter-impact',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_run=str(source),input_sha256=hashes,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],trained_actor_sha256=sha(actors),panel_checkpoints={'655':old['panel_checkpoints']['655']},phases=[],wall_limit_seconds=600,storage_limit_bytes=512<<20,actual_optimizer_steps=0)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def execute(name,command,native):
        verify();gpu=None
        if native:
            for index in (6,4,5,2,0):
                try:gpu=admission(index);break
                except RuntimeError:pass
            if gpu is None:raise RuntimeError('no idle GPU; preserve unique run')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8');env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu,execution_device='cuda' if gpu else 'cpu');m['phases'].append(phase);save()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>600 or bytes_in(out)>512<<20:raise RuntimeError('bounded sensitivity probe budget')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; only stop owned child')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True)
        try:run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(240,600-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        directory=out/'s655';cmd[2]=str(ROOT/'scripts/run_inspire_filter_environment.py')
        for flag,value in (('--run-dir',directory),('--output',directory/'unused.json'),('--output_path',directory/'player')):cmd[cmd.index(flag)+1]=str(value)
        execute('s655',cmd,True)
        execute('s655_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_inspire_filter_panel.py'),'--directory',str(out),'--panel','655'],False)
        before=counts(source/'s655');after=counts(out/'s655');bp=before.sum(0);ap=after.sum(0)
        gates=dict(any_pooled_change_at_least5pp=bool(np.any(np.abs(ap-bp)/192>=.05)),previously_unsolved_motion0_at_least4=bool(np.any(after[0]>=4)))
        result=dict(run_status='COMPLETED',label='PROMISING' if any(gates.values()) else 'UNPROMISING',gates=gates,before_per_motion_per64=before.tolist(),after_per_motion_per64=after.tolist(),before_pooled_per192=bp.tolist(),after_pooled_per192=ap.tolist(),arm_names=['p0','cm','dynamics_off','cold_q'],new_native_episodes=768,env_control_ticks=155136,actual_optimizer_steps=0,physics_sensitive_policy_diagnosis_only=True,exact_paired_counterfactual=False,cm_training_benefit_not_tested=True)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');verify();m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True)
        print(json.dumps(result),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
