"""Owned true future query for one subject while all other instances use P0."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in
from scripts.run_oracle_replay_engineering import CONFIG,REFERENCES,POLICY

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--engineering',action='store_true');p.add_argument('--baseline',type=Path);a=p.parse_args();out=a.output.resolve();source=a.source.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique owned output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('clean fixed source')
    paths=[CONFIG,REFERENCES,POLICY,Path(__file__).resolve(),ROOT/'scripts/run_oracle_native_replay.py',ROOT/'src/task/CmResidual/oracle_native.py',ROOT/'src/task/CmResidual/oracle_features.py',ROOT/'scripts/select_single_focal_oracle.py',ROOT/'scripts/analyze_single_focal_oracle.py',ROOT/'docs/decisions/D-20261003-single-focal-oracle.md',source/'options.npy',source/'fit/selector.pt',source/'fit/results.json',source/'run_manifest.json']
    if a.baseline:
        a.baseline=a.baseline.resolve();paths.extend(p for p in a.baseline.iterdir() if p.is_file())
    hashes={str(path):sha(path) for path in paths};out.mkdir();begin=time.monotonic();limit=300 if a.engineering else 3600;storage=(512<<20) if a.engineering else (6<<30)
    manifest=dict(run_status='RUNNING',experiment_id='engineering-oracle-single-focal' if a.engineering else 'P-20261003-effect-interaction-oracle',run_id=out.name,engineering_only=a.engineering,pid=os.getpid(),phases=[],input_sha256=hashes,wall_limit_seconds=limit,storage_limit_bytes=storage,seed=763,envs=12,source_frozen_selectors=str(source),actual_new_optimizer_steps=0,inherited_optimizer_steps=6000,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if a.baseline:(out/'baseline').symlink_to(a.baseline,target_is_directory=True);manifest['retained_baseline']=str(a.baseline)
    def save():manifest['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def verify():
        if any(sha(Path(f))!=h for f,h in hashes.items()):raise RuntimeError('protected input drift')
    def run(name,command,timeout=270):
        verify();gpu=admission(6);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'),MAX_JOBS='2');env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,admission=gpu,run_status='RUNNING');manifest['phases'].append(phase);save()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>limit or bytes_in(out)>storage:raise RuntimeError('focal budget')
            rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            for row in rows.splitlines():
                fields=[x.strip() for x in row.split(',')]
                if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention')
        print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
        try:run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,timeout,spawned);verify();phase.update(run_status='COMPLETED')
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    def native(path,ticks,option=None):
        command=[PYTHON,'-u',str(ROOT/'scripts/run_oracle_native_replay.py'),'--output',str(path),'--env-config',str(CONFIG),'--references-manifest',str(REFERENCES),'--policy-checkpoint',str(POLICY),'--seed','763','--envs','12','--ticks',str(ticks),'--decision','36','--oracle-horizon','32','--contact-window-only','--retain-pd-target']
        if option is not None:command+=['--source',str(out/'baseline'),'--options',str(option)]
        return command
    save()
    try:
        import numpy as np
        options=np.load(source/'options.npy')
        if a.baseline is None:run('baseline',native(out/'baseline',202))
        (out/'queries').mkdir();(out/'deploy').mkdir();(out/'requests').mkdir()
        targets=(0,) if a.engineering else range(12)
        for target in targets:
            indices=(7,) if a.engineering else range(1,8)
            for index in indices:
                name='e%02d_c%02d'%(target,index);option=out/'requests'/(name+'.npy');matrix=np.zeros((12,12),np.float32);matrix[target]=options[index];np.save(option,matrix);hashes[str(option)]=sha(option)
                run('query_'+name,native(out/'queries'/name,68,option))
        if a.engineering:
            run('deploy_e00_c07',native(out/'deploy/e00_c07',202,out/'requests/e00_c07.npy'))
            import torch
            load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
            q=load(out/'queries/e00_c07/trace.pt');d=load(out/'deploy/e00_c07/trace.pt')
            errors={key:float((q[key]-d[key][:68]).abs().max()) for key in ('object_root','native_q','native_dq','rigid_state','net_force','target','action')}
            result=dict(run_status='COMPLETED',engineering_only=True,entire_world_query_deployed_max_errors=errors,entire_world_first68_bit_identical=all(value==0 for value in errors.values()),new_optimizer_steps=0)
            (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        else:
            run('select',[PYTHON,'-u',str(ROOT/'scripts/select_single_focal_oracle.py'),'--source',str(out),'--models',str(source/'fit/selector.pt')],180)
            choices=json.loads((out/'selection/choices.json').read_text());unique=sorted(set((int(env),int(option)) for arm in choices.values() for env,option in enumerate(arm) if option))
            for target,index in unique:
                name='e%02d_c%02d'%(target,index);run('deploy_'+name,native(out/'deploy'/name,202,out/'requests'/(name+'.npy')))
            run('analyze',[PYTHON,'-u',str(ROOT/'scripts/analyze_single_focal_oracle.py'),'--source',str(out),'--models',str(source/'fit/selector.pt')],240)
        manifest.update(run_status='COMPLETED',inputs_unchanged=True,bytes=bytes_in(out))
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
