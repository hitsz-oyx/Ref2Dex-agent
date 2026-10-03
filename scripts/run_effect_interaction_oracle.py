"""Fixed, owned, bounded no-Cm oracle utility experiment."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in
from scripts.run_oracle_replay_engineering import CONFIG,REFERENCES,POLICY

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique owned output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('clean fixed code')
    paths=[CONFIG,REFERENCES,POLICY,Path(__file__),ROOT/'scripts/run_oracle_native_replay.py',ROOT/'scripts/fit_oracle_selector.py',ROOT/'scripts/analyze_effect_interaction_oracle.py',ROOT/'src/task/CmResidual/oracle_native.py',ROOT/'src/task/CmResidual/oracle_features.py',ROOT/'docs/experiments/probes/P-20261003-effect-interaction-oracle.md',ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py']
    refs=json.loads(REFERENCES.read_text())
    for reference in refs['references']:paths.append(Path(reference['generated']))
    hashes={str(path):sha(path) for path in paths};begin=time.monotonic();out.mkdir()
    import numpy as np
    options=np.concatenate((np.zeros((1,12)),np.random.RandomState(805).normal(0,.25,(7,12))),0).astype(np.float32)
    np.save(out/'options.npy',options)
    hashes[str(out/'options.npy')]=sha(out/'options.npy')
    manifest=dict(run_status='RUNNING',experiment_id='P-20261003-effect-interaction-oracle',run_id=out.name,pid=os.getpid(),phases=[],input_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),wall_limit_seconds=3600,storage_limit_bytes=12<<30,training_seed=761,evaluation_seed=762,decision_tick=36,oracle_horizon=32,options=8,envs=96,no_learned_cm=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def verify():
        if any(sha(Path(f))!=h for f,h in hashes.items()):raise RuntimeError('protected input drift')
    def run(name,command,timeout=270):
        verify();gpu=admission(6)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'),MAX_JOBS='2')
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,admission=gpu,run_status='RUNNING');manifest['phases'].append(phase);save()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>3600 or bytes_in(out)>12<<30:raise RuntimeError('probe budget')
            rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            for row in rows.splitlines():
                fields=[x.strip() for x in row.split(',')]
                if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention')
        print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
        try:
            run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,timeout,spawned)
            verify();phase.update(run_status='COMPLETED')
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    def native(scene,path,seed,option=None):
        command=[PYTHON,'-u',str(ROOT/'scripts/run_oracle_native_replay.py'),'--output',str(path),'--env-config',str(CONFIG),'--references-manifest',str(REFERENCES),'--policy-checkpoint',str(POLICY),'--seed',str(seed),'--envs','96','--ticks','202','--decision','36','--oracle-horizon','32','--contact-window-only']
        if option is not None:command+=['--source',str(out/scene/'baseline'),'--options',str(option)]
        return command
    save()
    try:
        for scene,seed in (('train',761),('eval',762)):
            (out/scene).mkdir();run(scene+'_baseline',native(scene,out/scene/'baseline',seed))
            for index in range(1,8):
                option=out/scene/('option%02d.npy'%index);np.save(option,np.broadcast_to(options[index],(96,12)).copy());hashes[str(option)]=sha(option)
                name='c%02d'%index;run(scene+'_'+name,native(scene,out/scene/name,seed,option))
        run('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_oracle_selector.py'),'--source',str(out)],900)
        (out/'deploy').mkdir()
        for arm in ('state','effect','interaction','joint'):
            option=out/'fit'/(arm+'_options.npy');hashes[str(option)]=sha(option)
            run('deploy_'+arm,native('eval',out/'deploy'/arm,762,option))
        run('analyze',[PYTHON,'-u',str(ROOT/'scripts/analyze_effect_interaction_oracle.py'),'--source',str(out)],300)
        manifest.update(run_status='COMPLETED',inputs_unchanged=True,actual_optimizer_steps=6000,bytes=bytes_in(out))
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
