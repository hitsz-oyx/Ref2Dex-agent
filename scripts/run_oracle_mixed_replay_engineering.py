"""Minimal mixed-action counterfactual isolation qualification, no scientific fits."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in
from scripts.run_oracle_replay_engineering import CONFIG,REFERENCES,POLICY

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--retain-pd-target',action='store_true');p.add_argument('--physics',choices=('cpu','gpu'),default='gpu');a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique owned output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('clean fixed code')
    files=[CONFIG,REFERENCES,POLICY,Path(__file__).resolve(),ROOT/'src/task/CmResidual/oracle_native.py',ROOT/'scripts/run_oracle_native_replay.py',a.source/'eval/option07.npy',a.source/'fit/state_options.npy',a.source/'fit/state_choices.npy']
    hashes={str(f.resolve()):sha(f) for f in files};out.mkdir();begin=time.monotonic();m=dict(run_status='RUNNING',engineering_only=True,experiment_id='engineering-oracle-mixed-replay',pid=os.getpid(),phases=[],input_sha256=hashes,wall_limit_seconds=600,storage_limit_bytes=1<<30,physics=a.physics,retain_pd_target=a.retain_pd_target,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    save()
    try:
        for name in ('baseline','query','mixed'):
            if any(sha(Path(f))!=h for f,h in hashes.items()):raise RuntimeError('input drift')
            gpu=admission(6);command=[PYTHON,'-u',str(ROOT/'scripts/run_oracle_native_replay.py'),'--output',str(out/name),'--env-config',str(CONFIG),'--references-manifest',str(REFERENCES),'--policy-checkpoint',str(POLICY),'--seed','762','--envs','96','--ticks','72','--decision','36','--oracle-horizon','32','--contact-window-only','--physics',a.physics]
            if a.retain_pd_target:command+=['--retain-pd-target']
            if name!='baseline':command+=['--source',str(out/'baseline'),'--options',str(a.source/('eval/option07.npy' if name=='query' else 'fit/state_options.npy'))]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'),MAX_JOBS='2');env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
            phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu);m['phases'].append(phase);save()
            def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
            def guard():
                if time.monotonic()-begin>600 or bytes_in(out)>1<<30:raise RuntimeError('engineering budget')
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention')
            print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
            try:run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,270,spawned);phase.update(run_status='COMPLETED')
            except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
            finally:save()
        import torch,numpy as np
        load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
        q=load(out/'query/trace.pt');mixed=load(out/'mixed/trace.pt');ids=np.flatnonzero(np.load(a.source/'fit/state_choices.npy')==7)
        errors={key:float((q[key][36:68,ids]-mixed[key][36:68,ids]).abs().max()) for key in ('object_root','native_q','native_dq','rigid_state','net_force','target','action')}
        prefix=all(torch.equal(q[key][:36],mixed[key][:36]) for key in ('object_root','native_q','native_dq','rigid_state','net_force','target','action'))
        result=dict(run_status='COMPLETED',engineering_only=True,physics=a.physics,retain_pd_target=a.retain_pd_target,exact_prefix=prefix,same_option_envs=len(ids),mixed_query_errors=errors,all_short_states_commands_forces_bit_identical=bool(prefix and all(v==0 for v in errors.values())),no_new_training=True)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');m.update(run_status='COMPLETED',inputs_unchanged=all(sha(Path(f))==h for f,h in hashes.items()),bytes=bytes_in(out));print(json.dumps(result),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
