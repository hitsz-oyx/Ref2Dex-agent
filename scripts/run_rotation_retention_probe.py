"""Fixed two-seed controller feasibility with pinned inputs and owned GPU guard."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED';hashes=dict(old['input_sha256'])
    for f in ['scripts/run_rotation_retention_probe.py','scripts/run_rotation_retention_environment.py','scripts/audit_rotation_retention_feasibility.py','scripts/run_natural_retention_environment.py','scripts/audit_natural_retention_feedback.py','src/task/CmResidual/rotation_retention_feedback.py','docs/decisions/D-20261002-after-response-field-route-review.md','docs/experiments/probes/P-20261002-rotation-retention-feasibility.md', 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    hashes[str(source/'run_manifest.json')]=sha(source/'run_manifest.json')
    def verify():
        for f,h in hashes.items():assert sha(Path(f))==h,f
    verify();gpu=admission(4);out.mkdir();begin=time.monotonic();m=dict(experiment_id='P-20261002-rotation-retention-feasibility',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,evaluation_seeds=[571,572],phases=[],gpu=gpu,wall_limit_seconds=900,storage_limit_bytes=512<<20,policy_sha256=old['policy_sha256'],base_checkpoint=old['base_checkpoint'],no_training=True,cm_policy_utility_unproved=True)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def execute(name,cmd,gpu_compute,timeout):
        verify();admitted=admission(4) if gpu_compute else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu_compute else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'));env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=admitted,run_status='RUNNING');m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>900 or bytes_in(out)>512<<20:raise RuntimeError('whole experiment budget')
            if gpu_compute:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for line in apps.splitlines():
                    fields=[q.strip() for q in line.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop owned child only')
        print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,900-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        template=list(old['phases'][0]['command'])
        for flag in ['--continuous-checkpoint','--continuous-sha256','--expected-update']:
            at=template.index(flag);del template[at:at+2]
        assert '--deterministic' not in template;template[2]=str(ROOT/'scripts/run_rotation_retention_environment.py')
        for seed in (571,572):
            cmd=list(template);d=out/f's{seed}'
            for flag,value in [('--run-dir',d),('--eval-seed',seed),('--seed',seed),('--output',d/'unused.json'),('--output_path',d/'player')]:cmd[cmd.index(flag)+1]=str(value)
            execute(d.name,cmd,True,300)
            for f in d.iterdir():
                if f.is_file():hashes[str(f.resolve())]=sha(f)
        m['run_status']='COLLECTION_COMPLETED';save();execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_rotation_retention_feasibility.py'),'--directory',str(out)],False,300)
        result=json.loads((out/'results.json').read_text());m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
