#!/usr/bin/env python3
"""Complete only the missing frozen panel after GPU admission rejection."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import PYTHON,sha,admission

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    root=args.directory.resolve();original=root/'run_manifest.json';m=json.loads(original.read_text())
    if ROOT not in root.parents or m['run_status']!='FAILED' or m['error']!="RuntimeError('GPU occupied; no interference')":raise ValueError('exact resource rejection required')
    if len(m['phases'])!=1 or m['phases'][0]['name']!='s506' or m['phases'][0]['run_status']!='COMPLETED':raise ValueError('only missing507 may resume')
    try:os.kill(m['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('original parent still exists')
    if (root/'s507').exists():raise ValueError('never overwrite existing pending panel')
    hashes={**m['input_sha256'],str(original):sha(original),str(Path(__file__).resolve()):sha(Path(__file__))}
    for path in (root/'s506').rglob('*'):
        if path.is_file():hashes[str(path)]=sha(path)
    if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('frozen input drift')
    output=root/'continuation_r1';output.mkdir(exist_ok=False);begin=time.monotonic()
    manifest=dict(experiment_id=m['experiment_id'],run_id=root.name+'/continuation_r1',run_status='RUNNING',pid=os.getpid(),
        phases=[],evaluation_seeds=[506,507],input_sha256=hashes,original_failure_manifest_sha256=sha(original),
        original_code_commit=m['git_commit'],continuation_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        no_completed_physics_repeated=True,gpu_index=args.gpu,wall_limit_seconds=600-m['wall_seconds'])
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>manifest['wall_limit_seconds']:raise TimeoutError('remaining total runtime budget')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>128*(1<<20):raise ValueError('same storage budget')
        if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('protected input drift')
    def execute(name,command,native):
        check();gpu=admission(args.gpu) if native else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if native else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu);manifest['phases'].append(phase);save();start=time.monotonic();process=None
        print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid);save();code=process.wait(timeout=min(240,manifest['wall_limit_seconds']-(time.monotonic()-begin)))
            if code:raise RuntimeError(f'{name} exit{code}; retained log')
            phase['run_status']='COMPLETED'
        except BaseException as error:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        command=list(m['phases'][0]['command'])
        for flag,value in (('--run-dir',root/'s507'),('--eval-seed',507),('--seed',507),('--output',root/'s507/unused.json'),('--output_path',root/'s507/player')):
            command[command.index(flag)+1]=str(value)
        execute('s507',command,True)
        for seed in (506,507):(output/f's{seed}').symlink_to(root/f's{seed}',target_is_directory=True)
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_frame0_tracking_feasibility.py'),'--directory',str(output)],False)
        check();result=json.loads((output/'results.json').read_text());manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()
if __name__=='__main__':main()
