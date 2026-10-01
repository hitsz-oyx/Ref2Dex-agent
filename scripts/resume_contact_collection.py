#!/usr/bin/env python3
"""Finish missing frozen phases after resource admission failure; preserve prior data."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    old_path=args.prior/'run_manifest.json';old=json.loads(old_path.read_text())
    if old['run_status']!='FAILED' or old.get('error')!="RuntimeError('GPU occupied; no interference allowed')":raise ValueError('resource admission failure only')
    if old['experiment_id']!='P-20261001-contact-supported-height-control':raise ValueError('frozen experiment required')
    if any(p['run_status']!='COMPLETED' for p in old['phases']):raise ValueError('incomplete native phases require separate audit')
    if any(sha(Path(p))!=v for p,v in old['input_sha256'].items()):raise ValueError('frozen inputs/sources changed')
    for phase in old['phases']:
        if sha(Path(phase['directory'])/'records.pt')!=phase['record_sha256']:raise ValueError('prior record changed')
    missing=sorted(set(range(351,357))-{p['seed'] for p in old['phases']})
    if not missing:raise ValueError('no missing phase')
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.resolve()
    if output.parent!=base or args.output.is_symlink():raise ValueError('owned new output required')
    output.mkdir(exist_ok=False);start=time.monotonic()
    prior_bytes=sum(p.stat().st_size for p in args.prior.rglob('*') if p.is_file())+old['setup_bytes']
    prior_seconds=old['elapsed_seconds']+old['setup_seconds']
    manifest=dict(old,run_status='RUNNING',error=None,pid=os.getpid(),command=sys.argv,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  continuation_of=str(args.prior.resolve()),continuation_reason='other GPU task arrived; no native phase rerun or outcome-based selection',
                  phases=list(old['phases']),input_sha256=dict(old['input_sha256'],**{str(old_path.resolve()):sha(old_path),str(Path(__file__).resolve()):sha(Path(__file__))}))
    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        for seed in missing:
            admission=gpu_admission(args.gpu);directory=output/f'seed{seed}'
            command=list(old['phases'][0]['command'])
            for flag,value in {'--output-dir':directory,'--panel-seed':seed,'--assignment-seed':7000+seed,'--seed':seed,
                               '--output':directory/'unused.json','--output_path':directory/'player'}.items():command[command.index(flag)+1]=str(value)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
                     PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed),
                     TORCH_EXTENSIONS_DIR=str(args.prior.resolve()/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
            env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
            phase=dict(seed=seed,assignment_seed=7000+seed,run_status='STARTED',gpu=admission,command=command,directory=str(directory));manifest['phases'].append(phase)
            process=None;begin=time.monotonic();save()
            try:
                with (output/f'seed{seed}.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    phase.update(pid=process.pid,pgid=process.pid);save()
                    code=process.wait(timeout=min(290,3540-prior_seconds-(time.monotonic()-start)))
                if code:raise RuntimeError(f'native exit{code}')
                result=json.loads((directory/'results.json').read_text())
                if result['run_status']!='COMPLETED':raise ValueError('incomplete collection')
                phase.update(run_status='COMPLETED',result=result,record_sha256=sha(directory/'records.pt'));print(json.dumps(dict(seed=seed,result=result)),flush=True)
            except BaseException as error:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                phase.update(run_status='FAILED',error=repr(error));raise
            finally:phase['elapsed_seconds']=time.monotonic()-begin;save()
        if any(sha(Path(p))!=v for p,v in manifest['input_sha256'].items()):raise ValueError('frozen inputs changed')
        byte_count=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if byte_count>8<<30 or prior_seconds+time.monotonic()-start>3600:raise ValueError('budget exceeded')
        manifest.update(run_status='COMPLETED',stage='SELECTOR_POOL_COMPLETE_AWAITING_ANALYSIS',input_hashes_unchanged=True,cumulative_seconds=prior_seconds+time.monotonic()-start,output_bytes=byte_count)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-start;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    run(p.parse_args())
