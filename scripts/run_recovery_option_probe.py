#!/usr/bin/env python3
"""Bounded HF11 engineering and matched option-policy training/evaluation."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON,R7,MOTIONS


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve();output=args.output.resolve()
    if output.parent!=base or args.output.is_symlink():raise ValueError('new owned output required')
    prior_seconds=0.;prior_bytes=0
    if not args.smoke_only:
        if not args.prior_smoke:raise ValueError('terminal engineering smoke required')
        old=json.loads((args.prior_smoke/'run_manifest.json').read_text())
        if old['run_status']!='COMPLETED' or not old['smoke_only']:raise ValueError('engineering smoke incomplete')
        if any(sha(Path(k))!=v for k,v in old['input_sha256'].items()):raise ValueError('smoke input/code drift')
        if not all(p['result']['frozen_cm_experts'] and p['result']['policy_parameters_changed'] and p['result']['optimizer_updates']>0 for p in old['phases']):raise ValueError('smoke learning contract failed')
        prior_seconds=old['elapsed_seconds'];prior_bytes=sum(p.stat().st_size for p in args.prior_smoke.rglob('*') if p.is_file())
    trajectory=(base/'P-20261001-contact-supported-height-control-setup-r1/calibrated_trajectory.pt')
    if sha(trajectory)!='027202015c32ba783aa1bbef5a0a3c501643bfa904e0b460cdf1877193971355':raise ValueError('frozen physical Cm required')
    route_path=ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json';route=json.loads(route_path.read_text())
    inputs=[trajectory,route_path,R7/'environment.yaml',R7/'training.yaml']
    for spec in route['experts'].values():
        path=(ROOT/spec['checkpoint']).resolve()
        if sha(path)!=spec['sha256']:raise ValueError('expert drift')
        inputs.append(path)
    for spec in route['motions']:
        path=MOTIONS/spec['name']/'interaction_hand_inspire.pt'
        if sha(path)!=spec['interaction_hand_sha256']:raise ValueError('motion drift')
        inputs.append(path)
    sources=[Path(__file__),ROOT/'scripts/train_recovery_options.py',ROOT/'scripts/collect_contact_consequences.py']
    sources += [ROOT/'src/task/CmResidual'/name for name in ['recovery_option_policy.py','contact_trajectory.py','contact_ranker.py','trajectory_selector.py','physical_value_live.py','physical_value_contract.py','paired_evaluation.py']]
    sources += [ROOT/'third_party/DExplore/dexplore'/name for name in ['evaluate.py','env/tasks/dexplore_inspire.py','env/tasks/base_dexplore_task.py']]
    inputs+=sources
    if args.prior_smoke:inputs.append(args.prior_smoke/'run_manifest.json')
    hashes={str(p.resolve()):sha(p) for p in inputs}
    output.mkdir(exist_ok=False);start=time.monotonic()
    manifest=dict(experiment_id='P-20261001-recovery-option-learning',family='HF11',probe_index_in_family=1,
                  smoke_only=args.smoke_only,run_status='RUNNING',pid=os.getpid(),command=sys.argv,phases=[],input_sha256=hashes,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),prior_smoke_seconds=prior_seconds,
                  wall_limit_seconds=3600,output_limit_bytes=8<<30,cm_training=False,expert_training=False,option_policy_training=True)
    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-start>3540:raise TimeoutError('HF11 total budget')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>8<<30:raise ValueError('storage limit')
        if any(sha(Path(p))!=v for p,v in hashes.items()):raise ValueError('frozen input/code drift')
    def phase_run(name,cm_on,seed,evaluate=False,policy=None,smoke=False):
        check();admission=None
        for index in args.gpus:
            try:admission=gpu_admission(index);break
            except RuntimeError:continue
        if admission is None:raise RuntimeError('all allowed GPUs occupied; preserve completed phases')
        directory=output/name
        command=[PYTHON,'-u',str(ROOT/'scripts/train_recovery_options.py'),'--output-dir',str(directory),'--trajectory',str(trajectory),
                 '--rollouts','1' if evaluate or smoke else '4','--epochs','2' if smoke else '20','--assignment-seed',str(7000+seed),
                 '--wall-seconds','1200','--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
                 '--checkpoint',str((ROOT/route['experts']['source_e260']['checkpoint']).resolve()),'--motion_file',str(MOTIONS),
                 '--headless','--num_envs','96','--seed',str(seed),'--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0',
                 '--disable-early-termination','--output',str(directory/'unused.json'),'--output_path',str(directory/'player')]
        if cm_on:command.append('--cm-on')
        if evaluate:command.append('--evaluate')
        if policy:command.extend(['--policy',str(policy)])
        if smoke:command.append('--smoke')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
                 PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed),
                 TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,cm_on=cm_on,seed=seed,evaluate=evaluate,smoke=smoke,run_status='STARTED',directory=str(directory),gpu=admission,command=command)
        manifest['phases'].append(phase);save();process=None;begin=time.monotonic();print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:
            with (output/f'{name}.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid);save()
                code=process.wait(timeout=min(1250,3540-prior_seconds-(time.monotonic()-start)))
            if code:raise RuntimeError(f'native exit{code}: {name}')
            result=json.loads((directory/'results.json').read_text())
            if result['run_status']!='COMPLETED':raise ValueError('incomplete phase')
            phase.update(run_status='COMPLETED',result=result);print(json.dumps(dict(name=name,status='COMPLETED',updates=result['optimizer_updates'],env_steps=result['native_env_steps'])),flush=True)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['elapsed_seconds']=time.monotonic()-begin;save()
    save()
    try:
        if args.smoke_only:
            for mode in [True,False]:phase_run('smoke_on' if mode else 'smoke_off',mode,380,smoke=True)
        else:
            for mode in [True,False]:phase_run('train_on' if mode else 'train_off',mode,381)
            first,second=[p['result']['rollouts'] for p in manifest['phases']]
            if any(a['episode_steps']!=b['episode_steps'] for a,b in zip(first,second)):
                raise ValueError('rollout episode budget differs; investigate matched contract')
            for seed in range(391,395):
                for mode in [True,False]:phase_run(f'eval_{"on" if mode else "off"}_s{seed}',mode,seed,evaluate=True,policy=output/('train_on' if mode else 'train_off')/'policy.pt')
            for mode in [True,False]:phase_run('prior_on_s391' if mode else 'prior_off_s391',mode,391,evaluate=True)
        check();manifest.update(run_status='COMPLETED',input_hashes_unchanged=True,cumulative_seconds=prior_seconds+time.monotonic()-start,
                                output_bytes=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-start;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=[2,3]);p.add_argument('--smoke-only',action='store_true');p.add_argument('--prior-smoke',type=Path)
    run(p.parse_args())
