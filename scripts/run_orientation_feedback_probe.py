#!/usr/bin/env python3
"""Bounded HF15 engineering/randomized executable-option opportunity."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON,R7,MOTIONS
EXPERIMENT='P-20261002-orientation-feedback-opportunity'


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve();output=args.output.resolve()
    if output.parent!=base or args.output.is_symlink():raise ValueError('new owned output required')
    prior_seconds=0.;prior_bytes=0;prior=None # preserved preflight syntax failure, no GPU/data
    if not args.smoke_only:
        if not args.prior_smoke:raise ValueError('completed source-matched engineering required')
        prior=json.loads((args.prior_smoke/'run_manifest.json').read_text())
        if prior['run_status']!='COMPLETED' or not prior['smoke_only'] or prior['experiment_id']!=EXPERIMENT:raise ValueError('engineering contract')
        if any(sha(Path(k))!=v for k,v in prior['input_sha256'].items()):raise ValueError('engineering input drift')
        if not all(p['result']['wrist_anchor_target_contract_passed'] and p['result']['feedback_execution_contract_passed'] for p in prior['phases']):raise ValueError('engineering execution failed')
        prior_seconds=prior['cumulative_seconds'];prior_bytes=prior['output_bytes']
    if args.prior_failed:
        if not args.smoke_only or args.prior_smoke:raise ValueError('engineering retry only')
        prior=json.loads((args.prior_failed/'run_manifest.json').read_text())
        if prior['run_status']!='FAILED' or not prior['smoke_only'] or prior['experiment_id']!=EXPERIMENT:raise ValueError('terminal failed engineering required')
        prior_seconds=prior['elapsed_seconds']+prior.get('prior_seconds',0)
        prior_bytes=sum(p.stat().st_size for p in args.prior_failed.rglob('*') if p.is_file())+prior.get('prior_bytes',0)
    if args.prior_superseded:
        if not args.smoke_only or args.prior_smoke or args.prior_failed:raise ValueError('superseded completed engineering only')
        prior=json.loads((args.prior_superseded/'run_manifest.json').read_text())
        if prior['run_status']!='COMPLETED' or not prior['smoke_only'] or prior['experiment_id']!=EXPERIMENT:raise ValueError('same terminal engineering required')
        prior_seconds=prior['cumulative_seconds']
        prior_bytes=sum(p.stat().st_size for p in args.prior_superseded.rglob('*') if p.is_file())+prior.get('prior_bytes',0)
    force_audit_path=ROOT/'docs/experiments/probes/P-20261002-contact-force-units-r2-audit.json'
    force_audit=json.loads(force_audit_path.read_text())
    if force_audit['status']!='COMPLETED' or force_audit['normalized_positive_recall']!=1. or force_audit['normalized_negative_false_positives']!=0.:raise ValueError('known force conditions required')
    route_path=ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json';route=json.loads(route_path.read_text())
    inputs=[force_audit_path,route_path,R7/'environment.yaml',R7/'training.yaml',Path(__file__),ROOT/'scripts/collect_orientation_feedback_options.py',ROOT/'scripts/collect_contact_consequences.py']
    inputs += [ROOT/'src/task/CmResidual'/n for n in ['executable_contact_options.py','orientation_anchored_options.py','weight_normalized_contact.py','contact_consequence.py','paired_evaluation.py','physical_value_live.py']]
    inputs += [ROOT/'third_party/DExplore/dexplore'/n for n in ['evaluate.py','env/tasks/dexplore_inspire.py','env/tasks/base_dexplore_task.py','data/assets/mjcf/airplane.urdf','data/assets/mjcf/table.urdf','data/assets/mjcf/objects/airplane/airplane.obj','data/assets/mjcf/objects/table/table.obj']]
    for s in route['experts'].values():
        p=(ROOT/s['checkpoint']).resolve()
        if sha(p)!=s['sha256']:raise ValueError('expert drift')
        inputs.append(p)
    for s in route['motions']:
        p=MOTIONS/s['name']/'interaction_hand_inspire.pt'
        if sha(p)!=s['interaction_hand_sha256']:raise ValueError('motion drift')
        inputs.append(p)
    if args.prior_smoke:inputs.append(args.prior_smoke/'run_manifest.json')
    if args.prior_failed:inputs.append(args.prior_failed/'run_manifest.json')
    if args.prior_superseded:inputs.append(args.prior_superseded/'run_manifest.json')
    hashes={str(p.resolve()):sha(p) for p in inputs};output.mkdir(exist_ok=False);begin=time.monotonic()
    m=dict(experiment_id=EXPERIMENT,family='HF15',probe_index_in_family=1,smoke_only=args.smoke_only,run_status='RUNNING',pid=os.getpid(),command=sys.argv,
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,phases=[],prior_seconds=prior_seconds,prior_bytes=prior_bytes,
           cm_training=False,expert_training=False,wall_limit_seconds=3600,output_limit_bytes=8<<30,scientific_seeds=list(range(461,473)),assignment_seed_offset=10000)
    def save():(output/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def check():
        if time.monotonic()-begin+prior_seconds>3540:raise TimeoutError('family wall limit')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>8<<30:raise ValueError('storage limit')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('input/source drift')
    save()
    try:
        for seed in ([460] if args.smoke_only else range(461,473)):
            check();admission=None
            for gpu in args.gpus:
                try:admission=gpu_admission(gpu);break
                except RuntimeError:continue
            if admission is None:raise RuntimeError('allowed GPUs occupied')
            d=output/f'seed{seed}'
            command=[PYTHON,'-u',str(ROOT/'scripts/collect_orientation_feedback_options.py'),'--output-dir',str(d),'--panel-seed',str(seed),'--assignment-seed',str(10000+seed),
                '--windows-per-stratum','2' if args.smoke_only else '8','--max-steps','650','--wall-seconds','240','--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
                '--checkpoint',str((ROOT/route['experts']['source_e260']['checkpoint']).resolve()),'--motion_file',str(MOTIONS),'--headless','--num_envs','96','--seed',str(seed),'--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0','--disable-early-termination',
                '--output',str(d/'unused.json'),'--output_path',str(d/'player')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed),TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
            env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
            p=dict(seed=seed,assignment_seed=10000+seed,run_status='STARTED',directory=str(d),command=command,gpu=admission);m['phases'].append(p);save();process=None;t=time.monotonic();print(json.dumps(dict(seed=seed,status='STARTED')),flush=True)
            try:
                with (output/f'seed{seed}.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);p.update(pid=process.pid,pgid=process.pid);save()
                    code=process.wait(timeout=min(290,3540-prior_seconds-(time.monotonic()-begin)))
                if code:raise RuntimeError('native exit'+str(code))
                r=json.loads((d/'results.json').read_text())
                if r['run_status']!='COMPLETED' or sha(d/'records.pt')!=r['record_sha256']:raise ValueError('record contract')
                p.update(run_status='COMPLETED',result=r);print(json.dumps(dict(seed=seed,status='COMPLETED',rows=r['rows'],initially_clear=r['initially_clear'])),flush=True)
            except BaseException as e:
                if process and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                p.update(run_status='FAILED',error=repr(e));raise
            finally:p['elapsed_seconds']=time.monotonic()-t;save()
        check();m.update(run_status='COMPLETED',input_hashes_unchanged=True,cumulative_seconds=prior_seconds+time.monotonic()-begin,output_bytes=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:m['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);p.add_argument('--smoke-only',action='store_true');p.add_argument('--prior-smoke',type=Path);p.add_argument('--prior-failed',type=Path);p.add_argument('--prior-superseded',type=Path);run(p.parse_args())
