"""Bounded Shadow BC -> Stage-0 PPO, retaining the pinned native collector setup."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def worker(a):
    # Asset initialization can be slow: freeze the matching package before importing Gym.
    frozen=a.output/'runtime-code'
    shutil.copytree(TASK/'src/wam_rl_policy',frozen/'wam_rl_policy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=TASK.parent/'video-data-preparation/tools/run/replay_unidexgrasp.py'
    text=source.read_text()
    start=text.index('    records = []\n    begin = time.monotonic()')
    end=text.index('\n\ndef supervise(args):',start)
    replacement='''    from wam_rl_policy.shadow_loop import run
    try:
        run(task,snapshot,env_objects,actor,args)
    finally:
        task.gym.destroy_sim(task.sim)
'''
    text=text[:start]+replacement+text[end:]
    # Scale runs separately audit the complete linked closure, including symlink targets.
    if a.experiment_id=='P-20261016-shadow-scale-continuous':
        manifest=json.loads(a.object_manifest.read_text())
        if manifest.get('real_asset_closure_bytes',513*2**20)>512*2**20:raise ValueError('Scale shard asset closure unaudited')
        text=text.replace('> 128*2**20:', '> 512*2**20:')
    interventions=[]
    if a.retain_reset_indices:
        helper='''def _keep_gpu_index(task, value):
    if not hasattr(task, '_retained_gpu_indices'):task._retained_gpu_indices=[]
    task._retained_gpu_indices.append(value)
    return gymtorch.unwrap_tensor(value)


'''
        interventions.append(('class ShadowHandGrasp(BaseTask):',helper+'class ShadowHandGrasp(BaseTask):'))
        for key in ('all_hand_indices','all_indices','goal_object_indices'):
            interventions.append(('gymtorch.unwrap_tensor('+key+')','_keep_gpu_index(self, '+key+')'))
    if a.single_reset_target_setter:
        old='''        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))'''
        interventions.insert(0,(old,'        # The same pre_physics_step sets the final targets once, after reset.'))
    if interventions:
        insert=''
        for old,new in interventions:
            insert+='    if '+repr(old)+' not in source:raise ValueError("Pinned reset intervention source changed")\n'
            insert+='    source=source.replace('+repr(old)+','+repr(new)+')\n'
        text=text.replace('    source = source_path.read_text()\n','    source = source_path.read_text()\n'+insert)
        (a.output/'reset-api-interventions.json').write_text(json.dumps(dict(retain_reset_indices=a.retain_reset_indices,
            single_reset_target_setter=a.single_reset_target_setter,replacements=interventions),indent=2)+'\n')
    dest=a.output/'runtime_replay.py';dest.write_text(text)
    for task in ('wam-rl-policy','shadow-finetune'):
        sys.path.insert(0,str(TASK.parent/task/'src'))
    sys.path.insert(0,str(frozen))
    spec=importlib.util.spec_from_file_location('wam_rl_shadow_runtime',dest)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a.vendor=ROOT/'third_party/UniDexGrasp2'
    a.subset=ROOT/'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1'
    a.assets=None;a.scale=.08;a.episodes=1;a.gpu_seconds=a.seconds
    module.worker(a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--object-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('bc','zero','train','evaluate'),required=True)
    p.add_argument('--rl-checkpoint',type=Path)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--seed',type=int,default=216)
    p.add_argument('--updates',type=int,default=20)
    p.add_argument('--seconds',type=int,default=2400)
    p.add_argument('--alpha',type=float,default=.15)
    p.add_argument('--bc-kind',choices=('wam','mlp'),default='wam')
    p.add_argument('--experiment-id',default='P-20261016-shadow-bc-ppo')
    p.add_argument('--policy-seed',type=int,default=231)
    p.add_argument('--exploration-seed',type=int,default=232)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--rendezvous',default='tcp://127.0.0.1:29541')
    p.add_argument('--trace-every',type=int,default=1)
    p.add_argument('--initial-log-std',type=float,default=-2.5)
    p.add_argument('--synchronize-warm-step',action='store_true')
    p.add_argument('--capture-warm-state',action='store_true')
    p.add_argument('--retain-reset-indices',action='store_true')
    p.add_argument('--single-reset-target-setter',action='store_true')
    p.add_argument('--force-zero-residual',action='store_true')
    p.add_argument('--outside-support-zero',action='store_true')
    p.add_argument('--observation-kind',choices=('native','normalized','bc-hidden','wam-none','wam-aligned','wam-shuffled'),default='native')
    p.add_argument('--network-kind',choices=('small','large'),default='small')
    p.add_argument('--action-kind',choices=('residual','reference-gaussian'),default='residual')
    p.add_argument('--reward-kind',choices=('native','hold'),default='native')
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--ppo-minibatches',type=int,default=1)
    p.add_argument('--rollout-steps',type=int,default=199)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve();a.cache=a.cache.resolve()
    if a.rl_checkpoint:a.rl_checkpoint=a.rl_checkpoint.resolve()
    identity=ROOT/'outputs/wam-rl-policy/deployed-source.json'
    a.code_commit=(json.loads(identity.read_text())['git_commit'] if identity.exists() else
        os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip())
    if not 1<=a.envs<=256 or not 1<=a.updates<=200 or a.alpha!=.15 or not 1<=a.world_size<=4 or not -5<=a.initial_log_std<=-1.2:
        raise ValueError('Scale bounds:1..256env/rank,1..200updates,1..4ranks,alpha0.15')
    if a.mode=='evaluate' and a.rl_checkpoint is None:raise ValueError('Evaluate requires RL checkpoint')
    if a.rl_checkpoint and a.mode not in ('train','evaluate'):raise ValueError('RL checkpoint only for train resume/evaluate')
    if a.capture_warm_state and not a.synchronize_warm_step:raise ValueError('Warm capture requires explicitly synchronized probe')
    if a.force_zero_residual and a.mode!='evaluate':raise ValueError('Matched loaded-policy zero control is evaluate-only')
    if not 1<=a.rollout_steps<=199:raise ValueError('PPO chunk must contain 1..199 native transitions')
    if not 1<=a.ppo_epochs<=10 or not 1<=a.ppo_minibatches<=8:raise ValueError('PPO optimization budget bound')
    if a.worker:worker(a);return
    if a.output.exists() or ROOT/'outputs/wam-rl-policy' not in a.output.parents:raise ValueError('Fresh Task output')
    occupied=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('Assigned GPU occupied')
    a.output.mkdir(parents=True);(a.output/'tmp').mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(a.output/'tmp'),
        TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR',str(ROOT/'tmp/shadow-isaac-extensions')),MAX_JOBS='2',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH']=str(Path(sys.executable).resolve().parents[1]/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    env['PATH']=str(Path(sys.executable).resolve().parent)+os.pathsep+env.get('PATH','')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();reason='CHILD_EXIT'
    with (a.output/'worker.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        process=dict(status='RUNNING',supervisor_pid=os.getpid(),pid=child.pid,command=command,
            run_id=a.output.name,git_commit=a.code_commit,budget_seconds=a.seconds,gpu=a.gpu)
        def save():
            part=a.output/'process.json.part';part.write_text(json.dumps(process,indent=2)+'\n');part.replace(a.output/'process.json')
        save()
        while child.poll() is None:
            elapsed=time.monotonic()-start
            if stop[0] or (a.output/'STOP').exists() or elapsed>a.seconds+60:
                reason='USER_STOP' if stop[0] or (a.output/'STOP').exists() else 'TIME_BUDGET'
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                break
            resource=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (a.output/'resources.jsonl').open('a') as stream:stream.write(json.dumps(dict(elapsed_s=elapsed,gpu=resource))+'\n')
            time.sleep(10)
        code=child.wait();process.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,reason=reason,elapsed_s=time.monotonic()-start)
        save()
    if code:raise SystemExit(code)


if __name__=='__main__':main()
