"""Ordinary random-plan rollouts and prospective rolling planning; never fork."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT),str(ROOT/'third_party/DExplore/dexplore'),
              str(ROOT/'src/task/cm-interaction-oracle/src')]
from consequence_evaluator.contracts import is_within, HAND_LINKS
from consequence_evaluator.retarget_collection import (
    ACTIVE_FINGERS, MODE_NAMES, PHASE_NAMES, STRUCTURED_PROFILE_NAMES,
    finger_pulse_residual, phase_code, sample_structured_residual,
    validate_residual_family)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--envs',type=int,default=64);p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--mode',choices=('random','baseline','planner','retarget'),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--bridge',type=Path)
    p.add_argument('--initial-jitter',action='store_true',help='fixed small seeded q perturbation for distinct matched episodes')
    p.add_argument('--save-actor-observation',action='store_true',
                   help='save the actor observation at every retained frame for H-matched audits')
    p.add_argument('--structured-profile',choices=STRUCTURED_PROFILE_NAMES,default='random',
                   help='retarget residual schedule; finger-pulse is a serial diagnostic only')
    p.add_argument('--pulse-start-tick',type=int,default=50)
    p.add_argument('--pulse-end-tick',type=int,default=51)
    p.add_argument('--pulse-finger-index',type=int,choices=tuple(int(i) for i in ACTIVE_FINGERS),default=6)
    p.add_argument('--pulse-value',type=float,default=.08)
    a=p.parse_args();out=a.output.resolve();cfg=json.loads(a.inputs.read_text())
    if a.bridge is not None:a.bridge=a.bridge.resolve()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not 1<=a.envs<=96 or not 1<=a.seconds<=900:
        raise ValueError('fresh bounded rollout required')
    if a.mode=='planner' and a.bridge is None:raise ValueError('frozen hand bridge required')
    if a.structured_profile != 'random' and a.mode != 'retarget':
        raise ValueError('structured profile is only valid for retarget mode')
    if a.structured_profile == 'finger-pulse':
        if not 0 <= a.pulse_start_tick < a.pulse_end_tick <= 542:
            raise ValueError('finger-pulse interval must lie within the 542 commands')
        if not np.isfinite(a.pulse_value) or abs(a.pulse_value) > .120001:
            raise ValueError('finger-pulse value exceeds the registered bound')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch=ROOT/'tmp/hand-execution';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'),
                      TRITON_CACHE_DIR=str(scratch/'triton'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    from isaacgym import gymtorch
    import torch
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.reset_kinematics import task_kinematics
    from consequence_evaluator.physical_geometry import PhysicalGeometry,poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.hand_execution import SCHEMA
    from oracle_y_utility import candidate_deltas,align_native_reference_tables
    from src.task.CmResidual.paired_evaluation import fingerprint
    if torch.__version__!='2.4.1+cu121':raise ValueError('pinned graspenv runtime required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    install_reset_patch()
    reset=DexploreTask._reset_ref_state_init
    def aligned(task,ids):
        align_native_reference_tables(task)
        return reset(task,ids)
    DexploreTask._reset_ref_state_init=aligned
    loader=native.torch_ext.load_checkpoint
    def load(path):
        state=loader(path);keys=list(state['model']);compiled=[k.startswith('_orig_mod.') for k in keys]
        if any(compiled) and not all(compiled):raise ValueError('mixed compiled keys')
        if all(compiled):state=dict(state,model={k[10:]:v for k,v in state['model'].items()})
        return state
    native.torch_ext.load_checkpoint=load
    # The original input manifest predates this optional observation field.  A
    # history-preserving probe is a new capture contract, so refresh only the
    # runner hash in memory and record the override instead of mutating the
    # frozen input JSON on disk.
    runner_key = str(Path(__file__).resolve())
    runner_hash_override = None
    if a.save_actor_observation:
        declared_runner_hash = cfg.get('input_sha256', {}).get(runner_key)
        actual_runner_hash = sha(__file__)
        if declared_runner_hash != actual_runner_hash:
            cfg = dict(cfg, input_sha256=dict(cfg.get('input_sha256', {})))
            cfg['input_sha256'][runner_key] = actual_runner_hash
            runner_hash_override = dict(declared=declared_runner_hash, actual=actual_runner_hash)
    hashes={str(a.inputs.resolve()):sha(a.inputs),runner_key:sha(__file__)}
    hashes.update(cfg['input_sha256'])
    for path in (TASK/'src/consequence_evaluator').glob('*.py'):hashes[str(path)]=sha(path)
    if a.bridge:hashes[str(a.bridge.resolve())]=sha(a.bridge)
    if any(sha(path)!=h for path,h in hashes.items()):raise ValueError('frozen input drift')
    out.mkdir(parents=True);begin=time.monotonic()
    action_semantics = (f'captured full native action: actor action plus {a.structured_profile} structured residual; '
                        'retarget model input excludes actor action' if a.mode == 'retarget' else
                        'requested24x18 residual, first8nonzero then16zero; replanning every8only in planner')
    manifest=dict(schema=SCHEMA,status='RUNNING',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        mode=a.mode,seed=a.seed,num_envs=a.envs,physical_gpu=a.gpu,budget_s=a.seconds,smoke=a.smoke,
        input_sha256=hashes,actor_sha256=cfg['actor_sha256'],rollout_kind='ordinary single-world feedback, no forks',
        action_semantics=action_semantics,
        actor_observation_saved=bool(a.save_actor_observation),
        actor_observation_contract=('obs["obs"] at every reset/state frame, aligned with trajectory frame t'
                                    if a.save_actor_observation else None),
        runner_hash_override=runner_hash_override,
        initialization='full-reference frame0 with fixed seeded q jitter' if a.initial_jitter else 'full-reference frame0',
        initial_jitter=a.initial_jitter,fps=30,old_Y_unchanged=True,
        contact_capture=dict(hand_field='native_contact_forces', object_field='native_object_contact_forces',
                             pair_rule='hand force norm>.1 any AND object force norm>.1'))
    if a.mode == 'retarget':
        manifest.update(structured_residual_schema='ref2dex.structured-residual.v1',
                        structured_profile=a.structured_profile,
                        structured_pulse=(dict(start_tick=a.pulse_start_tick,
                                               end_tick=a.pulse_end_tick,
                                               finger_index=a.pulse_finger_index,
                                               value=a.pulse_value)
                                         if a.structured_profile == 'finger-pulse' else None),
                        structured_modes=list(MODE_NAMES), structured_phases=list(PHASE_NAMES),
                        phase_boundaries={'approach_end_tick':120, 'contact_end_tick':240},
                        residual_bounds={'wrist_translation':[0.01, 0.01, 0.015],
                                         'wrist_rotation':[0.035, 0.035, 0.05],
                                         'active_finger':0.12},
                        saved_state_fields=['dof_position','dof_velocity'],
                        target_field='action[t:t+24] captured after actor+residual composition',
                        native_pd_target_field=('task.real_pd_tar captured during pre_physics_step; '
                                                'Inspire active order [14,15,6,8,12,10]'))
    write(out/'manifest.json',manifest)
    class Player(native.EvalPlayer):
        @torch.no_grad()
        def run(self):
            task=self.env.task;device=task.device;n=task.num_envs
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False
            task._state_init=DexploreTask.StateInit.Start;task._hybrid_init_prob=1.
            self.model.eval();ids=torch.arange(n,device=device)
            obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
            if self.is_rnn:raise ValueError('feedforward frozen actor required')
            if (task.start_times!=0).any() or len(task.motion_file)!=1:raise ValueError('one full frame0 motion required')
            if a.initial_jitter:
                # Native Start alone has no seed-dependent task variation.
                # Apply known, matched perturbations before the first physics
                # step. The reset FK keeps caches/observations coherent.
                gen=np.random.default_rng(a.seed)
                scale=np.array([.001]*3+[.005]*3+[.01]*12,np.float32)
                noise=torch.from_numpy(gen.uniform(-1,1,(n,18)).astype('float32')*scale).to(device)
                q=task._dof_pos.clone()+noise;velocity=task._dof_vel.clone()
                task._set_env_state(ids,q,velocity)
                task._reset_env_tensors(ids)
                reset_fk=task_kinematics(task)
                bodies=task._rigid_body_state.view(n,-1,13)
                bodies[:,:task.num_bodies]=reset_fk.states(task._dof_pos,task._dof_vel,task._humanoid_root_states)
                task._contact_forces[:]=0;task._tar_contact_forces[:]=0
                task._compute_observations(ids)
                obs={'obs':task.obs_buf.to(self.device).clone()}
                manifest['initial_jitter_request_scale']=scale.tolist()
            manifest.update(actor_fingerprint=fingerprint(self.model.state_dict()),
                rms_fingerprint=fingerprint(self.running_mean_std.state_dict()),
                initial_q_sha256=fingerprint(task._dof_pos),initial_object_sha256=fingerprint(task._target_states),
                native_reference_steps=int(task.max_episode_length[task.data_id].max()-1),
                pipeline=str(device),physics_gpu=bool(task.gym.get_sim_params(task.sim).physx.use_gpu))
            geometry=PhysicalGeometry(task,ROOT/'third_party/DExplore/dexplore/data/assets',distance_device=self.device)
            support=TableSupport(ROOT/'third_party/DExplore/dexplore/data/assets',device)
            fk=task_kinematics(task);key_ids=geometry.key_ids
            delta=candidate_deltas(device);rng=np.random.default_rng(a.seed)
            active=np.ones(n,bool);lengths=np.zeros(n,int);commands=[];clipping=[];decisions=[]
            retarget_residuals=[];retarget_base=[];retarget_modes=[];retarget_phases=[]
            native_pd_targets=[]
            logs={k:[] for k in ('object_pose','hand_keypoints','surface_gap','support_gap','table_footprint',
                'object_velocity','reference_object_pose','pair','native_contact_forces',
                'native_object_contact_forces','dof_position','dof_velocity','done')}
            if a.save_actor_observation:
                logs['actor_observation'] = []
            plan=np.zeros((n,24,18),np.float32);ages=np.full(n,32);counts=np.zeros(n,int)
            structured=np.zeros((n,18),np.float32)
            structured_modes=np.full(n,-1,np.int8)
            capture=None;capture_pd=None;original_pre=task.pre_physics_step
            def pre(actions):
                nonlocal capture,capture_pd
                capture=actions.detach().cpu().numpy().copy()
                result=original_pre(actions)
                pd_target=getattr(task, 'real_pd_tar', None)
                if pd_target is None:
                    raise ValueError('native PD target capture failed')
                capture_pd=pd_target.detach().cpu().numpy().copy()
                return result
            task.pre_physics_step=pre
            planner=None
            if a.mode=='planner':
                from consequence_evaluator.hand_planner import HandPlanner
                planner=HandPlanner(a.bridge,cfg['C1'],cfg['PW'],cfg['canonical'])
            def measure():
                hand,gap=geometry.measure(task);sup,foot=support.measure(task,geometry)
                ref=task.hoi_data[task.data_id,task.progress_buf.clamp_max(task.hoi_data.shape[1]-1)]
                ref_state=torch.zeros(n,13,device=device);ref_state[:,:3]=ref[:,106:109];ref_state[:,3:7]=ref[:,109:113]
                contact=task._contact_forces[:,task._contact_body_ids]
                object_contact=task._tar_contact_forces
                if contact.ndim != 3 or contact.shape[-1] != 3 or object_contact.shape != (n,3):
                    raise ValueError('unexpected native contact-force tensor shape')
                pair=(contact.norm(dim=-1)>.1).any(-1)&(object_contact.norm(dim=-1)>.1)
                values=dict(object_pose=poses(task._target_states),hand_keypoints=hand,surface_gap=gap,
                    support_gap=sup,table_footprint=foot,object_velocity=task._target_states[:,7:13],
                    reference_object_pose=poses(ref_state),pair=pair,
                    native_contact_forces=contact,native_object_contact_forces=object_contact,
                    dof_position=task._dof_pos,dof_velocity=task._dof_vel)
                for k,v in values.items():logs[k].append(v.cpu().numpy().copy())
                return values
            measured=measure()
            if a.save_actor_observation:
                logs['actor_observation'].append(obs['obs'].detach().cpu().numpy().copy())
            for tick in range(700 if not a.smoke else 96):
                if time.monotonic()-begin>a.seconds:raise TimeoutError('rollout deadline')
                if tick%64==0:
                    pids=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
                    if any(int(pid)!=os.getpid() for pid in pids):raise RuntimeError('foreign GPU process')
                    if sum(x.stat().st_size for x in out.rglob('*') if x.is_file())>2*2**30:raise RuntimeError('artifact cap')
                    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                    print(json.dumps(dict(tick=tick,elapsed_s=round(time.monotonic()-begin,1),active=int(active.sum()),gpu=gpu)),flush=True)
                base=self.get_action(obs,True).clamp(-1,1).to(device).clone()
                base_np=base.cpu().numpy()
                if a.mode == 'retarget':
                    if a.structured_profile == 'random' and tick % 16 == 0:
                        phase = phase_code(tick)
                        structured, structured_modes = sample_structured_residual(rng, n, phase)
                    elif a.structured_profile == 'zero':
                        structured.fill(0.)
                        structured_modes.fill(-1)
                    elif a.structured_profile == 'finger-pulse':
                        structured.fill(0.)
                        structured_modes.fill(-1)
                        if a.pulse_start_tick <= tick < a.pulse_end_tick:
                            structured[:] = finger_pulse_residual(
                                n, a.pulse_finger_index, a.pulse_value)
                decide=tick>=40 and ((tick-40)%(8 if a.mode=='planner' else 32)==0)
                remaining=(task.max_episode_length[task.data_id]-1-task.progress_buf).cpu().numpy()
                selected=np.flatnonzero(active&(remaining>=32)) if decide else np.array([],int)
                if len(selected) and a.mode in ('random','planner'):
                    if a.mode=='random':
                        choices=(np.arange(n)+rng.integers(0,7,n))%7
                        plan[:]=0;plan[:,:8]=delta[torch.tensor(choices,device=device)].cpu().numpy()[:,None]
                        scores=None
                    else:
                        choices,scores=planner.choose(obs['obs'].cpu().numpy(),np.stack(logs['object_pose'][-4:]),np.stack(logs['hand_keypoints'][-4:]),selected)
                        plan[:]=0;plan[:,:8]=delta[torch.tensor(choices,device=device)].cpu().numpy()[:,None]
                    ages[selected]=0;counts[selected]+=choices[selected]!=0
                    # Nominal FK uses only current actor command held constant,
                    # current q/root and the known plan; no future feedback action.
                    nominal=[];current_q=task._dof_pos.clone();root=task._humanoid_root_states.clone()
                    for h in range(24):
                        control=(base+torch.tensor(plan[:,h],device=device)).clamp(-1,1)
                        native_q=task._action_to_pd_targets(control.clone())
                        nominal.append(fk.states(native_q,torch.zeros_like(native_q),root)[:,key_ids,:3].cpu().numpy())
                    for env in selected:
                        record=dict(tick=tick,env=int(env),history=obs['obs'][env].cpu().numpy().copy(),
                            action=plan[env].copy(),object_history=np.stack(logs['object_pose'][-4:])[:,env],
                            hand_history=np.stack(logs['hand_keypoints'][-4:])[:,env],
                            nominal=np.stack(nominal)[:,env],candidate=int(choices[env]))
                        if scores is not None:record['scores']=scores[env]
                        decisions.append(record)
                if a.mode == 'retarget':
                    requested=structured.copy()
                else:
                    requested=np.zeros((n,18),np.float32)
                    rows=np.flatnonzero(active&(ages<8));requested[rows]=plan[rows,ages[rows]]
                intended=base_np+requested;command=np.clip(intended,-1,1);command[~active]=0
                obs,_,done,info=self.env_step(self.env,torch.tensor(command,device=self.device))
                if not isinstance(obs,dict):obs={'obs':obs}
                if capture is None or capture.shape!=command.shape:raise ValueError('command capture failed')
                if not np.allclose(capture,command,atol=1e-7,rtol=0):raise ValueError('native action noise/command mutation')
                if capture_pd is None or capture_pd.shape[0] != n or not np.isfinite(capture_pd).all():
                    raise ValueError('native PD target capture failed')
                commands.append(capture.copy());clipping.append((np.abs(intended-command)>1e-7).any(-1)&active)
                native_pd_targets.append(capture_pd.copy())
                if a.mode == 'retarget':
                    retarget_residuals.append(requested.copy())
                    retarget_base.append(base_np.copy())
                    retarget_modes.append(structured_modes.copy())
                    retarget_phases.append(phase_code(tick))
                ended=done.cpu().numpy().astype(bool).reshape(-1);logs['done'].append(ended.copy())
                measured=measure();self._post_step(info)
                if a.save_actor_observation:
                    logs['actor_observation'].append(obs['obs'].detach().cpu().numpy().copy())
                lengths[active]+=1;active&=~ended;ages+=1
                if not active.any():break
                # Do not call env_reset after a terminal row. Its first episode
                # remains frozen for slicing even as the single world advances.
            if active.any() and not a.smoke:raise ValueError('incomplete full-reference episode')
            arrays={k:np.stack(v) for k,v in logs.items()};arrays.update(
                action=np.stack(commands), native_pd_target=np.stack(native_pd_targets),
                clipped=np.stack(clipping),length=lengths,interventions=counts)
            if a.mode == 'retarget':
                arrays.update(structured_residual=np.stack(retarget_residuals),
                              actor_action=np.stack(retarget_base), structured_mode=np.stack(retarget_modes),
                              structured_phase=np.asarray(retarget_phases,dtype=np.int8))
                validate_residual_family(arrays['structured_residual'])
            np.savez_compressed(out/'trajectory.npz',**arrays)
            windows=[]
            for r in decisions:
                t,e=r['tick'],r['env']
                if t+24>lengths[e]:continue
                r['hand_future']=arrays['hand_keypoints'][t+1:t+25,e]
                r['object_future']=arrays['object_pose'][t+1:t+25,e]
                windows.append(r)
            if windows:
                np.savez_compressed(out/'decisions.npz',**{k:np.stack([r[k] for r in windows]) for k in windows[0]})
            manifest.update(episodes=n,steps=lengths.tolist(),windows=len(windows),interventions=counts.tolist(),
                clipped_steps=int(arrays['clipped'].sum()),trajectory_sha256=sha(out/'trajectory.npz'))
            if windows:manifest['decisions_sha256']=sha(out/'decisions.npz')
            if planner is not None:
                planner.verify()
                manifest['planner_input_sha256']=planner.hashes
                manifest['PW_repeat_checked']=planner.first_repeat_checked
            write(out/'manifest.json',manifest)
    native.EvalPlayer=Player
    argv=['--task','Dexplore_Inspire','--cfg_env',cfg['cfg_env'],'--cfg_train',cfg['cfg_train'],
        '--checkpoint',cfg['actor'],'--motion_file',cfg['motions'],'--headless','--num_envs',str(a.envs),
        '--seed',str(a.seed),'--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','cpu',
        '--graphics_device_id','0','--num_threads','1','--disable-early-termination',
        '--output',str(out/'native-unused.json'),'--output_path',str(out/'native')]
    sys.argv=[sys.argv[0],*argv];cwd=Path.cwd()
    def deadline(signum,frame):raise KeyboardInterrupt('owned bounded rollout deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.alarm(a.seconds)
    try:
        os.chdir(ROOT/'third_party/DExplore');native.main()
        if any(sha(path)!=h for path,h in hashes.items()):raise ValueError('rollout input drift')
        manifest.update(status='SMOKE_COMPLETED' if a.smoke else 'COMPLETED')
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);os.chdir(cwd)
        manifest['elapsed_s']=time.monotonic()-begin;write(out/'manifest.json',manifest)


if __name__=='__main__':main()
