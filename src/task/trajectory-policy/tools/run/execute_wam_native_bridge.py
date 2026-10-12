"""Three nominal simulated rows: official teacher, oracle GT-R, live WAM-R."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
OLD=ROOT/'src/task/consequence-evaluator'
RUNTIME=Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit'),str(TASK/'tools/run')]
from audit_wam_ref2_shadow import sha,write
from trajectory_policy.wam_ipc import SCHEMA,send,receive,peer_pid
from collect_competent_pd import OFFICIAL,OFFICIAL_SHA,gpu_state


def partial_grasp_outcome(packet):
    """Measured hold45 component only; a truncated rollout cannot judge placement."""
    pose=np.asarray(packet['object_pose']);count=len(pose)
    gap=np.asarray(packet['surface_gap']);support=np.asarray(packet['support_gap'])
    footprint=np.asarray(packet['table_footprint']);velocity=np.asarray(packet['object_velocity'])
    if (count<2 or pose.shape!=(count,4,4) or gap.shape!=(count,)
            or support.shape!=(count,) or footprint.shape!=(count,)
            or velocity.shape!=(count,6) or any(not np.isfinite(x).all()
                for x in (pose,gap,support,velocity)) or np.any(gap<0)):
        raise ValueError('finite measured partial rollout required')
    height=pose[:,2,3]-pose[0,2,3]
    valid=np.arange(count)>0
    supported=valid&footprint.astype(bool)&(np.abs(support)<=.02)
    held=valid&(gap<=.01)&~supported&(height>=.03)
    run=np.zeros(count,np.int64)
    for i in range(1,count):
        run[i]=run[i-1]+1 if held[i] else 0
    stable=np.flatnonzero(run>=45)
    return dict(maximum_held_frames=int(run.max()),
        first_stable_tick=int(stable[0]) if len(stable) else None,
        last_stable_tick=int(stable[-1]) if len(stable) else None,
        terminal_height_m=float(height[-1]),metric='partial_hold45_only_no_placement')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--server-pid',type=int,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--expert-recovery',action='store_true')
    p.add_argument('--controls',type=int,default=542)
    p.add_argument('--takeover-tick',type=int,default=16)
    p.add_argument('--experiment-id',default=None)
    p.add_argument('--head-dir',type=Path,default=ROOT/'outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1')
    p.add_argument('--r-dir',type=Path,default=None)
    args=p.parse_args()
    args.experiment_id=args.experiment_id or ('P-20261012-goal-point-wam-ref2-recovery' if args.expert_recovery else 'P-20261012-goal-point-wam-ref2-execution')
    if not 1<=args.controls<=542 or (args.expert_recovery and not 0<=args.takeover_tick<args.controls):
        raise ValueError('bounded controls and in-horizon takeover required')
    out=args.output.resolve()
    roles=['teacher','gt_tau_R','wam_R']+([f'wam_command_replay_then_expert_tick{args.takeover_tick}'] if args.expert_recovery else [])
    env_count=len(roles)
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents or Path(sys.prefix).resolve()!=RUNTIME.resolve():
        raise ValueError('fresh owned output and pinned native runtime required')
    scratch=args.socket.parent.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/consequence-official-generator/torch201-extensions'),
        OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    native_root=ROOT/'third_party/DExplore/dexplore'
    sys.path[:0]=[str(Path('/home2/wyy/isaac-gym/isaacgym/python')),str(OLD/'src'),
        str(OLD/'tools/audit'),str(ROOT),str(native_root),str(ROOT/'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # must initialize before Torch
    import dexplore_ddp_rank_bootstrap
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat,model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry,poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.pd_retargeter import load_model,decode_target,command,pd_from_action
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.reference_actor import reference_from_processed
    if torch.__version__!='2.0.1+cu118' or sha(OFFICIAL)!=OFFICIAL_SHA:
        raise ValueError('pinned native runtime/teacher identity required')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(292)
    run_dir=ROOT/'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base'
    cfg=json.loads((run_dir/'config.json').read_text())
    source=ROOT/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    source_manifest=json.loads((source/'manifest.json').read_text())
    if (source_manifest['status']!='COMPLETED' or not source_manifest['training_allowed']
        or not json.loads((source/'result.json').read_text())['teacher_readiness']
        or sha(source/'trajectory.npz')!=source_manifest['trajectory_sha256']):
        raise ValueError('authenticated source oracle calibration required')
    producer_updates=[]
    for name,digest in source_manifest['input_sha256'].items():
        path=Path(name)
        if sha(path)==digest:
            continue
        if path!=TASK/'tools/run/collect_competent_pd.py':
            raise ValueError('calibrated source native/data/config drift: '+name)
        relative=path.relative_to(ROOT).as_posix()
        blob=subprocess.check_output(['git','show',source_manifest['git_commit']+':'+relative],cwd=ROOT)
        import hashlib
        if hashlib.sha256(blob).hexdigest()!=digest:
            raise ValueError('source producer Git blob differs from recorded identity')
        producer_updates.append(dict(path=name,producer_sha256=digest,consumer_sha256=sha(path),
                                     authenticated_git_commit=source_manifest['git_commit']))
    with np.load(source/'trajectory.npz',allow_pickle=False) as stream:
        oracle=stream['hand'][:,24].copy()  # privileged GT-R arm only
    reference_path=ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz'
    reference_manifest=json.loads((reference_path.parent/'manifest.json').read_text())
    head_dir=args.head_dir.resolve()
    if ROOT/'outputs/trajectory-policy' not in head_dir.parents:
        raise ValueError('task-owned authenticated head required')
    head_manifest=json.loads((head_dir/'manifest.json').read_text())
    if (reference_manifest['status']!='COMPLETED' or not reference_manifest['planned_reference_only']
        or sha(reference_path)!=reference_manifest['reference_sha256']
        or sha(reference_path)!=head_manifest['input_sha256'][str(reference_path)]):
        raise ValueError('planned reference differs from export or head calibration identity')
    with np.load(reference_path,allow_pickle=False) as stream:
        reference={k:stream[k].copy() for k in stream.files}
    from trajectory_policy.wam_retargeter_admission import retargeter_admission
    r_path,r_identity,r_hashes=retargeter_admission(args.r_dir,head_manifest)
    r=load_model(torch.load(r_path,map_location='cuda:0'),'cuda:0')
    for parameter in r.parameters():
        parameter.requires_grad_(False)
    head_path=head_dir/'head.pt'
    head_result=json.loads((head_dir/'result.json').read_text())
    if head_manifest['status']!='COMPLETED' or sha(head_path)!=head_result['head_sha256']:
        raise ValueError('authenticated completed frozen head required')
    from wam_native_server import scene_patch_admission
    head_packet=torch.load(head_path,map_location='cpu')
    base_path=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    base_hash=sha(base_path)
    if (head_packet.get('base_checkpoint_sha256')!=base_hash
            or head_manifest.get('base_checkpoint_sha256')!=base_hash):
        raise ValueError('native peer head base identity differs')
    patch_identity=scene_patch_admission(head_packet,head_manifest,head_result)
    del head_packet
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    connection.settimeout(30);connection.connect(str(args.socket))
    if peer_pid(connection)!=args.server_pid:
        raise ValueError('wrong local inference server PID')
    handshake=receive(connection)
    if (handshake['schema']!=SCHEMA or handshake['pid']!=args.server_pid
        or handshake['physical_gpu']!=args.gpu or handshake['head_sha256']!=sha(head_path)
        or handshake.get('controls')!=args.controls
        or handshake.get('experiment_id')!=args.experiment_id
        or handshake['base_sha256']!=base_hash
        or handshake.get('scene_patch_identity')!=patch_identity
        or handshake.get('retargeter_identity')!=r_identity):
        raise ValueError('WAM inference peer identity differs')
    helper=TASK/'src/trajectory_policy/wam_scene_patch.py'
    if handshake['input_sha256'].get(str(helper))!=sha(helper):
        raise ValueError('WAM inference scene-patch helper source differs')
    r_helper=TASK/'src/trajectory_policy/wam_retargeter_admission.py'
    if handshake['input_sha256'].get(str(r_helper))!=sha(r_helper):
        raise ValueError('WAM inference retargeter-admission helper source differs')
    initial_gpu=gpu_state(args.gpu)
    if set(initial_gpu['pids'])-{os.getpid(),args.server_pid}:
        raise RuntimeError('foreign GPU process before simulation')
    files=[Path(__file__),TASK/'src/trajectory_policy/wam_ipc.py',TASK/'src/trajectory_policy/pd_retargeter.py',
        TASK/'src/trajectory_policy/retargeter_data.py',r_path,OFFICIAL,reference_path,source/'trajectory.npz',
        source/'manifest.json',run_dir/'config.json',Path(cfg['cfg_env']),head_path,
        helper,r_helper,TASK/'tools/run/wam_native_server.py',base_path]
    if args.expert_recovery:
        files.append(TASK/'tools/run/wam_expert_probe.py')
    files += [reference_path.parent/'manifest.json',head_dir/'manifest.json',head_dir/'result.json']
    files += list((OLD/'src/consequence_evaluator').glob('*.py'))+list(native_root.rglob('*.py'))
    files += list(Path(cfg['motion_root']).glob('*/interaction_hand_inspire.pt'))
    train_cfg=native_root/'data/cfg/train/rlg/inspire.yaml';files.append(train_cfg)
    hashes={str(path.resolve()):sha(path) for path in files}
    hashes.update(r_hashes)
    out.mkdir(parents=True)
    manifest=dict(schema='ref2dex.wam-native-recovery.v1' if args.expert_recovery else 'ref2dex.wam-native-execution.v1',status='RUNNING',seed=292,physical_gpu=args.gpu,
        task='goal-point-wam',experiment_id=args.experiment_id,run_id=out.parent.name,
        roles=roles,controls=args.controls,partial_episode=args.controls<542,dt=1/30,training_allowed=args.expert_recovery,
        training_scope='visited-state expert PD only; mixed-policy future hand forbidden' if args.expert_recovery else 'none',
        takeover_tick=args.takeover_tick if args.expert_recovery else None,
        expert_PD_training_allowed=args.expert_recovery,mixed_policy_future_hand_training_allowed=False,
        actor_future_hand_training_allowed=False,
        actual_consequence_scope='executed action plus measured next state; not expert motion',
        expert_observation_semantics='current measured native state + loaded planned HOI hand/body/object/contact reference at t+1/t+16',
        server_pid=args.server_pid,native_pid=os.getpid(),server_identity=handshake,input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_shadow_screen='FAILED; root Decision authorizes bounded simulation after frame/role gates',
        actor_period=1,oracle_source_env=24,wam_input='live measured H4 + planned object endpoint min(t+24,542)',
        no_wam_oracle_prefix=True,controller='frozen actual-PD R, native conversion, no extra feedforward')
    manifest['retargeter_identity']=r_identity
    manifest['producer_code_updates']=producer_updates
    write(out/'manifest.json',manifest)
    install_legacy_player_compat(native,common_player,base_player);install_reset_patch()
    started=time.monotonic()
    captured_logs={}
    executed_controls=0

    class Player(native.EvalPlayer):
        def restore(self,filename):
            value=native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(value['model']),strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(value['running_mean_std'],strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(value['amp_input_mean_std'],strict=True)

        def run(self):
            nonlocal executed_controls
            task=self.env.task
            if task.num_envs!=env_count or self.is_rnn or str(task.device)!='cuda:0':
                raise ValueError('exact declared-row nonrecurrent GPU pipeline required')
            task._state_init=DexploreTask.StateInit.Start;task._hybrid_init_prob=1.
            task._adaptive_kappa_enabled=False;task._enable_early_termination=False
            obs=self.env_reset(torch.arange(env_count,device=task.device));self.get_batch_size(obs['obs'],1)
            if (task.start_times!=0).any():
                raise ValueError('frame0 reset required')
            geometry=PhysicalGeometry(task,native_root/'data/assets',distance_device=task.device)
            support=TableSupport(native_root/'data/assets',task.device)
            offset,scale=task._pd_action_offset.clone(),task._pd_action_scale.clone()
            manifest['native_controller']=dict(offset=offset.tolist(),scale=scale.tolist())
            measured_reference=reference_from_processed(task.hoi_data[task.data_id[0]].cpu().numpy())
            errors={k:float(np.max(np.abs(measured_reference[k]-reference[k]))) for k in reference}
            if max(errors.values())>2e-5:
                raise ValueError('exported object goal reference differs from native HOI reference')
            manifest['native_reference_max_errors']=errors
            actual_pd=capture=None
            original_pre,original_pd=task.pre_physics_step,task._action_to_pd_targets

            def pre(action):
                nonlocal capture
                capture=action.clone();return original_pre(action)

            def pd(action):
                nonlocal actual_pd
                actual_pd=original_pd(action).clone();return actual_pd

            task.pre_physics_step,task._action_to_pd_targets=pre,pd
            logs={k:[] for k in ('q','dq','hand','obj','velocity','surface_gap','support_gap','table_footprint',
                'action','applied','pd_targets','intended','requested_pd','clipped','state_features','tau_features',
                'encoded_target','wam_object_history','wam_hand_history','wam_goal','server_latency_s','native_latency_s')}
            if args.expert_recovery:
                from wam_expert_probe import expert_on_rows,paired_prefix_error,prefix_physical_metrics,PREFIX_PHYSICAL_BOUNDS
                logs.update({key:[] for key in ('expert_raw','expert_action','expert_clipped','expert_PD',
                    'expert_observation','learner_state','learner_object_history','learner_hand_history','learner_goal',
                    'learner_wam_tau','learner_wam_tau_valid','expert_applied')})
                queries=out/'expert_queries';queries.mkdir()
                manifest['prefix_pair_max_errors']={}
                manifest['expert_query_sha256']={}
                manifest['prefix_physical_bounds']=PREFIX_PHYSICAL_BOUNDS
                manifest['prefix_physical_max_metrics']={}
                manifest['prefix_pairing']='same normalized command prefix, independent PhysX measured states; not exact hidden-state fork'
            captured_logs.update(logs)

            def record():
                hand,gap=geometry.measure(task);sup,foot=support.measure(task,geometry)
                current=dict(q=task._dof_pos.clone(),dq=task._dof_vel.clone(),hand=hand,
                    obj=poses(task._target_states),velocity=task._target_states[:,7:13].clone(),
                    surface_gap=gap,support_gap=sup,table_footprint=foot)
                for k,v in current.items():
                    logs[k].append(v.cpu().numpy().copy())
                return current

            current=record()
            reset={k:float(np.max(np.abs(logs[k][0]-logs[k][0][:1]))) for k in ('q','dq','hand','obj','velocity')}
            if max(reset.values())>1e-5:
                raise ValueError('teacher/GT/WAM observed resets differ')
            manifest['reset_max_errors']=reset;write(out/'manifest.json',manifest)
            for tick in range(args.controls):
                before=time.monotonic()
                with torch.no_grad():
                    if args.expert_recovery:
                        expert_raw,expert_action,expert_clipped=expert_on_rows(self,obs,verify_official=tick==0)
                        expert_PD=pd_from_action(expert_action,current['q'],offset,scale)
                        if not torch.isfinite(expert_PD).all():
                            raise FloatingPointError('nonfinite current-q expert PD')
                        action=expert_action.clone()
                        if tick<=args.takeover_tick:
                            errors=paired_prefix_error(current)
                            manifest['prefix_pair_max_errors']={key:max(value,manifest['prefix_pair_max_errors'].get(key,0.)) for key,value in errors.items()}
                            metrics=prefix_physical_metrics(current)
                            manifest['prefix_physical_max_metrics']={key:max(value,manifest['prefix_physical_max_metrics'].get(key,0.)) for key,value in metrics.items()}
                            if any(metrics[key]>bound for key,bound in PREFIX_PHYSICAL_BOUNDS.items()):
                                raise ValueError('pre-takeover command-replay rows exceed declared physical comparability bounds')
                    else:
                        raw=obs['obs'];padded=raw.repeat((64+2)//3,1)[:64]
                        action=self.get_action(dict(obs,obs=padded),True)[:3].clamp(-1,1).clone()
                    intended=expert_raw.clone() if args.expert_recovery else action.clone()
                    requested=pd_from_action(intended,current['q'],offset,scale)
                    past=np.maximum(tick+np.arange(-3,1),0)
                    oh=np.stack([logs['obj'][i][2] for i in past])
                    hh=np.stack([logs['hand'][i][2] for i in past])
                    goal=reference['obj'][min(tick+24,542)]
                    send(connection,dict(tick=tick,object_history=oh.tolist(),hand_history=hh.tolist(),planned_object_goal=goal.tolist()))
                    reply=receive(connection)
                    if reply['schema']!=SCHEMA or reply['tick']!=tick or not np.array_equal(np.asarray(reply['anchor_object_pose'],np.float32),oh[-1]):
                        raise ValueError('inference reply uses another live current anchor')
                    wamtau=np.asarray(reply['tau'],np.float32)
                    if wamtau.shape!=(24,33) or not np.isfinite(wamtau).all():
                        raise ValueError('full native-ordered finite WAM tau required')
                    future=oracle[np.minimum(tick+np.arange(1,25),542)]
                    state_gt,tau_gt=features(*(current[k][1].cpu().numpy() for k in ('q','dq','hand','obj','velocity')),future)
                    # Construct current-only R state; the helper's dummy future
                    # has zero displacement and is discarded, never a label.
                    state_wam,_=features(*(current[k][2].cpu().numpy() for k in ('q','dq','hand','obj','velocity')),
                        np.broadcast_to(current['hand'][2].cpu().numpy(),(24,11,3)))
                    states=np.stack((state_gt,state_wam));taus=np.stack((tau_gt,wamtau))
                    encoded=r(torch.as_tensor(states,device=task.device),torch.as_tensor(taus,device=task.device))
                    target=decode_target(encoded,current['q'][1:3])
                    first,applied,_=command(target,current['q'][1:3],offset,scale)
                    action[1:3]=applied;intended[1:3]=first;requested[1:3]=target
                    if args.expert_recovery:
                        if tick<args.takeover_tick:
                            action[3]=action[2];intended[3]=intended[2]
                            requested[3]=pd_from_action(intended[3:4],current['q'][3:4],offset,scale)[0]
                        learner_states=[];learner_oh=[];learner_hh=[]
                        for env in (2,3):
                            state,_=features(*(current[k][env].cpu().numpy() for k in ('q','dq','hand','obj','velocity')),
                                np.broadcast_to(current['hand'][env].cpu().numpy(),(24,11,3)))
                            learner_states.append(state)
                            learner_oh.append(np.stack([logs['obj'][i][env] for i in past]))
                            learner_hh.append(np.stack([logs['hand'][i][env] for i in past]))
                        extra=dict(expert_raw=expert_raw[2:],expert_action=expert_action[2:],expert_clipped=expert_clipped[2:],
                            expert_PD=expert_PD[2:],expert_observation=obs['obs'][2:].clone(),
                            learner_state=np.stack(learner_states),learner_object_history=np.stack(learner_oh),
                            learner_hand_history=np.stack(learner_hh),learner_goal=np.repeat(goal[None],2,axis=0),
                            learner_wam_tau=np.stack((wamtau,np.zeros_like(wamtau))),
                            learner_wam_tau_valid=np.array([True,False]),expert_applied=np.array([False,tick>=args.takeover_tick]))
                        extra={key:value.cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value).copy()
                               for key,value in extra.items()}
                        # A durable query record exists before advancing physics; it is not a future-hand label.
                        if not all(np.isfinite(value).all() for value in extra.values()):
                            raise FloatingPointError('nonfinite expert query data')
                        query_path=queries/f'{tick:04d}.npz'
                        np.savez_compressed(query_path,tick=np.asarray(tick),
                            current_q=current['q'][2:].cpu().numpy(),**extra)
                        manifest['expert_query_records']=tick+1
                        manifest['expert_query_sha256'][str(query_path.relative_to(out))]=sha(query_path)
                    expected=pd_from_action(action,current['q'],offset,scale)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('nonfinite requested native action')
                    obs,_,done,_=self.env_step(self.env,action.clone())
                    executed_controls+=1
                    if not isinstance(obs,dict):
                        obs={'obs':obs}
                    if capture is None or not torch.equal(capture,action) or actual_pd is None or (actual_pd-expected).abs().max()>1e-6:
                        raise ValueError('native applied command/actual PD capture mismatch')
                    if args.expert_recovery and tick>=args.takeover_tick and (actual_pd[3]-expert_PD[3]).abs().max()>1e-6:
                        raise ValueError('applied takeover PD differs from current-q expert label')
                    if tick<args.controls-1 and done.any():
                        raise ValueError('early reset invalidates full episode')
                    for k,v in dict(action=action,applied=capture,pd_targets=actual_pd,intended=intended,
                        requested_pd=requested,clipped=(intended-action).abs().amax(-1)>1e-6,
                        state_features=states,tau_features=taus,encoded_target=encoded,
                        wam_object_history=oh,wam_hand_history=hh,wam_goal=goal,
                        server_latency_s=reply['elapsed_s'],native_latency_s=time.monotonic()-before).items():
                        logs[k].append(v.cpu().numpy().copy() if isinstance(v,torch.Tensor) else np.asarray(v).copy())
                    if args.expert_recovery:
                        if tick<args.takeover_tick and not torch.equal(capture[2],capture[3]):
                            raise ValueError('pre-takeover paired learner normalized actions differ')
                        for key,value in extra.items():
                            logs[key].append(value)
                    current=record()
                if tick==0 or (tick+1)%64==0:
                    gpu=gpu_state(args.gpu)
                    if set(gpu['pids'])-{os.getpid(),args.server_pid}:
                        raise RuntimeError('foreign GPU process during episode')
                    row=dict(control=tick+1,elapsed_s=time.monotonic()-started,
                        eta_s=(time.monotonic()-started)/(tick+1)*(args.controls-1-tick),gpu=gpu)
                    print(json.dumps(row),flush=True)
                    with (out/'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
            packed={k:np.stack(v) for k,v in logs.items()}
            np.savez_compressed(out/'trajectory.npz',**packed)
            outcomes=[]
            for env,role in enumerate(manifest['roles']):
                outcome=partial_grasp_outcome if args.controls<542 else episode_outcome
                value=outcome(dict(object_pose=packed['obj'][:,env],object_velocity=packed['velocity'][:,env],
                    surface_gap=packed['surface_gap'][:,env],support_gap=packed['support_gap'][:,env],table_footprint=packed['table_footprint'][:,env]))
                value.update(env=env,role=role,clipping_rate=float(packed['clipped'][:,env].mean()))
                outcomes.append(value)
            result=dict(status='COMPLETED',outcomes=outcomes,
                partial_episode=args.controls<542,
                calibration_pass=None if args.controls<542 else all(v['success'] and v['maximum_held_frames']>=45 for v in outcomes[:2]),
                server_latency_median_s=float(np.median(packed['server_latency_s'])),
                native_latency_median_s=float(np.median(packed['native_latency_s'])),
                scope='one nominal simulated episode per row; expert takeover is training-time diagnosis, never autonomous success' if args.expert_recovery
                    else 'one nominal simulated episode per row; no hidden-state fork or robot-domain generalization claim')
            if args.controls<542:
                result.update(recovery_pass=None,
                    local_grasp_calibration=all(v['maximum_held_frames']>=45 for v in outcomes[:2]),
                    scope='truncated rollout; local grasp only, no placement verdict')
            if args.expert_recovery:
                value=outcomes[3]
                if args.controls<542:
                    result.update(recovery_pass=None,recovery_conclusion='UNCLEAR',
                        local_grasp_calibration=all(v['maximum_held_frames']>=45 for v in outcomes[:2]),
                        local_grasp_recovery=bool(value['maximum_held_frames']>=45 and value['clipping_rate']<.01),
                        scope='truncated training-time expert-query rollout; local grasp only, no placement or autonomous recovery verdict')
                else:
                    result['recovery_pass']=bool(result['calibration_pass'] and value['success']
                        and value['maximum_held_frames']>=45 and value['clipping_rate']<.01)
                    result['recovery_conclusion']='PROMISING' if result['recovery_pass'] else (
                        'UNPROMISING' if result['calibration_pass'] and value['maximum_held_frames']==0 else 'UNCLEAR')
                result['expert_query_records']=manifest['expert_query_records']
                result['expert_query_clipping_rates']=packed['expert_clipped'].mean(0).tolist()
                result['expert_query_scope']='actual learner observations, privileged planned HOI reference; PD labels only'
            if any(sha(path)!=digest for path,digest in hashes.items()):
                raise ValueError('native frozen input/source drift')
            write(out/'result.json',result)
            manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started,
                trajectory_sha256=sha(out/'trajectory.npz'),result_sha256=sha(out/'result.json'))
            write(out/'manifest.json',manifest);print(json.dumps(result,indent=2))

    native.EvalPlayer=Player
    sys.argv=[sys.argv[0],'--task','Dexplore_Inspire','--cfg_env',cfg['cfg_env'],'--cfg_train',str(train_cfg),
        '--motion_file',cfg['motion_root'],'--checkpoint',str(OFFICIAL),'--headless','--num_envs',str(env_count),'--seed','292',
        '--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0','--num_threads','1',
        '--disable-early-termination','--output',str(out/'unused.json'),'--output_path',str(out/'native')]
    cwd=Path.cwd()
    def deadline(signum,frame):
        raise TimeoutError('bounded native WAM closure deadline600s')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        os.chdir(ROOT/'third_party/DExplore');native.main()
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started,
                        executed_controls=executed_controls)
        try:
            if captured_logs:
                partial={key:np.stack(value) for key,value in captured_logs.items() if value}
                if partial:
                    np.savez_compressed(out/'partial.npz',**partial)
                    manifest.update(partial_controls=len(captured_logs.get('action',[])),
                            partial_measured_states=len(captured_logs.get('q',[])),
                        partial_sha256=sha(out/'partial.npz'),partial_only=True)
        except Exception as recording_error:
            manifest['failure_recording_error']=repr(recording_error)
        write(out/'manifest.json',manifest);raise
    finally:
        signal.alarm(0);os.chdir(cwd);connection.close()


if __name__=='__main__':
    main()
