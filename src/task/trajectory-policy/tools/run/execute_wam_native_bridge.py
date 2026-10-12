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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--server-pid',type=int,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--head-dir',type=Path,default=ROOT/'outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1')
    args=p.parse_args()
    out=args.output.resolve()
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
    r_path=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    r_hash=json.loads((ROOT/'outputs/trajectory-policy/competent-pd-execution-20261011-r1/manifest.json').read_text())['input_sha256'][str(r_path)]
    if sha(r_path)!=r_hash:
        raise ValueError('physically calibrated R identity drift')
    r=load_model(torch.load(r_path,map_location='cuda:0'),'cuda:0')
    for parameter in r.parameters():
        parameter.requires_grad_(False)
    head_path=head_dir/'head.pt'
    head_result=json.loads((head_dir/'result.json').read_text())
    if head_manifest['status']!='COMPLETED' or sha(head_path)!=head_result['head_sha256']:
        raise ValueError('authenticated completed frozen head required')
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    connection.settimeout(30);connection.connect(str(args.socket))
    if peer_pid(connection)!=args.server_pid:
        raise ValueError('wrong local inference server PID')
    handshake=receive(connection)
    if (handshake['schema']!=SCHEMA or handshake['pid']!=args.server_pid
        or handshake['physical_gpu']!=args.gpu or handshake['head_sha256']!=sha(head_path)):
        raise ValueError('WAM inference peer identity differs')
    initial_gpu=gpu_state(args.gpu)
    if set(initial_gpu['pids'])-{os.getpid(),args.server_pid}:
        raise RuntimeError('foreign GPU process before simulation')
    files=[Path(__file__),TASK/'src/trajectory_policy/wam_ipc.py',TASK/'src/trajectory_policy/pd_retargeter.py',
        TASK/'src/trajectory_policy/retargeter_data.py',r_path,OFFICIAL,reference_path,source/'trajectory.npz',
        source/'manifest.json',run_dir/'config.json',Path(cfg['cfg_env']),head_path]
    files += [reference_path.parent/'manifest.json',head_dir/'manifest.json',head_dir/'result.json']
    files += list((OLD/'src/consequence_evaluator').glob('*.py'))+list(native_root.rglob('*.py'))
    files += list(Path(cfg['motion_root']).glob('*/interaction_hand_inspire.pt'))
    train_cfg=native_root/'data/cfg/train/rlg/inspire.yaml';files.append(train_cfg)
    hashes={str(path.resolve()):sha(path) for path in files}
    out.mkdir(parents=True)
    manifest=dict(schema='ref2dex.wam-native-execution.v1',status='RUNNING',seed=292,physical_gpu=args.gpu,
        task='goal-point-wam',experiment_id='P-20261012-goal-point-wam-ref2-execution',run_id=out.parent.name,
        roles=['teacher','gt_tau_R','wam_R'],controls=542,dt=1/30,training_allowed=False,
        server_pid=args.server_pid,native_pid=os.getpid(),server_identity=handshake,input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_shadow_screen='FAILED; root Decision authorizes bounded simulation after frame/role gates',
        actor_period=1,oracle_source_env=24,wam_input='live measured H4 + planned object endpoint min(t+24,542)',
        no_wam_oracle_prefix=True,controller='frozen actual-PD R, native conversion, no extra feedforward')
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
            if task.num_envs!=3 or self.is_rnn or str(task.device)!='cuda:0':
                raise ValueError('exact three-row nonrecurrent GPU pipeline required')
            task._state_init=DexploreTask.StateInit.Start;task._hybrid_init_prob=1.
            task._adaptive_kappa_enabled=False;task._enable_early_termination=False
            obs=self.env_reset(torch.arange(3,device=task.device));self.get_batch_size(obs['obs'],1)
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
            for tick in range(542):
                before=time.monotonic()
                with torch.no_grad():
                    raw=obs['obs'];padded=raw.repeat((64+2)//3,1)[:64]
                    action=self.get_action(dict(obs,obs=padded),True)[:3].clamp(-1,1).clone()
                    intended=action.clone();requested=pd_from_action(action,current['q'],offset,scale)
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
                    target=decode_target(encoded,current['q'][1:])
                    first,applied,_=command(target,current['q'][1:],offset,scale)
                    action[1:]=applied;intended[1:]=first;requested[1:]=target
                    expected=pd_from_action(action,current['q'],offset,scale)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('nonfinite requested native action')
                    obs,_,done,_=self.env_step(self.env,action.clone())
                    executed_controls+=1
                    if not isinstance(obs,dict):
                        obs={'obs':obs}
                    if capture is None or not torch.equal(capture,action) or actual_pd is None or (actual_pd-expected).abs().max()>1e-6:
                        raise ValueError('native applied command/actual PD capture mismatch')
                    if tick<541 and done.any():
                        raise ValueError('early reset invalidates full episode')
                    for k,v in dict(action=action,applied=capture,pd_targets=actual_pd,intended=intended,
                        requested_pd=requested,clipped=(intended-action).abs().amax(-1)>1e-6,
                        state_features=states,tau_features=taus,encoded_target=encoded,
                        wam_object_history=oh,wam_hand_history=hh,wam_goal=goal,
                        server_latency_s=reply['elapsed_s'],native_latency_s=time.monotonic()-before).items():
                        logs[k].append(v.cpu().numpy().copy() if isinstance(v,torch.Tensor) else np.asarray(v).copy())
                    current=record()
                if tick==0 or (tick+1)%64==0:
                    gpu=gpu_state(args.gpu)
                    if set(gpu['pids'])-{os.getpid(),args.server_pid}:
                        raise RuntimeError('foreign GPU process during episode')
                    row=dict(control=tick+1,elapsed_s=time.monotonic()-started,
                        eta_s=(time.monotonic()-started)/(tick+1)*(541-tick),gpu=gpu)
                    print(json.dumps(row),flush=True)
                    with (out/'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
            packed={k:np.stack(v) for k,v in logs.items()}
            np.savez_compressed(out/'trajectory.npz',**packed)
            outcomes=[]
            for env,role in enumerate(manifest['roles']):
                value=episode_outcome(dict(object_pose=packed['obj'][:,env],object_velocity=packed['velocity'][:,env],
                    surface_gap=packed['surface_gap'][:,env],support_gap=packed['support_gap'][:,env],table_footprint=packed['table_footprint'][:,env]))
                value.update(env=env,role=role,clipping_rate=float(packed['clipped'][:,env].mean()))
                outcomes.append(value)
            result=dict(status='COMPLETED',outcomes=outcomes,
                calibration_pass=all(v['success'] and v['maximum_held_frames']>=45 for v in outcomes[:2]),
                server_latency_median_s=float(np.median(packed['server_latency_s'])),
                native_latency_median_s=float(np.median(packed['native_latency_s'])),
                scope='one nominal simulated episode per row; no hidden-state fork or robot-domain generalization claim')
            if any(sha(path)!=digest for path,digest in hashes.items()):
                raise ValueError('native frozen input/source drift')
            write(out/'result.json',result)
            manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started,
                trajectory_sha256=sha(out/'trajectory.npz'),result_sha256=sha(out/'result.json'))
            write(out/'manifest.json',manifest);print(json.dumps(result,indent=2))

    native.EvalPlayer=Player
    sys.argv=[sys.argv[0],'--task','Dexplore_Inspire','--cfg_env',cfg['cfg_env'],'--cfg_train',str(train_cfg),
        '--motion_file',cfg['motion_root'],'--checkpoint',str(OFFICIAL),'--headless','--num_envs','3','--seed','292',
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
