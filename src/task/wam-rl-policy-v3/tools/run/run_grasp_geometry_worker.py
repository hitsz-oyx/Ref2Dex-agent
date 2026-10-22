"""Size-derived SE3 grasp and bounded R-intent residual worker: one physical GPU, fresh native process/trial."""
import argparse,json,os,shutil,socket,subprocess,sys,time,tempfile
from pathlib import Path
import numpy as np
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'tools/audit'))
from repeat_frozen_tau_chains import freeze,write
from freeze_grasp_geometry import copy_geometry_runtime


def control_counts(infos,actions):
    """JSON-native integers even when NumPy comparisons enter Python sum."""
    return dict(reach_filter_controls=int(sum(bool(i['reachability'] and i['reachability']['filter_active']) for i in infos)),
        recovery_controls=int(sum(bool(i['reachability'] and 'recovery_candidate' in i['reachability']) for i in infos)),
        goal_servo_saturated_controls=int(sum(i['phase'] in ('lift','hold') and np.abs(ac[:3]).max()>=.024999 for i,ac in zip(infos,actions))))


class NativeEpisode:
    def __init__(self,frozen,folder,private,gpu,sequence,seed):
        self.folder=folder;self.private=private;private.mkdir(parents=True,exist_ok=False)
        fixed=private/'motions.json';write(fixed,dict(sequences=[sequence]));sock=private/'rpc.sock'
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.server.settimeout(2);self.server.bind(str(sock));self.server.listen(1);self.connection=None
        self.log=(folder/'native.log').open('x')
        self.command=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(frozen/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),
            '--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(folder/'physics'),'--socket',str(sock),'--server-pid',str(os.getpid()),
            '--envs','1','--gpu',str(gpu),'--seed',str(seed),'--seconds','1800','--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
        self.child=subprocess.Popen(self.command,cwd=ROOT,env=os.environ.copy(),stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        write(folder/'process.json',dict(client_pid=os.getpid(),native_pid=self.child.pid,command=self.command))

    def connect(self,receive,peer_pid):
        start=time.monotonic()
        while self.connection is None:
            if self.child.poll() is not None:raise RuntimeError('native startup exited: '+str(self.child.returncode))
            if time.monotonic()-start>180:raise TimeoutError('native handshake')
            try:self.connection,_=self.server.accept()
            except socket.timeout:continue
        self.connection.settimeout(120)
        if peer_pid(self.connection)!=self.child.pid:raise ValueError('native peer pid mismatch')
        return receive(self.connection)

    def finish(self,send,receive):
        send(self.connection,dict(op='close'));receive(self.connection);self.child.wait(timeout=20)
        if self.child.returncode:raise RuntimeError('native nonzero exit')

    def cleanup(self):
        if self.connection:self.connection.close()
        self.server.close();self.log.close()
        if self.child.poll() is None:
            self.child.terminate()
            try:self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
        write(self.folder/'exit-receipt.json',dict(native_pid=self.child.pid,native_returncode=self.child.returncode))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--worker',type=int,choices=[0,1,2],required=True);p.add_argument('--gpu',type=int,choices=[4,5,6],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--carry-run',type=Path);p.add_argument('--residual-checkpoint',type=Path);p.add_argument('--ppo',action='store_true');a=p.parse_args()
    if a.carry_run:raise ValueError('new geometry contract requires a fresh run')
    out=a.output.resolve();reg=json.loads(a.registration.read_text());jobs=[j for j in reg['jobs'] if j['worker']==a.worker]
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh task-local output required')
    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if int(gpu.split(',')[1])>64:raise RuntimeError('GPU occupied, queue this worker: '+gpu)
    out.mkdir(parents=True);oldrun=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1';frozen=freeze(out,oldrun)
    copy_geometry_runtime(frozen)
    shutil.copyfile(TASK/'tools/audit/measure_s2d_controls.py',out/'s2d_mapping.py');shutil.copyfile(Path(__file__),out/'runner.py');shutil.copyfile(a.registration,out/'registration.json')
    private=Path(tempfile.mkdtemp(prefix='gv10-',dir=str(ROOT/'tmp')))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen),TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(out),str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    import torch
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM,WAM_TO_NATIVE,expand_action,ACTIVE
    from wam_rl_policy_v3.inspire_shadow import current_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_scheduler import TaskSchedulerConfig
    from wam_rl_policy_v3.goal_scheduler import SchedulerConfig
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.executor_contract import task_condition
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.tau_replay_contract import pd_targets
    from wam_rl_policy_v3.pose_consumer import pose7,reduce_to_wrist_plan,grasp_intent_from_retargeter
    from wam_rl_policy_v3.grasp_repair import clean_tau_future,support_plane
    from wam_rl_policy_v3.grasp_geometry import GeometryGraspConsumer,GraspGeometry
    from wam_rl_policy_v3.source_pd import source_pd_action
    from wam_rl_policy_v3.intent_residual import IntentResidualPolicy,residual_observation,apply_intent_residual,ppo_update,RESIDUAL_CONTRACT
    from wam_rl_policy_v3.evaluation_sources import freeze_template,template_intent,gt_future
    from wam_rl_policy_v3.geometry_tau import object_local_hand,geometry_future
    from wam_rl_policy_v3.stage_evaluation import PalmSurface,PALM_FEATURE_CONTRACT,terminal_success
    from wam_rl_policy_v3.measured_reward_bridge import reward_features
    from ptv3_shared_wam_v6.staged_progress import StagedProgress,StagedDenseReward,STAGE_CONTRACT
    from ptv3_shared_wam_v6.execution_contract import grasp_lift_task,encoder_domain_available
    from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
    from s2d_mapping import object_anchored_targets
    torch.set_num_threads(2);torch.manual_seed(196);np.random.seed(196)
    original=json.loads((oldrun/'manifest.json').read_text());models=[]
    for path in (Path(original['base_checkpoint']),oldrun/'calibrated.pt'):
        model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models.append(model.requires_grad_(False))
    model=models[0]
    residual=None;optimizer=None
    if a.residual_checkpoint:
        residual=IntentResidualPolicy().cuda();residual.load_state_dict(torch.load(a.residual_checkpoint,map_location='cpu',weights_only=False)['model'])
        optimizer=torch.optim.Adam(residual.parameters(),lr=1e-4) if a.ppo else None
    if a.ppo and (residual is None or not reg.get('ppo_authorized') or reg['maximum_controls']>4800):raise ValueError('PPO requires frozen BC, staged reward and registered <=4800 control budget')
    wam=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)
    palm_surface=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    with np.load(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz') as z:template_ref={k:z[k].copy() for k in z.files}
    template=freeze_template(template_ref,30,50)
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];cache={}
    report=dict(status='STARTING',run_id=out.name,protocol=reg['protocol'],registration=str(a.registration.resolve()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        worker=a.worker,gpu=a.gpu,gpu_before=gpu,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,stages=STAGE_CONTRACT,palm_feature=PALM_FEATURE_CONTRACT,
        base_checkpoint=original['base_checkpoint'],wam_step=wam.step,planned_episodes=sum(len(j['order']) for j in jobs),environment_controls=0,learning_updates=0,runs=[],maximum_pd_reconstruction_error=0.,residual_contract=RESIDUAL_CONTRACT if residual else None,residual_checkpoint=str(a.residual_checkpoint) if residual else None,ppo=a.ppo)
    report['carried_elapsed_s']=0.
    if a.carry_run:
        previous=json.loads((a.carry_run/'manifest.json').read_text())
        if json.loads((a.carry_run/'registration.json').read_text())!=reg or previous['worker']!=a.worker:raise ValueError('carry protocol/assignment mismatch')
        for name in ('pose_consumer','evaluation_sources','geometry_tau','stage_evaluation','grasp_repair','grasp_geometry','intent_residual'):
            relative=Path('src/task/wam-rl-policy-v3/src/wam_rl_policy_v3')/(name+'.py')
            if (frozen/relative).read_bytes()!=(a.carry_run/'runtime-code'/relative).read_bytes():raise ValueError('carry consumer/scorer changed: '+name)
        report['environment_controls']=previous['environment_controls'];report['carried_elapsed_s']=previous['elapsed_s']
        report['carry_run']=str(a.carry_run.resolve());report['partial_controls_retained']=previous['environment_controls']-sum(r['controls'] for r in previous['runs'])
        for row in previous['runs']:
            report['runs'].append(dict(row,carried_unchanged=True,source_run=str(a.carry_run.resolve())))
            (out/row['id']).symlink_to(a.carry_run.resolve()/row['id'],target_is_directory=True)
    write(out/'manifest.json',report);started=time.monotonic()
    def save_trace(folder,trace,observations,actions,raw_actions,adapter_actions,pds,taus,valids):
        if trace:np.savez_compressed(folder/'trajectory.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]},observation=np.asarray(observations),action=np.asarray(actions),raw_action=np.asarray(raw_actions),adapter_raw_action=np.asarray(adapter_actions),pd_targets_actual=np.asarray(pds),tau=np.asarray(taus),tau_valid=np.asarray(valids),forecast_hand=np.asarray(forecast_hands))
    try:
        for job in jobs:
            name=job['sequence']
            if name not in cache:
                record=next(r for r in catalog if r['sequence']==name);source=Path(record['source'])
                with np.load(source/'canonical.npz') as z:points=z['points'].copy()
                with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/name/'admission.npz') as z:bounds=z['table_bounds'].copy()
                with np.load(job['reference']) as z:reference={k:z[k].copy() for k in z.files}
                if not np.array_equal(reference['tick'].reshape(-1),np.arange(len(reference['q']))) or len(reference['action'])+1!=len(reference['q']):raise ValueError('reference source clock invalid')
                with torch.inference_mode():shape=model.shape(torch.tensor(points[None],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
                cache[name]=(record,points,bounds,reference,shape,object_local_hand(reference['hand'][:,0],reference['objects'][:,0]))
            record,points,bounds,reference,shape,reference_local=cache[name]
            for arm in job['order']:
                if time.monotonic()-started+report['carried_elapsed_s']>reg['worker_wall_s'] or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('fixed worker wall/storage budget')
                label=job['id']+'-'+arm
                if any(r['id']==label for r in report['runs']):continue
                folder=out/label;folder.mkdir();episode_started=time.monotonic();native=None;trace=[];observations=[];actions=[];raw_actions=[];adapter_actions=[];pds=[];taus=[];valids=[];stage_info=[];control_info=[];forecast_hands=[]
                report['active_episode']=dict(id=label,sequence=name,subset=job['subset'],tick=0);write(out/'manifest.json',report)
                try:
                    torch.manual_seed(job['model_seed']);np.random.seed(job['model_seed'])
                    native=NativeEpisode(frozen,folder,private/label,a.gpu,name,job['native_seed']);hello=native.connect(receive,peer_pid)
                    send(native.connection,dict(op='reset',ids=[0]));state=unpack_state(receive(native.connection)['state'])
                    offset=np.asarray(hello['audit']['controller']['offset'],np.float32);scale=np.asarray(hello['audit']['controller']['scale'],np.float32);limits=hello['audit']['physical_joint_limits']
                    write(folder/'native-audit.json',hello['audit'])
                    qerr=np.asarray(state['q'])[0]-reference['q'][0,0];qerr[3:6]=(qerr[3:6]+np.pi)%(2*np.pi)-np.pi
                    initial_errors=dict(q=float(np.abs(qerr).max()),dq=float(np.abs(np.asarray(state['dq'])[0]-reference['dq'][0,0]).max()),hand=float(np.abs(np.asarray(state['hand'])[0]-reference['hand'][0,0]).max()))
                    if max(initial_errors.values())>2e-6 or state['sequences']!=[name] or state['contact_force_valid'][0] or int(state['tick'][0])!=0:raise ValueError('reference/reset mismatch: '+str(initial_errors))
                    initial={k:np.asarray(state[k]).copy() for k in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')};np.savez_compressed(folder/'initial.npz',**initial)
                    pose=pose7(np.asarray(state['objects'])[0]);target=grasp_lift_task(pose);geometry=GraspGeometry.from_points(points);context=TaskController(target,points,scheduler_config=TaskSchedulerConfig(episode_timeout_s=30.),path_config=SchedulerConfig(stall_timeout_s=60.))
                    consumer=GeometryGraspConsumer(np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),offset,scale,plane=support_plane(np.asarray(state['table_pose'])[0],bounds),palm_vertices=palm_surface.vertices[::32],table_geometry=(pose7(np.asarray(state['table_pose'])[0]),bounds),object_points=points,joint_limits=limits);stages=StagedProgress(pose[2,3],geometry=geometry);rewarder=None;rollout=dict(observations=[],latents=[],old_logp=[],rewards=[],values=[]);reward_parts=[];history=[];hands=[];decision=None
                    wam_calls=actor_calls=covered=final_clipped=adapter_clipped=source_failures=0;ever_primary=False;physical_abort=None;lastsafe=None;errors_m=[];drop_after_hold=False
                    for t in range(reg['horizon_controls']+1):
                        report['active_episode']['tick']=t
                        if t%60==0:
                            if t==60:report['gpu_monitor']=dict(id=label,tick=t,reading=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),episode_eta_s=(time.monotonic()-episode_started)/t*(reg['horizon_controls']-t))
                            write(out/'manifest.json',report)
                        if time.monotonic()-started+report['carried_elapsed_s']>reg['worker_wall_s']:raise TimeoutError('registered wall budget')
                        obj=np.asarray(state['objects'])[0];pose=pose7(obj);palm=pose7(np.asarray(state['palm_pose'])[0]);hand=np.asarray(state['hand'])[0,NATIVE_TO_WAM];forces=np.asarray(state['contact_forces'])[0]
                        safe=physical_safety(state,0,points,bounds,limits);lastsafe=safe;physical_abort=safe['reason'] if safe['abort'] else None
                        supported=online_support(state,0,pose,points,bounds)
                        if decision is None or not decision.terminated:
                            value=context.observe(pose,hand,obj[7:13],t/30,contact_forces=forces,force_valid=bool(state['contact_force_valid'][0]),supported=supported,workspace_ok=not safe['abort']);decision=value['decision'];feedback=value['feedback']
                        else:feedback=context.observer.update(pose,hand,obj[7:13],t/30,contact_forces=forces,force_valid=bool(state['contact_force_valid'][0]),supported=supported,workspace_ok=not safe['abort'])
                        f=reward_features(pose,hand,points,decision,obj[7:13],forces);features=palm_surface.features(f,palm,pose,points)
                        stage=stages.update(obj[2],t/30,features,feedback,forces,release_authorized=False,physical_abort=safe['abort'],approach_tip_gap_m=geometry.paired_tip_gap(hand,pose,points));stage_info.append(stage)
                        goal_error=float(np.linalg.norm(pose[:3,3]-target.object_path[-1,:3,3]));errors_m.append(goal_error)
                        current_primary=bool(stage['current']['hold'] and goal_error<=geometry.position_tolerance_m and not safe['abort']);ever_primary|=current_primary
                        drop_after_hold|=bool(stage['achieved']['hold'] and feedback.released)
                        if a.ppo and not actions:rewarder=StagedDenseReward(stages)
                        if a.ppo and actions:
                            reward_decision=SimpleNamespace(task_complete=bool(t==reg['horizon_controls'] and stage['current']['hold'] and geometry.arrived(goal_error) and not safe['abort']),failure_reason='physical' if safe['abort'] else None,terminated=bool(safe['abort']))
                            reward,parts,_=rewarder.step(stages,reward_decision,np.asarray(actions[-1]),terminal=bool(safe['abort'] or t==reg['horizon_controls']),failed=bool(safe['abort']))
                            rollout['rewards'].append(reward);reward_parts.append(parts)
                        history.append(obj[:7].copy());hands.append(hand.copy());history=history[-4:];hands=hands[-4:]
                        previous=np.asarray(actions[-1])[None] if actions else np.zeros((1,12),np.float32)
                        current=current_state(np.asarray(state['q']),np.asarray(state['dq']),np.asarray(state['physical']),np.asarray(state['objects']),target.object_path[-1,:3,3][None],previous)[0]
                        dv=bool(state['dof_force_valid'][0]);cv=bool(state['contact_force_valid'][0]);measured=np.r_[np.clip(current,-5,5),np.asarray(state['dof_force'])[0] if dv else np.zeros(18),forces.reshape(-1) if cv else np.zeros(15),dv,cv].astype(np.float32)
                        robot=np.r_[measured[:345],shape,measured[345:]];tau=np.zeros((8,11,3),np.float32);valid=False;future=None;plan=None
                        available=bool(len(history)==4 and encoder_domain_available(points,hand,pose))
                        if available and consumer.phase in ('approach','close'):
                            if arm in ('A1','A2'):
                                with torch.random.fork_rng(devices=[0]):prediction=wam.predict(np.asarray(history)[None],np.asarray(hands)[None],decision.local_goal_pose[None],[decision.next_event.wam_condition()],['dexplore/'+record['object']])[0][:,WAM_TO_NATIVE]
                                future=np.asarray(state['hand'])[0][None]+prediction;valid=True;wam_calls+=1
                            elif arm=='A3':future,valid=geometry_future(reference_local,t,pose)
                            elif arm=='A4':future,valid=gt_future(object_anchored_targets,reference,t,obj,True)
                        future,valid=clean_tau_future(future,np.asarray(state['hand'])[0],valid,zero=arm=='A2')
                        if future is not None:
                            future=np.asarray(future,np.float32);tau=(future[:8]-np.asarray(state['hand'])[0]).astype(np.float32)
                            plan=reduce_to_wrist_plan(np.asarray(state['hand'])[0],palm,future,pose,points)
                        observation=np.r_[robot,task_condition(pose,decision),tau.reshape(-1),float(valid)].astype(np.float32)
                        if observation.shape!=(686,) or not np.isfinite(observation).all():raise ValueError('observation contract')
                        trace.append({k:np.asarray(state[k]).copy() for k in ('q','dq','objects','hand','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')});observations.append(observation);taus.append(tau.copy());valids.append(valid);forecast_hands.append(np.zeros((24,11,3),np.float32) if future is None else future.copy())
                        # Never terminate on a successful stage or encoder failure.
                        if safe['abort'] or t==reg['horizon_controls']:break
                        with torch.inference_mode():adapter=model.actor(torch.tensor(observation[None],device='cuda')).cpu().numpy()
                        if arm=='A0' and consumer.phase in ('approach','close'):
                            ref_t=min(t,len(reference['action'])-1)
                            source_action=reference['action'][ref_t]
                            if source_action.shape==(18,):source_action=source_action[list(ACTIVE)]
                            source_target=pd_targets(source_action,reference['q'][ref_t,0],offset,scale)
                            try:adapter=source_pd_action(source_target,np.asarray(state['q'])[0],offset,scale)[0][None]
                            except ValueError:
                                # Explicit source admission failure; continue the measured
                                # same-controller recovery, report separately from task hardness.
                                source_failures+=1
                        actor_calls+=1;adapter_clipped+=int((np.abs(adapter)>1+1e-6).any())
                        intent=grasp_intent_from_retargeter(np.asarray(state['q'])[0],palm,pose,target.object_path[-1],np.clip(adapter[0],-1,1),offset,scale,plan=plan,tau_valid=valid)
                        if residual is not None:
                            residual_input=residual_observation(observation,intent)
                            with torch.no_grad():
                                dist,value=residual.distribution(torch.tensor(residual_input[None],device='cuda'))
                                latent=dist.sample() if a.ppo else dist.mean
                                logp=dist.log_prob(latent).sum(-1)
                            intent=apply_intent_residual(intent,latent[0].cpu().numpy())
                            if a.ppo:
                                rollout['observations'].append(residual_input);rollout['latents'].append(latent[0].cpu().numpy());rollout['old_logp'].append(float(logp[0]));rollout['values'].append(float(value[0]))
                        evidence=context.observer.last_evidence
                        raw,info=consumer.step(hand=np.asarray(state['hand'])[0],q=np.asarray(state['q'])[0],dq=np.asarray(state['dq'])[0],palm=palm,object_pose=pose,object_velocity=obj[7:13],forces=forces,feedback=feedback,near_fingers=evidence['near_fingers'],fingertip_gap_m=evidence['fingertip_gap_m'],intent=intent,now=t/30)
                        action=np.clip(raw[None],-1,1).astype(np.float32);final_clipped+=int((np.abs(raw-action[0])>1e-6).any());full=expand_action(action)
                        predicted=pd_targets(action,np.asarray(state['q']),offset,scale);before=list(state['episodes']);tick=np.asarray(state['tick']).copy()
                        send(native.connection,dict(op='step',action=full.tolist(),episodes=before,return_pd=True));reply=receive(native.connection);report['environment_controls']+=1;nextstate=unpack_state(reply['state']);actual=np.asarray(reply['pd_targets'])
                        pd_error=float(np.abs(predicted-actual).max());report['maximum_pd_reconstruction_error']=max(report['maximum_pd_reconstruction_error'],pd_error)
                        if pd_error>2e-5 or nextstate['episodes']!=before or not np.array_equal(np.asarray(nextstate['tick']),tick+1):raise ValueError('native action/PD/clock contract')
                        actions.append(action[0]);raw_actions.append(raw);adapter_actions.append(adapter[0]);pds.append(actual[0]);control_info.append(info);covered+=int(valid);state=nextstate
                    if actor_calls!=len(actions) or (arm=='A2' and np.abs(np.asarray(taus)).max()!=0):raise ValueError('clean shared-R ablation contract')
                    save_trace(folder,trace,observations,actions,raw_actions,adapter_actions,pds,taus,valids);write(folder/'stages.json',stage_info);write(folder/'consumer.json',control_info)
                    row=dict(id=label,trial=job['id'],subset=job['subset'],sequence=name,object=record['object'],arm=arm,order_position=job['order'].index(arm),seed=job['native_seed'],model_seed=job['model_seed'],controls=len(actions),
                        final_success=bool(t==reg['horizon_controls'] and not safe['abort'] and stage['current']['hold'] and geometry.arrived(goal_error)),ever_success=ever_primary,final_hold=bool(t==reg['horizon_controls'] and stage['current']['hold'] and not safe['abort']),
                        stage_achieved=stage['achieved'],stage_final=stage['current'],longest_hold_s=stages.longest_hold_s,release_applicable=False,drop_after_hold=drop_after_hold,
                        final_goal_error_m=goal_error,goal_within15mm=bool(not safe['abort'] and t==reg['horizon_controls'] and goal_error<=.015),goal_error_quantiles_m=np.quantile(errors_m,[0,.5,.9,1]).tolist(),
                        physical_abort=physical_abort,physical_safety_final=lastsafe,wam_calls=wam_calls,actor_calls=actor_calls,encoder_covered_controls=covered,final_clipped_controls=final_clipped,adapter_clipped_controls=adapter_clipped,initial_errors=initial_errors,seconds_observed=t/30)
                    row.update(control_counts(control_info,actions));row['source_admission_failures']=source_failures;row['geometry']=geometry.manifest();row['goal_within_geometry']=bool(t==reg['horizon_controls'] and not safe['abort'] and geometry.arrived(goal_error))
                    if a.ppo:
                        bootstrap=0.
                        if not safe['abort'] and not row['final_success']:
                            # The 20s evaluation horizon is a time limit, not a
                            # physical terminal. Bootstrap with its observed state.
                            with torch.no_grad():terminal_adapter=model.actor(torch.tensor(observation[None],device='cuda')).cpu().numpy()
                            row['bootstrap_R_calls']=1
                            terminal_intent=grasp_intent_from_retargeter(np.asarray(state['q'])[0],palm,pose,target.object_path[-1],np.clip(terminal_adapter[0],-1,1),offset,scale,plan=plan,tau_valid=valid)
                            with torch.no_grad():_,terminal_value=residual(torch.tensor(residual_observation(observation,terminal_intent)[None],device='cuda'))
                            bootstrap=float(terminal_value[0])
                        update_started=time.monotonic();update=ppo_update(residual,optimizer,bootstrap=bootstrap,**rollout);update['wall_s']=time.monotonic()-update_started;update['cuda_peak_bytes']=torch.cuda.max_memory_allocated();row['ppo_update']=update;report['learning_updates']+=update['optimizer_updates']
                        torch.save(dict(model=residual.state_dict(),optimizer=optimizer.state_dict(),contract=RESIDUAL_CONTRACT),out/('residual-'+label+'.pt'))
                        np.savez_compressed(folder/'ppo-rollout.npz',**{k:np.asarray(v) for k,v in rollout.items()});write(folder/'reward-parts.json',reward_parts)
                    native.finish(send,receive);write(folder/'result.json',row);report['runs'].append(row);report.update(status='RUNNING',elapsed_s=time.monotonic()-started+report['carried_elapsed_s']);write(out/'manifest.json',report);print(json.dumps(row),flush=True)
                except BaseException:
                    save_trace(folder,trace,observations,actions,raw_actions,adapter_actions,pds,taus,valids);write(folder/'partial-stages.json',stage_info);raise
                finally:
                    if native:native.cleanup()
        report.update(status='COMPLETED',elapsed_s=time.monotonic()-started+report['carried_elapsed_s'],wam_seconds=wam.seconds);write(out/'manifest.json',report)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started+report['carried_elapsed_s']);write(out/'manifest.json',report);raise
    return 0


if __name__=='__main__':raise SystemExit(main())
