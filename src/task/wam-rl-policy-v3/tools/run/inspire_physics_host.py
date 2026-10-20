"""Pure Inspire simulation RPC: no teacher, DExplore actor, reward or PPO updates.

Pinned Isaac/Torch2.0 runtime only. The Shadow PPO and WAM process own goals,
conditioning, rewards and episode termination. References are initialization
and object-path instructions, never actor observations.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
NATIVE=ROOT/'third_party/DExplore/dexplore'


def physical248(task):
    import torch
    from utils import torch_utils
    n=task.num_envs;key=task._key_body_ids
    if len(key)!=16 or len(task._contact_body_ids)!=5:raise ValueError('native16 key/5 contacts required')
    root=task._rigid_body_pos[:,7];rotation=task._rigid_body_rot[:,7]
    heading=torch_utils.calc_heading_quat_inv(rotation)
    expanded=heading[:,None].expand(-1,16,-1).reshape(-1,4)
    position=(task._rigid_body_pos[:,key]-root[:,None]).reshape(-1,3)
    position=torch_utils.quat_rotate(expanded,position).reshape(n,48)[:,3:]
    rotations=torch_utils.quat_mul(expanded,task._rigid_body_rot[:,key].reshape(-1,4))
    rotations=torch_utils.quat_to_tan_norm(rotations).reshape(n,96)
    if task._local_root_obs:rotations[:,:6]=torch_utils.quat_to_tan_norm(rotation)
    linear=torch_utils.quat_rotate(expanded,task._rigid_body_vel[:,key].reshape(-1,3)).reshape(n,48)
    angular=torch_utils.quat_rotate(expanded,task._rigid_body_ang_vel[:,key].reshape(-1,3)).reshape(n,48)
    contacts=(task._contact_forces[:,task._contact_body_ids].abs()>.1).any(-1).float()
    return torch.cat((task._rigid_body_vel[:,6],task._rigid_body_ang_vel[:,6],position,rotations,linear,angular,contacts),-1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--server-pid',type=int,required=True);p.add_argument('--envs',type=int,default=256)
    p.add_argument('--gpu',type=int,default=7);p.add_argument('--seed',type=int,default=246)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--motions',type=int,default=0)
    p.add_argument('--object-manifest',type=Path)
    p.add_argument('--admission-manifest',type=Path,help='Restrict motion sampling to admitted same-execution TRAIN prefixes')
    p.add_argument('--state-transport',choices=['json','packed'],default='json')
    p.add_argument('--sequence',help='Exact TRAIN motion for bounded canonical-command replay')
    p.add_argument('--fixed-motions-json',type=Path,help='Engineering replay only: exact collection batch membership, no object sampler')
    p.add_argument('--extended-state',action='store_true',help='Measured DOF/contact forces and per-episode validity; legacy actor input is unchanged')
    a=p.parse_args();a.cache=a.cache.resolve();a.output=a.output.resolve();a.socket=a.socket.resolve()
    if a.output.exists() or ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents or not 2<=a.envs<=512:
        raise ValueError('fresh bounded native host required')
    a.output.mkdir(parents=True);scratch=ROOT/'tmp'/a.output.parent.name;scratch.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/consequence-official-generator/torch201-extensions'),PYTHONDONTWRITEBYTECODE='1')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    sys.path[:0]=['/home2/wyy/isaac-gym/isaacgym/python',str(NATIVE),str(ROOT),str(TASK/'src'),
        str(ROOT/'src/task/consequence-evaluator/src'),str(ROOT/'src/task/CmResidual/tools'),str(ROOT/'src/task/trajectory-policy/src')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    if Path(sys.prefix).resolve()!=Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train').resolve():raise ValueError('pinned native interpreter required')
    from isaacgym import gymtorch, gymapi
    import dexplore_ddp_rank_bootstrap
    import numpy as np
    import torch
    import run as native
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    from trajectory_policy.wam_shadow import NATIVE_LINKS
    from trajectory_policy.pd_retargeter import pd_from_action
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import pack_state
    if torch.__version__!='2.0.1+cu118':raise ValueError('pinned native Torch2.0.1 required')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    meta=json.loads((a.cache/'manifest.json').read_text());records=[r for r in meta['sequences'] if r['split']=='train']
    if a.admission_manifest:
        admission=json.loads(a.admission_manifest.read_text())
        if admission.get('schema')!='ref2dex.teacher-measured-rollout-admission.v2' or admission.get('status')!='COMPLETED':
            raise ValueError('corrected completed admission required')
        allowed={r['sequence'] for r in admission['records'] if r['split']=='train' and r['engineering']=='PASS' and r['training_prefix_commands']>0}
        records=[r for r in records if r['sequence'] in allowed]
        if {r['sequence'] for r in records}!=allowed:raise ValueError('native geometry does not cover admitted TRAIN')
    if a.fixed_motions_json:
        names=json.loads(a.fixed_motions_json.read_text())['sequences'];by_name={r['sequence']:r for r in meta['sequences']}
        if len(names)!=a.envs or len(set(names))!=len(names):raise ValueError('fixed replay needs one motion per environment')
        records=[by_name[name] for name in names]
        if a.sequence or a.object_manifest:raise ValueError('fixed replay cannot use policy sampling flags')
    if a.sequence:
        records=[r for r in records if r['sequence']==a.sequence]
        if len(records)!=1:raise ValueError('one known TRAIN replay sequence required')
    if a.object_manifest:
        selected=json.loads(a.object_manifest.read_text())['object_names']
        if not selected or len(set(selected))!=len(selected) or set(selected)-{r['object'] for r in records}:
            raise ValueError('distinct known TRAIN objects required')
        records=[r for r in records if r['object'] in selected]
    if a.motions:records=records[:a.motions]
    if len({r['object'] for r in records})>a.envs:raise ValueError('each selected object needs an environment')
    original=json.loads((ROOT/'data/processed_data/inspire_rl_object_dexplore/manifest.json').read_text())
    geometric=Path(original['source_geometric_root'])
    files=[str(geometric/r['sequence']) for r in records]
    runtime=a.output/'gym_runtime';asset=runtime/'dexplore/data/assets';asset.parent.mkdir(parents=True)
    asset.symlink_to('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')
    install_reset_patch()
    from wam_rl_policy_v3.inspire_reset_submission import install_reset_submission
    install_reset_submission(DexploreTask,gymtorch)
    original_load=DexploreTask._load_table
    def table(task,requested):
        exported=task.export_rl;task.export_rl=True
        try:original_load(task,list(files))
        finally:task.export_rl=exported
        if list(task.motion_file)!=files:raise ValueError('full selected TRAIN membership changed')
    DexploreTask._load_table=table
    original_create=native.create_rlgpu_env
    def create(**kwargs):
        native.cfg['env'].update(motion_file=files,objectMotionSampling=not bool(a.fixed_motions_json),hardObjectOversampling=False,
            stateInit='Start',hybridInitProb=1.,enableEarlyTermination=False,saveImages=False,export_rl=False,playdataset=False)
        native.cfg_train['params']['config']['multi_gpu']=False
        return original_create(**kwargs)
    native.create_rlgpu_env=create
    original_parse=native.parse_sim_params
    def parse(*args,**kwargs):
        sim=original_parse(*args,**kwargs);sim.physx.max_gpu_contact_pairs=8*1024*1024;sim.physx.default_buffer_size_multiplier=5.;return sim
    native.parse_sim_params=parse
    started=time.monotonic();audit=dict(schema='ref2dex.inspire-physics-host.v1',status='RUNNING',
        python=sys.executable,torch=torch.__version__,gpu=a.gpu,envs=a.envs,seed=a.seed,
        policy='none; Shadow PPO in remote process',native_ref_observation_consumed=False,
        native_ref_reward_used=False,native_ref_termination_used=False,source_sequences=len(files),
        source_sequence_names=[r['sequence'] for r in records],source_object_names=sorted({r['object'] for r in records}),
        state_transport=a.state_transport,
        reset_submission='logical_FK_now_once_indexed_setters_before_simulate.v1',
        indexed_resets=0,steps=0,episode_action_mismatches=0,maximum_pd_error=0.,extended_state=a.extended_state)
    def save():
        (a.output/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    save()

    class Host:
        def restore(self,filename):
            if filename is not None:raise ValueError('pure host forbids policy checkpoints')
        @torch.no_grad()
        def run(self):
            env=create();task=env.task
            if task.num_envs!=a.envs or not np.isclose(task.dt,1/30):raise ValueError('native30Hz contract changed')
            task._adaptive_kappa_enabled=False
            ids=torch.arange(task.num_envs,device=task.device);task.reset(ids)
            current=physical248(task)
            legacy=task._compute_observations_iter(ids,0)[:,9:257]
            error=float((current-legacy).abs().max())
            if error>2e-6:raise ValueError('pure physical248 mismatch '+str(error))
            # Real negative control: arbitrary changes to every hand/object ref
            # cannot influence the actor's directly measured physical fields.
            reference_ids=task.data_id.unique()
            original_hoi=task.hoi_data[reference_ids,0].clone()
            task.hoi_data[reference_ids,0]=0
            changed=physical248(task)
            task.hoi_data[reference_ids,0]=original_hoi;del original_hoi
            if not torch.equal(current,changed):raise ValueError('physical actor state consumed reference')
            audit.update(physical248_max_error=error,live_reference_invariance_passed=True,native_dt_s=task.dt,
                objects=task.object_name,controller=dict(offset=task._pd_action_offset.tolist(),scale=task._pd_action_scale.tolist()))
            # Reset patch updates q/body/contact caches with FK; no stale
            # post-setter refresh and no hidden physics step in reset().
            task._compute_observations=lambda env_ids=None:None
            def post():
                task.progress_buf+=1;task._refresh_sim_tensors()
                task.reset_buf.zero_();task.rew_buf.zero_();task._terminate_buf.zero_()
            task.post_physics_step=post
            names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0]);hand_ids=[names.index(x) for x in NATIVE_LINKS]
            episodes=np.zeros(a.envs,np.int64)
            force_valid=torch.zeros(a.envs,dtype=torch.bool,device=task.device)
            table_indices=[task.gym.get_actor_index(env, handle, gymapi.DOMAIN_ENV)
                           for env,handle in zip(task.envs,task._table_handles)]
            table_indices=torch.tensor(table_indices,device=task.device,dtype=torch.long)
            def packet():
                packet_started=time.perf_counter()
                bodies=task._rigid_body_state.view(a.envs,-1,13)
                objects=task._target_states.clone();sign=torch.where(objects[:,6:7]<0,-1.,1.);objects[:,3:7]*=sign
                values=dict(q=task._dof_pos,dq=task._dof_vel,physical=physical248(task),objects=objects,hand=bodies[:,hand_ids,:3],tick=task.progress_buf)
                result=(pack_state({k:v.cpu().numpy() for k,v in values.items()}) if a.state_transport=='packed'
                        else {k:v.cpu().tolist() for k,v in values.items()})
                result.update(episodes=episodes.tolist(),sequences=[Path(task.motion_file[i]).name for i in task.data_id.cpu().tolist()])
                if a.extended_state:
                    roots=task._root_states.view(a.envs,task.get_num_actors_per_env(),-1)
                    result.update(dof_force=torch.where(force_valid[:,None],task.dof_force_tensor,0.).cpu().tolist(),
                        contact_forces=torch.where(force_valid[:,None,None],task._contact_forces[:,task._contact_body_ids],0.).cpu().tolist(),
                        object_contact_force=torch.where(force_valid[:,None],task._tar_contact_forces,0.).cpu().tolist(),
                        table_pose=roots[torch.arange(a.envs,device=task.device),table_indices,:7].cpu().tolist(),
                        dof_force_valid=force_valid.cpu().tolist(),contact_force_valid=force_valid.cpu().tolist())
                audit['last_packet_build_s']=time.perf_counter()-packet_started
                return result
            connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.settimeout(min(180,a.seconds));connection.connect(str(a.socket))
            if peer_pid(connection)!=a.server_pid:raise ValueError('wrong policy peer')
            captured=actual_pd=None;pre,pd=task.pre_physics_step,task._action_to_pd_targets
            def before(action):
                nonlocal captured
                captured=action.clone();return pre(action)
            def convert(action):
                nonlocal actual_pd
                actual_pd=pd(action).clone();return actual_pd
            task.pre_physics_step,task._action_to_pd_targets=before,convert
            send(connection,dict(schema='ref2dex.inspire-shadow-rpc.v1',kind='hello',state=packet(),audit=audit))
            save()
            with connection:
                while True:
                    command=receive(connection);op=command['op']
                    if op=='close':send(connection,dict(ok=True));break
                    if op=='reset':
                        selected=np.asarray(command['ids'],np.int64)
                        if len(selected)!=len(np.unique(selected)) or (selected<0).any() or (selected>=a.envs).any():raise ValueError('invalid indexed reset')
                        reset_started=time.perf_counter()
                        task.reset(torch.as_tensor(selected,device=task.device));episodes[selected]+=1
                        force_valid[torch.as_tensor(selected,device=task.device)]=False
                        audit['last_reset_s']=time.perf_counter()-reset_started
                        audit['indexed_resets']+=len(selected)
                    elif op=='step':
                        if command['episodes']!=episodes.tolist():
                            audit['episode_action_mismatches']+=1;raise ValueError('cross-episode action')
                        action=torch.tensor(command['action'],device=task.device,dtype=torch.float32)
                        if action.shape!=(a.envs,18) or not torch.isfinite(action).all() or action.abs().max()>1.00001:raise ValueError('finite canonical18 action required')
                        original_action=action.clone();expected=pd_from_action(action,task._dof_pos,task._pd_action_offset,task._pd_action_scale)
                        physics_started=time.perf_counter();task.step(action.clone())
                        force_valid[:]=True
                        pd_error=float((expected-actual_pd).abs().max());audit['maximum_pd_error']=max(audit['maximum_pd_error'],pd_error)
                        if not torch.equal(action,original_action) or not torch.equal(captured,action) or pd_error>1e-6:raise ValueError('canonical/PD corruption')
                        audit['last_physics_and_pd_check_s']=time.perf_counter()-physics_started
                        audit['steps']+=1
                    else:raise ValueError('unknown native operation')
                    reply=dict(state=packet())
                    if op=='step' and command.get('return_pd',False):reply['pd_targets']=actual_pd.cpu().tolist()
                    send(connection,reply)
                    if audit['steps']%128==0:
                        # This counts motion initializations, not completed or
                        # successful trajectories; keep coverage visible live.
                        audit['motion_visits']=task._motion_sampler.visits.cpu().tolist() if task._motion_sampler is not None else []
                        audit['reset_setter_submissions']=getattr(task,'_inspire_reset_setter_submissions',0)
                        save()
            audit.update(status='COMPLETED',elapsed_s=time.monotonic()-started,
                motion_visits=task._motion_sampler.visits.cpu().tolist() if task._motion_sampler is not None else [])
            save();task.gym.destroy_sim(task.sim)

    def host_run(runner,command):
        if command.get('checkpoint') is not None:raise ValueError('pure host forbids policy weights')
        Host().run()
    native.Runner.run=host_run
    cfg=ROOT/'outputs/consequence-evaluator/baseline-transfer-inputs-20261007-r1/cfg/inspire.yaml'
    sys.argv=[sys.argv[0],'--task','Dexplore_Inspire','--cfg_env',str(cfg),'--cfg_train',str(NATIVE/'data/cfg/train/rlg/inspire.yaml'),
        '--test','--checkpoint','Base','--headless','--motion_file',str(geometric),'--num_envs',str(a.envs),'--seed',str(a.seed),
        '--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0','--num_threads','1','--output_path',str(a.output/'native')]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native host deadline')));signal.alarm(a.seconds)
    try:
        os.chdir(runtime);native.main()
    except BaseException as error:
        audit.update(status='FAILED',error=repr(error));save();raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
