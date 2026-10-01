#!/usr/bin/env python3
"""Known-static-contact force calibration in an owned native GPU simulation."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON,R7,MOTIONS


def native(args,remaining):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():raise ValueError('new owned native output required')
    args.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
    from isaacgym import gymtorch
    import torch
    import evaluate as original
    from src.task.CmResidual.executable_contact_options import hold_target,hold_action,TableClearance,obj_vertices
    class CalibrationPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
            t=self.env.task;n=t.num_envs
            if n!=96 or abs(t.dt-1/30)>1e-8:raise ValueError('native calibration contract')
            t._hybrid_init_prob=1.;t._enable_early_termination=False;t._adaptive_kappa_enabled=False
            ids=torch.arange(n,device=self.device);obs=self.env_reset(ids)
            if self.get_batch_size(obs['obs'],1)!=n:raise ValueError('native batch')
            # Reference frame0, zero initial velocities, hand translated2m up.
            # Only this owned simulation state is changed, not assets/checkpoints.
            t._dof_pos[:,2]+=2.;t._dof_vel[:]=0;t._target_states[:,7:]=0
            goal=hold_target(t._dof_pos,t._pd_action_offset,t._pd_action_scale)
            t._dof_pos[:]=goal
            t.gym.set_actor_root_state_tensor(t.sim,gymtorch.unwrap_tensor(t._root_states))
            t.gym.set_dof_state_tensor(t.sim,gymtorch.unwrap_tensor(t._dof_state))
            masses=[];flags=[]
            for env,handle in zip(t.envs,t._target_handles):
                properties=t.gym.get_actor_rigid_body_properties(env,handle)
                if len(properties)!=1:raise ValueError('single rigid airplane required')
                masses.append(properties[0].mass);flags.append(properties[0].flags)
            assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
            geometry=TableClearance(obj_vertices(assets/'objects/airplane/airplane.obj',self.device),obj_vertices(assets/'objects/table/table.obj',self.device),t.ball_size)
            forces=[];states=[];clearances=[];hand_forces=[];distances=[]
            for tick in range(150):
                action=hold_action(goal,t._dof_pos,t._pd_action_offset,t._pd_action_scale)
                _,_,done,_=self.env_step(self.env,action)
                if done.any():raise ValueError('calibration ended unexpectedly')
                forces.append(t._tar_contact_forces.clone());states.append(t._target_states.clone())
                clearances.append(geometry.clearance(t._target_states,t._table_states))
                hand_forces.append(t._contact_forces[:,t._contact_body_ids].norm(dim=-1).amax(-1))
                distances.append((t._rigid_body_pos[:,t._key_body_ids[0]]-t._target_states[:,:3]).norm(dim=-1))
            f=torch.stack(forces);s=torch.stack(states);c=torch.stack(clearances);h=torch.stack(hand_forces);distance=torch.stack(distances)
            quiet=(s[-60:,:,7:10].norm(dim=-1)<.005)&(s[-60:,:,10:13].norm(dim=-1)<.05)&(distance[-60:]>1.)
            force=f[-60:].norm(dim=-1);weight=torch.tensor(masses,device=self.device)*abs(t.sim_params.gravity.z)
            ratio=force/weight[None];selected=force[quiet]
            if not len(selected):raise ValueError('no quiet known-contact diagnostic frames')
            torch.save(dict(object_force=f.cpu(),object_state=s.cpu(),mesh_clearance=c.cpu(),hand_force_max=h.cpu(),hand_distance=distance.cpu(),mass=masses,
                            flags=flags,gravity=[t.sim_params.gravity.x,t.sim_params.gravity.y,t.sim_params.gravity.z],quiet_final_frames=quiet.cpu(),
                            substeps=t.sim_params.substeps,contact_collection=int(t.sim_params.physx.contact_collection)),args.output/'telemetry.pt')
            result=dict(run_status='COMPLETED',kind='ENGINEERING_FORCE_UNIT_DIAGNOSTIC',episodes=n,frames=150,quiet_observations=int(quiet.sum()),
                        mass_kg_range=[min(masses),max(masses)],object_flags=sorted(set(flags)),substeps=t.sim_params.substeps,contact_collection=int(t.sim_params.physx.contact_collection),
                        weight_N_range=[float(weight.min()),float(weight.max())],quiet_force_N_quantiles=torch.quantile(selected,torch.tensor([0.,.1,.5,.9,1.],device=self.device)).cpu().tolist(),
                        quiet_force_over_weight_quantiles=torch.quantile(ratio[quiet],torch.tensor([.1,.5,.9],device=self.device)).cpu().tolist(),
                        quiet_fraction_above_legacy_threshold=float((selected>.1).float().mean()),quiet_min_hand_distance_m=float(distance[-60:][quiet].min()),
                        quiet_clearance_m_quantiles=torch.quantile(c[-60:][quiet],torch.tensor([.1,.5,.9],device=self.device)).cpu().tolist(),
                        telemetry_sha256=sha(args.output/'telemetry.pt'),elapsed_seconds=time.monotonic()-begin,
                        scope='known resting-contact telemetry with hand>1m away and object quiet; verifies net-force scale, does not prove grasp or change old labels')
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    original.EvalPlayer=CalibrationPlayer;sys.argv=[sys.argv[0],*remaining];original.main()


def launch(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve();output=args.output.resolve()
    if output.parent!=base or args.output.is_symlink():raise ValueError('new owned output required')
    route_path=ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json';route=json.loads(route_path.read_text());spec=route['experts']['source_e260'];checkpoint=(ROOT/spec['checkpoint']).resolve()
    if sha(checkpoint)!=spec['sha256']:raise ValueError('checkpoint drift')
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    inputs=[Path(__file__),ROOT/'src/task/CmResidual/executable_contact_options.py',route_path,R7/'environment.yaml',R7/'training.yaml',checkpoint]
    inputs += [assets/p for p in ['airplane.urdf','table.urdf','objects/airplane/airplane.obj','objects/table/table.obj']]
    inputs += [ROOT/'third_party/DExplore/dexplore'/p for p in ['evaluate.py','env/tasks/base_dexplore_task.py','env/tasks/dexplore_inspire.py','env/tasks/base_task.py']]
    for spec in route['motions']:
        p=MOTIONS/spec['name']/'interaction_hand_inspire.pt'
        if sha(p)!=spec['interaction_hand_sha256']:raise ValueError('motion drift')
        inputs.append(p)
    hashes={str(p.resolve()):sha(p) for p in inputs};admission=None
    for gpu in args.gpus:
        try:admission=gpu_admission(gpu);break
        except RuntimeError:continue
    if admission is None:raise RuntimeError('all allowed GPUs occupied')
    output.mkdir(exist_ok=False);begin=time.monotonic();child=output/'native'
    command=[PYTHON,'-u',str(Path(__file__).resolve()),'--native','--output-dir',str(child),'--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
             '--checkpoint',str(checkpoint),'--motion_file',str(MOTIONS),'--headless','--num_envs','96','--seed','439','--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0','--disable-early-termination','--output',str(child/'unused.json'),'--output_path',str(child/'player')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    m=dict(run_status='RUNNING',experiment_id='P-20261002-contact-force-units',kind='ENGINEERING_BLOCKER_DIAGNOSTIC',pid=os.getpid(),gpu=admission,input_sha256=hashes,command=command,
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),wall_limit_seconds=240,output_limit_bytes=100<<20)
    path=output/'run_manifest.json'
    def save():path.write_text(json.dumps(m,indent=2)+'\n')
    save();process=None
    try:
        with (output/'native.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);m.update(native_pid=process.pid,pgid=process.pid);save();code=process.wait(timeout=210)
        if code:raise RuntimeError('native exit'+str(code))
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('input drift')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>100<<20:raise ValueError('storage limit')
        m.update(run_status='COMPLETED',result=json.loads((child/'results.json').read_text()),input_hashes_unchanged=True)
        print(json.dumps(m['result']),flush=True)
    except BaseException as e:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        m.update(run_status='FAILED',error=repr(e));raise
    finally:m['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output-dir',dest='output',type=Path,required=True);p.add_argument('--native',action='store_true');p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);args,remaining=p.parse_known_args()
    if args.native:native(args,remaining)
    elif remaining:raise ValueError('unknown parent arguments')
    else:launch(args)
