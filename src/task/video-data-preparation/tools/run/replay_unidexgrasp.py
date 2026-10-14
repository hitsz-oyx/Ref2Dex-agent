#!/usr/bin/env python3
"""Bounded inference-only replay of the official UniDexGrasp++ state policy."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CAR = 'sem/Car-669043a8ce40d9d78781f76a6db4ab62'


def write_json(path, value):
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, indent=2))
    partial.replace(path)


def object_specs(args):
    if args.object_manifest is None:
        return [dict(object_code=CAR, scale=args.scale, split='engineering')]
    manifest = json.loads(args.object_manifest.read_text())
    if manifest['status'] != 'COMPLETE' or manifest['upstream_commit'] != '6c9e710b04769bf5d9765e3233820101cb3d718f':
        raise ValueError('completed pinned asset manifest required')
    objects = manifest['objects']
    if not 1 <= len(objects) <= 50 or len({r['object_code'] for r in objects}) != len(objects):
        raise ValueError('unique bounded object list required')
    for row in objects:
        path = Path(row['object_code'])
        if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or row['scale'] not in (.06,.08,.10,.12,.15):
            raise ValueError('invalid object/scale in prepared manifest')
        if row['official_split'] != 'TRAIN' or row['split'] not in ('train','dev'):
            raise ValueError('pilot may only use declared official TRAIN assets')
    if args.envs % len(objects):
        raise ValueError('equal environments per object required')
    expected = Path(manifest['asset_root']).resolve()
    if args.assets is not None and args.assets != expected:
        raise ValueError('manifest and supplied asset root disagree')
    args.assets = expected
    return objects


def task_class(vendor, output):
    """Preserve upstream physics while fixing heterogeneous aggregate capacity."""
    source_path = vendor/'dexgrasp/tasks/shadow_hand_grasp.py'
    source = source_path.read_text()
    replacements = {
        'max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1':
            'max_agg_bodies = self.num_shadow_hand_bodies + 2 * max(self.gym.get_asset_rigid_body_count(a) for scales in object_asset_dict.values() for a in scales.values()) + 1\n        self._pilot_aggregate_bodies = max_agg_bodies',
        'max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1':
            'max_agg_shapes = self.num_shadow_hand_shapes + 2 * max(self.gym.get_asset_rigid_shape_count(a) for scales in object_asset_dict.values() for a in scales.values()) + 1\n        self._pilot_aggregate_shapes = max_agg_shapes',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('pinned aggregate allocation source changed')
        source = source.replace(old, new)
    dest = output/'tmp/shadow_hand_grasp_capacity.py'
    dest.write_text(source)
    write_json(output/'upstream-runtime-patch.json',dict(source=str(source_path),copy=str(dest),
        replacements=replacements,reason='reserve capacity for largest loaded object, not last object',
        physics_parameters_changed=False,collision_geometry_changed=False))
    spec = importlib.util.spec_from_file_location('tasks.shadow_hand_grasp_capacity', dest)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ShadowHandGrasp


def worker(args):
    # Isaac Gym must load its native library before importing torch.
    import numpy as np
    if 'float' not in np.__dict__:
        np.float = float
    from isaacgym import gymapi, gymtorch, gymutil
    import torch
    import yaml
    from transforms3d.euler import euler2quat
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    vendor = args.vendor.resolve()
    sys.path.insert(0, str(vendor/'dexgrasp'))
    ShadowHandGrasp = task_class(vendor, args.output)
    from utils.data_info import plane2euler

    objects = object_specs(args)
    object_dict = {r['object_code']: [r['scale']] for r in objects}

    scratch = args.output/'tmp'
    assets = args.assets if args.assets else scratch/'assets'
    if args.assets is None:
        shutil.copytree(vendor/'assets', assets)
    if sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()) > 128*2**20:
        raise ValueError('bounded prepared assets exceeded 128MiB')
    pose_data = {CAR: {args.scale: dict(target_qpos=[], target_hand_pos=[], target_hand_rot=[],
                                        object_euler_xy=[], object_init_z=[])}}
    row = pose_data[CAR][args.scale]
    for path in ([] if args.object_manifest else sorted((assets/'datasetv4.1'/CAR).glob('*.npz'))):
        with np.load(path, allow_pickle=True) as label:
            if round(1/float(label['scale']), 2) != args.scale:
                continue
            q = label['qpos'].item()
            translation, angles = plane2euler(label['plane'], axes='sxyz')
            # Same original loader inclusion rule, preserving source support pose.
            if translation[2] > .05:
                continue
            rotation = euler2quat(*(q['WRJR'+a] for a in 'xyz'), axes='sxyz')
            row['target_qpos'].append([v for k,v in q.items() if k.startswith('robot0:')])
            row['target_hand_pos'].append([q['WRJT'+a] for a in 'xyz'])
            row['target_hand_rot'].append(rotation[[1,2,3,0]])
            row['object_euler_xy'].append(angles[:2])
            row['object_init_z'].append([translation[2]])
    if not row['object_init_z'] and not args.object_manifest:
        raise ValueError('no real single-object support-pose labels at selected scale')
    for key in row:
        row[key] = np.asarray(row[key], np.float32)
    pose_file = assets/'datasetv4.1_posedata.npy'
    if args.object_manifest:
        pose_data = np.load(pose_file, allow_pickle=True).item()
        widths = dict(target_qpos=22,target_hand_pos=3,target_hand_rot=4,object_euler_xy=2,object_init_z=1)
        for spec in objects:
            prior = pose_data[spec['object_code']][spec['scale']]
            count = len(prior['object_init_z'])
            if count != spec['initialization_priors'] or any(np.asarray(prior[k]).shape != (count,w)
                    or not np.isfinite(prior[k]).all() for k,w in widths.items()):
                raise ValueError('prepared official initialization prior contract changed')
        prior_counts = {r['object_code']:len(pose_data[r['object_code']][r['scale']]['object_init_z']) for r in objects}
    elif args.assets is None:
        np.save(pose_file, pose_data)
    else:
        existing = np.load(pose_file, allow_pickle=True).item()[CAR][args.scale]
        if any(not np.array_equal(existing[key], row[key]) for key in row):
            raise ValueError('reused posedata differs from the pinned source labels')
    if not args.object_manifest:
        prior_counts = {CAR:len(row['object_init_z'])}
    # The upstream state task hard-codes ../assets even when assetRoot is set.
    # Reusing assets therefore also requires its original adjacent work folder.
    work = assets.parent/'dexgrasp'
    work.mkdir(exist_ok=True)
    os.chdir(work)  # Official task expects ../assets.
    cfg = yaml.safe_load((vendor/'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
    cfg['env'].update(numEnvs=args.envs, random_time=False, printNumSuccesses=False,
                      object_code_dict=object_dict)
    cfg['env']['asset']['assetRoot'] = str(assets)
    cfg['graphics_device_id'] = -1
    sim = gymapi.SimParams()
    sim.dt = 1/60
    sim.num_client_threads = 0
    sim.physx.solver_type = 1
    sim.physx.num_position_iterations = 4
    sim.physx.num_velocity_iterations = 0
    sim.physx.num_threads = 4
    sim.physx.use_gpu = True
    sim.physx.num_subscenes = 0
    sim.physx.max_gpu_contact_pairs = 8*1024*1024
    sim.use_gpu_pipeline = True
    # The public config overrides contact offsets, substeps and solver
    # iterations. These are part of the trained policy's physical environment.
    gymutil.parse_sim_config(cfg['sim'], sim)
    write_json(args.output/'simulation-config.json', dict(dt=sim.dt,substeps=sim.substeps,
        use_gpu_pipeline=sim.use_gpu_pipeline,physx={key:getattr(sim.physx,key)
        for key in cfg['sim']['physx']},environment=cfg['env'],prepared_priors=prior_counts))
    write_json(args.output/'gpu-started.json', dict(time=time.time(), gpu=args.gpu))
    task = ShadowHandGrasp(cfg, sim, gymapi.SIM_PHYSX, 'cuda', 0, True)
    env_objects = []
    for env_id in range(args.envs):
        obj_id = int(task.object_id_buf[env_id])
        code = task.object_code_list[obj_id]
        scale = float(task.object_scale_buf[env_id])
        spec = objects[obj_id]
        if code != spec['object_code'] or scale != spec['scale']:
            raise ValueError('actual simulator object assignment disagrees with manifest')
        env_objects.append(dict(env=env_id,object_id=obj_id,object_code=code,scale=scale,split=spec['split']))
    write_json(args.output/'environment-objects.json',env_objects)
    write_json(args.output/'aggregate-capacity.json',dict(bodies=task._pilot_aggregate_bodies,
        shapes=task._pilot_aggregate_shapes,reservation='maximum over all loaded objects/scales'))
    actor_layers = []
    widths = [300,1024,1024,512,512,24]
    for i,(lo,hi) in enumerate(zip(widths,widths[1:])):
        actor_layers.append(torch.nn.Linear(lo,hi))
        if i < 4:
            actor_layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*actor_layers).to('cuda:0').eval()
    weights = torch.load(args.subset/'plus/dexgrasp/example_model/state_based_model.pt',
                         map_location='cuda:0', weights_only=True)
    actor.load_state_dict({key[len('actor.'):]:value for key,value in weights.items()
                          if key.startswith('actor.')}, strict=True)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    # The official replay asset removes the independent wrist body. Palm origin
    # is the explicitly named root landmark, not an invented missing wrist.
    names = ['robot0:palm','robot0:thhub','robot0:ffknuckle','robot0:mfknuckle',
             'robot0:rfknuckle','robot0:lfknuckle','robot0:thdistal','robot0:ffdistal',
             'robot0:mfdistal','robot0:rfdistal','robot0:lfdistal']
    indices = [task.gym.find_actor_rigid_body_index(task.envs[0], task.shadow_hands[0],
                                                  name, gymapi.DOMAIN_ACTOR) for name in names]
    if any(i < 0 for i in indices):
        present = task.gym.get_actor_rigid_body_names(task.envs[0],task.shadow_hands[0])
        raise ValueError('missing named ShadowHand landmark: '+str(dict(zip(names,indices)))+'; '+str(present))
    tip_offsets = torch.tensor([[0,0,.0275]] + [[0,0,.026]]*4,
                                dtype=torch.float32,device='cuda:0')
    from isaacgym.torch_utils import quat_apply
    def snapshot():
        rigid = task.rigid_body_states[:,indices].clone()
        hand = rigid[:,:,:3].clone()
        hand[:,6:] += quat_apply(rigid[:,6:,3:7].reshape(-1,4),
                                 tip_offsets[None].expand(args.envs,-1,-1).reshape(-1,3)).reshape(args.envs,5,3)
        return dict(hand=hand.cpu().numpy(), object=task.object_pose.detach().cpu().numpy().copy(),
                    q=task.shadow_hand_dof_pos.detach().cpu().numpy().copy(),
                    dq=task.shadow_hand_dof_vel.detach().cpu().numpy().copy(),
                    hand_root=task.root_state_tensor[task.hand_indices,:13].cpu().numpy().copy(),
                    observation=task.obs_buf.detach().cpu().numpy().copy(),
                    goal=task.goal_pos.detach().cpu().numpy().copy(),
                    success=task.successes.cpu().numpy().copy(), reset=task.reset_buf.cpu().numpy().copy())
    records = []
    begin = time.monotonic()
    try:
        with torch.inference_mode():
            for episode in range(args.episodes):
                task.reset_buf[:] = 1
                task.step(torch.zeros((args.envs,24), device='cuda:0'), -1)
                sequence = [snapshot()]
                controls, applied, forces, torques, targets = [], [], [], [], []
                for tick in range(1,task.max_episode_length):
                    if time.monotonic()-begin > args.gpu_seconds:
                        raise TimeoutError('bounded GPU replay deadline')
                    if task.reset_buf.any():
                        raise ValueError('unexpected reset inside episode; do not splice')
                    if not torch.isfinite(task.obs_buf).all():
                        raise ValueError('nonfinite policy observation')
                    if not torch.allclose(task.obs_buf[:,207:236], torch.zeros_like(task.obs_buf[:,207:236])):
                        raise ValueError('UniDexGrasp++ goal observation mismatch')
                    if not torch.allclose(task.obs_buf[:,236:300], .1*task.visual_feat_buf):
                        raise ValueError('official shape feature mismatch')
                    action = actor(task.obs_buf.clamp(-5,5)).clamp(-1,1)
                    controls.append(action.cpu().numpy().copy())
                    task.pre_physics_step(action.clone())
                    applied.append(task.actions.cpu().numpy().copy())
                    forces.append(task.apply_forces.cpu().numpy().copy())
                    torques.append(task.apply_torque.cpu().numpy().copy())
                    targets.append(task.cur_targets.cpu().numpy().copy())
                    for _ in range(task.control_freq_inv):
                        task.gym.simulate(task.sim)
                        task.gym.fetch_results(task.sim, True)
                    task.post_physics_step()
                    sequence.append(snapshot())
                    if tick % 50 == 0:
                        write_json(args.output/'progress.json', dict(episode=episode,tick=tick,
                            elapsed_s=time.monotonic()-begin,success=task.successes.cpu().tolist(),
                            device='cuda:0',training_allowed=False))
                        print('episode',episode,'tick',tick,'success',task.successes.cpu().tolist(),flush=True)
                data = {key:np.stack([frame[key] for frame in sequence]) for key in sequence[0]}
                if any(not np.isfinite(value).all() for value in data.values()):
                    raise ValueError('nonfinite recorded physics state')
                dt = float(task.dt*task.control_freq_inv)
                data.update(action=np.stack(controls),applied=np.stack(applied),force=np.stack(forces),
                            torque=np.stack(torques),target=np.stack(targets),
                            time=np.arange(1,len(sequence)+1,dtype=np.float64)*dt,
                            training_allowed=False)
                path = args.output/('episode-%02d.npz'%episode)
                np.savez_compressed(path,**data)
                records.append(dict(path=path.name,frames=len(sequence),dt=dt,
                    official_success=data['success'][-1].tolist(),
                    max_lift_m=(data['object'][:,:,2].max(0)-data['object'][0,:,2]).tolist()))
    finally:
        task.gym.destroy_sim(task.sim)
    result = dict(status='COMPLETE',episodes=records,envs=args.envs,
                  scale=objects[0]['scale'] if len(objects)==1 else None,
                  object_code=objects[0]['object_code'] if len(objects)==1 else None,
                  objects=objects,environment_objects=env_objects,seed=args.seed,elapsed_s=time.monotonic()-begin,
                  upstream_commit='6c9e710b04769bf5d9765e3233820101cb3d718f',
                  policy='official deterministic state actor, strict weights',
                  hand_landmarks=names,hand_tip_local_offsets_m=tip_offsets.cpu().tolist(),
                  first_observation='after official reset zero-action warm step',asset_root=str(assets),
                  training_allowed=False)
    write_json(args.output/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.output/'tmp'
    scratch.mkdir()
    metrics = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    write_json(args.output/'gpu-preflight.json',dict(gpu=args.gpu,raw_memory_mib=metrics.strip(),
        status='IDLE' if int(metrics.strip())<=64 else 'BUSY',checked_at=time.time()))
    if int(metrics.strip()) > 64:
        raise RuntimeError('selected GPU is no longer idle')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
               TORCH_EXTENSIONS_DIR=str(scratch/'extensions'), MAX_JOBS='2')
    env['LD_LIBRARY_PATH'] = str(Path(sys.executable).resolve().parents[1]/'lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start = time.monotonic()
    reason = 'CHILD_EXIT'
    peak = 0
    with (args.output/'replay.log').open('x') as log:
        child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity = dict(pid=child.pid,supervisor_pid=os.getpid(),uid=os.getuid(),gpu=args.gpu,
                        command=command,git_commit=commit,work_version='unidexgrasp-state-replay',
                        run_id=args.output.name,max_seconds=args.seconds,gpu_seconds=args.gpu_seconds,
                        input=str(args.subset),vendor=str(args.vendor),seed=args.seed,started_at=time.time())
        identity['object_manifest'] = str(args.object_manifest) if args.object_manifest else None
        write_json(args.output/'launch.json',identity)
        try:
            while child.poll() is None:
                metrics = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                    '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                peak = max(peak,int(metrics.split(',')[1]))
                row = dict(elapsed_s=time.monotonic()-start,gpu=metrics,peak_mib=peak)
                with (args.output/'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                marker = args.output/'gpu-started.json'
                if stop[0] or (args.output/'STOP').exists():
                    reason='USER_STOP';break
                if time.monotonic()-start >= args.seconds or (marker.exists() and
                        time.time()-json.loads(marker.read_text())['time'] >= args.gpu_seconds):
                    reason='HARD_DEADLINE';break
                if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 1024*2**20:
                    reason='DISK_BUDGET';break
                for _ in range(10):
                    if child.poll() is not None or stop[0]:break
                    time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            write_json(args.output/'supervisor.json',dict(identity,status='COMPLETE' if child.returncode==0 else 'FAILED',
                reason=reason,returncode=child.returncode,elapsed_s=time.monotonic()-start,peak_mib=peak))
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset',type=Path,required=True)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--assets',type=Path,help='Reuse an audited prepared asset root read-only')
    parser.add_argument('--object-manifest',type=Path,help='Completed official multi-object preparation manifest')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--envs',type=int,default=8)
    parser.add_argument('--episodes',type=int,default=2)
    parser.add_argument('--scale',type=float,default=.08)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--gpu-seconds',type=int,default=300)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    for key in ('subset','vendor','output'):
        setattr(args,key,getattr(args,key).resolve())
    if args.assets:
        args.assets=args.assets.resolve()
    if args.object_manifest:
        args.object_manifest=args.object_manifest.resolve()
    if (not 1<=args.envs<=256 or not 1<=args.episodes<=2 or not 60<=args.seconds<=600
            or not 30<=args.gpu_seconds<=300 or args.output==Path('/tmp') or Path('/tmp') in args.output.parents):
        raise ValueError('bounded replay arguments required')
    object_specs(args)
    worker(args) if args.worker else supervise(args)
