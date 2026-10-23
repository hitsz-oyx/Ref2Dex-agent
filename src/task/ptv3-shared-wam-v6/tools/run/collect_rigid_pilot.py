#!/usr/bin/env python3
"""Fresh source4 pilot: producer-owned intent/reset evidence and measured labels.

Stages prepare, collect and audit use independent receipts. No production recipe
is edited. Trusted local author assets are read-only; all output is run-local.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK.parent/'ptv3-shared-wam-v5/src'),
                str(TASK.parent/'wam-rl-policy-v3/src'), str(TASK/'tools/audit')]
from ptv3_shared_wam_v6.rigid_dataset import geometry_class, pca_shape, admission_reasons
from ptv3_shared_wam_v5.contracts import JOINT_NAMES, PART_NAMES, LABEL_SEMANTICS
from wam_rl_policy_v3.goal_channel import (bind_source, instruction_from_displacement,
    ResetWaypointChannel, digest, validate_instruction)

EXPERIMENT = 'P-20261023-rigid-source-bound-pilot'
VERSION = 'source-bound-rigid-pilot.v1'
PRIOR = ROOT/'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1'
COVERAGE = ROOT/'outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/coverage-after.csv'
PHASES = ('approach', 'close', 'contact', 'hold')
CLASSES = ('flat', 'elongated', 'compact')
GIB = 2**30


def write(path, value):
    path = Path(path)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    guard(path.parent, len(content.encode()))
    pending = path.with_suffix(path.suffix+'.part')
    pending.write_text(content)
    pending.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def guard(folder, need=0):
    free = shutil.disk_usage(folder).free
    if free-need < 30*GIB:
        raise RuntimeError('30GiB free-space floor: free=%d need=%d' % (free, need))
    return free


def split_entries(path):
    node = ast.parse('{'+Path(path).read_text()+'}', mode='eval').body
    result = {}
    for key, value in zip(node.keys, node.values):
        result.setdefault(ast.literal_eval(key), set()).update(ast.literal_eval(value))
    return result


def quotas(rows, count=96, source=4):
    """Water-fill the lowest existing mechanism cells; identities never enter."""
    old = {(r['phase'], r['geometry_class']):int(r['windows']) for r in rows
           if int(r['source_id']) == source and r['phase'] in PHASES and r['geometry_class'] in CLASSES}
    if len(old) != 12: raise ValueError('complete source phase/geometry coverage required')
    allocation = {k:0 for k in old}
    for _ in range(count):
        key = min(old, key=lambda k:(old[k]+allocation[k], PHASES.index(k[0]), CLASSES.index(k[1])))
        allocation[key] += 1
    return [dict(phase=p, geometry_class=g, prior_windows=old[p,g], rollouts=allocation[p,g])
            for p in PHASES for g in CLASSES]


def sample_mesh(mesh, rng):
    triangles = np.asarray(mesh.triangles)
    cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    area = np.linalg.norm(cross, axis=-1)
    keep = area > 1e-14
    triangles, cross, area = triangles[keep], cross[keep], area[keep]
    ids = rng.choice(len(area), 2048, p=area/area.sum())
    uv = rng.random((2048,2)); uv[uv.sum(-1)>1] = 1-uv[uv.sum(-1)>1]
    t = triangles[ids]
    points = (t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])).astype('f4')
    normals = (cross[ids]/area[ids,None]).astype('f4')
    return dict(surface=points, normals=normals, query=points[::64].copy())


def prepare(out):
    import trimesh
    if out.exists(): raise FileExistsError(out)
    free = guard(out.parent, 2*GIB)
    out.mkdir(); (out/'tmp').mkdir(); (out/'registry').mkdir()
    write(out/'storage-preflight.json', dict(free_before_bytes=free, floor_bytes=30*GIB,
        output_cap_bytes=2*GIB, estimated_tick_payload_gib=[.44,.88],
        concurrent_conversion=True, checked_at_ns=time.time_ns()))
    train = split_entries(PRIOR/'inputs/train_set.yaml')
    excluded = set(split_entries(PRIOR/'inputs/test_set_seen_cat.yaml')) | set(split_entries(PRIOR/'inputs/test_set_unseen_cat.yaml'))
    priors = np.load(PRIOR/'inputs/datasetv4.1_posedata.npy', allow_pickle=True).item()
    chosen = []; counts = Counter(); meshes = {}
    started = time.monotonic()
    # Trusted local author archive, no extraction of arbitrary archive paths.
    with tarfile.open(PRIOR/'inputs/meshdatav3_scaled.tar.xz', 'r|xz') as archive:
        for entry in archive:
            parts = Path(entry.name).parts
            if not entry.isfile() or len(parts)!=5 or not parts[-1].startswith('decomposed_') or not parts[-1].endswith('.obj'): continue
            code = '/'.join(parts[1:3])
            if code not in train or code in excluded or code not in priors: continue
            scale = int(parts[-1][11:-4])/100
            if scale not in train[code] or scale not in priors[code]: continue
            if any(r['object_code']==code for r in chosen): continue
            if entry.size > 16*1024**2: raise ValueError('oversized mesh')
            mesh = trimesh.load(io.BytesIO(archive.extractfile(entry).read()), file_type='obj', process=False, force='mesh')
            # Classify the same deterministic surface used by the coverage audit.
            geo = sample_mesh(mesh, np.random.default_rng(198))
            shape = pca_shape(geo['surface']); kind = geometry_class(shape)
            if kind not in CLASSES or counts[kind]>=2: continue
            values = priors[code][scale]
            count = len(values['object_init_z'])
            if not count: continue
            row = dict(object_code=code, scale=scale, geometry_class=kind, pca_extent_m=shape.tolist(),
                       official_split='TRAIN', split='train', initialization_priors=count)
            meshes[code] = geo; chosen.append(row); counts[kind]+=1
            print(json.dumps(dict(stage='asset-selection', selected=row, elapsed_s=time.monotonic()-started)), flush=True)
            if len(chosen)==6: break
    if len(chosen)!=6: raise ValueError('two real assets per numeric geometry class required')
    assets = out/'assets'; assets.mkdir()
    for name in ('mjcf','textures'):
        (assets/name).symlink_to(PRIOR/'assets'/name, target_is_directory=True)
    needed = {r['object_code']:r for r in chosen}
    with tarfile.open(PRIOR/'inputs/meshdatav3_scaled.tar.xz', 'r|xz') as archive:
        for entry in archive:
            parts = Path(entry.name).parts
            if len(parts)<5 or '/'.join(parts[1:3]) not in needed or not entry.isfile(): continue
            if '..' in parts or Path(entry.name).is_absolute() or entry.size>16*1024**2: raise ValueError('unsafe asset')
            guard(out, entry.size)
            dest = assets/entry.name; dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.extractfile(entry).read())
    with zipfile.ZipFile(PRIOR/'inputs/meshdatav3_pc_feat.zip') as archive:
        for row in chosen:
            name = 'meshdatav3_pc_feat/'+row['object_code']+'/pc_feat_%03d.npy'%round(row['scale']*100)
            dest = assets/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(archive.read(name))
    filtered = {r['object_code']:{r['scale']:priors[r['object_code']][r['scale']]} for r in chosen}
    np.save(assets/'datasetv4.1_posedata.npy', filtered)
    for i,row in enumerate(chosen):
        filename = 'registry/geometry-%02d.npz'%i
        np.savez(out/filename, **meshes[row['object_code']])
        row.update(registry=filename, registry_sha256=sha(out/filename))
    with COVERAGE.open() as f: baseline = list(csv.DictReader(f))
    allocation = quotas(baseline)
    jobs = []
    for cell in allocation:
        eligible = [r for r in chosen if r['geometry_class']==cell['geometry_class']]
        for i in range(cell['rollouts']):
            asset = eligible[i%len(eligible)]
            jobs.append(dict(rollout_id='rollout-%03d'%len(jobs), intended_phase=cell['phase'],
                             geometry_class=cell['geometry_class'], object_code=asset['object_code'], scale=asset['scale']))
    # The upstream loader assigns equal contiguous env blocks to object codes.
    # Use one asset/process, allowing mechanism quotas independent of identity.
    for row in chosen:
        row['jobs'] = [j for j in jobs if j['object_code']==row['object_code']]
    write(out/'plan.json', dict(experiment_id=EXPERIMENT, work_version=VERSION, run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), branch='main',
        seed=198, source_id=4, source_name='UniDexGrasp++ Shadow same physical source4',
        objects=chosen, quotas=allocation, jobs=jobs, fps=30, ticks_per_rollout=300,
        state_frames_per_rollout=301, boundary_convention='initial state plus 300 output ticks; actor/physics60Hz, record every2 controls',
        native_controls_per_rollout=600, output_cap_bytes=2*GIB, seconds_cap=7200,
        assets_selection='first two per PCA surface class in local official TRAIN archive order; heldout meshes excluded; no outcome selection',
        phase_protocol={'approach':'neutral first2s then frozen actor', 'close':'frozen actor',
                        'contact':'frozen actor6s, neutral/open1s, frozen actor3s', 'hold':'frozen actor'},
        training_allowed=False, baseline_coverage=str(COVERAGE), baseline_coverage_sha256=sha(COVERAGE),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,trimesh=trimesh.__version__,host=platform.node()),
        prepared_elapsed_s=time.monotonic()-started))


def install_bound_reset(cls, frozen, before_reset):
    """Restore the predeclared goal on EVERY native reset_target_pose call."""
    original = cls.reset_target_pose
    def reset_target_pose(task, env_ids, apply_reset=False):
        stamp = time.time_ns()
        original(task, env_ids, apply_reset=False)
        task.goal_states[env_ids] = frozen[env_ids]
        task.goal_init_state[env_ids] = frozen[env_ids]
        task.root_state_tensor[task.goal_object_indices[env_ids], :13] = frozen[env_ids]
        before_reset(task, env_ids, stamp)
        if apply_reset:
            from isaacgym import gymtorch
            ids = task.goal_object_indices[env_ids].to(dtype=__import__('torch').int32).contiguous()
            task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task.root_state_tensor), gymtorch.unwrap_tensor(ids), len(ids))
        task.reset_goal_buf[env_ids] = 0
    cls.reset_target_pose = reset_target_pose


def reuse_prepared(out, previous):
    """Reuse immutable prepared assets after a failed pre-control initialization."""
    if out.exists(): raise FileExistsError(out)
    guard(out.parent,2*GIB);out.mkdir();(out/'tmp').mkdir()
    plan=json.loads((previous/'plan.json').read_text())
    for name in ('assets','registry'):(out/name).symlink_to(previous/name,target_is_directory=True)
    plan.update(run_id=out.name,prepared_from=str(previous),prepared_plan_sha256=sha(previous/'plan.json'))
    write(out/'plan.json',plan)
    write(out/'storage-preflight.json',dict(free_before_bytes=guard(out,2*GIB),floor_bytes=30*GIB,
        output_cap_bytes=2*GIB,estimated_tick_payload_gib=[.44,.88],checked_at_ns=time.time_ns(),prepared_assets_reused_read_only=True))


def worker(out, object_index, gpu):
    # Preserve required Isaac-before-Torch import order.
    from isaacgym import gymapi, gymutil
    import torch
    import yaml
    from scipy.spatial.transform import Rotation
    from ptv3_shared_wam_v5.shadow import ShadowSurface
    plan = json.loads((out/'plan.json').read_text()); asset = plan['objects'][object_index]
    jobs = asset['jobs']; n = len(jobs)
    batch = out/('batch-%02d'%object_index); batch.mkdir(); (batch/'tmp').mkdir()
    torch.set_num_threads(2); torch.manual_seed(198+object_index); np.random.seed(198+object_index); random.seed(198+object_index)
    if 'float' not in np.__dict__: np.float = float
    vendor = ROOT/'third_party/UniDexGrasp2'
    sys.path.insert(0,str(vendor/'dexgrasp'))
    spec = importlib.util.spec_from_file_location('pilot_upstream', ROOT/'src/task/video-data-preparation/tools/run/replay_unidexgrasp.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    cls = helper.task_class(vendor,batch)
    cfg = yaml.safe_load((vendor/'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
    cfg['env'].update(numEnvs=n, random_time=False, printNumSuccesses=False,
        object_code_dict={asset['object_code']:[asset['scale']]}, episodeLength=602)
    cfg['env']['asset']['assetRoot'] = str(out/'assets'); cfg['graphics_device_id']=-1
    sim = gymapi.SimParams(); sim.dt=1/60; sim.num_client_threads=0
    sim.physx.use_gpu=True; sim.physx.num_subscenes=0; sim.physx.num_threads=2
    sim.physx.max_gpu_contact_pairs=8*1024*1024; sim.use_gpu_pipeline=True
    gymutil.parse_sim_config(cfg['sim'],sim)
    work = out/'dexgrasp'; work.mkdir(exist_ok=True); os.chdir(work)
    write(batch/'simulation-config.json',dict(config=cfg, dt=sim.dt, substeps=sim.substeps, gpu=gpu,
        seed=198+object_index, python=sys.version, executable=sys.executable, numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda))
    task = cls(cfg,sim,gymapi.SIM_PHYSX,'cuda',0,True)
    try:
        if abs(task.dt-1/60)>1e-7 or task.control_freq_inv!=1 or task.num_shadow_hand_dofs!=22:
            raise ValueError('pinned60Hz/q22 source4 physics required')
        frozen = task.goal_init_state.clone()
        frozen_np = frozen.cpu().numpy()
        intent_at = time.time_ns()
        # Written before resetting; no read of target rollout future.
        intents = [dict(source_role='explicit_task_instruction', source_execution_id=out.name+'-constructor',
            target_execution_id=out.name+'-'+j['rollout_id'], reset_id='reset-0',
            object_code=asset['object_code'], object_slot=0, hand_slot=0,
            goal_state=frozen_np[e].tolist(), available_at_ns=intent_at,
            goal_semantics='spatial_path_waypoint', independent_of_target_rollout=True) for e,j in enumerate(jobs)]
        write(batch/'intent-before-reset.json',intents); intent_hash=sha(batch/'intent-before-reset.json')
        geo = dict(np.load(out/asset['registry'],allow_pickle=False))
        channels = {}; reset_rows = {}; initializing = [True]
        def on_reset(native, ids, stamp):
            for env in ids.cpu().tolist():
                if env in channels:
                    if not initializing[0]: raise ValueError('unexpected second reset; never splice rollouts')
                    if not torch.equal(native.goal_states[env],frozen[env]): raise ValueError('repeated logical reset changed goal')
                    write(out/jobs[env]['rollout_id']/'repeated-initial-reset.json',dict(at_ns=stamp,
                        logical_reset=reset_rows[env]['reset'],restored_goal_state=native.goal_states[env].cpu().tolist(),
                        before_first_physics_control=True))
                    continue
                reset = dict(execution_id=intents[env]['target_execution_id'],reset_id='reset-0', clock_id='host-realtime-ns',at_ns=stamp)
                pose = np.eye(4); pose[:3,3]=frozen_np[env,:3]; pose[:3,:3]=Rotation.from_quat(frozen_np[env,3:7]).as_matrix()
                def bound(payload,role,name):
                    return bind_source(payload,source_role=role,source_kind='declared_task',source_id=name,
                        reset=reset, observed_at_ns=intent_at,available_at_ns=intent_at)
                instruction=instruction_from_displacement(bound(pose,'task_instruction','frozen-constructor-goal'),
                    bound(dict(rule='fixed_world_translation',translation_m=[0,0,0],target_stage='task_completion'),'task_instruction','declared-goal'),
                    bound(geo['query'],'reset_geometry','query32'),bound(geo['surface'],'reset_geometry','surface2048'),
                    reset=reset,object_slot=0,hand_slot=0,goal_type='se3_pose')
                folder=out/jobs[env]['rollout_id']; folder.mkdir()
                channels[env]=ResetWaypointChannel(instruction,expected_reset=reset,artifact_dir=folder/'goal-channel')
                row=dict(source_role='explicit_task_instruction', available_before_reset=intent_at<stamp,
                    source_available_timestamp_ns=intent_at,reset_timestamp_ns=stamp,intent_artifact=str(batch/'intent-before-reset.json'),
                    source_artifact_sha256=intent_hash, reset=reset, restored_goal_state=native.goal_states[env].cpu().tolist(),
                    reset_target_pose_goal_restored=bool(torch.equal(native.goal_states[env],frozen[env])),
                    reset_root_goal_restored=bool(torch.equal(native.root_state_tensor[native.goal_object_indices[env]],frozen[env])),
                    instruction_sha256=instruction['instruction_sha256'],job=jobs[env])
                write(folder/'reset-binding.json',row); reset_rows[env]=row
        install_bound_reset(cls,frozen,on_reset)
        task.reset_buf[:]=1; task.step(torch.zeros((n,24),device='cuda:0'),-1)
        initializing[0] = False
        widths=[300,1024,1024,512,512,24]; layers=[]
        for i,(a,b) in enumerate(zip(widths,widths[1:])):
            layers.append(torch.nn.Linear(a,b))
            if i<4: layers.append(torch.nn.ELU())
        actor=torch.nn.Sequential(*layers).cuda().eval()
        checkpoint=ROOT/'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/plus/dexgrasp/example_model/state_based_model.pt'
        weights=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
        actor.load_state_dict({k[len('actor.'):]:v for k,v in weights.items() if k.startswith('actor.')},strict=True)
        for p in actor.parameters(): p.requires_grad_(False)
        robot=out/'assets/mjcf/open_ai_assets/hand/robot.xml'
        surface=ShadowSurface(robot)
        names=task.gym.get_actor_rigid_body_names(task.envs[0],task.shadow_hands[0])
        # All21 joints independently obtained from named live rigid bodies.
        landmarks=['robot0:palm']
        for f in ('th','ff','mf','rf','lf'):
            landmarks += ['robot0:'+x for x in (['thbase','thhub','thdistal'] if f=='th' else [f+'knuckle',f+'middle',f+'distal'])]
            landmarks += ['robot0:'+f+'distal']
        ids=[names.index(x) for x in landmarks]
        write(batch/'semantic-map.json',dict(joint_semantics=list(JOINT_NAMES),part_semantics=list(PART_NAMES),
            joint_body_names=landmarks,tip_local_z_m=[.0275,.026,.026,.026,.026],
            wrist_definition='source4 palm origin; replay asset has no independent wrist body',
            native_dof_names=task.gym.get_actor_dof_names(task.envs[0],task.shadow_hands[0]),
            fk_dof_names=surface.names, asset=str(robot),asset_sha256=sha(robot),
            actor_checkpoint=str(checkpoint),actor_sha256=sha(checkpoint),policy_updates=0))
        if task.gym.get_actor_dof_names(task.envs[0],task.shadow_hands[0])!=surface.names:
            raise ValueError('FK native DOF order mismatch')
        from isaacgym.torch_utils import quat_apply
        frames=[]; actions=[]; targets=[]; forces=[]
        def snapshot():
            body=task.rigid_body_states[:,ids].clone(); hand=body[:,:,:3].clone()
            offset=torch.zeros((n,5,3),device='cuda:0');offset[:,:,2]=torch.tensor([.0275,.026,.026,.026,.026],device='cuda:0')
            tip_ids=[4,8,12,16,20]
            hand[:,tip_ids]+=quat_apply(body[:,tip_ids,3:7].reshape(-1,4),offset.reshape(-1,3)).reshape(n,5,3)
            return dict(q=task.shadow_hand_dof_pos.cpu().numpy().copy(),hand_root=task.root_state_tensor[task.hand_indices,:13].cpu().numpy().copy(),
                object=task.root_state_tensor[task.object_indices,:13].cpu().numpy().copy(),
                native_joint21=hand.cpu().numpy(), goal_state=task.goal_states.cpu().numpy().copy(),native_progress=task.progress_buf.cpu().numpy().copy(),
                observed_at_ns=np.full(n,time.time_ns(),np.int64))
        frames.append(snapshot()); started=time.monotonic()
        neutral=torch.zeros((n,24),device='cuda:0')
        neutral[:,6:]=(2*(task.shadow_hand_default_dof_pos[task.actuated_dof_indices]-task.shadow_hand_dof_lower_limits[task.actuated_dof_indices]) /
            (task.shadow_hand_dof_upper_limits[task.actuated_dof_indices]-task.shadow_hand_dof_lower_limits[task.actuated_dof_indices])-1)
        with torch.inference_mode():
            for tick in range(600):
                if time.monotonic()-started>1000: raise TimeoutError('batch cap1000s')
                if task.reset_buf.any() or task.reset_goal_buf.any(): raise ValueError('unexpected native reset')
                if not torch.isfinite(task.obs_buf).all(): raise ValueError('nonfinite observation')
                action=actor(task.obs_buf.clamp(-5,5)).clamp(-1,1)
                for e,j in enumerate(jobs):
                    if (j['intended_phase']=='approach' and tick<120) or (j['intended_phase']=='contact' and 360<=tick<420): action[e]=neutral[e]
                actions.append(action.cpu().numpy().copy())
                task.pre_physics_step(action.clone())
                targets.append(task.cur_targets.cpu().numpy().copy()); forces.append(task.apply_forces.cpu().numpy().copy())
                task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task.post_physics_step()
                if tick%2==1: frames.append(snapshot())
                if tick%120==119:
                    guard(out, 24*1024**2)
                    write(batch/'progress.json',dict(native_controls=tick+1,envs=n,elapsed_s=time.monotonic()-started))
                    print(json.dumps(dict(stage='collect',batch=object_index,native_controls=tick+1,envs=n,elapsed_s=time.monotonic()-started)),flush=True)
        data={k:np.stack([f[k] for f in frames]) for k in frames[0]}
        data.update(action=np.stack(actions),pd_target=np.stack(targets),hand_force=np.stack(forces))
        for e,j in enumerate(jobs):
            folder=out/j['rollout_id']; guard(out,16*1024**2)
            np.savez(folder/'raw.npz',**{k:v[:,e] for k,v in data.items()})
        write(batch/'completed.json',dict(status='COMPLETED',rollouts=n,native_controls=n*600,
            warmup_controls=n,output_ticks=n*300,elapsed_s=time.monotonic()-started,gpu=gpu,policy_updates=0))
    finally: task.gym.destroy_sim(task.sim)


def collect(out,gpu,resume=False):
    plan=json.loads((out/'plan.json').read_text())
    if any(out.glob('batch-*')) and not resume: raise ValueError('use explicit --resume at completed batch boundary')
    prior=out/'execution-manifest.json'
    if resume:
        previous=json.loads(prior.read_text())
        if previous['plan_sha256']!=sha(out/'plan.json'):raise ValueError('cannot resume changed plan')
        receipt=out/('resume-%d.json'%time.time_ns())
    else:receipt=prior
    write(receipt,dict(work_version=VERSION,run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_path=str(Path(__file__).resolve()),code_sha256=sha(__file__),command=sys.argv,
        executable=sys.executable,python=sys.version,physical_gpu=gpu,started_at_ns=time.time_ns(),
        plan_sha256=sha(out/'plan.json'),source_role='official data generator only; no learned weights changed'))
    shutil.copyfile(__file__,out/(receipt.stem+'-collector.py'))
    elapsed_before=(time.time_ns()-json.loads(prior.read_text())['started_at_ns'])/1e9
    started=time.monotonic()-elapsed_before; log=(out/'collection.log').open('a' if resume else 'x')
    records=json.loads((out/'processes.json').read_text()) if resume else []
    for i,asset in enumerate(plan['objects']):
        if not asset['jobs']: continue
        batch=out/('batch-%02d'%i)
        if batch.exists():
            done=batch/'completed.json'
            if not resume or not done.exists():raise ValueError('incomplete batch cannot be overwritten: '+str(batch))
            completed=json.loads(done.read_text())
            if completed['status']!='COMPLETED' or completed['rollouts']!=len(asset['jobs']):raise ValueError('batch receipt mismatch')
            if not all((out/j['rollout_id']/'raw.npz').exists() for j in asset['jobs']):raise ValueError('completed batch missing raw')
            continue
        while True:
            memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if memory<=64:break
            guard(out,24*1024**2)
            if time.monotonic()-started>7200 or (out/'STOP').exists():raise TimeoutError('waiting assigned idle GPU reached cap/STOP')
            write(out/'waiting-for-gpu.json',dict(gpu=gpu,memory_mib=memory,at_ns=time.time_ns(),next_batch=i,
                reason='leave other jobs and two idle GPUs intact'))
            time.sleep(10)
        guard(out, 2*GIB-sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()))
        scratch=out/'tmp'/('batch-%02d'%i);scratch.mkdir()
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--output',str(out),'--gpu',str(gpu),'--object-index',str(i)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1',
                 TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/rigid-pilot-20261023/extensions'),MAX_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        env['PYTHONPATH']='/home2/wyy/isaac-gym/isaacgym/python'+os.pathsep+env.get('PYTHONPATH','')
        env['LD_LIBRARY_PATH']=str(Path(sys.executable).resolve().parents[1]/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        records.append(dict(pid=child.pid,command=command,object_index=i));write(out/'processes.json',records)
        try:
            while child.poll() is None:
                if time.monotonic()-started>7200: raise TimeoutError('2h hard cap')
                if (out/'STOP').exists(): raise InterruptedError('run-owned STOP')
                guard(out,24*1024**2)
                size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
                if size>2*GIB: raise RuntimeError('2GiB hard output cap')
                gpu_state=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                with (out/'resources.jsonl').open('a') as f: f.write(json.dumps(dict(elapsed_s=time.monotonic()-started,gpu=gpu_state,free_bytes=shutil.disk_usage(out).free,output_bytes=size,batch=i))+'\n')
                time.sleep(10)
            if child.returncode: raise RuntimeError('batch worker failed: '+str(child.returncode))
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill();child.wait(timeout=5)
    log.close()
    write(out/'collection-complete.json',dict(status='COMPLETED',elapsed_s=time.monotonic()-started,gpu=gpu,
        rollouts=96,output_ticks=28800,native_controls=57600,warmup_native_controls=96,policy_updates=0))


def audit(out):
    from scipy.spatial.transform import Rotation
    from ptv3_shared_wam_v5.shadow import ShadowSurface
    from ptv3_shared_wam_v5.preparation import surface_contact
    from ptv3_shared_wam_v5.event_labels import proximity_counts, annotate_events
    from ptv3_shared_wam_v5.data import sample_window
    from certify_waypoint_sources import classify, LEGAL
    from curate_tau_training_set import phase_for
    plan=json.loads((out/'plan.json').read_text()); started=time.monotonic()
    if (out/'audit-summary.json').exists(): raise FileExistsError('audit already published')
    surface=ShadowSurface(out/'assets/mjcf/open_ai_assets/hand/robot.xml')
    assets={r['object_code']:r for r in plan['objects']}; verdicts=[]; windows=[]; certified=[]; phase_counts=Counter()
    for job in plan['jobs']:
        folder=out/job['rollout_id'];asset=assets[job['object_code']]
        raw=dict(np.load(folder/'raw.npz',allow_pickle=False)); n=len(raw['q'])
        if n!=301 or len(raw['action'])!=600: raise ValueError('length/tick drift')
        if not np.array_equal(raw['native_progress'],np.arange(1,602,2)): raise ValueError('reset/timing drift')
        joints,jvalid,vertices,svalid,parts=surface.reconstruct(raw['q'],raw['hand_root'])
        fk_error=float(np.max(np.abs(joints[:,0]-raw['native_joint21'])))
        if fk_error>.0001: raise ValueError('FK vs independently measured21 error '+str(fk_error))
        poses=np.broadcast_to(np.eye(4,dtype='f4'),(n,1,4,4)).copy()
        poses[:,0,:3,3]=raw['object'][:,:3];poses[:,0,:3,:3]=Rotation.from_quat(raw['object'][:,3:7]).as_matrix()
        finite=np.isfinite(raw['object']).all(-1)&np.isclose(np.linalg.norm(raw['object'][:,3:7],axis=-1),1.,atol=1e-4)
        seq=dict(joints=joints,joint_valid=jvalid,surface_offsets=(vertices-joints[:,:,None,0]).astype('f2'),
            surface_valid=svalid,surface_parts=parts,object_poses=poses,object_pose_valid=finite[:,None])
        geo=dict(np.load(out/asset['registry'],allow_pickle=False))
        if sha(out/asset['registry'])!=asset['registry_sha256']: raise ValueError('registry drift')
        contact=dict(contact=np.zeros((n,1,2048),'f2'),part=np.zeros((n,1,2048),'u1'),valid=finite[:,None].copy())
        for t in range(n):
            observed=(seq['surface_offsets'][t,0].astype('f4')+joints[t,0,0])
            query=geo['surface']@poses[t,0,:3,:3].T+poses[t,0,:3,3]
            contact['contact'][t,0],contact['part'][t,0]=surface_contact(observed,parts[0],query)
        enter,leave=proximity_counts(contact['contact'],contact['valid'])
        events=annotate_events(enter,leave,contact['valid'])
        binding=json.loads((folder/'reset-binding.json').read_text())
        intents=json.loads(Path(binding['intent_artifact']).read_text())
        intent=next(r for r in intents if r['target_execution_id']==binding['reset']['execution_id'])
        instruction=json.loads((folder/'goal-channel/instruction.json').read_text());validate_instruction(instruction)
        replay=ResetWaypointChannel(instruction,expected_reset=binding['reset'])
        source_ok=sha(binding['intent_artifact'])==binding['source_artifact_sha256']
        goal=np.eye(4);goal[:3,3]=intent['goal_state'][:3];goal[:3,:3]=Rotation.from_quat(intent['goal_state'][3:7]).as_matrix()
        reset_ok=(binding['reset_target_pose_goal_restored'] and binding['reset_root_goal_restored']
            and np.array_equal(binding['restored_goal_state'],intent['goal_state'])
            and np.array_equal(raw['goal_state'],np.broadcast_to(intent['goal_state'],raw['goal_state'].shape)))
        proof=dict(target_tuple_bound=(binding['job']==job),source_role=intent['source_role'],
            source_execution_id=intent['source_execution_id'],target_execution_id=binding['reset']['execution_id'],
            available_before_reset=(source_ok and intent['available_at_ns']<binding['reset_timestamp_ns'] and binding['available_before_reset']),
            independent_of_target_rollout=intent['independent_of_target_rollout'],artifact_digest_verified=source_ok,
            reset_binding_verified=bool(reset_ok),derivation_verified=bool(np.array_equal(goal,instruction['task_goal_pose'])),
            goal_semantics=intent['goal_semantics'])
        certification=classify(proof)
        evidence=dict(joint_semantics=list(JOINT_NAMES),part_semantics=list(PART_NAMES),label_semantics=LABEL_SEMANTICS,
            binding_verified=source_ok and fk_error<=.0001,geometry_binding_verified=(job['object_code']==intent['object_code']),event_binding_verified=True)
        reasons=Counter(); phases=Counter()
        for t in range(n):
            stamp=max(int(raw['observed_at_ns'][t]),binding['reset_timestamp_ns'])
            condition,receipt=replay.update(poses[t,0],raw['native_joint21'][t,[0,1,5,9,13,17,4,8,12,16,20]],
                tick=t,observed_at_ns=stamp,available_at_ns=stamp,decision_at_ns=stamp)
            if not np.array_equal(receipt['goal_pose'],goal): raise ValueError('goal changed after reset')
            if t>=3 and t<n-24:
                rejected=admission_reasons(seq,geo,contact,events,evidence,t,0); reasons.update(rejected)
                phase=phase_for(events,0,t); phases[phase]+=1
                record=dict(rollout_id=job['rollout_id'],source_id=4,slot=0,tick=t,phase=phase,
                    geometry_class=job['geometry_class'],admitted=not rejected,goal_certification=certification,
                    goal_condition_source='reset-declared task terminal pose',goal_condition_sha256=instruction['instruction_sha256'],reasons=rejected)
                windows.append(record)
                if not rejected and certification==LEGAL: certified.append([len(verdicts),0,t])
                if t==3:
                    from ptv3_shared_wam_v6.fresh_rigid import reset_bound_sample
                    sampled=reset_bound_sample(seq,geo,contact,t,0,4,[len(verdicts),0,t],instruction,binding['reset'])
                    if sampled['trajectory_target'].shape!=(24,2,21,3): raise ValueError('reader contract')
                    for key in condition:
                        if not np.array_equal(sampled[key],condition[key]):raise ValueError('offline/online goal mismatch')
        for sub,arrays in [('sequence',seq),('contact',contact),('events',events)]:
            dest=folder/sub;dest.mkdir()
            for key,value in arrays.items(): guard(out,value.nbytes+4096);np.save(dest/(key+'.npy'),value,allow_pickle=False)
        row=dict(job=job,frames=n,output_ticks=300,joint_semantics=list(JOINT_NAMES),part_semantics=list(PART_NAMES),
            label_semantics=LABEL_SEMANTICS,registry=asset['registry'],registry_sha256=asset['registry_sha256'],
            admitted_windows=sum(w['admitted'] for w in windows if w['rollout_id']==job['rollout_id']),
            candidate_windows=n-27,rejection_reasons=dict(reasons),goal_certification=certification,certification_proof=proof,
            native_joint21_vs_fk_max_m=fk_error,phase_windows=dict(phases),requested_phase_observed=phases[job['intended_phase']]>0,
            source_role='explicit_task_instruction',available_before_reset=proof['available_before_reset'],
            training_allowed=False,goal_input_policy='reset-bound-task-goal-required; default sample_window future goal forbidden')
        write(folder/'admission.json',row);verdicts.append(row)
        for w in windows[-(n-27):]:
            if w['admitted'] and w['goal_certification']==LEGAL:
                phase_counts[w['phase'],job['geometry_class']]+=1
        print(json.dumps(dict(stage='audit',rollout=job['rollout_id'],admitted=row['admitted_windows'],certification=certification,phases=dict(phases))),flush=True)
    with COVERAGE.open() as f: baseline=list(csv.DictReader(f))
    after=[]
    for r in baseline:
        row=dict(r);row['windows']=int(row['windows'])+(phase_counts[r['phase'],r['geometry_class']] if int(r['source_id'])==4 else 0);after.append(row)
    for name,rows in [('coverage-before.csv',baseline),('coverage-after.csv',after)]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['phase','geometry_class','source_id','windows']);writer.writeheader();writer.writerows(rows)
    for name,rows in [('window-verdicts.jsonl',windows),('rollout-verdicts.jsonl',verdicts)]:
        with (out/name).open('x') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    np.save(out/'certified-candidate-tuples.npy',np.asarray(certified,dtype='i8'))
    required=lambda rows:[r for r in rows if r['phase'] in PHASES and r['geometry_class'] in CLASSES]
    result=dict(experiment_id=EXPERIMENT,run_id=out.name,work_version=VERSION,
        collection=json.loads((out/'collection-complete.json').read_text()),
        git_commit=json.loads((out/'execution-manifest.json').read_text())['git_commit'],audit_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        rollouts=len(verdicts),fully_admitted_rollouts=sum(r['admitted_windows']==r['candidate_windows'] for r in verdicts),
        certified_goal_rollouts=sum(r['goal_certification']==LEGAL for r in verdicts),candidate_windows=len(windows),
        admitted_windows=sum(w['admitted'] for w in windows),certified_admitted_windows=len(certified),
        maximum_fk_error_m=max(r['native_joint21_vs_fk_max_m'] for r in verdicts),
        required_cells=len(required(baseline)),zero_cells_before=sum(int(r['windows'])==0 for r in required(baseline)),
        zero_cells_after=sum(int(r['windows'])==0 for r in required(after)),
        intended_phase_observed_rollouts=sum(r['requested_phase_observed'] for r in verdicts),
        actual_phase_geometry_windows={p+'/'+g:v for (p,g),v in phase_counts.items()},
        training_allowed=False,train_updates=0,model_changed=False,fixed_panel_prediction='NOT_RUN',
        audit_environment=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),
        audit_elapsed_s=time.monotonic()-started,free_after_bytes=shutil.disk_usage(out).free,
        output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()))
    write(out/'audit-summary.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','collect','worker','audit'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);p.add_argument('--object-index',type=int)
    p.add_argument('--prepared-from',type=Path)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if ROOT/'outputs/ptv3-shared-wam-v6' not in out.parents: raise ValueError('task-owned output required')
    if a.stage=='prepare':
        if a.prepared_from: reuse_prepared(out,a.prepared_from.resolve())
        else: prepare(out)
    elif a.stage=='collect':collect(out,a.gpu,a.resume)
    elif a.stage=='worker':worker(out,a.object_index,a.gpu)
    else:audit(out)


if __name__=='__main__': main()
