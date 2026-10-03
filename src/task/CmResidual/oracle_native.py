"""Own-policy native task with CPU contact reads and GPU physics/inference.

The emitted task copy only fixes implicit CUDA placement for the CPU pipeline.
External task source is not edited. No root/DOF rewind after first simulation.
"""
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]


def install_cpu_task_source(output):
    source_file = ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py'
    source = source_file.read_text()
    replacements = {
        'to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).cuda()':
        'to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long, device="cpu")',
        'to_torch(self.max_episode_length, dtype=torch.long)':
        'to_torch(self.max_episode_length, dtype=torch.long, device=self.device)',
        'to_torch(self.start_contact_idx, dtype=torch.long)':
        'to_torch(self.start_contact_idx, dtype=torch.long, device=self.device)',
        'object_points = to_torch(object_points)':
        'object_points = to_torch(object_points, device=self.device)',
        'self.object_points.append(to_torch(object_points))':
        'self.object_points.append(to_torch(object_points, device=self.device))',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('CPU device-only source adaptation drift: '+old)
        source = source.replace(old, new)
    # Table loading precedes BaseTask.device initialization; all twelve
    # reference-data transfers are explicitly CPU for this CPU-only copy.
    old, new = ".to('cuda')", ".to('cpu')"
    if source.count(old) != 12:
        raise ValueError('reference-data CUDA placement count drift')
    source = source.replace(old, new)
    replacements[old] = dict(replacement=new, occurrences=12)
    destination = output/'native_sources/base_dexplore_task.py'
    destination.parent.mkdir()
    destination.write_text(source)
    name = 'env.tasks.base_dexplore_task'
    if name in sys.modules:
        raise ValueError('task already imported; preserve module identity')
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return dict(original=str(source_file), adapted=str(destination), replacements=replacements)


def create_session(args):
    from isaacgym import gymapi, gymtorch, gymutil
    import torch
    import yaml
    from scripts.run_contact_response_probe import sha
    from src.task.CmResidual.native_reset_transaction import NativeResetQueue
    from src.task.CmResidual.dexplore_bc_policy import DExploreBcPolicy
    from src.task.CmResidual.paired_evaluation import fingerprint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT/'third_party/DExplore/dexplore'))
    for name, value in (('float', float), ('int', int)):
        if not hasattr(np, name):
            setattr(np, name, value)
    source_adaptation = install_cpu_task_source(args.output)
    from env.tasks.dexplore_inspire import Dexplore_Inspire
    generation = json.loads(args.references_manifest.read_text())
    references = sorted(generation['references'], key=lambda r: r['name'])
    if generation['run_status'] != 'COMPLETED' or len(references) != 3 or not generation['synthetic']:
        raise ValueError('fixed three synthetic references')
    cfg = yaml.safe_load(args.env_config.read_text())
    cfg['env'].update(numEnvs=args.envs, motion_file=[str(Path(r['generated']).parent) for r in references],
                      is_test=True, enableEarlyTermination=False, hybridInitProb=1.,
                      hardObjectOversampling=False)
    cfg.update(seed=args.seed, name='Dexplore_Inspire', headless=True, task={'randomize': False})
    params = gymapi.SimParams()
    params.dt = 1 / 60.
    gymutil.parse_sim_config(cfg['sim'], params)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = args.physics == 'gpu'
    params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    params.physx.num_subscenes = 0
    task = Dexplore_Inspire(cfg, params, gymapi.SIM_PHYSX, 'cpu', 0, True)
    task._enable_early_termination = False
    task._adaptive_kappa_enabled = False
    if str(task._dof_pos.device) != 'cpu' or abs(task.dt - 1/30.) > 1e-8:
        raise ValueError('CPU read pipeline / native control clock')
    native = task.gym
    queue = NativeResetQueue(native, gymtorch.wrap_tensor)
    task.gym = queue
    task.reset(torch.arange(args.envs))
    initial_unshifted = task._target_states.clone()
    generator = torch.Generator(device='cpu').manual_seed(args.seed + 11000)
    offsets = (torch.rand((args.envs, 2), generator=generator)*2-1)*.01
    task._target_states[:, :2] += offsets
    ids = task._tar_actor_ids.to(torch.int32)
    queue.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                              gymtorch.unwrap_tensor(ids), len(ids))
    root_ids, dof_ids = queue.commit(task.sim, task._root_states, task._dof_state, gymtorch.unwrap_tensor)
    task.gym = native
    if task.progress_buf.any() or task.ref_index.any() or task.start_times.any():
        raise ValueError('frame0 reset')
    cp = torch.load(args.policy_checkpoint, map_location='cpu', weights_only=False)
    if cp['official_policy_checkpoint'] is not None or cp['source_actor_weights_used'] or cp['updates'] != 2000:
        raise ValueError('own scratch P0 only')
    model = DExploreBcPolicy(70, 18, tuple(cp['hidden_dims'])).to('cuda').eval()
    model.load_state_dict(cp['model'])
    if fingerprint(model.state_dict()) != cp['final_model_fingerprint']:
        raise ValueError('P0 identity')
    stops = torch.tensor([r['plateau_reference_frames_inclusive'][1] for r in references])
    lift = torch.tensor([r['first_lift_interval'][0] for r in references])
    motion = task.data_id.clone()
    if not torch.equal(motion, torch.arange(args.envs) % 3):
        raise ValueError('fixed balanced motion assignment')
    names = native.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    ranges = native.get_actor_rigid_body_shape_indices(task.envs[0], task.humanoid_handles[0])
    shape_owners = [None]*len(native.get_actor_rigid_shape_properties(task.envs[0], task.humanoid_handles[0]))
    for name, span in zip(names, ranges):
        start = int(span.start if hasattr(span, 'start') else span['start'])
        count = int(span.count if hasattr(span, 'count') else span['count'])
        for index in range(start, start+count):
            shape_owners[index] = name
    expected = [3 if ('thumb' in name and 'distal' in name) or ('thumb' not in name and 'intermediate' in name) else 2
                for name in shape_owners]
    body_metadata, filters = [], []
    for i, env in enumerate(task.envs):
        hand = task.humanoid_handles[i]
        actual = [int(p.filter) for p in native.get_actor_rigid_shape_properties(env, hand)]
        table_filter = [int(p.filter) for p in native.get_actor_rigid_shape_properties(env, task._table_handles[i])]
        if actual != expected or any(v != 1 for v in table_filter):
            raise ValueError('corrected native contact ownership')
        filters.append(actual)
        actor_records = []
        for handle in (hand, task._table_handles[i], task._target_handles[i]):
            props = native.get_actor_rigid_body_properties(env, handle)
            body_names = native.get_actor_rigid_body_names(env, handle)
            actor_records.append(dict(names=body_names, properties=[dict(mass=float(p.mass), flags=int(p.flags),
                com=[p.com.x,p.com.y,p.com.z], inertia=[[getattr(getattr(p.inertia, row), col)
                for col in ('x','y','z')] for row in ('x','y','z')]) for p in props]))
        body_metadata.append(actor_records)
    initial = dict(root_states=task._root_states.clone(), dof_state=task._dof_state.clone(),
                   motion=motion, before_placement=initial_unshifted, offsets=offsets,
                   root_commit_ids=root_ids, dof_commit_ids=dof_ids, refreshes_suppressed=queue.refreshes_suppressed,
                   object_root=task._target_states.clone(), table_root=task._table_states.clone(),
                   initial_height=task._target_states[:,2].clone(), phase_stop=stops, lift_start=lift,
                   lower=task.dof_limits_lower.clone(), upper=task.dof_limits_upper.clone(),
                   pd_offset=task._pd_action_offset.clone(), pd_scale=task._pd_action_scale.clone(),
                   native_reference_q=task.hoi_refs[:,0,:int(stops.max())+1,119:137].clone(),
                   policy_sha256=sha(args.policy_checkpoint), p0_fingerprint=fingerprint(model.state_dict()))
    metadata = dict(cpu_data_pipeline=True, physics=args.physics, inference_device='cuda',
                    hand_body_names=names, shape_owners=shape_owners, hand_shape_filters=filters,
                    actor_body_properties=body_metadata, source_adaptation=source_adaptation,
                    contact_collection='CC_ALL_SUBSTEPS', dt=task.dt, simulation_dt=params.dt,
                    substeps=params.substeps, control_frequency_inv=task.control_freq_inv,
                    gravity=[params.gravity.x, params.gravity.y, params.gravity.z],
                    body_names=names+['table','airplane'], actors_per_env=task.get_num_actors_per_env())
    return task, model, cp, initial, metadata
