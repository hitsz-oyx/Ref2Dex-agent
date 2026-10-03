"""Engineering only: attributed contacts on GPU/CPU pipeline combinations."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=('gpu', 'cpu'), required=True)
    parser.add_argument('--physics', choices=('gpu', 'cpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from isaacgym import gymapi, gymtorch
    import torch
    import numpy as np
    torch.set_num_threads(2)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60.
    params.substeps = 4
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = args.physics == 'gpu'
    params.use_gpu_pipeline = args.pipeline == 'gpu'
    params.physx.num_threads = 2
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 1
    params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('sim creation')
    try:
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        support = gym.create_box(sim, 1., 1., .1, fixed)
        dynamic = gymapi.AssetOptions()
        dynamic.density = 1000
        cube = gym.create_box(sim, .1, .1, .1, dynamic)
        envs, records = [], []
        for i in range(2):
            env = gym.create_env(sim, gymapi.Vec3(-2, -2, -2), gymapi.Vec3(2, 2, 2), 2)
            envs.append(env)
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0, 0, -.05 if i == 0 else -10.)
            table = gym.create_actor(env, support, table_pose, 'support', i, 0)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, .20 if i == 0 else 20.)
            target = gym.create_actor(env, cube, pose, 'target', i, 0)
            for handle in (table, target):
                props = gym.get_actor_rigid_shape_properties(env, handle)
                props[0].friction = .5
                gym.set_actor_rigid_shape_properties(env, handle, props)
            records.append(dict(mass=float(gym.get_actor_rigid_body_properties(env, target)[0].mass),
                                target_body_env=gym.get_actor_rigid_body_index(env, target, 0, gymapi.DOMAIN_ENV),
                                support_body_env=gym.get_actor_rigid_body_index(env, table, 0, gymapi.DOMAIN_ENV)))
        gym.prepare_sim(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        net = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        for _ in range(120):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        native_contact_api_error = None
        contacts = []
        try:
            for env in envs:
                raw = gym.get_env_rigid_contacts(env)
                names = list(raw.dtype.names)
                contacts.append(dict(fields=names, count=len(raw),
                                     rows=[{name: np.asarray(contact[name]).tolist() for name in names}
                                           for contact in raw]))
        except Exception as error:
            native_contact_api_error = repr(error)
        result = dict(engineering_only=True, run_status='COMPLETED', pipeline=args.pipeline,
                      physics=args.physics, device=str(root.device), dt=params.dt, substeps=params.substeps,
                      records=records, roots=root.cpu().tolist(), net_force=net.cpu().tolist(),
                      attributed_contact_api_available=native_contact_api_error is None,
                      native_contact_api_error=native_contact_api_error, contacts=contacts,
                      contact_collection='CC_ALL_SUBSTEPS', physical_steps=120)
        if native_contact_api_error is None:
            active = contacts[0]['rows']
            pair_ok = bool(active) and all({row['body0'], row['body1']} == {0, 1} for row in active)
            no_false_positive = contacts[1]['count'] == 0
            force = net.cpu().numpy().reshape(2, 2, 3)[0, 1]
            weight = records[0]['mass'] * 9.81
            force_error = float(np.abs(force - [0, 0, weight]).max() / weight)
            lambdas = sum(row['lambda'] for row in active)
            result.update(pair_indices_verified=pair_ok, free_object_has_no_contacts=no_false_positive,
                          stable_net_force_relative_error=force_error,
                          lambda_sum=lambdas, lambda_to_weight_ratio=lambdas/weight,
                          lambda_to_impulse_ratio=lambdas/(weight*params.dt),
                          basic_contact_contract_pass=pair_ok and no_false_positive and force_error < .05)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('contacts', 'roots', 'net_force', 'records')}), flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
