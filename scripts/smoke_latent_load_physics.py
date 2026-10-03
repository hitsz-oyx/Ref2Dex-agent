"""Engineering only: actual airplane GPU free-body force and torque response."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from isaacgym import gymapi, gymtorch  # must precede torch
    import torch
    import numpy as np
    from src.task.CmResidual.latent_load import scale_body, HEAVY_FACTOR
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1. / 60
    params.substeps = 4
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.use_gpu_pipeline = True
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('GPU sim creation')
    try:
        options = gymapi.AssetOptions()
        options.density = 20
        options.linear_damping = options.angular_damping = 0
        options.vhacd_enabled = True
        options.vhacd_params.max_convex_hulls = 20
        options.vhacd_params.max_num_vertices_per_ch = 16
        options.vhacd_params.resolution = 50000
        asset = gym.load_asset(sim, str(ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf'),
                               'airplane.urdf', options)
        records = []
        for index, factor in enumerate((1., HEAVY_FACTOR)):
            env = gym.create_env(sim, gymapi.Vec3(-2, -2, -2), gymapi.Vec3(2, 2, 2), 2)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1)
            handle = gym.create_actor(env, asset, pose, 'airplane', index, 0)
            before, after = scale_body(gym, env, handle, factor)
            records.append(dict(before=before, after=after, factor=factor))
        gym.prepare_sim(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        gym.refresh_actor_root_state_tensor(sim)
        initial = root.clone()
        force = torch.zeros((2, 3), device='cuda')
        torque = torch.zeros_like(force)
        force[:, 0] = .001
        torque[:, 2] = 1e-6
        if not gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(force),
                                                 gymtorch.unwrap_tensor(torque), gymapi.ENV_SPACE):
            raise RuntimeError('force/torque call')
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        delta = (root - initial).cpu().numpy()
        expected = np.array([.001 * params.dt / r['after']['mass'] for r in records])
        predicted_angular = np.array([np.linalg.solve(np.array(r['after']['inertia']),
                                                      [0, 0, 1e-6]) * params.dt for r in records])
        linear_error = float(np.max(np.abs(delta[:, 7] / expected - 1)))
        angular_relative = float(np.max(np.linalg.norm(delta[:, 10:13] - predicted_angular, axis=1)
                                        / np.linalg.norm(predicted_angular, axis=1)))
        ratio_linear = float(delta[0, 7] / delta[1, 7])
        ratio_angular = float(np.linalg.norm(delta[0, 10:13]) / np.linalg.norm(delta[1, 10:13]))
        passed = linear_error < .02 and angular_relative < .02 and abs(ratio_linear / HEAVY_FACTOR - 1) < .02 and abs(ratio_angular / HEAVY_FACTOR - 1) < .02
        result = dict(engineering_only=True, run_status='COMPLETED', passed=passed,
                      no_contacts_no_gravity=True, gpu_pipeline=True, dt=params.dt,
                      force_newtons=[.001, 0, 0], torque_newton_metres=[0, 0, 1e-6],
                      records=records, initial_root=initial.cpu().tolist(), final_root=root.cpu().tolist(),
                      velocity_delta=delta[:, 7:13].tolist(), expected_linear_x=expected.tolist(),
                      expected_angular=predicted_angular.tolist(), linear_relative_error=linear_error,
                      angular_relative_error=angular_relative, linear_ratio=ratio_linear, angular_ratio=ratio_angular)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('passed', 'linear_relative_error', 'angular_relative_error', 'linear_ratio', 'angular_ratio')}), flush=True)
        if not passed:
            raise ValueError('native inverse mass/inertia response failed')
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
