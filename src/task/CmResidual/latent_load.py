"""Persistent density contrast, set before prepare_sim; never a policy feature."""
import numpy as np

HEAVY_FACTOR = 50.0  # source density20 -> effective density1000 kg/m^3
EVALUATION_SEED = 701


def mass_and_inertia(properties):
    return dict(mass=float(properties.mass),
                inertia=[[float(getattr(getattr(properties.inertia, row), col))
                          for col in ('x', 'y', 'z')] for row in ('x', 'y', 'z')])


def scale_body(gym, env, handle, factor):
    """Same geometry/COM, uniform density scale; explicit inertia and inverse."""
    props = gym.get_actor_rigid_body_properties(env, handle)
    if len(props) != 1:
        raise ValueError('single rigid airplane body required')
    body = props[0]
    before = mass_and_inertia(body)
    body.mass *= factor
    body.invMass /= factor
    for row in ('x', 'y', 'z'):
        for col in ('x', 'y', 'z'):
            setattr(getattr(body.inertia, row), col,
                    getattr(getattr(body.inertia, row), col) * factor)
            setattr(getattr(body.invInertia, row), col,
                    getattr(getattr(body.invInertia, row), col) / factor)
    if not gym.set_actor_rigid_body_properties(env, handle, props, False):
        raise ValueError('SDK rejected explicit mass/inertia update')
    after = mass_and_inertia(gym.get_actor_rigid_body_properties(env, handle)[0])
    if not np.isclose(after['mass'], before['mass'] * factor, rtol=2e-6):
        raise ValueError('mass SDK roundtrip')
    if not np.allclose(after['inertia'], np.array(before['inertia']) * factor,
                       rtol=2e-6, atol=1e-12):
        raise ValueError('inertia SDK roundtrip')
    return before, after


def install_preparation_hook(seed):
    """Local process hook; source project is never edited."""
    import torch
    from env.tasks.base_dexplore_task import DexploreTask
    from src.task.CmResidual.continuous_critic_cm import policy_groups
    original = DexploreTask._build_target

    def build_target(task, env_id, env_ptr):
        original(task, env_id, env_ptr)
        if env_id == 0:
            motion = task._env_initial_motion.detach().cpu()
            if len(motion) != 768:
                raise ValueError('fixed panel size')
            groups = policy_groups(motion, seed).cpu()
            load = np.zeros(768, dtype=np.int64)
            generator = torch.Generator(device='cpu').manual_seed(seed + 31000)
            for mo in range(3):
                for arm in range(4):
                    indices = torch.where((motion == mo) & (groups == arm))[0]
                    if len(indices) != 64:
                        raise ValueError('balanced creation-time motion/arm')
                    chosen = indices[torch.randperm(64, generator=generator)[:32]]
                    load[chosen.numpy()] = 1
            task._latent_load = dict(creation_motion=motion.tolist(),
                                    creation_groups=groups.tolist(), load=load.tolist(),
                                    before=[], after=[], changed_before_prepare=True)
        factor = HEAVY_FACTOR if task._latent_load['load'][env_id] else 1.0
        before, after = scale_body(task.gym, env_ptr, task._target_handles[-1], factor)
        task._latent_load['before'].append(before)
        task._latent_load['after'].append(after)

    DexploreTask._build_target = build_target
