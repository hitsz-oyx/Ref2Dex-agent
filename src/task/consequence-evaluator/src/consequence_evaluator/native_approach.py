"""Process-local correction of the native approach reward's object mesh.

The shared agent uses an airplane surface for every task. Keep its original
airplane-only behavior, sampling stride, hand FK and potential shaping intact;
mixed/non-airplane tasks select their actual object URDF and actor scale.
"""
from pathlib import Path

import torch


def approach_gap(agent, task, original, bridge_factory=None):
    names = tuple(task.object_name)
    if names == ('airplane',):
        return original(agent, task)
    if agent.grasp_link_reward_coef or agent.min_grasp_links:
        raise ValueError('object-aware approach adapter only supports the baseline no-link-gate recipe')
    from src.task.CmResidual.dexplore_approach import sampled_surface_gap
    if bridge_factory is None:
        from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
        bridge_factory = DExploreCmv2GeometryBridge
    cache = getattr(agent, '_ref2dex_object_approach_bridges', None)
    if cache is None:
        cache = agent._ref2dex_object_approach_bridges = {}
    assets = Path(__file__).resolve().parents[5] / 'third_party/DExplore/dexplore/data/assets'
    object_ids = task.object_id[task.data_id]
    gap = torch.empty(task._target_states.shape[0], device=task._target_states.device)
    for index, name in enumerate(names):
        ids = torch.nonzero(object_ids == index).flatten()
        if not len(ids):
            continue
        if name not in cache:
            cache[name] = bridge_factory(
                hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=assets/('mjcf/'+name+'.urdf'), device=agent.ppo_device, seed=42)
        geometry = cache[name].current(task._dof_pos[ids], task._target_states[ids])
        center = geometry.object_pose[:, None, :3, 3]
        points = center + (geometry.object_points - center) * task.ball_size
        gap[ids] = sampled_surface_gap(geometry.hand_points, points, agent.approach_config)
    return gap


def install_approach_patch():
    """Install after Isaac Gym has loaded Torch, before the actor is built."""
    from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
    original = DExploreApproachAgent._approach_gap
    if getattr(original, '_ref2dex_object_aware', False):
        return

    @torch.inference_mode()
    def corrected(agent, task):
        return approach_gap(agent, task, original)

    corrected._ref2dex_object_aware = True
    DExploreApproachAgent._approach_gap = corrected
