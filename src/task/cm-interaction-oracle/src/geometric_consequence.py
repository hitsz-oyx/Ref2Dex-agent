"""Decision-time nominal surface action and regularized innovation models.

FK uses the measured actor root. PD endpoints are intended geometry, not an
eight-step execution forecast. No post-assignment state enters these features.
"""
import numpy as np
import torch

from intervention import physical_targets
from src.task.CmResidual.dexplore_cm_geometry import (
    dexplore_action_to_native_targets, dexplore_root_pose, native_joint_limits,
)
from src.task.CmResidual.surface_execution import area_hand_samples
from src.task.CmResidual.v118_planner import TorchInspireKinematics, QUERY_LINKS

ACTION_WIDTH = 32
STATE_GEOMETRY_WIDTH = 16
INTERACTION_STATE_WIDTH = 16
RIDGE = 32.


def physics_baseline(before, dt):
    """Constant world velocity/omega for E; measured current interaction for I."""
    result = physical_targets(before, before).clone()
    result[:, :3] = before[:, 7:10] * dt
    rotation = before[:, 10:13] * dt
    angle = rotation.norm(dim=-1, keepdim=True)
    short_angle = torch.remainder(angle + torch.pi, 2*torch.pi) - torch.pi
    result[:, 3:6] = rotation * short_angle / angle.clamp_min(1e-8)
    return result


def persistence_baseline(before):
    return physical_targets(before, before)


def standardize_fit(x, ids, floor=1e-3):
    mean = x[ids].mean(0)
    scale = x[ids].std(0, unbiased=False).clamp_min(floor)
    return dict(mean=mean, scale=scale)


def normalize(x, norm):
    return (x - norm['mean']) / norm['scale']


def pca_fit(x, ids, width):
    # Coordinates of geometry/flow share metres; avoid whitening near-zero
    # point axes before PCA. Final scores are train-only standardized.
    mean = x[ids].mean(0)
    _, _, v = torch.linalg.svd(x[ids] - mean, full_matrices=False)
    projection = v[:width].T
    scores = (x - mean) @ projection
    return dict(mean=mean, projection=projection, score=standardize_fit(scores, ids))


def pca_apply(x, norm):
    return normalize((x - norm['mean']) @ norm['projection'], norm['score'])


def pad(x, width=ACTION_WIDTH):
    if x.shape[-1] > width:
        raise ValueError('feature wider than matched action slots')
    return torch.nn.functional.pad(x, (0, width-x.shape[-1]))


class NominalSurfaceActions:
    def __init__(self, urdf, device, seed=241):
        self.device = torch.device(device)
        self.kinematics = TorchInspireKinematics(urdf, device)
        self.lower, self.upper = native_joint_limits(urdf, device)
        # Sample by area, then deterministically retain 24 points per finger.
        # Equal per-finger support avoids losing small thumb regions by area.
        samples = area_hand_samples(urdf, count=4096, seed=seed)
        selected = []
        names = ('index', 'middle', 'pinky', 'ring', 'thumb')
        for prefix in names:
            ids = np.array([i for i, link in enumerate(samples['links'])
                            if QUERY_LINKS[link].startswith(prefix)])
            if len(ids) < 24:
                raise ValueError('insufficient sampled finger surface')
            selected.extend(ids[np.linspace(0, len(ids)-1, 24).round().astype(int)])
        self.samples = {key: torch.as_tensor(samples[key][selected], device=device)
                        for key in ('points', 'normals', 'links')}
        self.seed = seed

    def points(self, links):
        selected = links[:, :, self.samples['links']]
        points = torch.einsum('bknij,nj->bkni', selected[..., :3, :3], self.samples['points'])
        points = points + selected[..., :3, 3]
        normals = torch.einsum('bknij,nj->bkni', selected[..., :3, :3], self.samples['normals'])
        return points, normals

    @torch.no_grad()
    def build(self, p):
        # Collector compact state starts with native q18 + dq18.
        q = p['history'][:, -1, :18].to(self.device)
        base = p['base_action'].to(self.device)
        delta = p['delta'].to(self.device)
        roots = dexplore_root_pose(p['hand_root'].to(self.device))
        object_pose = dexplore_root_pose(p['before'][:, :13].to(self.device))
        candidates = base[:, None] + delta[None]
        assert ((candidates >= -1) & (candidates <= 1)).all()
        current = self.kinematics.forward(q[:, None])
        targets = dexplore_action_to_native_targets(
            candidates.flatten(0, 1), q[:, None].expand(-1, 15, -1).flatten(0, 1),
            self.lower, self.upper).reshape(-1, 15, 18)
        nominal = self.kinematics.forward(targets)
        current_world = roots[:, None, None] @ current
        nominal_world = roots[:, None, None] @ nominal
        points, normals = self.points(current_world)
        endpoints, _ = self.points(nominal_world)
        rotation = object_pose[:, :3, :3]
        flow_world = endpoints - points
        flow = torch.einsum('bknj,bji->bkni', flow_world, rotation)
        normal_obj = torch.einsum('bknj,bji->bkni', normals, rotation)
        point_obj = torch.einsum('bknj,bji->bkni', points-object_pose[:, None, None, :3, 3], rotation)
        # Baseline nominal geometry is supplied to every learner as state.
        geometry = torch.cat((point_obj[:, 0].flatten(1), normal_obj[:, 0].flatten(1), flow[:, 0].flatten(1)), -1)
        incremental = flow - flow[:, :1]
        normal_component = (incremental * normal_obj).sum(-1)
        # Normal component also has units metres, and preserves inward/outward.
        action_flow = torch.cat((incremental.flatten(2), normal_component), -1)
        joint = targets - targets[:, :1]
        tip_ids = [QUERY_LINKS.index(name+'_tip') for name in ('index', 'middle', 'pinky', 'ring', 'thumb')]
        tips = current_world[:, 0, tip_ids, :3, 3]
        tip_error = (tips-p['before_fingertip_positions'].to(self.device)).norm(dim=-1)
        base_pose = dexplore_root_pose(torch.nn.functional.pad(p['before_hand_base_pose'].to(self.device), (0, 6)))
        fk_base = current_world[:, 0, 0]
        assert tip_error.max() < 1e-4, 'FK/live current fingertips mismatch'
        assert (base_pose-fk_base).abs().max() < 1e-4, 'FK/live handbase mismatch'
        assert action_flow[:, 0].abs().max() == 0 and joint[:, 0].abs().max() == 0
        return dict(geometry=geometry, flow=action_flow, joint=joint,
            points=point_obj[:, 0], normals=normal_obj[:, 0], nominal_flow=flow,
            incremental_flow=incremental, targets=targets,
            fingertip_error_max_m=float(tip_error.max()),
            handbase_matrix_error_max=float((base_pose-fk_base).abs().max()), samples=self.samples)


def action_design(h, action):
    interaction = (h[:, :INTERACTION_STATE_WIDTH, None] * action[:, None]).flatten(1)
    return torch.cat((h, action, interaction), -1)


def ridge_fit(x, y, ids, alpha=RIDGE):
    # Mean-centered solve leaves intercept unpenalized; float64 GPU statistics.
    x, y = x.double(), y.double()
    xm, ym = x[ids].mean(0), y[ids].mean(0)
    centered, target = x[ids]-xm, y[ids]-ym
    gram = centered.T @ centered
    weight = torch.linalg.solve(gram + alpha*torch.eye(x.shape[1], device=x.device, dtype=x.dtype), centered.T @ target)
    return dict(weight=weight, x_mean=xm, y_mean=ym, alpha=alpha)


def ridge_predict(x, model):
    return ((x.double()-model['x_mean']) @ model['weight'] + model['y_mean']).float()
