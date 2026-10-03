"""Frozen surface features with a common causal, persistence-residual head."""
import numpy as np
import torch
from torch import nn
from src.task.CmResidual.surface_motion_prior import SurfaceMotionPrior, FLOW_COLUMNS, encode_raw
from src.task.CmResidual.surface_execution import predict_execution
from src.task.CmResidual.v118_planner import TorchInspireKinematics
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose

ARMS = ('pretrained', 'shuffled', 'scratch', 'hand_flow_removed')
STEPS = 1200
HEAD_SEED = 4101
SCHEDULE_SEED = 4102


class PersistenceResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, encoded, persistence):
        return persistence + self.layers(encoded)


def encoder_states(prior):
    good = torch.load(prior/'fit/mano_7168.pt', map_location='cpu', weights_only=False)
    shuffled = torch.load(prior/'fit/mano_shuffled.pt', map_location='cpu', weights_only=False)
    assert good['steps'] == shuffled['steps'] == 1500
    assert good['size'] == shuffled['size'] == 7168
    assert not good['shuffled'] and shuffled['shuffled']
    assert torch.equal(good['batch_schedule'], shuffled['batch_schedule'])
    assert all(torch.equal(v, shuffled['common_initial'][k]) for k,v in good['common_initial'].items())
    sources = dict(pretrained=good['state'], shuffled=shuffled['state'],
                   scratch=good['common_initial'], hand_flow_removed=good['state'])
    return {name:{k[len('encoder.'):]:v.clone() for k,v in state.items() if k.startswith('encoder.')}
            for name,state in sources.items()}


def load_encoder(state, device):
    encoder = SurfaceMotionPrior().encoder.to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_frozen(encoder, features, removed=False):
    x = features.clone() if removed else features
    if removed:
        x[..., list(FLOW_COLUMNS)] = 0
    local = encoder(x)
    return torch.cat((local, local.mean(1, keepdim=True).expand_as(local)), -1)


def gates_and_label(reports, persistence):
    e = reports['pretrained']['parent_epe_mm']
    gates = dict(persistence=e <= .9*persistence,
                 scratch=e <= .9*reports['scratch']['parent_epe_mm'],
                 shuffled=e <= .95*reports['shuffled']['parent_epe_mm'],
                 hand_flow=e <= .95*reports['hand_flow_removed']['parent_epe_mm'])
    positive = all(gates.values())
    evidence = gates['persistence'] and (gates['scratch'] or gates['shuffled'] or gates['hand_flow'])
    return gates, 'PROMISING' if positive else ('UNCLEAR' if evidence else 'UNPROMISING')


@torch.no_grad()
def causal_features(rows, geometry, fitted, lower, upper, urdf, device, progress=None):
    """The next object pose supplies labels only; hand input uses train-fit actuator."""
    predicted = predict_execution(rows, fitted, lower, upper)['action_velocity']
    fk = TorchInspireKinematics(urdf, device)
    local = torch.from_numpy(geometry['points']).to(device)
    normals = torch.from_numpy(geometry['normals']).to(device)
    links = torch.from_numpy(geometry['links']).to(device)
    object_local = torch.from_numpy(geometry['object_local']).to(device)
    object_normal = torch.from_numpy(geometry['object_normal']).to(device)
    features, targets, indices = [], [], []
    for start in range(0, len(rows['env']), 16):
        stop = min(start+16, len(rows['env']))
        subset = {k:v[start:stop] for k,v in rows.items()}
        base = dexplore_root_pose(torch.from_numpy(subset['hand_root']).to(device))
        def surface(q):
            transforms = base[:,None] @ fk.forward(q[:,None])[:,0]
            rotation, translation = transforms[:,links,:3,:3], transforms[:,links,:3,3]
            points = (rotation @ local[None,:,:,None]).squeeze(-1)+translation
            n = (rotation @ normals[None,:,:,None]).squeeze(-1)
            return points, n
        hand, hn = surface(torch.from_numpy(subset['q']).to(device))
        next_hand, _ = surface(torch.from_numpy(predicted[start:stop]).to(device))
        poses = {k:dexplore_root_pose(torch.from_numpy(subset[k]).to(device))
                 for k in ('current_obj', 'previous_obj', 'next_obj')}
        objects = {k:object_local[None] @ po[:,:3,:3].transpose(-1,-2)+po[:,None,:3,3]
                   for k,po in poses.items()}
        on = object_normal[None] @ poses['current_obj'][:,:3,:3].transpose(-1,-2)
        distance = torch.cdist(objects['current_obj'], hand, compute_mode='donot_use_mm_for_euclid_dist')
        knn = distance.topk(4, largest=False, sorted=True).indices
        batch = torch.arange(stop-start, device=device)[:,None,None]
        raw = dict(pose=poses['current_obj'].cpu().numpy(), obj=objects['current_obj'].cpu().numpy(),
                   normal=on.cpu().numpy(), previous_obj=objects['previous_obj'].cpu().numpy(),
                   next_obj=objects['next_obj'].cpu().numpy(), hand=hand[batch,knn].cpu().numpy(),
                   hand_normal=hn[batch,knn].cpu().numpy(), next_hand=next_hand[batch,knn].cpu().numpy(),
                   global_hand_flow=(next_hand-hand).mean(1).cpu().numpy())
        x,y = encode_raw(raw)
        features.append(x); targets.append(y); indices.append(knn.cpu().numpy())
        if progress is not None and (start//16+1)%64 == 0:
            progress(stop, len(rows['env']))
    return dict(features=np.concatenate(features), target=np.concatenate(targets),
                knn_indices=np.concatenate(indices), predicted_q=predicted)
