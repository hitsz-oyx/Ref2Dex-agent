"""Ref14_3 immutable panel joins, nested anchor OOF and matched readouts."""
import numpy as np
import torch
from torch import nn
from oracle_hand_flow import current_state
from geometric_consequence import standardize_fit, normalize
from ranking_tolerance import same_state_ranking, assert_environment_oof

ARMS = ('H', 'Direct', 'Bottleneck', 'Hybrid', 'GT_EI')


def join_panels(packets, dense):
    groups = {(g['seed'], g['group']): g for g in dense['groups']}
    before, history, geometry, flow, ei, y, keys = [], [], [], [], [], [], []
    for packet in packets:
        dg = groups[(packet['seed'], packet['group'])]
        if not torch.equal(packet['rows'], dg['rows']):
            raise ValueError('dense/asset anchor identity mismatch')
        rounds = {r['offset']: r for r in dg['rounds']}
        for rd in packet['rounds']:
            dr = rounds[rd['offset']]
            if not torch.equal(rd['candidate_observed'], dr['candidate_observed']):
                raise ValueError('dense/asset observed-mask mismatch')
            if not rd['candidate_observed'].all():
                continue
            reference = rd['candidates'][0]['H']
            for k, record in enumerate(rd['candidates']):
                derived = dr['candidates'][k]
                expected = dict(seed=packet['seed'], group=packet['group'], offset=rd['offset'], candidate=k)
                if derived['join'] != expected or record['candidate'] != k:
                    raise ValueError('dense candidate join mismatch')
                # Use the common baseline H: solver replay roundoff is not an action input.
                for name in ('before', 'history', 'hand_root'):
                    if not torch.allclose(record['H'][name], reference[name], atol=1e-4, rtol=0):
                        raise ValueError('same-current H mismatch')
            n = len(packet['rows'])
            before.append(reference['before'][:, None].expand(-1, 7, -1).reshape(n*7, -1))
            history.append(reference['history'][:, None].expand(-1, 7, -1, -1).reshape(n*7, *reference['history'].shape[1:]))
            points = dr['candidates'][0]['points_current'].flatten(1)/.1
            geometry.append(points[:, None].expand(-1, 7, -1).reshape(n*7, -1))
            flow.append(torch.stack([c['flow_normalized_002m'].flatten(1) for c in dr['candidates']], 1).reshape(n*7, 720))
            ei.append(torch.stack([torch.cat((c['E8'], c['I8']), -1) for c in rd['candidates']], 1).reshape(n*7, 26))
            y.append(torch.stack([c['Y32'] for c in rd['candidates']], 1).reshape(n*7, 8))
            keys.extend([(packet['seed'], int(row), packet['group'], rd['offset']) for row in packet['rows']])
    if not keys:
        raise ValueError('no complete panels')
    result = dict(before=torch.cat(before), history=torch.cat(history), geometry=torch.cat(geometry),
                  flow=torch.cat(flow), ei=torch.cat(ei), y=torch.cat(y), panel_keys=keys)
    for name in ('before', 'history', 'geometry', 'flow', 'ei', 'y'):
        if not torch.isfinite(result[name]).all():
            raise ValueError('nonfinite joined '+name)
    return result


def anchor_folds(keys, count, seed):
    anchors = sorted(set(tuple(k[:2]) for k in keys))
    np.random.default_rng(seed).shuffle(anchors)
    table = {k: i % count for i, k in enumerate(anchors)}
    return np.array([table[tuple(k[:2])] for k in keys])


def rows_for(panels, device):
    return torch.as_tensor((np.asarray(panels)[:, None]*7+np.arange(7)).reshape(-1), device=device)


def partition(keys, fit_panels, hold_panels):
    assert_environment_oof([keys[i][:2] for i in fit_panels], [keys[i][:2] for i in hold_panels])
    return dict(fit_panels=np.asarray(fit_panels), hold_panels=np.asarray(hold_panels))


def h_inputs(data, ids, norms=None):
    return current_state(data, data['geometry'], ids, norms)


class Head(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, outputs))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, x):
    return torch.cat([model(chunk) for chunk in x.split(256)])


def fit(x, target, ids, updates):
    torch.manual_seed(273)
    model = Head(x.shape[1], target.shape[1]).to(x.device)
    gen = torch.Generator(device=x.device).manual_seed(274)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses = []
    for _ in range(updates):
        batch = ids[torch.randint(len(ids), (64,), generator=gen, device=x.device)]
        loss = (model(x[batch])-target[batch]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        optim.zero_grad(); loss.backward()
        if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 2.)):
            raise ValueError('nonfinite gradient')
        optim.step(); losses.append(float(loss.detach()))
    model.eval()
    return model, dict(parameters=sum(p.numel() for p in model.parameters()), updates=updates,
                       initial_loss=losses[0], final_loss=losses[-1])


def physical_fit(data, ids, updates):
    h, norms = h_inputs(data, ids)
    target_norm = standardize_fit(data['ei'], ids)
    model, stats = fit(torch.cat((h, data['flow']), -1), normalize(data['ei'], target_norm), ids, updates)
    with torch.no_grad():
        out = predict(model, torch.cat((h, data['flow']), -1))*target_norm['scale']+target_norm['mean']
    return out, dict(h_norms=norms, target_norm=target_norm, weights=model.state_dict(), stats=stats)


def readout_input(h, flow, physical, arm):
    return torch.cat((h, flow if arm in ('Direct', 'Hybrid') else torch.zeros_like(flow),
                      physical if arm in ('Bottleneck', 'Hybrid', 'GT_EI') else torch.zeros_like(physical)), -1)


def utility(y):
    return y[..., 7]+.25*y[..., 3]-y[..., 6]


def ranking(y, prediction):
    truth = utility(y).reshape(-1, 7).cpu().numpy()
    scores = utility(prediction).reshape(-1, 7).cpu().numpy()
    value = same_state_ranking(truth, scores, np.ones_like(truth, dtype=bool))
    value['median_top1_regret'] = float(np.median(value['top1_regret']))
    value['all8_mse'] = float((y-prediction).square().mean())
    value['per_channel_mse'] = (y-prediction).square().mean(0).cpu().tolist()
    return value


def bootstrap_difference(y, a, b, keys):
    """Anchor-clustered strict pair credits, pooled ratio under each draw."""
    anchors = sorted(set(tuple(k[:2]) for k in keys))
    truth, pa, pb = [utility(v).reshape(-1, 7).cpu().numpy() for v in (y, a, b)]
    totals = np.zeros((len(anchors), 3))
    index = {k:i for i,k in enumerate(anchors)}
    for gt, x, z, key in zip(truth, pa, pb, keys):
        at = index[tuple(key[:2])]
        for i in range(7):
            for j in range(i+1, 7):
                delta = gt[i]-gt[j]
                if delta == 0:
                    continue
                totals[at, 0] += 1
                totals[at, 1] += .5 if x[i] == x[j] else float(delta*(x[i]-x[j]) > 0)
                totals[at, 2] += .5 if z[i] == z[j] else float(delta*(z[i]-z[j]) > 0)
    rng = np.random.default_rng(271)
    draws = totals[rng.integers(len(anchors), size=(2000, len(anchors)))].sum(1)
    valid = draws[:, 0] > 0
    values = (draws[valid, 1]-draws[valid, 2])/draws[valid, 0]
    point = totals.sum(0)
    return dict(difference=float((point[1]-point[2])/point[0]) if point[0] else None,
                lower95=float(np.quantile(values, .025)) if len(values) else None,
                upper95=float(np.quantile(values, .975)) if len(values) else None,
                anchors=len(anchors), strict_pair_anchors=int((totals[:,0]>0).sum()),
                valid_bootstrap_fraction=float(valid.mean()))
