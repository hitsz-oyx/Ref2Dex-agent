"""Ref12 cross-fitted oracle hand-flow consequence to task prognosis contracts."""
import hashlib
import numpy as np
import torch
from torch import nn
from consequence_sufficiency import PRIMARY, cluster_gain
from conditional_consequence import oracle_retention

TASK_SPECS = {
    'H': (False, None),
    'Flow': (True, None),
    'PredEI': (False, 'Chunk'),
    'GT_EI': (False, 'GT'),
    'Flow_PredEI': (True, 'Chunk'),
    'StatePredEI': (False, 'State'),
}


def validate_folds(train, test, clusters, folds):
    """Exclude outer-test and whole environments from every inner fit."""
    train, test = np.asarray(train), np.asarray(test)
    assert not set(train) & set(test)
    assert not set(clusters[train]) & set(clusters[test])
    assert np.all(folds[test] == -1) and set(folds[train]) == {0, 1, 2}
    records = {}
    covered = []
    for fold in range(3):
        fit = train[folds[train] != fold]
        hold = train[folds[train] == fold]
        assert len(fit) and len(hold)
        assert not set(clusters[fit]) & set(clusters[hold])
        records['fold'+str(fold)] = dict(fit=fit, hold=hold)
        covered.extend(hold.tolist())
    assert sorted(covered) == sorted(train.tolist())
    return records


def assemble_crossfit(train, test, partitions, predictions, full):
    """Only each fold's held rows enter source features; full fits supply test."""
    result = torch.full_like(full, float('nan'))
    coverage = torch.zeros(len(full), dtype=torch.int64, device=full.device)
    for name, ids in partitions.items():
        hold = torch.as_tensor(ids['hold'], device=full.device)
        assert not set(ids['fit']) & set(ids['hold'])
        result[hold] = predictions[name][hold]
        coverage[hold] += 1
    tr = torch.as_tensor(train, device=full.device)
    te = torch.as_tensor(test, device=full.device)
    assert torch.all(coverage[tr] == 1) and torch.all(coverage[te] == 0)
    result[te] = full[te]
    assert torch.isfinite(result[torch.cat((tr, te))]).all()
    return result


def task_input(h, flow, physical, name):
    include_flow, physical_kind = TASK_SPECS[name]
    return torch.cat((h, flow if include_flow else torch.zeros_like(flow),
                      physical[physical_kind] if physical_kind else torch.zeros_like(physical['GT'])), -1)


class TaskHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 32),
                                 nn.Tanh(), nn.Linear(32, 8))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_task(model, x):
    return torch.cat([model(x[ids]) for ids in torch.arange(len(x), device=x.device).split(64)])


def fit_task(x, target, train, updates):
    torch.manual_seed(259)
    model = TaskHead(x.shape[1]).to(x.device)
    initial = hashlib.sha256(torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy().tobytes()).hexdigest()
    generator = torch.Generator(device=x.device).manual_seed(260)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses = []
    for _ in range(updates):
        ids = train[torch.randint(len(train), (64,), generator=generator, device=x.device)]
        loss = (model(x[ids])-target[ids]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite task loss')
        optimizer.zero_grad(); loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 2.)
        if not torch.isfinite(gradient):
            raise ValueError('nonfinite task gradient')
        optimizer.step(); losses.append(float(loss.detach()))
    model.eval(); prediction = evaluate_task(model, x)
    assert torch.isfinite(prediction).all()
    return model, prediction, dict(initial_hash=initial, parameters=sum(p.numel() for p in model.parameters()),
                                  updates=updates, losses=losses,
                                  source_all8_mse=float((prediction[train]-target[train]).square().mean()))


def task_summary(y, scale, predictions, train, test, clusters):
    errors = {name: ((raw-y)/scale)**2 for name, raw in predictions.items()}
    metrics = {name: dict(primary_mse=float(err[test][:, PRIMARY].mean()),
                         all8_mse=float(err[test].mean()),
                         physical_failure_mse=float(err[test, 6].mean()),
                         per_head_mse=err[test].mean(0).tolist(),
                         train_primary_mse=float(err[train][:, PRIMARY].mean()))
               for name, err in errors.items()}
    comparisons = {}
    pairs = [(name, 'H') for name in ('Flow', 'PredEI', 'GT_EI', 'Flow_PredEI', 'StatePredEI')]
    pairs += [('PredEI', 'Flow'), ('PredEI', 'StatePredEI'), ('Flow_PredEI', 'Flow'),
              ('Flow_PredEI', 'PredEI'), ('Flow', 'Flow_test_shuffled'),
              ('PredEI', 'PredEI_test_shuffled'), ('Flow_PredEI', 'Flow_PredEI_test_shuffled')]
    for new, base in pairs:
        comparisons[new+'_vs_'+base] = {
            key: cluster_gain(errors[base][test][:, axes].mean(1), errors[new][test][:, axes].mean(1), clusters[test], 261)
            for key, axes in (('primary', PRIMARY), ('physical', (6,)), ('all8', tuple(range(8))))}
    retention = oracle_retention(errors['H'][test][:, PRIMARY].mean(1),
                                errors['GT_EI'][test][:, PRIMARY].mean(1),
                                errors['PredEI'][test][:, PRIMARY].mean(1), clusters[test], 261)
    def positive(key, minimum):
        item = comparisons[key]['primary']
        return item['gain'] >= minimum and item['lower95'] > 0
    # A valid task oracle denominator is required before interpreting R.
    r = retention
    gates = dict(gt_task_information=positive('GT_EI_vs_H', .10),
                 predicted_task_information=positive('PredEI_vs_H', .05),
                 flow_mediated_increment=positive('PredEI_vs_StatePredEI', .03),
                 oracle_gain_retention=r['R'] is not None and r['R'] >= .5 and r['lower95'] is not None
                                      and r['lower95'] > 0 and r['valid_bootstrap_fraction'] >= .95,
                 predicted_feature_sensitivity=positive('PredEI_vs_PredEI_test_shuffled', .03))
    unique = positive('PredEI_vs_Flow', .03) or positive('Flow_PredEI_vs_Flow', .03)
    return dict(status='PROMISING' if all(gates.values()) else 'UNPROMISING', gates=gates,
                unique_added_value='PROMISING' if unique else 'UNCLEAR', metrics=metrics,
                comparisons=comparisons, oracle_retention=r,
                limits='Fixed-fit post-treatment oracle prognosis; no candidate ranking, exogenous flow intervention or policy utility.')
