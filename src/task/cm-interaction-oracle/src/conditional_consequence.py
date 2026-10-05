"""Action-conditioned consequence prediction: OOF and contrast contracts."""
import numpy as np
import torch


def environment_folds(train, motion, clusters, seed=237, count=3):
    """Every outer-train environment belongs to one fold; test stays -1."""
    folds = np.full(len(motion), -1, dtype=int)
    rng = np.random.default_rng(seed)
    for m in np.unique(motion[train]):
        envs = np.unique(clusters[train[motion[train] == m]])
        rng.shuffle(envs)
        for index, env in enumerate(envs):
            folds[train[clusters[train] == env]] = index % count
    return folds


def permute_within(groups, ids, seed):
    """Permutation never crosses the requested fit/OOF/test partition."""
    perm = np.arange(len(groups))
    rng = np.random.default_rng(seed)
    for group in np.unique(groups[ids]):
        rows = ids[groups[ids] == group]
        perm[rows] = rng.permutation(rows)
    assert set(perm[ids]) == set(ids)
    return perm


def task_slots(h, action, physical, use_action=False):
    """Keep ref8's 162D scorer width; signed sensitivity slot stays zero."""
    return torch.cat((h, action if use_action else torch.zeros_like(action),
                      physical, torch.zeros(len(h), 18, device=h.device, dtype=h.dtype)), -1)


def contrast_scores(gt, predicted, scale, axes):
    """Signed arm-vector agreement, independent of mean-reconstruction MSE."""
    g, p = gt[:, axes] / scale[axes], predicted[:, axes] / scale[axes]
    g, p = g.flatten(), p.flatten()
    energy = float(np.mean(g*g))
    variable = (g.std() > 1e-10 and p.std() > 1e-10)
    active = np.abs(g) >= .1
    return dict(normalized_mse=float(np.mean((p-g)**2)),
        gain_vs_zero=float(1-np.mean((p-g)**2)/max(energy,1e-12)),
        correlation=float(np.corrcoef(g,p)[0,1]) if variable else None,
        signed_agreement=float(np.mean(g[active]*p[active]>0)) if active.any() else None,
        signed_axes=int(active.sum()), amplitude_ratio=float(np.sqrt(np.mean(p*p)/max(energy,1e-12))),
        gt_rms=float(np.sqrt(energy)), dimensions=len(g))


def oracle_retention(errors_h, errors_gt, errors_pred, clusters, seed=238):
    """R on current matched optimization budget; undefined for weak oracle gain."""
    unique = np.unique(clusters)
    values = np.array([[errors_h[clusters==c].sum(),errors_gt[clusters==c].sum(),
                        errors_pred[clusters==c].sum()] for c in unique])
    point = (errors_h.mean()-errors_pred.mean()) / (errors_h.mean()-errors_gt.mean()) if errors_h.mean()>errors_gt.mean() else None
    samples = values[np.random.default_rng(seed).integers(len(unique),size=(2000,len(unique)))].sum(1)
    denominator = samples[:,0]-samples[:,1]
    valid = denominator>1e-10
    ratios = (samples[valid,0]-samples[valid,2])/denominator[valid]
    return dict(R=float(point) if point is not None else None,
        lower95=float(np.quantile(ratios,.025)) if len(ratios) else None,
        upper95=float(np.quantile(ratios,.975)) if len(ratios) else None,
        valid_bootstrap_fraction=float(valid.mean()),
        limits='Paired fixed-fit cluster ratios; uncertain denominator may make R unstable')
