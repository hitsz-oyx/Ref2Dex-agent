import math

import torch

from src.task.CmResidual.tools.probe_cm_weight_components import auroc, pearson


def test_auroc_perfect_reversed_and_ties():
    label = torch.tensor([0, 0, 1, 1]).bool()
    assert auroc(torch.tensor([0., 1., 2., 3.]), label) == 1.0
    assert auroc(torch.tensor([3., 2., 1., 0.]), label) == 0.0
    assert auroc(torch.zeros(4), label) == 0.5


def test_auroc_undefined_and_pearson():
    assert math.isnan(auroc(torch.ones(3), torch.ones(3).bool()))
    assert pearson(torch.tensor([1., 2., 3.]), torch.tensor([2., 4., 6.])) > 0.999
