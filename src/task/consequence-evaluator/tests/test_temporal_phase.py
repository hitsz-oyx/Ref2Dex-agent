"""Phase-only TCC loss and causal encoder checks; no evaluator fit."""
from pathlib import Path
import sys
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.temporal_phase import TemporalPhaseEncoder, context_clips
from consequence_evaluator.xirl_tcc_loss import compute_tcc_loss


def test_encoder_cannot_observe_later_geometry():
    torch.manual_seed(17)
    model = TemporalPhaseEncoder().eval()
    features = np.random.default_rng(17).normal(size=(80, 90)).astype('float32')
    changed = features.copy(); changed[41:] += 100
    with torch.inference_mode():
        before = model(torch.as_tensor(context_clips(features)))
        after = model(torch.as_tensor(context_clips(changed)))
        truncated = model(torch.as_tensor(context_clips(features[:41])))
    assert torch.allclose(before[:41], after[:41], atol=1e-7)
    assert torch.allclose(before[:41], truncated, atol=1e-7)
    assert before.shape == (80, 64)
    assert torch.allclose(before.norm(dim=1), torch.ones(80), atol=1e-6)


def test_upstream_tcc_cycle_loss_has_finite_nonzero_encoder_gradient():
    torch.manual_seed(17)
    model = TemporalPhaseEncoder()
    clips = torch.randn(2, 32, 720)
    embeddings = model(clips.flatten(0, 1)).reshape(2, 32, 64)
    indices = torch.arange(32)[None].expand(2, -1)
    loss = compute_tcc_loss(embeddings, indices, torch.tensor([32, 32]),
        stochastic_matching=False, normalize_embeddings=True,
        loss_type='regression_mse', temperature=.1, normalize_indices=True)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
