"""Small TCC alignment encoder; never a consequence value/evaluator head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .reference_progress import AlignmentConfig, FeatureScale, track_cost
from .xirl_alignment import get_scaled_similarity, trailing_context_indices

ENCODER_SCHEMA = 'ref2dex.consequence-reference-progress.tcc-encoder.v1'


class TemporalPhaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8 * 90, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, clips):
        if clips.ndim != 2 or clips.shape[1] != 720:
            raise ValueError('eight standardized90D history frames required')
        return F.normalize(self.network(clips), dim=-1)


def context_clips(standardized_features):
    indices = trailing_context_indices(len(standardized_features), 8)
    return standardized_features[indices].reshape(len(indices), 720)


class LearnedReferenceProgress:
    """Freeze encoder and references; bank weights never depend on actual future."""
    def __init__(self, reference_features, bundle, device):
        if bundle.get('schema') != ENCODER_SCHEMA:
            raise ValueError('alignment encoder checkpoint required, not evaluator weights')
        self.device = device
        self.config = AlignmentConfig(temperature=.1)
        references = reference_features if isinstance(reference_features, list) else [reference_features]
        if len({len(r) for r in references}) != 1:
            raise ValueError('this bank Probe requires equal full reference lengths')
        self.normalizer = FeatureScale(references)
        if self.normalizer.dictionary() != bundle['standardizer']:
            raise ValueError('encoder must use this exact frozen reference standardizer')
        self.encoder = TemporalPhaseEncoder().to(device)
        self.encoder.load_state_dict(bundle['model'], strict=True)
        self.encoder.eval()
        self.references = [self.embed(r) for r in references]
        self.reference = self.references[0]

    @torch.inference_mode()
    def embed(self, features):
        clips = context_clips(self.normalizer.apply(features))
        # Fixed-size batches keep CUDA GEMM rounding identical when an actual
        # prefix is truncated. Padding never enters the retained rows and this
        # encoder has no batch statistics or cross-row attention.
        results = []
        for start in range(0, len(clips), 64):
            block = clips[start:start + 64]
            padded = np.zeros((64, 720), np.float32)
            padded[:len(block)] = block
            encoded = self.encoder(torch.as_tensor(padded, device=self.device))
            results.append(encoded[:len(block)])
        return torch.cat(results, dim=0)

    @torch.inference_mode()
    def align(self, features):
        # Each embedding already contains8causal frames. No additional context
        # sampling, actual-frame index, reverse cycle or actual future at inference.
        encoded = self.embed(features)
        traces = []
        for reference in self.references:
            cost = -get_scaled_similarity(encoded.double(), reference.double(), 'l2', 1., False)
            traces.append(track_cost(cost.cpu().numpy(), self.config))
        if len(traces) == 1:
            return traces[0]
        # Each reference has its own causal prior. Equal weights are fixed
        # before evaluating any actual trajectory or candidate suffix.
        q = np.mean([t['distribution'] for t in traces], axis=0)
        index = np.arange(q.shape[1], dtype=np.float64)
        progress = q @ (index / (len(index) - 1))
        return dict(progress=progress, distribution=q,
            member_progress=np.stack([t['progress'] for t in traces], axis=1),
            index_std=np.sqrt(np.sum(q * (index[None] - progress[:, None] * (len(index) - 1)) ** 2, axis=1)),
            matched_cost=np.mean([t['matched_cost'] for t in traces], axis=0),
            unconstrained_index=np.mean([t['unconstrained_index'] for t in traces], axis=0))
