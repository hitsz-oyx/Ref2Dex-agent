# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small upstream XIRL alignment helpers, with explicit provenance.

The two distance functions are copied from xirl/xirl/losses.py at Google
Research commit a1e7371c5e006f4e8b314bd23d99220d2fe44c51. Only the first
row-softmax leg is used: no reverse cycle over the actual episode. The
trailing-context indexing adapts FrameSampler._get_context_steps at that
same commit to an array interface. We supply geometric clips, not trained
XIRL embeddings; causal bounded tracking is implemented separately.
"""
import numpy as np
import torch

UPSTREAM_COMMIT = 'a1e7371c5e006f4e8b314bd23d99220d2fe44c51'


def pairwise_l2_sq(x1, x2):
    return torch.cdist(x1, x2).pow(2)


def get_scaled_similarity(emb1, emb2, similarity_type, temperature,
                          normalize_dimension):
    """Return pairwise similarity (upstream implementation)."""
    if similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_sq(emb1, emb2)
        if normalize_dimension:
            similarity = similarity / emb1.shape[1]
    else:
        similarity = torch.mm(emb1, emb2.t())
    return similarity / temperature


def trailing_context_indices(length, context=8):
    """Adapt XIRL's causal context sampling, repeating frame zero at startup."""
    if length < 1 or context < 2:
        raise ValueError('nonempty sequence and multi-frame history required')
    offsets = np.arange(-(context - 1), 1)
    return np.clip(np.arange(length)[:, None] + offsets[None], 0, length - 1)
