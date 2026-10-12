"""Strict optional PTv3 final-decoder patch; never admit arbitrary scene weights."""
import hashlib

import torch


SCENE_PATCH_PREFIX = 'backbone.dec.dec0.'
SCENE_PATCH_SCHEMA = 'ref2dex.wam-scene-dec0.v1'


def _expected(scene):
    values = {k: v for k, v in scene.state_dict().items() if k.startswith(SCENE_PATCH_PREFIX)}
    if not values:
        raise ValueError('original PTv3 final decoder dec0 is missing')
    return values


def extract_scene_patch(scene):
    """Copy only the explicit final stage, with scene-relative full state keys."""
    return {k: v.detach().cpu().clone() for k, v in _expected(scene).items()}


def load_scene_patch(scene, patch):
    """Validate the complete whitelist before any mutation, including buffers."""
    expected = _expected(scene)
    if not isinstance(patch, dict) or set(patch) != set(expected):
        raise ValueError('scene patch must contain exactly the dec0 state keys')
    for key, target in expected.items():
        value = patch[key]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or value.layout != target.layout):
            raise ValueError('scene patch shape/dtype/layout differs: ' + key)
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite scene patch: ' + key)
    state = scene.state_dict()
    state.update(patch)
    scene.load_state_dict(state, strict=True)


def scene_remainder_hash(scene):
    """Authenticate all frozen scene parameters AND buffers outside dec0."""
    _expected(scene)
    digest = hashlib.sha256()
    for key, value in sorted(scene.state_dict().items()):
        if key.startswith(SCENE_PATCH_PREFIX):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
