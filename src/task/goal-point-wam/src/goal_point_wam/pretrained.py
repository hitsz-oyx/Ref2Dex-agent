"""Strict transfer of repaired temporal PointWorld scene weights into WAM."""
from pathlib import Path
import hashlib
import torch

from .model import VENDOR


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def transfer_scene(scene, state, config, stats_sha256):
    """Validate every transferable tensor before mutating the destination."""
    identity, source_config = state['identity'], state['config']
    if state.get('checkpoint_kind') != 'pointworld-temporal.ddp.v1':
        raise ValueError('expected repaired temporal PW checkpoint')
    if identity.get('stats_sha256') != stats_sha256:
        raise ValueError('PW normalization hash mismatch')
    for key in ('hidden', 'patch_size', 'fps', 'history', 'horizon', 'voxel_m', 'voxel_origin_m'):
        if source_config.get(key) != config.get(key):
            raise ValueError('PW config mismatch: ' + key)
    if source_config.get('voxel_origin_m') != [-2., -2., -2.] or source_config.get('voxel_m') != .01:
        raise ValueError('PW fixed workspace contract required')
    vendor = identity.get('vendor_sources', {})
    if not vendor or 'ptv3/ptv3.py' not in vendor or 'ptv3/ptv3_arch.yaml' not in vendor:
        raise ValueError('PW vendor identity missing')
    for relative, expected in vendor.items():
        if file_digest(VENDOR / relative) != expected:
            raise ValueError('PW vendor hash mismatch: ' + relative)
    source, target = state['model'], scene.state_dict()
    mapped = {}
    for key in target:
        if key.startswith('backbone.'):
            origin = 'backbone.core.' + key[len('backbone.'):]
        elif key.startswith('scene_proj.'):
            origin = key
        elif key in ('scene_mean', 'scene_std'):
            origin = key
        else:
            continue  # WAM object projection has a different architecture.
        value = source.get(origin)
        if value is None or value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise ValueError('PW tensor incompatible: ' + origin)
        if not torch.isfinite(value).all():
            raise ValueError('PW tensor nonfinite: ' + origin)
        if key in ('scene_mean', 'scene_std') and not torch.equal(value.cpu(), target[key].cpu()):
            raise ValueError('PW normalization buffer mismatch: ' + key)
        mapped[key] = value
    expected_source = {key for key in source if key.startswith(('backbone.core.', 'scene_proj.'))}
    actual_source = {'backbone.core.' + key[len('backbone.'):] if key.startswith('backbone.') else key
                     for key in mapped if key.startswith(('backbone.', 'scene_proj.'))}
    if expected_source != actual_source:
        raise ValueError('PW backbone/projection key coverage mismatch')
    h = hashlib.sha256()
    for key, value in sorted(mapped.items()):
        h.update(key.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    scene.load_state_dict({**target, **mapped}, strict=True)
    parameters = dict(scene.named_parameters())
    return dict(parent_git_commit=identity['git_commit'], parent_step=int(state['step']),
                parent_checkpoint_kind=state['checkpoint_kind'], transferred_tensors=len(mapped),
                transferred_parameters=sum(parameters[key].numel() for key in mapped if key in parameters),
                transferred_sha256=h.hexdigest(), vendor_sources=vendor,
                scope='backbone + scene projection + verified scene normalization; new WAM heads and object projection')


def initialize_pw(model, path, config, stats_sha256):
    state = torch.load(path, map_location='cpu', weights_only=False)
    report = transfer_scene(model.scene, state, config, stats_sha256)
    report.update(parent_checkpoint=str(Path(path).resolve()), parent_sha256=file_digest(path))
    return report
