"""Authenticate evolved source code without rewriting the original calibration."""
import ast
import hashlib
from pathlib import Path
import subprocess


def verify_default_R_compatibility(old_blob, current_path, source, base_R, device):
    """GPU differential replay of legacy default K24 in the ACTUAL runtime."""
    import numpy as np
    import torch
    from . import pd_retargeter as current
    namespace = {'__name__': 'trajectory_policy._legacy_calibrated_R', '__package__': 'trajectory_policy'}
    exec(compile(old_blob, '<authenticated-legacy-R>', 'exec'), namespace)
    old_tree = ast.parse(old_blob)
    new_tree = ast.parse(Path(current_path).read_bytes())
    def unchanged(tree):
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if not (isinstance(node, ast.ClassDef) and node.name == 'PDRetargeter')
                and not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'load_model')]
    if unchanged(old_tree) != unchanged(new_tree):
        raise ValueError('PD conversion/features/constants changed since calibration')
    packet = torch.load(base_R, map_location='cpu', weights_only=False)
    old_model = namespace['load_model'](packet, device)
    new_model = current.load_model(packet, device)
    if new_model.execution_horizon != 24:
        raise ValueError('legacy default execution horizon changed')
    if any(not torch.equal(v, old_model.state_dict()[k]) for k, v in new_model.state_dict().items()):
        raise ValueError('legacy loaded weights/normalization differ')
    with np.load(source/'trajectory.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets')}
    state, tau, _, _ = current.windows(data, [24])
    state, tau = [torch.as_tensor(v, device=device) for v in (state, tau)]
    with torch.no_grad():
        old_pd = old_model(state, tau)
        new_pd = new_model(state, tau)
    if not torch.equal(old_pd, new_pd):
        raise ValueError('legacy default K24 output changed on measured calibration windows')
    return dict(pass_=True, runtime_torch=torch.__version__, device=str(device),
                windows=len(state), encoded_PD_max_difference=float((old_pd-new_pd).abs().max()),
                unchanged_PD_conversion_AST=True, old_source_sha256=hashlib.sha256(old_blob).hexdigest(),
                current_source_sha256=hashlib.sha256(Path(current_path).read_bytes()).hexdigest(),
                scope='legacy default K24 code compatibility only; new R must earn fresh physical qualification')


def authenticate_calibration_inputs(manifest, root, task, compatibility):
    """Only two named code evolutions are allowed; all other inputs stay exact."""
    updates = []
    r_path = task/'src/trajectory_policy/pd_retargeter.py'
    for name, digest in manifest['input_sha256'].items():
        path = Path(name)
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current == digest:
            continue
        if path not in (task/'tools/run/collect_competent_pd.py', r_path):
            raise ValueError('calibrated source native/data/config drift: '+name)
        relative = path.relative_to(root).as_posix()
        blob = subprocess.check_output(['git', 'show', manifest['git_commit']+':'+relative], cwd=root)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('source Git blob differs from recorded identity')
        proof = None
        if path == r_path:
            proof = compatibility(blob, path)
            if (proof.get('pass_') is not True or proof.get('old_source_sha256') != digest
                    or proof.get('current_source_sha256') != current):
                raise ValueError('default K24 compatibility evidence required')
        updates.append(dict(path=name, producer_sha256=digest, consumer_sha256=current,
                            authenticated_git_commit=manifest['git_commit'], compatibility=proof))
    return updates
