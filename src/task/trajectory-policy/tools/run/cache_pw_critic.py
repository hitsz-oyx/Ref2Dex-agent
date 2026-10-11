"""Frozen PW(H, proposed trajectory) features and actual physical return labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
PW = ROOT / 'src/task/cm-pointflow-effect-pretrain'
OLD = ROOT / 'src/task/consequence-evaluator'
sys.path[:0] = [str(TASK / 'src'), str(OLD / 'src'), str(PW / 'src')]
from trajectory_policy.pw_critic import SCHEMA, PW_SHA, local_inputs, rewards, truncated_returns


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    out, collection = a.output.resolve(), a.collection.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned cache required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch = ROOT / 'tmp/trajectory-policy/pw-critic'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
        TRITON_CACHE_DIR=str(scratch / 'triton'), TORCH_EXTENSIONS_DIR=str(scratch / 'extensions'))
    import torch
    from consequence_evaluator.old_utility import pw_sample, deterministic_group_mean
    from oakink_wm.pointworld_temporal import model_from_config, capped_collate, VENDOR
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2)
    torch.manual_seed(188)
    torch.backends.cuda.matmul.allow_tf32 = True
    m = json.loads((collection / 'manifest.json').read_text())
    audit = json.loads((collection / 'audit.json').read_text())
    if (m['status'] != 'COMPLETED' or m.get('pw_mode') != 'collect' or not m['training_allowed']
            or m['seed'] != 187 or not audit['pass']
            or sha(collection / 'trajectory.npz') != m['trajectory_sha256']):
        raise ValueError('authenticated fresh training-eligible PW collection required')
    for path, digest in audit['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError('collection audit evidence drift: ' + path)
    reference_path = Path(m['reference'])
    with np.load(reference_path, allow_pickle=False) as f:
        reference = {k: f[k].copy() for k in f.files}
    with np.load(collection / 'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    checkpoint = ROOT / 'outputs/cm-pointflow-effect-pretrain/pointworld-main3-20261007/train-action/best.pt'
    canonical_path = ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    canonical_manifest = json.loads((canonical_path.parent / 'manifest.json').read_text())
    if sha(checkpoint) != PW_SHA or sha(canonical_path) != canonical_manifest['canonical_sha256']:
        raise ValueError('frozen PW/canonical identity differs')
    mesh = ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj'
    cfg = Path(json.loads((ROOT / 'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base/config.json').read_text())['cfg_env'])
    import yaml
    if (sha(mesh) != canonical_manifest['input_sha256'][str(mesh)]
            or yaml.safe_load(cfg.read_text())['env']['ballSize'] != 1):
        raise ValueError('PW canonical mesh/scale does not match physical simulator')
    with np.load(canonical_path, allow_pickle=False) as f:
        canonical = {k: f[k].copy() for k in f.files}
    paths = [collection / name for name in ('manifest.json', 'trajectory.npz', 'audit.json')]
    paths += [checkpoint, canonical_path, canonical_path.parent / 'manifest.json', mesh, cfg, reference_path,
              Path(__file__), TASK / 'src/trajectory_policy/pw_critic.py', OLD / 'src/consequence_evaluator/old_utility.py']
    hashes = {str(path.resolve()): sha(path) for path in paths}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for base, key in ((PW, 'implementation_sources'), (VENDOR, 'vendor_sources')):
        for name, digest in state['identity'][key].items():
            path = (base / name).resolve()
            if sha(path) != digest:
                raise ValueError('PW implementation identity drift: ' + str(path))
            hashes[str(path)] = digest
    install_fused_hilbert()
    import oakink_wm.pointworld as spatial
    import oakink_wm.pointworld_temporal as temporal
    spatial.mean_groups = temporal.mean_groups = deterministic_group_mean
    model = model_from_config(state['identity']['stats'], state['config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del state
    ticks = np.repeat(np.arange(0, 519, 4), 16)
    envs = np.tile(np.arange(8, 24), 130)
    row = np.arange(len(ticks))
    high = d['high_input'][ticks, envs - 8]
    tau = d['high_tau'][ticks, envs - 8]
    mean = d['high_tau_mean'][ticks, envs - 8]
    reward = rewards(d, reference)
    target = truncated_returns(reward, ticks, envs)
    z = np.zeros((len(ticks), 2, 24, 12), np.float32)
    out.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status='RUNNING', collection=str(collection), training_allowed=True,
        physical_gpu=a.gpu, batch=8, anchors_per_batch=4, checkpoint=str(checkpoint), checkpoint_sha256=PW_SHA,
        frozen_pw=True, rows=len(row), representation='predicted current-frame object SE3 24x3x4',
        conditioning='measured past/current4 + proposed future hand24; no measured future input',
        train_envs=list(range(8, 20)), validation_envs=list(range(20, 24)), gamma=.99, horizon=24,
        input_sha256=hashes, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        batch_context='4 chronological anchors with sample/mean paired; fixed for both evaluations',
        group_mean_backend='deterministic sorted CSR')
    write(out / 'manifest.json', manifest)
    start = time.monotonic()
    repeat_error = permutation_error = None
    with torch.inference_mode():
        for begin in range(0, len(row), 4):
            if time.monotonic() - start > 300:
                raise TimeoutError('bounded PW cache exceeded300s')
            samples = []
            for i in row[begin:begin + 4]:
                for candidate in (tau[i], mean[i]):
                    oh, hh, hf = local_inputs(d, ticks[i], envs[i], candidate)
                    samples.append(pw_sample(oh, hh, hf, canonical, len(samples)))
            batch = {k: v.cuda() for k, v in capped_collate(samples).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(batch, 'action')
            r = pred['rotation'][:, 0].float()
            t = pred['translation'][:, 0].float()
            if begin == 0:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    repeat = model(batch, 'action')
                repeat_error = max(float((pred[k] - repeat[k]).abs().max()) for k in ('rotation', 'translation'))
                swapped = {k: v.cuda() for k, v in capped_collate(samples[::-1]).items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    permutation = model(swapped, 'action')
                permutation_error = max(float((pred[k] - permutation[k].flip(0)).abs().max()) for k in ('rotation', 'translation'))
                if repeat_error > 0 or permutation_error > 1e-5:
                    raise ValueError('PW repeatability/permutation contract differs')
            value = torch.cat((r, t[..., None]), -1).reshape(-1, 24, 12).cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError('PW prediction nonfinite')
            z[begin:begin + 4] = value.reshape(-1, 2, 24, 12)
            if begin == 0 or (begin + 4) % 256 == 0 or begin + 4 == len(row):
                elapsed = time.monotonic() - start
                pids = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                              '--format=csv,noheader,nounits'], text=True).split()
                if set(map(int, pids)) - {os.getpid()}:
                    raise RuntimeError('foreign GPU process')
                monitor = dict(rows=begin + 4, total=len(row), elapsed_s=elapsed,
                    eta_s=elapsed / (begin + 4) * (len(row) - begin - 4),
                    gpu=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=utilization.gpu,memory.used',
                                                 '--format=csv,noheader'], text=True).strip())
                print(json.dumps(monitor), flush=True)
    np.savez_compressed(out / 'windows.npz', high=high, tau=tau, mean=mean, z=z[:, 0], z_mean=z[:, 1],
        returns=target, ticks=ticks, envs=envs, train=envs < 20, reward=reward)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('PW cache input drift')
    manifest.update(status='COMPLETED', elapsed_s=time.monotonic() - start, windows_sha256=sha(out / 'windows.npz'),
        repeat_error=repeat_error, permutation_error=permutation_error,
        proposal_mean_feature_delta_rms=float(np.sqrt(np.mean((z[:, 0] - z[:, 1]) ** 2))),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    write(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('elapsed_s', 'proposal_mean_feature_delta_rms', 'repeat_error', 'permutation_error')}), flush=True)


if __name__ == '__main__':
    main()
