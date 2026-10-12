"""Localize a failed WAM-R shadow using frozen-input oracle substitutions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/goal-point-wam/src'),
               str(ROOT / 'src/task/cm-pointflow-effect-pretrain/src'), str(Path(__file__).parent)]
from audit_wam_ref2_shadow import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shadow', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from trajectory_policy.pd_retargeter import load_model, decode_target
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.wam_shadow import robot_goal_sample, active_hand_tau
    from goal_point_wam.data import goal_collate
    out = a.shadow.resolve()
    target = out / 'diagnosis.json'
    if target.exists():
        raise FileExistsError(target)
    m = json.loads((out / 'manifest.json').read_text())
    if m['status'] != 'COMPLETED' or m['shadow_sha256'] != sha(out / 'shadow.npz'):
        raise ValueError('completed authenticated shadow required')
    for path, digest in m['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('shadow source identity differs: ' + path)
    torch.set_num_threads(2)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    with np.load(out / 'shadow.npz', allow_pickle=False) as f:
        s = {k: f[k].copy() for k in f.files}
    source = ROOT / 'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    with np.load(source / 'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets')}
    with np.load(ROOT / 'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz', allow_pickle=False) as f:
        ref = f['obj'].copy()
    with np.load(ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz', allow_pickle=False) as f:
        canonical = {k: f[k].copy() for k in f.files}
    fit = ROOT / 'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    r = load_model(torch.load(fit, map_location='cpu', weights_only=False), 'cuda')
    ticks, envs = s['ticks'], s['envs']
    state = np.stack([features(*(d[k][tick, env] for k in ('q', 'dq', 'hand', 'obj', 'velocity')),
        d['hand'][tick + np.arange(1, 25), env])[0] for tick, env in zip(ticks, envs)])
    st = torch.as_tensor(state, device='cuda')
    q = torch.as_tensor(d['q'][ticks, envs], device='cuda')
    delta_stats = lambda x, y: dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(x[:, :3]-y[:, :3], axis=-1), 95)*1000),
        rotation_axis_p95_rad=float(np.percentile(np.abs(x[:, 3:6]-y[:, 3:6]).max(-1), 95)),
        finger_coordinate_p95_rad=float(np.percentile(np.abs(x[:, [6,8,10,12,14,15]]-y[:, [6,8,10,12,14,15]]).max(-1), 95)))
    substitution = {}
    with torch.inference_mode():
        for name in ('gt_prefix_wam_tail', 'wam_prefix_gt_tail', 'zero_trajectory'):
            if name == 'zero_trajectory':
                tau = np.zeros_like(s['prediction'])
            else:
                tau = s['prediction'].copy()
                if name == 'gt_prefix_wam_tail':
                    tau[:, :8] = s['teacher_tau'][:, :8]
                else:
                    tau[:, 8:] = s['teacher_tau'][:, 8:]
            pd = decode_target(r(st, torch.as_tensor(tau, device='cuda')), q).cpu().numpy()
            substitution[name] = delta_stats(pd, s['pd_gt'])
    # Goal intervention uses the identical cached scene/observed-hand tokens.
    checkpoint = ROOT / 'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    stats = ROOT / 'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    spec = importlib.util.spec_from_file_location('frozen_wam_ref2_diagnosis', out / 'code/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from oakink_wm.pointworld_performance import install_fused_hilbert
    install_fused_hilbert()
    model = module.GoalPointWAM(json.loads(stats.read_text()), packet['config']).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    del packet
    real, static, norm = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(ticks), 8):
            samples = [robot_goal_sample(d['obj'][np.maximum(np.arange(t-3,t+1),0), e],
                d['hand'][np.maximum(np.arange(t-3,t+1),0), e], ref[t+24], canonical, sample_id=i)
                for i, (t,e) in enumerate(zip(ticks[begin:begin+8], envs[begin:begin+8]), begin)]
            batch = {k:v.cuda() for k,v in goal_collate(samples).items()}
            norm.append(((batch['features'] - model.scene.scene_mean) / model.scene.scene_std)
                        [batch['point_valid']].cpu().numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scene = model.scene(batch)
                observed = model.observed_hand_condition(batch)
                for name, goal in (('real', batch['goal_object_points']), ('static', batch['start_object_points'])):
                    tokens, mask = model.goal(batch['start_object_points'], goal, batch['object_goal_valid'])
                    pred = model.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, mask, observed).float()
                    value = active_hand_tau(pred.cpu().numpy(), batch['start_hand_valid'].cpu().numpy())
                    (real if name == 'real' else static).append(value)
    real = np.concatenate(real)
    static = np.concatenate(static)
    expected = s['teacher_tau'].reshape(-1,24,11,3)
    trajectory_summary = lambda tau: dict(first8_epe_mm=float(np.linalg.norm(tau.reshape(-1,24,11,3)-expected,axis=-1)[:,:8].mean()*1000),
        full24_epe_mm=float(np.linalg.norm(tau.reshape(-1,24,11,3)-expected,axis=-1).mean()*1000))
    normalized = np.abs(np.concatenate(norm))
    with torch.inference_mode():
        static_pd = decode_target(r(st, torch.as_tensor(static, device='cuda')), q).cpu().numpy()
    result = dict(status='COMPLETED', engineering_only=True, elapsed_s=time.monotonic()-started,
        substitution=substitution, gt_R_vs_source_actual_PD=delta_stats(s['pd_gt'], d['pd_targets'][ticks,envs]),
        epe=dict(real=trajectory_summary(real), static_goal=trajectory_summary(static),
                 stationary_hand=trajectory_summary(np.zeros_like(real))),
        rerun_prediction_max_m=float(np.max(np.abs(real-s['prediction']))),
        real_static_prediction_first8_difference_mm=float(np.linalg.norm((real-static).reshape(-1,24,11,3),axis=-1)[:,:8].mean()*1000),
        static_goal_pd_vs_gt_R=delta_stats(static_pd,s['pd_gt']),
        normalized_scene_feature_abs_p95_by_channel=np.percentile(normalized,95,axis=0).tolist(),
        normalized_scene_feature_abs_max_by_channel=normalized.max(0).tolist(),
        source_sha256=dict(shadow=sha(out / 'shadow.npz'), manifest=sha(out / 'manifest.json'),
                           diagnosis_tool=sha(Path(__file__))),
        limitations=['GT prefix/tail substitutions are privileged diagnostic labels, not policies',
                     'this checks R sensitivity to the retained forecasts, not complete execution or causal domain attribution'])
    np.savez_compressed(out / 'diagnosis.npz', static_prediction=static, real_prediction=real)
    write(target,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
