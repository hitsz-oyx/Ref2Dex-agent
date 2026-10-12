"""Replay a frozen WAM -> frozen R -> native PD chain without executing WAM."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
WAM = ROOT / 'src/task/goal-point-wam'
CM = ROOT / 'src/task/cm-pointflow-effect-pretrain'
sys.path[:0] = [str(TASK / 'src'), str(WAM / 'src'), str(CM / 'src')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--seed', type=int, default=289)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied before shadow inference')
    scratch = ROOT / 'tmp/trajectory-policy/wam-ref2-shadow'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TRITON_CACHE_DIR=str(scratch / 'triton'))
    import torch
    from trajectory_policy.wam_shadow import (NATIVE_TO_WAM, WAM_TO_NATIVE,
        NATIVE_LINKS, WAM_SEMANTICS, active_hand_tau, robot_goal_sample)
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.pd_retargeter import load_model, decode_target, command, pd_from_action
    from goal_point_wam.data import goal_collate
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.monotonic()
    source = ROOT / 'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    fit = ROOT / 'outputs/trajectory-policy/competent-pd-fit-20261011-r1'
    reference = ROOT / 'outputs/trajectory-policy/competent-reference-20261011-r1'
    calibration = ROOT / 'outputs/trajectory-policy/competent-pd-execution-20261011-r1'
    canonical_path = ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    checkpoint = ROOT / 'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    stats = ROOT / 'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (state['step'] != 10000 or state.get('artificial_labels')
            or state.get('checkpoint_kind') != 'goal-point-wam.single.v1'):
        raise ValueError('fixed real-data final checkpoint required')
    if sha(stats) != state['identity']['stats_sha256']:
        raise ValueError('frozen WAM stats differ')
    identity = state['identity']
    # Preserve and authenticate the original implementation after the video
    # route changes model.py. Never disable the checkpoint source identity gate.
    relative = 'src/task/goal-point-wam/src/goal_point_wam/model.py'
    original = subprocess.check_output(['git', 'show', 'e8b81994:' + relative], cwd=ROOT)
    if hashlib.sha256(original).hexdigest() != identity['model_sha256']:
        raise ValueError('original Git blob differs from checkpoint implementation')
    for folder, filename, key in ((source, 'trajectory.npz', 'trajectory_sha256'),
                                  (reference, 'reference.npz', 'reference_sha256')):
        m = json.loads((folder / 'manifest.json').read_text())
        if m['status'] != 'COMPLETED' or sha(folder / filename) != m[key]:
            raise ValueError('source/reference artifact identity differs')
    if not json.loads((source / 'audit.json').read_text())['pass']:
        raise ValueError('teacher data audit failed')
    calibration_manifest = json.loads((calibration / 'manifest.json').read_text())
    if not json.loads((calibration / 'audit.json').read_text())['pass']:
        raise ValueError('original R/PD calibration audit failed')
    fit_manifest = json.loads((fit / 'manifest.json').read_text())
    for filename in ('pd_retargeter.py', 'retargeter_data.py'):
        path = TASK / 'src/trajectory_policy' / filename
        if sha(path) != fit_manifest['input_sha256'][str(path)]:
            raise ValueError('frozen R implementation differs: ' + filename)
    # Calibration manifest authenticates the model as an input file.
    calibrated_r_hash = calibration_manifest['input_sha256'][str(fit / 'tau.pt')]
    if sha(fit / 'tau.pt') != calibrated_r_hash:
        raise ValueError('retained R checkpoint differs from physical calibration')
    canonical_manifest = json.loads((canonical_path.parent / 'manifest.json').read_text())
    if sha(canonical_path) != canonical_manifest['canonical_sha256']:
        raise ValueError('canonical geometry differs')
    with np.load(source / 'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets')}
    with np.load(reference / 'reference.npz', allow_pickle=False) as f:
        ref = f['obj'].copy()
    with np.load(canonical_path, allow_pickle=False) as f:
        canonical = {k: f[k].copy() for k in f.files}
    out.mkdir(parents=True)
    (out / 'code').mkdir()
    frozen_source = out / 'code/model.py'
    frozen_source.write_bytes(original)
    spec = importlib.util.spec_from_file_location('frozen_wam_ref2_model', frozen_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from oakink_wm.pointworld_performance import install_fused_hilbert
    install_fused_hilbert()
    model = module.GoalPointWAM(json.loads(stats.read_text()), state['config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    del state
    rmodel = load_model(torch.load(fit / 'tau.pt', map_location='cpu', weights_only=False), 'cuda')
    for network in (model, rmodel):
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    native = calibration_manifest['native_controller']
    offset = torch.tensor(native['offset'], device='cuda')
    scale = torch.tensor(native['scale'], device='cuda')
    ticks = np.repeat([0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 518], 4)
    envs = np.tile(np.arange(24, 28), 16)
    inputs = [Path(__file__), TASK / 'src/trajectory_policy/wam_shadow.py',
        TASK / 'src/trajectory_policy/pd_retargeter.py', TASK / 'src/trajectory_policy/retargeter_data.py',
        WAM / 'src/goal_point_wam/data.py', checkpoint, stats, fit / 'tau.pt', fit / 'manifest.json',
        source / 'trajectory.npz', source / 'manifest.json', source / 'audit.json',
        reference / 'reference.npz', reference / 'manifest.json',
        canonical_path, canonical_path.parent / 'manifest.json',
        calibration / 'manifest.json', calibration / 'audit.json', frozen_source]
    hashes = {str(path): sha(path) for path in inputs}
    manifest = dict(status='RUNNING', schema='ref2dex.wam-ref2-shadow.v1',
        engineering_only=True, training_allowed=False, physical_gpu=args.gpu, seed=args.seed,
        checkpoint_step=10000, wam_checkpoint_sha256=sha(checkpoint), r_checkpoint_sha256=calibrated_r_hash,
        wam_original_git='e8b81994', wam_implementation_sha256=identity['model_sha256'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, windows=len(ticks), source_envs=[24, 25, 26, 27],
        query_ticks=ticks[::4].tolist(), fps=30, goal='planned object reference pose at t+24',
        active_hand='right; absent left masked', current_frame='one measured current-object inverse',
        semantic_map=[dict(wam=WAM_SEMANTICS[i], native=NATIVE_LINKS[int(NATIVE_TO_WAM[i])])
                      for i in range(11)],
        anatomy_equivalence=False, execution='shadow only; no WAM command executed',
        screen=dict(frame_max_m=1e-6, pd_xyz_p95_m=.01,
                    pd_wrist_rotation_p95_rad=.05, pd_finger_p95_rad=.05, clipping_rate=.01))
    write(out / 'manifest.json', manifest)
    arrays = {k: [] for k in ('prediction', 'teacher_tau', 'pd_wam', 'pd_gt',
                              'action_wam', 'action_gt', 'clip_wam', 'clip_gt', 'pd_roundtrip')}
    frame_error = 0.
    with torch.inference_mode():
        for begin in range(0, len(ticks), 8):
            if time.monotonic() - start > 300:
                raise TimeoutError('shadow budget exceeded 300s')
            sample, teacher_state, teacher_tau = [], [], []
            for i in range(begin, min(begin + 8, len(ticks))):
                tick, env = int(ticks[i]), int(envs[i])
                past = np.maximum(np.arange(tick - 3, tick + 1), 0)
                sample.append(robot_goal_sample(d['obj'][past, env], d['hand'][past, env],
                    ref[tick + 24], canonical, sample_id=i))
                future = d['hand'][tick + np.arange(1, 25), env]
                st, tau = features(d['q'][tick, env], d['dq'][tick, env], d['hand'][tick, env],
                                   d['obj'][tick, env], d['velocity'][tick, env], future)
                teacher_state.append(st)
                teacher_tau.append(tau)
                # Independent affine point transformation, followed by WAM
                # semantic reorder then adapter back to native.
                anchor = np.linalg.inv(d['obj'][tick, env].astype(np.float64))
                local_now = d['hand'][tick, env] @ anchor[:3, :3].T + anchor[:3, 3]
                local_future = future @ anchor[:3, :3].T + anchor[:3, 3]
                wamtau = np.zeros((24, 2, 11, 3), np.float64)
                wamtau[:, 0] = (local_future - local_now)[:, NATIVE_TO_WAM]
                recovered = active_hand_tau(wamtau, sample[-1]['start_hand_valid'])
                frame_error = max(frame_error, float(np.max(np.abs(recovered - tau))))
            batch = {k: v.cuda() for k, v in goal_collate(sample).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scene = model.scene(batch)
                tokens, mask = model.goal(batch['start_object_points'], batch['goal_object_points'],
                                           batch['object_goal_valid'])
                predicted = model.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, mask,
                                              model.observed_hand_condition(batch)).float()
            pred_np = active_hand_tau(predicted.cpu().numpy(), batch['start_hand_valid'].cpu().numpy())
            state_tensor = torch.as_tensor(np.stack(teacher_state), device='cuda')
            gt_np = np.stack(teacher_tau)
            gt = torch.as_tensor(gt_np, device='cuda')
            q = torch.as_tensor(d['q'][ticks[begin:begin + len(sample)], envs[begin:begin + len(sample)]], device='cuda')
            pd_wam = decode_target(rmodel(state_tensor, torch.as_tensor(pred_np, device='cuda')), q)
            pd_gt = decode_target(rmodel(state_tensor, gt), q)
            _, action_wam, clip_wam = command(pd_wam, q, offset, scale)
            _, action_gt, clip_gt = command(pd_gt, q, offset, scale)
            unclipped = ~clip_gt
            roundtrip = pd_from_action(action_gt, q, offset, scale) - pd_gt
            arrays['pd_roundtrip'].append(roundtrip[unclipped].cpu().numpy())
            for key, value in dict(prediction=pred_np, teacher_tau=gt_np, pd_wam=pd_wam,
                                   pd_gt=pd_gt, action_wam=action_wam, action_gt=action_gt,
                                   clip_wam=clip_wam, clip_gt=clip_gt).items():
                arrays[key].append(value.cpu().numpy() if isinstance(value, torch.Tensor) else value)
            gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used',
                                             '--format=csv,noheader,nounits'], text=True).strip()
            print(json.dumps(dict(windows=begin + len(sample), total=len(ticks),
                                  elapsed_s=time.monotonic() - start, gpu=gpu)), flush=True)
            pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True).split()
            if set(map(int, pids)) - {os.getpid()}:
                raise RuntimeError('foreign GPU process appeared')
    packed = {k: np.concatenate(v) for k, v in arrays.items()}
    error = np.linalg.norm((packed['prediction'] - packed['teacher_tau']).reshape(-1, 24, 11, 3), axis=-1)
    delta = packed['pd_wam'] - packed['pd_gt']
    xyz = np.linalg.norm(delta[:, :3], axis=-1)
    rot = np.abs(delta[:, 3:6]).max(-1)
    fingers = np.abs(delta[:, [6, 8, 10, 12, 14, 15]]).max(-1)
    p95 = lambda values: float(np.percentile(values, 95))
    clip_rate = float(packed['clip_wam'].mean())
    screen = dict(frame=frame_error <= 1e-6, pd_xyz=p95(xyz) <= .01,
                  pd_rotation=p95(rot) <= .05, pd_fingers=p95(fingers) <= .05,
                  clipping=clip_rate < .01)
    result = dict(status='COMPLETED', scope='fixed 64 robot teacher windows; shadow control screen only',
        trajectory_epe_mm_by_horizon={str(h): float(error[:, h-1].mean() * 1000) for h in (1, 2, 4, 8, 24)},
        trajectory_first8_epe_mm=float(error[:, :8].mean() * 1000),
        trajectory_full24_epe_mm=float(error.mean() * 1000), frame_max_error_m=frame_error,
        pd_xyz_p95_mm=p95(xyz) * 1000, pd_rotation_max_axis_p95_rad=p95(rot),
        pd_finger_max_coordinate_p95_rad=p95(fingers), wam_clipping_rate=clip_rate,
        gt_clipping_rate=float(packed['clip_gt'].mean()),
        pd_unclipped_roundtrip_max=float(np.abs(packed['pd_roundtrip']).max()),
        startup_first8_epe_mm=float(error[ticks <= 8, :8].mean() * 1000), screen=screen,
        may_start_small_simulation=all(screen.values()), semantic_index_inverse_exact=bool(
            np.array_equal(NATIVE_TO_WAM[WAM_TO_NATIVE], np.arange(11))),
        elapsed_s=time.monotonic() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        limitations=['role mapping does not equate human and robot landmark offsets',
            'human-trained WAM tested on robot-domain geometry, no robot-domain adaptation',
            'GT-R success on source trajectory does not rule out R error on WAM distribution',
            'shadow gates are conservative predeclared screens, not validated success guarantees'])
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('shadow evidence/source drift')
    np.savez_compressed(out / 'shadow.npz', ticks=ticks, envs=envs, **packed)
    result['shadow_sha256'] = sha(out / 'shadow.npz')
    write(out / 'result.json', result)
    manifest.update(status='COMPLETED', elapsed_s=result['elapsed_s'],
                    shadow_sha256=result['shadow_sha256'], result_sha256=sha(out / 'result.json'))
    write(out / 'manifest.json', manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
