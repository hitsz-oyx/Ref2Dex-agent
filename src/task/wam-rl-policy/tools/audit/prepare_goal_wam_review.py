"""Cache bounded, frozen WAM-v4 predictions for the Viser goal review."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler, transform_points
from wam_rl_policy.goal_review import VARIANTS, goals_as_delta, prediction_in_world, review_goals
from wam_rl_policy.wam_v4_runtime import V4Runtime, collate_v4, v4_online_samples


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def poses(values):
    result = np.repeat(np.eye(4)[None], len(values), axis=0)
    result[:, :3, :3] = Rotation.from_quat(values[:, 3:7]).as_matrix()
    result[:, :3, 3] = values[:, :3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dense-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=600)
    args = parser.parse_args()
    import torch
    output = args.output.resolve()
    if ROOT/'outputs' not in output.parents or args.stride < 1 or not 0 < args.seconds <= 1200:
        raise ValueError('bounded workspace output, positive stride and <=1200s required')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'predictions.json').exists():
        raise FileExistsError('predictions already exist; choose a new output')
    if shutil.disk_usage(output).free < 20*2**30:
        raise RuntimeError('less than 20GiB free')
    if not torch.cuda.is_available():
        raise RuntimeError('real WAM inference requires an available CUDA device')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    replay = json.loads((args.replay/'replay.json').read_text())
    manifest = json.loads((args.replay/'manifest.json').read_text())
    cases = [c for c in replay if c['metadata'].get('variant') == 'nominal']
    identities = [c['metadata']['identity'] for c in cases]
    cache = Path(manifest['cache'])
    runtime = V4Runtime(args.checkpoint, identities, cache,
                        dense_manifest=args.dense_manifest, device=f'cuda:{args.gpu}')
    source = Path(manifest['source'])
    with np.load(source/'last-episode.npz', allow_pickle=False) as arrays:
        objects, hands, times, reset = [arrays[k].copy() for k in ('object', 'hand', 'time', 'reset')]
    if reset[1:-1].any() or not np.allclose(np.diff(times), 1/60, atol=1e-5):
        raise ValueError('contiguous 60Hz source required; H4 uses ticks t-6,t-4,t-2,t')
    started = time.monotonic()
    record = dict(status='RUNNING', run_id=output.name,
        experiment_id='P-20261018-goal-wam-viser',
        git_commit=os.environ.get('REF2DEX_REVIEW_COMMIT') or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=runtime.step,
        source=str(source), replay=str(args.replay.resolve()), cache=str(cache),
        dense_manifest=str(args.dense_manifest.resolve()),
        device=f'cuda:{args.gpu}', gpu=torch.cuda.get_device_name(args.gpu),
        python=sys.executable, python_version=sys.version.split()[0], torch=torch.__version__,
        numpy=np.__version__, precision='CUDA bfloat16 autocast; physical outputs float32',
        pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), stride=args.stride,
        input_lineage={k:manifest[k] for k in ('original_source','original_replay','compaction') if k in manifest},
        seconds_budget=args.seconds, output_budget_gib=2, seed=0,
        history='actual t-6,t-4,t-2,t @60Hz -> H4@30Hz; current object anchor',
        scheduler='recomputed at60Hz using the original32 flow identities and dense observed surface',
        reference='known full recorded object path, not independently generated task instruction',
        scope='frozen inference and presentation; no new training, simulation or task-success evaluation')
    write_json(output/'inference-manifest.json', record)
    catalog = []
    for case_id, (case, identity) in enumerate(zip(cases, identities)):
        env = identity['env']
        reference = poses(objects[:, env])
        surface, normals, flow, interaction, interaction_normals = runtime.geometry[case_id]
        scheduler = OrderedGoalScheduler(reference, times-times[0], flow, interaction_points=surface)
        snapshots, decisions = [], []
        for tick, current in enumerate(reference):
            decision = scheduler.update(current, float(times[tick]-times[0]), hand_points=hands[tick, env])
            if tick < 6 or (tick-6) % args.stride:
                continue
            if time.monotonic()-started > args.seconds:
                raise TimeoutError('frozen inference budget exceeded')
            goals = review_goals(scheduler, current, decision)
            deltas = goals_as_delta(current, goals, flow)
            predicted_hands, predicted_scenes, contacts, parts, validities, student_tau = [], [], [], [], [], []
            indices = tick+np.arange(25)*2
            future_valid = indices < len(times)
            observed_hand = np.zeros((25, 11, 3), np.float32)
            observed_scene = np.zeros((25, 32, 3), np.float32)
            observed_hand[future_valid] = hands[indices[future_valid], env]
            for k in np.flatnonzero(future_valid):
                observed_scene[k] = transform_points(reference[indices[k]], flow)
            for variant in range(3):
                # Single-sample inference avoids changing the PTv3 batch context
                # when comparing goals at the exact same observed state.
                samples = v4_online_samples(objects[tick-6:tick+1:2, env][None],
                    hands[tick-6:tick+1:2, env][None], [runtime.geometry[case_id]],
                    goal_delta=deltas[variant][None])
                batch = {k:v.to(runtime.device) for k,v in collate_v4(samples).items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = runtime.predict(batch)
                hand_world, scene_world = prediction_in_world(current, hands[tick, env], flow,
                    prediction['trajectory'][0].cpu().numpy(), prediction['scene'][0].cpu().numpy())
                field = prediction['interaction'][0, 0].cpu().numpy()
                valid = prediction['interaction_prediction_valid'][0, 0].cpu().numpy()
                if (not np.isfinite(field).all() or field.min() < 0 or field.max() > 1):
                    raise ValueError('finite C probabilities required')
                predicted_hands.append(hand_world);predicted_scenes.append(scene_world)
                contacts.append(field[..., 0]);parts.append(field[..., 1:]);validities.append(valid)
                student_tau.append(runtime.tau_for_student(prediction, current))
                del prediction, batch
            snapshots.append(dict(tick=tick, time=times[tick]-times[0], object_pose=current,
                hand=hands[tick, env], goals=goals, goal_delta=deltas,
                predicted_hand=predicted_hands, predicted_scene=predicted_scenes,
                contact=contacts, parts=parts, interaction_valid=validities, student_tau=student_tau,
                observed_hand=observed_hand, observed_scene=observed_scene, observed_future_valid=future_valid))
            decisions.append({k:decision[k] for k in ('mode','reason','confirmed_index','projected_index',
                'target_reference_index','target_on_reference','confirmed_progress','reference_error_m',
                'lookahead_m','interaction','milestone_witnesses')})
        packed = {key:np.asarray([row[key] for row in snapshots]) for key in snapshots[0]}
        packed.update(surface=surface, flow=flow, interaction_points=interaction,
                      reference=reference, reference_times=times-times[0], milestones=scheduler.milestones)
        path = output/f'case-{case_id:02d}.npz'
        np.savez_compressed(path, **packed)
        goal_gap = np.linalg.norm(packed['goal_delta'], axis=-1).mean(-1)
        sensitivity = np.linalg.norm(packed['predicted_hand'][:, :, -1]-packed['predicted_hand'][:, 1:2, -1], axis=-1).mean(-1)
        catalog.append(dict(name=case['name'].split(' / ')[0], identity=identity,
            file=path.name, frames=len(snapshots), decisions=decisions,
            mean_goal_point_mm=(goal_gap.mean(0)*1000).tolist(),
            mean_hand_endpoint_difference_from_selected_mm=(sensitivity.mean(0)*1000).tolist()))
        print(json.dumps(dict(case=catalog[-1]['name'], frames=len(snapshots),
                             seconds=time.monotonic()-started)), flush=True)
    torch.cuda.synchronize(runtime.device)
    write_json(output/'predictions.json', dict(variants=VARIANTS, fps=30, cases=catalog))
    record.update(status='COMPLETED', cases=len(catalog), frames=sum(c['frames'] for c in catalog),
                  predictions=sum(c['frames'] for c in catalog)*3, seconds=time.monotonic()-started,
                  peak_cuda_mib=torch.cuda.max_memory_allocated(runtime.device)/2**20)
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 2*2**30:
        raise RuntimeError('2GiB output budget exceeded; artifacts retained')
    write_json(output/'inference-manifest.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
