"""Extract frozen WAM/forecast and matched raw inputs from existing TRAIN episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / 'src'), str(TASK.parent / 'shadow-finetune/src')]
from shadow_finetune import paths
from wam_rl_policy.wam_features import FeatureController
from wam_rl_policy.fast_bc import batched_inputs, type_summaries
from wam_rl_policy.task_readout import (object_split, finite_returns, transition_targets,
                                        raw_tokens, forecast_from_encoded, observed_batch)
from video_data_preparation.unidexgrasp_data import object_poses


def write(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2)
    torch.manual_seed(246)
    a.output.mkdir(parents=True)
    started = time.monotonic()
    objects = json.loads((a.source / 'environment-objects.json').read_text())
    source_manifest = json.loads((a.source / 'manifest.json').read_text())
    assert source_manifest['status'] == 'COMPLETED'
    fit, held = object_split(objects)
    controller = FeatureController(a.checkpoint, objects, a.cache)
    model = controller.model.forecaster
    assert controller.schema == 'ref2dex.ptv3-shared-wam.v3' and controller.source_step == 80000
    ticks = np.linspace(6, 166, 8, dtype=int).tolist()
    names = ['first-episode.npz', 'last-episode.npz']
    if a.smoke:
        ticks, names = ticks[:1], names[:1]
    manifest = dict(status='RUNNING', run_id=a.output.parent.name,
        experiment_id='P-20261016-wam-task-readout', task='wam-rl-policy',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        pid=os.getpid(), checkpoint=str(a.checkpoint.resolve()), source=str(a.source.resolve()),
        checkpoint_step=controller.source_step, checkpoint_config=model.config,
        ticks=ticks, input_files=names, heldout_train_objects=held, gamma=.99,
        scope='TRAIN-only readout qualification, no policy improvement claim', gpu=a.gpu,
        no_teacher_action_target_or_bc_hidden_input=True,
        native_observation_includes_previous_executed_action=True,
        max_seconds=1200, max_output_gib=4)
    write(a.output / 'manifest.json', manifest)
    collected = {k: [] for k in ['state', 'raw', 'raw_valid', 'wam', 'wam_valid', 'mean',
                                'forecast', 'targets', 'fit', 'object_id', 'tick', 'episode']}
    total_batches = len(names) * len(ticks) * (1 if a.smoke else len(objects) // 16)
    done = 0
    with torch.inference_mode():
        for episode, name in enumerate(names):
            with np.load(a.source / name, allow_pickle=False) as d:
                data = {key: d[key] for key in ('observation', 'object', 'hand', 'goal', 'reward', 'native_reward', 'reset')}
            np.testing.assert_array_equal(data['reward'], data['native_reward'])
            returns = finite_returns(data['reward'], data['reset'])
            assert len(returns) == 199 and len(objects) == returns.shape[1]
            for tick in ticks:
                past = tick + np.array([-6, -4, -2, 0])
                rotations = object_poses(data['object'][tick])[:, :3, :3]
                labels = np.concatenate((returns[tick, :, None], transition_targets(data, tick, rotations)), -1)
                # Raw observation is clipped exactly as in existing policy, but
                # there is no reference action or BC hidden representation.
                state = np.concatenate((data['observation'][tick].clip(-5, 5),
                    data['goal'][tick] - data['object'][tick, :, :3],
                    np.full((len(objects), 1), (199 - tick) / 199)), -1).astype(np.float32)
                for start in range(0, 16 if a.smoke else len(objects), 16):
                    end = min(start + 16, len(objects))
                    samples = batched_inputs(data['object'][past, start:end].swapaxes(0, 1),
                        data['hand'][past, start:end].swapaxes(0, 1), data['goal'][tick, start:end],
                        controller.geometry[start:end])
                    batch = observed_batch(samples,max(len(g['surface']) for g in controller.geometry))
                    shared, valid, geometry = model.encode(batch)
                    raw, raw_valid = raw_tokens(batch)
                    count = geometry['tokens'].shape[1]
                    mean = type_summaries(shared, valid, count, 22)
                    hand_future, scene_future = forecast_from_encoded(model, batch, shared, valid, count)
                    if done == 0:
                        reference = model(batch)
                        decoder_error = max(float((reference['trajectory'] - hand_future).abs().max()),
                                            float((reference['scene'] - scene_future).abs().max()))
                        if decoder_error > 1e-5:
                            raise AssertionError(('Decoder differs from original V3', decoder_error))
                        manifest['decoder_max_abs_error_m'] = decoder_error
                        manifest['tensor_shapes_per_batch'] = dict(wam=list(shared.shape), raw=list(raw.shape))
                        assert all(not p.requires_grad for p in model.parameters()) and not model.training
                    forecast = torch.cat((hand_future[:, [7, 15], 0].flatten(1),
                                          scene_future[:, :, [7, 15]].flatten(1)), -1)
                    values = dict(state=torch.from_numpy(state[start:end]), raw=raw.half().cpu(),
                        raw_valid=raw_valid.cpu(), wam=shared.half().cpu(), wam_valid=valid.cpu(),
                        mean=mean.cpu(), forecast=forecast.cpu(), targets=torch.from_numpy(labels[start:end]),
                        fit=torch.from_numpy(fit[start:end]), object_id=torch.arange(start, end),
                        tick=torch.full((end-start,), tick), episode=torch.full((end-start,), episode))
                    for key, value in values.items():
                        if value.is_floating_point() and not torch.isfinite(value).all():
                            raise FloatingPointError(key)
                        collected[key].append(value)
                    done += 1
                    if done == 1 or done % 16 == 0:
                        elapsed = time.monotonic() - started
                        resource = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip()
                        row = dict(batches=done, total=total_batches, elapsed=elapsed,
                                   eta_seconds=elapsed / done * (total_batches - done), resource=resource)
                        print(json.dumps(row), flush=True)
                        with (a.output / 'resources.jsonl').open('a') as f:
                            f.write(json.dumps(row) + '\n')
                    if time.monotonic() - started > 1200:
                        raise TimeoutError('Extraction budget exceeded')
    packet = {key: torch.cat(value) for key, value in collected.items()}
    packet['objects'] = objects
    packet['manifest'] = manifest.copy()
    torch.save(packet, a.output / 'features.pt')
    manifest.update(status='COMPLETED', samples=len(packet['state']), seconds=time.monotonic() - started,
                    bytes=(a.output / 'features.pt').stat().st_size,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if manifest['bytes'] > 4 * 1024**3:
        raise RuntimeError('Artifact budget exceeded')
    write(a.output / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
