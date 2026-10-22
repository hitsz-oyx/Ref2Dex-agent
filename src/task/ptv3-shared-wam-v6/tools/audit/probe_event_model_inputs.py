#!/usr/bin/env python3
"""Observe the real training reader/Steps/collate boundary without running main()."""
import argparse
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
sys.path.insert(0, str(ROOT/'src/task/wam-rl-policy-v3/src'))
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.events import event_features


def capture(batch):
    features, valid = event_features(batch)
    keys = ('event_type', 'event_delta_steps', 'event_valid', 'event_hz',
            'event_raw_invalid', 'event_condition_dropout')
    result = dict(tensors={k: dict(shape=list(batch[k].shape), dtype=str(batch[k].dtype),
                                   values=batch[k].tolist()) for k in keys},
        model_features=dict(shape=list(features.shape), dtype=str(features.dtype), values=features.tolist()),
        masks=dict(valid=int(valid.sum()), raw_invalid=int(batch['event_raw_invalid'].sum()),
                   dropout=int(batch['event_condition_dropout'].sum())),
        examples=[dict(sample_id=batch['sample_id'][i].tolist(), source_id=int(batch['source_id'][i]),
                       **{k: batch[k][i].item() for k in keys}, feature=features[i].tolist())
                  for i in range(min(16, len(valid)))],
        goal_tensor=dict(shape=list(batch['goal_object_delta'].shape), dtype=str(batch['goal_object_delta'].dtype),
            valid_shape=list(batch['object_goal_valid'].shape), valid_count=int(batch['object_goal_valid'].sum())))
    assert torch.equal(features[~valid], torch.zeros_like(features[~valid]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    entry = TASK/'tools/run/train_pretrain.py'
    spec = importlib.util.spec_from_file_location('audited_v6_training_entry', entry)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)  # __name__ != __main__: no model, optimizer, GPU or training.
    assert train.V6Windows is V6Windows and train.collate is collate
    config = json.loads(args.training_config.read_text())
    reader_options = dict(geometry_gt_object=config.get('geometry_object_source') == 'gt_dense2048')
    dataset = train.V6Windows(args.recipe, 'train', **reader_options)
    steps = train.Steps(dataset, 32, 170, 0, 1, 1)
    draw = next(iter(steps))
    loader = DataLoader(dataset, batch_sampler=steps, collate_fn=train.collate, num_workers=0)
    actual = next(iter(loader))
    direct = collate([dataset[index] for index in draw])
    assert actual.keys() == direct.keys()
    assert all(torch.equal(actual[k], direct[k]) for k in actual)
    # Verify the actual main() still forwards this batch with only .to(device).
    source = inspect.getsource(train.main)
    assert 'batch=next(iterator)' in source and 'batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}' in source
    assert 'prediction=wrapped(batch)' in source
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                   and isinstance(n.value, ast.Name) and n.value.id == 'batch' for n in ast.walk(tree))
    # Controlled coverage of all four raw categories, both retained and dropped.
    lookup = {tuple(map(int, row)): i for i, row in enumerate(dataset.index)}
    audit = json.loads(args.source_audit.read_text())
    examples = audit['populations']['train']['examples']
    diagnostic = collate([dataset[(lookup[tuple(row['sample_id'])], drop)]
                         for row in examples for drop in (False, True)])
    from wam_rl_policy_v3.task_spec import TaskSpec, TaskEvent, EventType, TerminalMode
    from wam_rl_policy_v3.task_scheduler import TaskScheduler
    from wam_rl_policy_v3.task_feedback import InteractionFeedback
    poses = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(4)*.02
    instruction = TaskSpec(poses, np.array([0, 30, 60, 90]), 30., 91, TerminalMode.RELEASE,
        (TaskEvent(30, EventType.CONTACT), TaskEvent(60, EventType.RELEASE)))
    query = np.zeros((32, 3)); query[:, 0] = np.linspace(-.01, .01, 32); query[:, 1] = np.tile([-.01, .01], 16)
    scheduler = TaskScheduler(instruction, query)
    deployment = []
    for now in (0., .5):
        decision = scheduler.update(poses[0], now, InteractionFeedback())
        deployment.append(dict(now=now, phase=int(decision.phase), packet=decision.next_event.wam_condition()))
    scheduler.recovering = True
    deployment.append(dict(case='recovering_acquire', packet=scheduler._event(.5).wam_condition()))
    result = dict(schema='ref2dex.v6.actual-event-input-probe.v1', run_id='tau-contract-20261023-r1',
        entry=str(entry), equivalent=True, reader_options=reader_options,
        training_config=str(args.training_config), entire_batch_tensor_keys_compared=len(actual),
        proof=['same imported V6Windows/collate function objects', 'actual training Steps and torch DataLoader',
               'all collated tensor values equal direct reader path', 'main forwards batch unchanged except device transfer',
               'SharedWAMV6.encode calls event_goal(batch), whose forward calls event_features(batch)'],
        environment=dict(python=sys.version, torch=torch.__version__, device='CPU reader/feature probe only'),
        actual_steps_batch=capture(actual), controlled_eight_examples=capture(diagnostic),
        deployment_causal_task_scheduler_examples=deployment,
        training_updates=0, model_instantiated=False, optimizer_instantiated=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(equivalent=result['equivalent'], compared_tensor_keys=len(actual),
        actual_masks=result['actual_steps_batch']['masks'], examples=result['actual_steps_batch']['examples'][:8],
        controlled_masks=result['controlled_eight_examples']['masks'], deployment=deployment), indent=2))


if __name__ == '__main__': main()
