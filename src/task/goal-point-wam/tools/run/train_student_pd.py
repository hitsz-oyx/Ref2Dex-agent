"""One robot adaptation round: demonstration reference, frozen competent R, no effect loss.

Student states supervise the expert's current PD only. Nominal teacher windows
also supervise that executed hand trajectory. A small human set limits forgetting.
"""
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
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/trajectory-policy/src'),
               str(ROOT/'src/task/cm-pointflow-effect-pretrain/src'),
               str(ROOT/'src/task/consequence-evaluator/src')]
from goal_point_wam.data import GoalWindows
from goal_point_wam.reference_adapter import ReferenceStudentWAM
from oakink_wm.multisource import MixedWindows
from oakink_wm.pointworld_performance import install_fused_hilbert
from trajectory_policy.pd_retargeter import load_model
from trajectory_policy.student_pd import action_loss, native_tau, trajectory_loss
from trajectory_policy.student_reference_data import nominal_sample, reference_collate, student_sample
from trajectory_policy.wam_shadow import WAM_TO_NATIVE

SCHEMA = 'ref2dex.student-pd-adapter.v1'


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1000)
    args = parser.parse_args()
    out = args.output.resolve()
    student_dir = args.student.resolve()
    if out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents:
        raise ValueError('fresh owned output required')
    if not 1 <= args.updates <= 2000:
        raise ValueError('this round is capped at 2000 updates')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=False)
    os.environ.update(TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch/'triton'))
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(295)
    torch.backends.cuda.matmul.allow_tf32 = True
    install_fused_hilbert()
    started = time.monotonic()
    base = ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    retargeter = ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    corpus = ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    def load_npz(path):
        with np.load(path, allow_pickle=False) as stream:
            return {key: stream[key].copy() for key in stream.files}
    reference = load_npz(ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz')
    canonical = load_npz(ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz')
    nominal = load_npz(ROOT/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2/trajectory.npz')
    manifest_in = json.loads((student_dir/'manifest.json').read_text())
    if not manifest_in.get('record_expert_pd') or manifest_in.get('student_expert_future_hand_stored') is not False:
        raise ValueError('student collection must record expert PD and store no future-hand label')
    visited = load_npz(student_dir/'trajectory.npz')
    if visited['student_expert_pd'].shape[0] != manifest_in['controls']:
        raise ValueError('expert PD must cover every actor control and nothing beyond it')
    stats = json.loads((corpus.parent/'norm_stats.json').read_text())
    state = torch.load(base, map_location='cpu', weights_only=False)
    model = ReferenceStudentWAM(stats, state['config']).cuda().eval()
    fresh = {k: v for k, v in model.state_dict().items() if k.startswith('reference_adapter.')}
    model.load_state_dict(dict(state['model'], **fresh), strict=True)
    del state
    retargeter_packet = torch.load(retargeter, map_location='cpu', weights_only=False)
    frozen_r = load_model(retargeter_packet, 'cuda')
    for parameter in frozen_r.parameters():
        parameter.requires_grad_(False)
    order = torch.tensor(WAM_TO_NATIVE, device='cuda')
    human_base = GoalWindows(MixedWindows(corpus, 'train'), 'object_points', 32)
    rng = np.random.default_rng(295)
    human_ids = rng.choice(len(human_base), 32, replace=False)

    def human_sample(index):
        sample = human_base[int(index)]
        sample.update(proprio=np.zeros(48, np.float32), reference_goal=np.zeros(408, np.float32),
                      robot_mask=np.bool_(False), native_state=np.zeros(87, np.float32),
                      pd_target=np.zeros(12, np.float32), has_pd=np.bool_(False),
                      tau_supervised=np.bool_(True))
        return sample

    student_train = [('student', t, 2) for t in range(0, 400, 2)]
    student_val = [('student', t, 2) for t in range(400, 530, 2)]
    nominal_train = [('nominal', t, e) for t in range(0, 128, 8) for e in range(8)]
    keys = dict(student=student_train, nominal=nominal_train, human=[('human', int(i), 0) for i in human_ids])

    def build(kind, tick, role):
        if kind == 'student':
            return student_sample(visited, reference, canonical, tick, role)
        if kind == 'nominal':
            return nominal_sample(nominal, reference, canonical, tick, role)
        return human_sample(tick)

    def cache(pairs):
        contexts, batches = [], []
        for begin in range(0, len(pairs), 8):
            batch = {k: v.cuda() for k, v in reference_collate(
                [build(*pair) for pair in pairs[begin:begin + 8]]).items()}
            contexts.append(model.contexts(batch))
            batches.append({k: batch[k] for k in (
                'trajectory_target', 'trajectory_valid', 'proprio', 'reference_goal', 'robot_mask',
                'native_state', 'pd_target', 'has_pd', 'tau_supervised')})
            if time.monotonic() - started > 1500:
                raise TimeoutError('context preparation exceeded 1500s')
        return contexts, batches

    pools = {name: cache(pairs) for name, pairs in keys.items()}
    val_pool = cache(student_val)
    if any(bool(batch['tau_supervised'].any()) for batch in pools['student'][1]):
        raise ValueError('student pool is supervising a hand trajectory')
    freeze = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith(('reference_adapter.', 'trajectory.'))}
    r_freeze = {k: v.detach().cpu().clone() for k, v in frozen_r.state_dict().items()}
    write(out/'manifest.json', dict(schema=SCHEMA, status='RUNNING', task='goal-point-wam',
        experiment_id='P-20261013-goal-point-wam-student-pd', run_id=out.name,
        base_checkpoint=str(base), retargeter=str(retargeter), student_collection=str(student_dir),
        effect_loss=False, student_future_hand_supervised=False,
        reference_offsets=[1, 16], updates=args.updates, mix=dict(student=16, nominal=8, human=4),
        training_allowed=True))

    def wrist_mm(predicted, target, has_pd):
        delta = torch.linalg.vector_norm((predicted - target)[:, :3], dim=-1) * 1000
        return float(delta[has_pd].mean()) if bool(has_pd.any()) else None

    @torch.no_grad()
    def evaluate(step):
        values = []
        for context, batch in zip(*val_pool):
            prediction = model.actor(context, batch)
            command = frozen_r(batch['native_state'], native_tau(prediction, order))
            values.append(wrist_mm(command, batch['pd_target'], batch['has_pd']))
        report = dict(step=step, student_val_wrist_mm=float(np.mean(values)))
        write(out/f'evaluation_step{step}.json', report)
        return report

    def save(step, report):
        payload = dict(schema=SCHEMA, step=step, report=report,
                       adapter=model.reference_adapter.state_dict(),
                       trajectory=model.trajectory.state_dict(),
                       base_checkpoint=str(base), retargeter=str(retargeter))
        path = out/f'step{step}.pt'
        torch.save(payload, path)
        return payload

    initial = evaluate(0)
    best = (float('inf'), args.updates, None)
    parameters = [{'params': model.reference_adapter.parameters(), 'lr': 1e-4},
                  {'params': [p for p in model.trajectory.parameters() if p.requires_grad], 'lr': 1e-5}]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    counts = dict(student=16, nominal=8, human=4)
    draw = np.random.default_rng(295)
    for step in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for name, count in counts.items():
            contexts, batches = pools[name]
            choice = int(draw.integers(0, len(batches)))
            context, batch = contexts[choice], batches[choice]
            take = torch.as_tensor(draw.choice(len(batch['proprio']), size=min(count, len(batch['proprio'])),
                                               replace=False), device='cuda')
            batch = {k: v[take] for k, v in batch.items()}
            context = tuple(part[take] if torch.is_tensor(part) else
                            {k: v[take] for k, v in part.items()} for part in context)
            prediction = model.actor(context, batch)
            command = frozen_r(batch['native_state'], native_tau(prediction, order))
            loss = (trajectory_loss(prediction, batch['trajectory_target'], batch['trajectory_valid'],
                                    batch['tau_supervised'])
                    + action_loss(command, batch['pd_target'], batch['has_pd']))
            if step == 1 and name == 'student':
                actor_parameter = next(p for p in model.trajectory.parameters() if p.requires_grad)
                actor_grad = torch.autograd.grad(loss, actor_parameter, retain_graph=True)[0]
                adapter_grad = torch.autograd.grad(loss, model.reference_adapter.alpha_g, retain_graph=True)[0]
                if float(actor_grad.abs().sum()) == 0. or float(adapter_grad.abs().sum()) == 0.:
                    raise ValueError('student PD loss did not reach the actor and reference adapter')
            if not torch.isfinite(loss):
                raise FloatingPointError('student PD loss nonfinite')
            (loss * count / sum(counts.values())).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step % 50 == 0 or step == args.updates:
            if any(not torch.equal(model.state_dict()[k].detach().cpu(), v) for k, v in freeze.items()):
                raise ValueError('scene or World weights moved')
            if any(not torch.equal(frozen_r.state_dict()[k].detach().cpu(), v) for k, v in r_freeze.items()):
                raise ValueError('competent R moved')
        if step == 1 or step % 50 == 0:
            record = dict(step=step, loss=total, elapsed_s=time.monotonic() - started,
                          gpu_mib=torch.cuda.max_memory_allocated() / 2**20)
            print(json.dumps(record), flush=True)
            write(out/'progress.json', record)
        if step % 250 == 0 or step == args.updates:
            report = evaluate(step)
            payload = save(step, report)
            if report['student_val_wrist_mm'] < best[0]:
                best = (report['student_val_wrist_mm'], step, payload)
                torch.save(payload, out/'selected.pt')
    final = json.loads((out/'manifest.json').read_text())
    final.update(status='COMPLETED', selected_step=best[1], selected_wrist_mm=best[0],
                 initial_wrist_mm=initial['student_val_wrist_mm'], elapsed_s=time.monotonic() - started,
                 offline_gate=False, rollout_required=True)
    write(out/'manifest.json', final)
    write(out/'result.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
