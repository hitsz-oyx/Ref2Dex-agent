"""Fixed matched H/G trajectory BC with frozen actual-PD supervision."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--retargeter', type=Path, required=True)
    p.add_argument('--physical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--idle-context-pid', type=int, action='append', default=[])
    p.add_argument('--warm', type=Path, help='reference joint checkpoint for matched continuation')
    p.add_argument('--queries', type=Path, help='audited learner-state expert-PD query collection')
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    sm = json.loads((args.source / 'manifest.json').read_text())
    if sm['status'] != 'COMPLETED' or not sm['training_allowed']:
        raise ValueError('eligible official source required')
    for path, digest in sm['input_sha256'].items():
        if sha(path) != digest:
            relative = str(Path(path).resolve().relative_to(ROOT))
            if relative != 'src/task/trajectory-policy/tools/run/collect_competent_pd.py':
                raise ValueError('source input drift: ' + path)
            blob = subprocess.check_output(['git', 'show', sm['git_commit'] + ':' + relative], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError('source producer provenance differs')
    if sha(args.source / 'trajectory.npz') != sm['trajectory_sha256']:
        raise ValueError('source trajectory drift')
    for folder in (args.source, args.retargeter.parent, args.physical):
        if not json.loads((folder / 'audit.json').read_text())['pass']:
            raise ValueError('independent source/R/physical audit required')
    if json.loads((args.physical / 'result.json').read_text())['status'] != 'PROMISING':
        raise ValueError('competent R physical gate required')
    if sha(args.retargeter) != '414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f':
        raise ValueError('fixed competent R required')
    row = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).split()
    util, used = map(int, row.split(','))
    allowed_contexts = set(args.idle_context_pid)
    if set(map(int, pids)) - allowed_contexts or util > (5 if allowed_contexts else 10) or used > 512:
        raise RuntimeError('GPU occupied')
    scratch = ROOT / 'tmp/trajectory-policy/competent-pd'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch), OMP_NUM_THREADS='2')
    import torch
    from trajectory_policy.competent_actor import CompetentActor, SCHEMA, actor_windows
    reference = None
    if args.reference is not None:
        from trajectory_policy.reference_actor import ReferenceActor, SCHEMA, reference_windows, load_reference_actor
        ref_manifest = json.loads((args.reference.parent / 'manifest.json').read_text())
        if sha(args.reference) != ref_manifest['reference_sha256'] or not ref_manifest['planned_reference_only']:
            raise ValueError('authenticated planned reference required')
        with np.load(args.reference, allow_pickle=False) as f:
            reference = {k: f[k].copy() for k in f.files}
        def actor_windows(data, envs):
            return reference_windows(data, envs, reference)
        CompetentActor = ReferenceActor
    from trajectory_policy.pd_retargeter import load_model, command, decode_target
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(183)
    with np.load(args.source / 'trajectory.npz', allow_pickle=False) as f:
        data = {k: f[k].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets', 'goal')}
    groups = dict(train=list(range(16)), validation=list(range(16, 24)), test=list(range(24, 32)))
    arrays = {key: actor_windows(data, ids) for key, ids in groups.items()}
    tensors = {key: [torch.as_tensor(v, device='cuda') for v in values[:4]] for key, values in arrays.items()}
    x, state, tau, target = tensors['train']
    base = CompetentActor().cuda()
    if (args.warm is None) != (args.queries is None) or (args.warm is not None and reference is None):
        raise ValueError('reference warm checkpoint and query collection must be paired')
    queries = None
    if args.warm is not None:
        base = load_reference_actor(torch.load(args.warm, map_location='cuda', weights_only=False), 'cuda')
        qm = json.loads((args.queries / 'manifest.json').read_text())
        if (qm['status'] != 'COMPLETED' or not qm.get('dagger_rollin')
                or not json.loads((args.queries / 'audit.json').read_text())['pass']):
            raise ValueError('audited completed expert-PD query roll-in required')
        from trajectory_policy.pd_retargeter import encode_target
        with np.load(args.queries / 'trajectory.npz', allow_pickle=False) as f:
            query = {k: f[k].copy() for k in ('high_input', 'state_features', 'expert_pd', 'q')}
        queries = {}
        for name, envs in dict(train=np.arange(16, 22), validation=np.arange(22, 24)).items():
            queries[name] = [torch.as_tensor(v, device='cuda') for v in (
                query['high_input'][:, envs - 8].reshape(-1, len(base.input_mean)),
                query['state_features'][:, envs - 4].reshape(-1, 87),
                encode_target(query['expert_pd'][:, envs], query['q'][:-1, envs]).reshape(-1, 12))]
    with torch.no_grad():
        for name, values in (('input', x), ('tau', tau)):
            if args.warm is None:
                getattr(base, name + '_mean').copy_(values.mean(0))
                getattr(base, name + '_scale').copy_(values.std(0, unbiased=False).clamp_min(.001))
    models = dict(tau_only=base, joint=copy.deepcopy(base))
    optimizers = {key: torch.optim.Adam(model.parameters(), lr=3e-4) for key, model in models.items()}
    r = load_model(torch.load(args.retargeter, map_location='cuda', weights_only=False), 'cuda')
    for parameter in r.parameters():
        parameter.requires_grad_(False)
    frozen = {k: v.detach().clone() for k, v in r.state_dict().items()}
    offset = torch.tensor(sm['native_controller']['offset'], device='cuda')
    scale = torch.tensor(sm['native_controller']['scale'], device='cuda')
    files = [Path(__file__), TASK / 'src/trajectory_policy/competent_actor.py',
        TASK / 'src/trajectory_policy/history.py', TASK / 'src/trajectory_policy/pd_retargeter.py',
        ROOT / 'src/task/consequence-evaluator/src/consequence_evaluator/proposal_history.py',
        TASK / 'docs/experiments/probes/P-20261011-competent-joint-bc.md', args.retargeter,
        *[args.source / name for name in ('manifest.json', 'trajectory.npz', 'audit.json')],
        *[args.physical / name for name in ('result.json', 'audit.json')]]
    if reference is not None:
        files += [TASK / 'src/trajectory_policy/reference_actor.py', args.reference,
                  args.reference.parent / 'manifest.json', TASK / 'docs/experiments/probes/P-20261011-reference-goal-dagger.md']
    if args.warm is not None:
        files += [args.warm, *[args.queries / name for name in ('trajectory.npz', 'manifest.json', 'audit.json')]]
    hashes = {str(path.resolve()): sha(path) for path in files}
    out.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status='RUNNING', groups=groups, seed=183, updates=3000,
        batch_size=512, learning_rate=.0003, lambda_pd=.1, actor_period=8,
        sampling='quarter ticks<8; quarter8<=ticks<64; half all', physical_gpu=args.gpu,
        source=str(args.source.resolve()), retargeter=str(args.retargeter.resolve()),
        retargeter_sha256=sha(args.retargeter), input_sha256=hashes, gpu_before=row,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        checkpoint_selection='lowest own validation trajectory MSE + lambda PD MSE every250updates',
        inference='own measured H328 + known goal12 only; no clock/teacher', train_only_normalization=True)
    write(out / 'manifest.json', manifest)
    if reference is not None:
        manifest.update(reference=str(args.reference.resolve()), input_dim=len(base.input_mean), actor_period=1,
            inference='measured H328 + authorized planned human/object pose anchors t+1,t+16',
            experiment_id='P-20261011-reference-goal-dagger')
    if args.warm is not None:
        manifest.update(warm=str(args.warm.resolve()), queries=str(args.queries.resolve()), normalization_frozen=True,
            continuation_arms=dict(tau_only='offline_joint_continuation', joint='PD_DAgger_continuation'))
    write(out / 'manifest.json', manifest)
    best = {key: (float('inf'), None, None) for key in models}
    started = time.monotonic()

    def predict(model, split):
        xx, ss, tt, yy = tensors[split]
        output, pd = [], []
        with torch.no_grad():
            for i in range(0, len(xx), 512):
                value = model(xx[i:i + 512])
                output.append(value)
                pd.append(r(ss[i:i + 512], value))
        output, pd = torch.cat(output), torch.cat(pd)
        lt = float(((output - tt) / model.tau_scale).square().mean())
        lp = float(((pd - yy) / r.target_scale).square().mean())
        return output, pd, lt, lp

    def deadline(signum, frame):
        raise TimeoutError('fixed matched BC fit exceeded300s')

    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        early = torch.as_tensor(np.flatnonzero(arrays['train'][4] < 8), device='cuda')
        approach = torch.as_tensor(np.flatnonzero((arrays['train'][4] >= 8) & (arrays['train'][4] < 64)), device='cuda')
        for step in range(1, 3001):
            ids = torch.cat((early[torch.randint(len(early), (128,), device='cuda')],
                             approach[torch.randint(len(approach), (128,), device='cuda')],
                             torch.randint(len(x), (256,), device='cuda')))
            if queries is not None:
                ids = ids[torch.randperm(len(ids), device='cuda')]
            for name, model in models.items():
                source_ids = ids[:256] if queries is not None and name == 'joint' else ids
                prediction = model(x[source_ids])
                loss = ((prediction - tau[source_ids]) / model.tau_scale).square().mean()
                if name == 'joint' or queries is not None:
                    loss = loss + .1 * ((r(state[source_ids], prediction) - target[source_ids]) / r.target_scale).square().mean()
                if queries is not None and name == 'joint':
                    qx, qs, qy = queries['train']
                    qi = torch.randint(len(qx), (256,), device='cuda')
                    loss = loss + .1 * ((r(qs[qi], model(qx[qi])) - qy[qi]) / r.target_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite BC loss')
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizers[name].step()
            if step == 1 or step % 250 == 0:
                row = dict(update=step, elapsed_s=time.monotonic() - started,
                           eta_s=(time.monotonic() - started) / step * (3000 - step))
                for name, model in models.items():
                    _, _, lt, lp = predict(model, 'validation')
                    objective = lt + (.1 * lp if name == 'joint' or queries is not None else 0)
                    row[name] = dict(objective=objective, tau_mse=lt, pd_mse=lp)
                    if queries is not None and name == 'joint':
                        qx, qs, qy = queries['validation']
                        with torch.no_grad():
                            query_loss = torch.cat([((r(qs[i:i + 256], model(qx[i:i + 256])) - qy[i:i + 256])
                                / r.target_scale).square().mean(-1) for i in range(0, len(qx), 256)]).mean()
                        objective += .1 * float(query_loss)
                        row[name].update(objective=objective, query_pd_mse=float(query_loss))
                    if objective < best[name][0]:
                        best[name] = objective, step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                row['gpu'] = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                    '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                               '--format=csv,noheader,nounits'], text=True).split()
                from collect_competent_pd import gpu_state
                gpu_status = gpu_state(args.gpu)
                foreign = set(gpu_status['pids']) - {os.getpid()}
                if foreign - allowed_contexts or any(gpu_status['process_mib'][pid] > 512 for pid in foreign):
                    raise RuntimeError('foreign GPU process')
                with (out / 'monitor.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
        reports, saved, checkpoint_sha = {}, {}, {}
        for name, model in models.items():
            model.load_state_dict(best[name][2])
            reports[name] = {}
            for split in arrays:
                output, pd, lt, lp = predict(model, split)
                ticks = arrays[split][4]
                envs = np.tile(groups[split], 519)
                current = torch.as_tensor(data['q'][ticks, envs], device='cuda')
                _, _, clipped = command(decode_target(pd, current), current, offset, scale)
                error = (output - tensors[split][2]).cpu().numpy()
                report = dict(tau_mse=lt, pd_mse=lp, clipping_rate=float(clipped.float().mean()))
                for label, mask in (('startup', ticks < 8), ('all', np.ones(len(ticks), bool))):
                    report[label + '_first8_rms_m'] = float(np.sqrt(np.mean(error[mask, :8] ** 2)))
                reports[name][split] = report
                saved[name + '_' + split + '_tau'] = output.cpu().numpy()
                saved[name + '_' + split + '_pd'] = pd.cpu().numpy()
            torch.save(dict(schema=SCHEMA, width=256, model=best[name][2], selected_update=best[name][1],
                retargeter_sha256=sha(args.retargeter), input_sha256=hashes), out / (name + '.pt'))
            checkpoint_sha[name] = sha(out / (name + '.pt'))
        models['joint'].zero_grad(set_to_none=True)
        prediction = models['joint'](x[:32])
        (((r(state[:32], prediction) - target[:32]) / r.target_scale).square().mean()).backward()
        gradient = sum(float(p.grad.abs().sum()) for p in models['joint'].parameters() if p.grad is not None)
        if gradient <= 0 or any(not torch.equal(v, frozen[k]) for k, v in r.state_dict().items()):
            raise ValueError('detached PD objective or modified R')
        screens = {name: value['validation']['startup_first8_rms_m'] <= .02
                   and value['validation']['clipping_rate'] < .01 for name, value in reports.items()}
        np.savez_compressed(out / 'predictions.npz', **saved)
        result = dict(status='UNCLEAR', offline_only=True, reports=reports, physical_screen_pass=screens,
            pd_actor_gradient_l1=gradient, frozen_R_unchanged=True, checkpoint_sha256=checkpoint_sha,
            selected_updates={key: best[key][1] for key in best}, elapsed_s=time.monotonic() - started,
            torch_peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        write(out / 'result.json', result)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('fit input drift')
        manifest.update(status='COMPLETED', elapsed_s=result['elapsed_s'])
        write(out / 'manifest.json', manifest)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        write(out / 'manifest.json', manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    main()
