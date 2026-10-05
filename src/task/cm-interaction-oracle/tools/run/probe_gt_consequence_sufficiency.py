#!/usr/bin/env python3
"""ref8: frozen GT prognosis and arm-consequence transfer, without Cm fitting."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/tools/audit'))
from consequence_sufficiency import readouts, matched_inputs, cluster_gain, supported_ranking, cross_half_bridge, HEADS, PRIMARY
from intervention import per_finger_residuals, PER_FINGER_ARM_NAMES
from probe_interventions import standardized, split_stratified, sha, tensor_hash
from probe_early_hold import dropout_auc
from probe_amplitude_authority import audit_packet
from probe_duration_response import current_design, residual_fit
from audit_finger_amplitudes import finger_audit, markdown

EXPECTED_SHA = '138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149'


def fit(x, y, train, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.Tanh(), nn.Linear(64, 32),
                          nn.Tanh(), nn.Linear(32, y.shape[1])).to(x.device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    losses = []
    for _ in range(200):
        loss = (model(x[train]) - y[train]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite fitting loss')
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    return prediction, {k: v.detach().cpu() for k, v in model.state_dict().items()}, losses, tensor_hash(torch.cat([v.flatten() for v in initial.values()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    def save(name, value):
        (args.run_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    code = [SCRIPT, SCRIPT.with_name('probe_interventions.py'), SCRIPT.with_name('probe_early_hold.py'),
            ROOT / 'src/task/cm-interaction-oracle/src/consequence_sufficiency.py',
            ROOT / 'src/task/cm-interaction-oracle/src/intervention.py']
    code.extend(ROOT / 'src/task/cm-interaction-oracle/tools/audit' / name for name in
                ('probe_amplitude_authority.py', 'probe_duration_response.py', 'audit_finger_amplitudes.py'))
    manifest = dict(run_status='STARTED', command=sys.argv, dataset=str(args.dataset.resolve()),
        dataset_sha256=sha(args.dataset), collection_manifest_sha256=sha(args.dataset.parent / 'manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code}, seeds=[231, 232],
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device='cuda:0',
        torch_version=torch.__version__, created_at=datetime.now(timezone.utc).isoformat())
    save('manifest.json', manifest)
    try:
        if manifest['dataset_sha256'] != EXPECTED_SHA:
            raise ValueError('ref7 input SHA drift')
        collection = json.loads((args.dataset.parent / 'manifest.json').read_text())
        assert collection['run_status'] == 'COMPLETED'
        p = torch.load(args.dataset, weights_only=False, map_location='cpu')
        assert tuple(p['arm_names']) == PER_FINGER_ARM_NAMES and p['finger_range_fraction'] == .2
        audit = audit_packet(p, args.dataset, per_finger_residuals(.2)); save('engineering_audit.json', audit)
        env_count = int(collection['command'][collection['command'].index('--num_envs') + 1])
        assert env_count == 168 and collection['waves'] == 10
        arms, motion = p['arm'].numpy(), p['motion_id'].numpy()
        clusters = p['episode_id'].numpy() % env_count
        waves = p['episode_id'].numpy() // env_count
        half = waves >= 5
        for cluster in np.unique(clusters):
            assert len(np.unique(motion[clusters == cluster])) == 1
        train_np, test_np = split_stratified(motion, clusters, 231)
        assert not set(clusters[train_np]) & set(clusters[test_np])
        assert len(np.unique(p['episode_id'].numpy())) == len(arms)
        e, i, signed, y, details = readouts(p)
        assert details['risk'].all()
        joint = torch.cat((e, i, signed, y), -1).numpy().astype(float)
        design, groups = current_design(p, waves)
        full_beta, _, _, rank = residual_fit(design, arms, joint, 14)
        halves = [residual_fit(design[half == s], arms[half == s], joint[half == s], 14) for s in (False, True)]
        assert rank == 14 and all(v[3] == 14 for v in halves)
        bridge = cross_half_bridge([v[0] for v in halves]); save('arm_bridge.json', bridge)
        save('arm_contrasts.json', dict(names=p['arm_names'][1:], channels=['E12', 'I14', 'signed18', *HEADS],
            full=full_beta.tolist(), halves=[v[0].tolist() for v in halves],
            input_sha256=manifest['dataset_sha256'], limits='Adjusted marginal signed/vector arm-minus-zero estimates, not same-state contrasts'))
        amplitudes = finger_audit(p, design, arms); save('finger_amplitudes.json', amplitudes)
        (args.run_dir / 'finger_amplitudes.md').write_text(markdown(amplitudes))
        support_details = dict(test_trials=len(test_np), train_trials=len(train_np),
            test_environments=len(np.unique(clusters[test_np])), train_environments=len(np.unique(clusters[train_np])),
            test_motion_counts=np.bincount(motion[test_np], minlength=3).tolist(),
            test_arm_counts=np.bincount(arms[test_np], minlength=15).tolist(),
            height_failure_test=int(y[test_np, 6].sum()), height_nonfailure_test=int((y[test_np, 6] == 0).sum()),
            early_failure_test=int(details['early_failure'][test_np].sum()), early_failure_total=int(details['early_failure'].sum()))
        support = (len(test_np) >= 120 and support_details['test_environments'] >= 25 and
            min(support_details['test_motion_counts']) >= 5 and min(support_details['test_arm_counts']) >= 3 and
            min(support_details['height_failure_test'], support_details['height_nonfailure_test']) >= 15)
        save('support.json', support_details)
        device = torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        train = torch.as_tensor(train_np, device=device)
        h_raw = torch.cat((p['history'].flatten(1), p['actor_obs'], p['context'], p['base_action']), -1).to(device)
        hn, hm, hs = standardized(h_raw, train)
        _, _, v = torch.linalg.svd(hn[train], full_matrices=False)
        projection = v[:32].T
        hp, hpm, hps = standardized(hn @ projection, train)
        before, bm, bs = standardized(p['before'].to(device), train)
        h = torch.cat((hp, before), -1)
        a = torch.nn.functional.one_hot(p['arm'].to(device), 15)[:, 1:].float()
        action, am, ass = standardized(a, train)
        effect, em, es = standardized(e.to(device), train)
        interaction, im, iss = standardized(i.to(device), train)
        signed_n, sm, ss = standardized(signed.to(device), train)
        yn, ym, ys = standardized(y.to(device), train)
        inputs = matched_inputs(h, action, effect, interaction, signed_n)
        target_n, target = yn.cpu().numpy(), y.numpy()
        metrics, predictions, models, losses, initial_hashes, errors = {}, {}, {}, {}, {}, {}
        rank_groups = motion * 4 + np.minimum((p['context'][:, 3].numpy() * 4).astype(int), 3)
        for name, x in inputs.items():
            normalized, state, trace, initial = fit(x, yn, train, 231)
            prediction = normalized * ys.cpu().numpy() + ym.cpu().numpy()
            assert np.isfinite(prediction).all()
            error = (normalized - target_n) ** 2
            errors[name] = error[test_np]
            metrics[name] = dict(test_normalized_mse_per_head=error[test_np].mean(0).tolist(),
                test_primary_mse=float(error[test_np][:, PRIMARY].mean()), train_mse=float(error[train_np].mean()),
                test_all_heads_mse=float(error[test_np].mean()), test_MAE_per_head=np.abs(prediction[test_np] - target[test_np]).mean(0).tolist(),
                physical_failure_auc=dropout_auc(prediction[test_np, 6], target[test_np, 6]),
                physical_failure_brier=float(((np.clip(prediction[test_np, 6], 0, 1) - target[test_np, 6]) ** 2).mean()),
                ranking=supported_ranking(prediction, target, test_np, rank_groups, arms),
                last50_train_loss_fractional_change=float(1 - trace[-1] / trace[-50]),
                parameters=sum(v.numel() for v in state.values()))
            predictions[name], models[name], losses[name], initial_hashes[name] = prediction, state, trace, initial
            print(json.dumps(dict(model=name, primary_mse=metrics[name]['test_primary_mse'], physical_auc=metrics[name]['physical_failure_auc'])), flush=True)
        assert len(set(initial_hashes.values())) == 1
        comparisons = {}
        for base, new in (('H', 'Ha'), ('H', 'HE'), ('H', 'HI'), ('H', 'HEI'), ('HEI', 'HaEI'),
                          ('H', 'HI_signed'), ('H', 'HEI_signed'), ('HEI_signed', 'HaEI_signed')):
            comparisons[f'{new}_vs_{base}'] = {axis: cluster_gain(
                errors[base][:, indices].mean(1), errors[new][:, indices].mean(1), clusters[test_np], 232)
                for axis, indices in (('primary', PRIMARY), ('physical_failure', (6,)), ('contact', (3,)))}
        gt = comparisons['HEI_vs_H']
        prognosis = (gt['primary']['gain'] >= .10 and gt['physical_failure']['gain'] >= .05 and
                     min(gt['primary']['lower95'], gt['physical_failure']['lower95']) > 0)
        increment = comparisons['HaEI_vs_HEI']
        small_action = all(increment[axis]['gain'] <= .02 and increment[axis]['upper_one_sided95'] <= .05
                           for axis in ('primary', 'physical_failure'))
        arm_positive = all(bridge['EI']['gain_vs_zero'][j] >= .10 and bridge['EI']['gain_vs_source_mean'][j] >= .10
                           and bridge['EI']['correlation'][j] is not None and bridge['EI']['correlation'][j] >= .30 for j in (3, 6))
        chain_status = 'UNCLEAR'
        if support and not prognosis:
            chain_status = 'UNPROMISING'
        elif support and prognosis and small_action and arm_positive:
            chain_status = 'PROMISING'
        result = dict(status=chain_status, GT_prognosis_status='PROMISING' if support and prognosis else ('UNPROMISING' if support else 'UNCLEAR'),
            support_pass=bool(support), GT_prognosis_pass=bool(prognosis), small_remaining_action=bool(small_action),
            arm_bridge_pass=bool(arm_positive), support=support_details, comparisons=comparisons,
            models=metrics, heads=HEADS, primary_axes=PRIMARY, initial_weight_hashes=initial_hashes,
            input_width=next(iter(inputs.values())).shape[1], elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device), Cm_executed=False,
            limits='Fixed-fit GT prognosis and noisy aggregate action contrasts; no identified causal mediation, deployable same-state ranking or trained-policy utility')
        # Disclose already-failing-state contribution without selecting the causal cohort.
        survivor = ~details['early_failure'].numpy()[test_np]
        result['descriptive_no_early_failure'] = dict(trials=int(survivor.sum()),
            physical_failure=int(target[test_np[survivor], 6].sum()),
            primary_mse={name: float(value[survivor][:, PRIMARY].mean()) for name, value in errors.items()})
        save('result.json', result); save('loss_traces.json', losses)
        torch.save(dict(train=train_np, test=test_np, clusters=clusters, waves=waves, half=half, motion=motion,
            arms=arms, target=target, normalized_target=target_n, E=e, I=i, signed=signed,
            predictions=predictions, models=models, initial_hashes=initial_hashes,
            normalizers={key: value.detach().cpu() for key, value in dict(h_mean=hm, h_scale=hs, projection=projection,
                hp_mean=hpm, hp_scale=hps, before_mean=bm, before_scale=bs, a_mean=am, a_scale=ass,
                e_mean=em, e_scale=es, i_mean=im, i_scale=iss, signed_mean=sm, signed_scale=ss,
                y_mean=ym, y_scale=ys).items()}, design=design, groups=groups,
            full_contrasts=full_beta, half_contrasts=[v[0] for v in halves]), args.run_dir / 'diagnostic.pt')
        assert sha(args.dataset) == EXPECTED_SHA
        assert all(sha(ROOT / key) == value for key, value in manifest['code_sha256'].items())
        manifest['run_status'] = 'COMPLETED'
        print(json.dumps({k: result[k] for k in ('status', 'GT_prognosis_status', 'support_pass', 'small_remaining_action', 'arm_bridge_pass')}, indent=2))
    except BaseException as error:
        manifest.update(run_status='FAILED', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        manifest['elapsed_seconds'] = time.monotonic() - started
        save('manifest.json', manifest)


if __name__ == '__main__':
    main()
