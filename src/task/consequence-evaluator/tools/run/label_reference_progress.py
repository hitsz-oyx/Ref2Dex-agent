"""Bounded train-only ref4_1 label Probe on preserved continuous episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, K, is_within
from consequence_evaluator.data import sha
from consequence_evaluator.reference_motion import REFERENCE_SCHEMA
from consequence_evaluator.reference_progress import (SCHEMA, RULE, GROUPS, AlignmentConfig,
    ReferenceProgress, trajectory_features, label_window)
from consequence_evaluator.value_outcomes import RAW_SCHEMA, validate_plan_execution
from consequence_evaluator.xirl_alignment import UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--episode-limit', type=int, default=8)
    parser.add_argument('--seconds', type=int, default=180)
    parser.add_argument('--phase-encoder', type=Path, help='completed frozen TCC alignment run; never an evaluator')
    args = parser.parse_args()
    out = args.output.resolve(); source = args.source.resolve(); ref_dir = args.reference.resolve()
    if (out.exists() or not is_within(out, ROOT / 'outputs/consequence-evaluator')
            or not 2 <= args.episode_limit <= 64 or not 0 <= args.gpu <= 7 or not 10 <= args.seconds <= 900):
        parser.error('fresh owned output, 2..64 train episodes, GPU0..7 and <=900s required')
    occupied = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if occupied > 256:
        raise RuntimeError('selected physical GPU is occupied; do not interfere')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    frozen = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ('reference_progress.py', 'xirl_alignment.py', 'reference_motion.py', 'data.py', 'contracts.py', 'value_outcomes.py'):
        path = TASK / 'src/consequence_evaluator' / name; frozen[str(path)] = sha(path)
    raw_path = source / 'manifest.json'; ref_path = ref_dir / 'manifest.json'
    raw = json.loads(raw_path.read_text()); ref_meta = json.loads(ref_path.read_text())
    if (raw.get('schema') != RAW_SCHEMA or raw.get('status') != 'COMPLETED'
            or raw.get('rollout_kind') != 'continuous' or raw.get('audit_only') is not False
            or raw.get('horizon') != K or raw.get('execution_horizon') != K
            or raw.get('fps') != 30 or raw.get('units') != 'm'
            or ref_meta.get('schema') != REFERENCE_SCHEMA or ref_meta.get('status') != 'COMPLETED'
            or ref_meta.get('origin') != 'original_successful_retargeted_motion'
            or ref_meta.get('hand_links') != list(HAND_LINKS)):
        raise ValueError('completed original-reference/full continuous raw contracts required')
    frozen.update({str(raw_path): sha(raw_path), str(ref_path): sha(ref_path)})
    frozen.update(raw['sources']); frozen.update(ref_meta['sources'])
    file = ref_dir / 'reference.npz'; frozen[str(file)] = ref_meta['reference_sha256']
    if any(sha(p) != value for p, value in frozen.items()):
        raise ValueError('reference/controller/source drift')
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps(dict(schema=SCHEMA, rule=RULE,
        status='RUNNING', training_allowed=False, run_id=out.name, pid=os.getpid(),
        seconds_budget=args.seconds, source=str(source), reference=str(ref_dir),
        physical_gpu=args.gpu, sources=frozen,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()), indent=2) + '\n')
    with np.load(file, allow_pickle=False) as reference:
        ref_features = trajectory_features(reference['object_pose'], reference['hand_keypoints'], reference['timestamps'])
        ref_height = reference['object_pose'][:, 2, 3].copy()
    config = AlignmentConfig(); matcher = ReferenceProgress(ref_features, config, 'cuda:0')
    if args.phase_encoder is not None:
        from consequence_evaluator.temporal_phase import ENCODER_SCHEMA, LearnedReferenceProgress
        run = args.phase_encoder.resolve(); path = run / 'manifest.json'
        trained = json.loads(path.read_text())
        checkpoint = Path(trained['checkpoint']).resolve()
        if (not is_within(run, ROOT / 'outputs/consequence-evaluator') or not is_within(checkpoint, run)
                or trained.get('schema') != ENCODER_SCHEMA or trained.get('status') != 'COMPLETED'
                or trained.get('training_source_seed') != 230 or raw.get('seed') == 230
                or trained.get('reference_sha256') != ref_meta['reference_sha256']):
            raise ValueError('independent frozen phase-only encoder and exact original reference required')
        frozen.update(trained['sources']); frozen[str(path)] = sha(path)
        frozen[str(checkpoint)] = trained['checkpoint_sha256']
        for name in ('temporal_phase.py', 'xirl_tcc_loss.py'):
            file = TASK / 'src/consequence_evaluator' / name; frozen[str(file)] = sha(file)
        if any(sha(p) != h for p, h in frozen.items()):
            raise ValueError('trained phase encoder/source drift')
        matcher = LearnedReferenceProgress(ref_features, torch.load(checkpoint, map_location='cpu', weights_only=False), 'cuda:0')
        config = matcher.config
    self_trace = matcher.align(ref_features)
    stationary = np.repeat(ref_features[:1], len(ref_features), axis=0)
    stationary[:, 45:] = 0  # all motion channels are zero
    stationary_trace = matcher.align(stationary)
    # Balanced by assigned intervention only, never by future success or Y.
    eligible = [r for r in raw['episodes'] if r['split'] == 'train']
    clean = [r for r in eligible if r['perturbation_tick'] < 0]
    intervention = [r for r in eligible if r['perturbation_tick'] >= 0]
    records = []
    for n in range(args.episode_limit):
        group = clean if n % 2 == 0 else intervention
        if n // 2 >= len(group):
            raise ValueError('insufficient balanced train-only raw episodes')
        records.append(group[n // 2])
    windows = []; audits = []; traces = []; plots = []; prefix_error = 0.
    for record in records:
        if time.monotonic() - started > args.seconds:
            raise TimeoutError('bounded progress-label Probe deadline')
        if record['task'] != ref_meta['task'] or record['motion'] != ref_meta['motion']:
            raise ValueError('task/motion must have an explicit matching reference')
        path = source / record['path']
        if not is_within(path, source) or sha(path) != record['sha256']:
            raise ValueError('raw episode drift or path escape')
        frozen[str(path)] = record['sha256']
        with np.load(path, allow_pickle=False) as f:
            packet = {key: f[key] for key in f.files}
        if (len(packet['action']) != record['steps']
                or not np.allclose(packet['timestamps'], np.arange(record['steps'] + 1) / 30, atol=1e-9, rtol=0)):
            raise ValueError('full native30Hz packet clock mismatch')
        validate_plan_execution(packet, record, raw.get('requested_residual_bound', .2))
        features = trajectory_features(packet['object_pose'], packet['hand_keypoints'], packet['timestamps'])
        trace = matcher.align(features)
        tick = max(record['perturbation_tick'], 48); end = tick + K
        prefix = matcher.align(trajectory_features(packet['object_pose'][:end + 1],
            packet['hand_keypoints'][:end + 1], packet['timestamps'][:end + 1]))
        prefix_error = max(prefix_error, float(np.abs(trace['distribution'][:end + 1] - prefix['distribution']).max()))
        ticks = set(range(0, record['steps'] - K + 1, 8))
        if record['perturbation_tick'] >= 0:
            ticks.add(record['perturbation_tick'])
        local = []
        for tick in sorted(ticks):
            if not packet['plan_known'][tick]:
                continue
            label = label_window(trace, tick, config)
            windows.append(dict(episode=record['episode'], tick=tick, split='train',
                split_group=record['split_group'], **label))
            local.append(label)
        step = np.diff(trace['progress'])
        audit = dict(episode=record['episode'], assignment=record['assigned_phase'],
            perturbation_tick=record['perturbation_tick'], windows=len(local),
            final_progress=float(trace['progress'][-1]),
            total_negative_progress=float(-np.minimum(step, 0).sum()),
            max_progress_step=float(np.abs(step).max()),
            median_matched_cost=float(np.median(trace['matched_cost'])),
            max_matched_cost=float(trace['matched_cost'].max()),
            valid_window_fraction=float(np.mean([v['match_valid'] for v in local])),
            positive_windows=sum(v['value'] > config.epsilon for v in local),
            stagnant_windows=sum(abs(v['value']) <= config.epsilon for v in local),
            negative_windows=sum(v['value'] < -config.epsilon for v in local))
        if record['perturbation_tick'] >= 0:
            audit['intervention_value'] = label_window(trace, record['perturbation_tick'], config)
        audits.append(audit); traces.append((record['episode'], trace))
        plots.append((record, packet['object_pose'][:, 2, 3], trace))
        print(json.dumps(audit), flush=True)
    self_expected = np.linspace(0, 1, len(ref_features))
    controls = dict(reference_self_final=float(self_trace['progress'][-1]),
        reference_self_mean_index_error=float(np.abs(self_trace['progress'] - self_expected).mean()),
        stationary_final=float(stationary_trace['progress'][-1]), prefix_distribution_max_error=prefix_error,
        max_allowed_progress_step=config.max_step / (len(ref_features) - 1))
    if prefix_error > 1e-9 or any(r['max_progress_step'] > controls['max_allowed_progress_step'] + 1e-9 for r in audits):
        raise ValueError('causality or bounded posterior progress contract failed')
    if any(sha(p) != value for p, value in frozen.items()):
        raise ValueError('input/source changed during labeling')
    (out / 'traces').mkdir()
    for name, trace in traces:
        np.savez_compressed(out / 'traces' / (name + '.npz'), **trace)
    np.savez_compressed(out / 'reference-controls.npz', progress=self_trace['progress'],
        stationary_progress=stationary_trace['progress'], reference_index=self_expected)
    np.savez_compressed(out / 'labels.npz', **{key: np.asarray([w[key] for w in windows]) for key in windows[0]})
    scratch = ROOT / 'tmp/consequence-progress-plots'; scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(MPLCONFIGDIR=str(scratch), TMPDIR=str(ROOT / 'tmp'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, min(8, len(plots)), figsize=(4 * min(8, len(plots)), 6), squeeze=False)
    for column, (record, height, trace) in enumerate(plots[:8]):
        seconds = np.arange(len(height)) / 30
        axes[0, column].plot(seconds, trace['progress'], label='causal reference progress')
        axes[0, column].set_ylim(-.02, 1.02)
        axes[0, column].set_title(record['episode'] + '\n' + record['assigned_phase'], fontsize=8)
        axes[1, column].plot(seconds, height - height[0], label='measured object height')
        axes[1, column].plot(np.arange(len(ref_height)) / 30, ref_height - ref_height[0], alpha=.5,
                             label='original ref height (diagnostic only)')
        for ax in axes[:, column]:
            if record['perturbation_tick'] >= 0:
                tick = record['perturbation_tick']; ax.axvspan(tick / 30, (tick + K) / 30, color='orange', alpha=.2)
            ax.grid(alpha=.2); ax.legend(fontsize=6)
        axes[1, column].set_xlabel('time (s), diagnostic axis only')
    fig.tight_layout(); fig.savefig(out / 'train-progress-examples.png', dpi=130); plt.close(fig)
    if time.monotonic() - started > args.seconds or any(sha(p) != value for p, value in frozen.items()):
        raise ValueError('label deadline exceeded or final source guard failed')
    manifest = dict(schema=SCHEMA, rule=RULE, status='COMPLETED', training_allowed=False,
        run_id=out.name, task='consequence-evaluator', physical_gpu=args.gpu,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        phase='offline label validation only; no evaluator or policy fit',
        label_definition='Y_t=P(H_0:t+24|R)-P(H_0:t|R)', horizon=K,
        action_semantics='decision-known requested24 residual, frozen controller feedback during execution',
        alignment=config.dictionary(), features=[list(group) for group in GROUPS],
        filter='log-domain bounded transitions; minimum-KL posterior mean-step projection',
        standardizer=matcher.normalizer.dictionary(), upstream_xirl_commit=UPSTREAM_COMMIT,
        phase_encoder=str(args.phase_encoder.resolve()) if args.phase_encoder else None,
        reference=str(ref_dir), sources=frozen, episode_audit=audits, control_audit=controls,
        labels_sha256=sha(out / 'labels.npz'),
        counts=dict(episodes=len(audits), windows=len(windows), preferences=0),
        ranking='difference > epsilon; abstain ties; no unmatched-state pair generation',
        semantic_gate='pending inspection of nominal/failure trajectories against original reference',
        resource=dict(elapsed_s=time.monotonic() - started,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=manifest['status'], controls=controls, resource=manifest['resource'])), flush=True)


if __name__ == '__main__':
    main()
