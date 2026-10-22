"""CPU-only stopped-FM checkpoint/curve audit. Never signals or resumes a job.

Reads checkpoint metadata using mmap + FakeTensorMode, so optimizer storage is
not materialized. Real CPU reads are limited to small model sentinel tensors.
Original process/training manifests, checkpoints and links are never changed.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SHANGHAI = timezone(timedelta(hours=8))
WEIGHT_SUFFIXES = {'.pt', '.pth', '.ckpt'}


def timestamp(value):
    return dict(unix=value, utc=datetime.fromtimestamp(value, timezone.utc).isoformat(),
                asia_shanghai=datetime.fromtimestamp(value, SHANGHAI).isoformat())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def capture(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return dict(command=command, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def metadata(path, torch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model = saved.get('model', {})
    return dict(schema=saved.get('schema'), step=saved.get('step'),
                elapsed_s=saved.get('elapsed_s'), world_size=saved.get('world_size'),
                objective_semantics=saved.get('objective_semantics'),
                execution_contract=saved.get('execution_contract'),
                data_identity=saved.get('data_identity'),
                model_items=len(model), model_numel=sum(v.numel() for v in model.values()),
                model_bytes=sum(v.numel()*v.element_size() for v in model.values()),
                flow_parameter_items=sum(k.startswith('flow_future.') for k in model),
                optimizer_state_items=len(saved.get('optimizer', {}).get('state', {})),
                rng_rank_states=len(saved.get('random_state', [])),
                initialization_binding=saved.get('initialization_binding'),
                resume_binding=saved.get('resume_binding'), selection=saved.get('selection'),
                tensor_examples={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                 for k, v in model.items() if k in
                                 ('flow_future.output.weight', 'flow_future.output.bias',
                                  'hand_future.mlp.3.weight')},
                read_mode='torch.load CPU mmap + FakeTensorMode; optimizer storage not materialized')


def sentinels(path, torch):
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    result = {}
    for key in ('flow_future.output.weight', 'flow_future.output.bias', 'hand_future.mlp.3.weight'):
        value = saved['model'][key].detach().float().clone()
        result[key] = value
    return result


def inspect_processes(run, process):
    known = {int(process['supervisor_pid'])}
    known.update(int(s['pid']) for s in process['stages'] if s.get('pid'))
    # The SIGINT log explicitly names these three main training workers.
    log = (run/'training.log').read_text(errors='replace')
    known.update(int(v) for v in re.findall(r'Sending process (\d+) closing signal SIGINT', log))
    alive = []
    for pid in sorted(known):
        path = Path('/proc')/str(pid)/'cmdline'
        try:
            argv = path.read_bytes().decode(errors='replace').replace('\0', ' ').strip()
            alive.append(dict(pid=pid, exists=True, matches_this_run=run.name in argv, command=argv))
        except (FileNotFoundError, ProcessLookupError):
            alive.append(dict(pid=pid, exists=False, matches_this_run=False))
    # Capture only command lines of training/supervisor processes for this run,
    # not other users' unrelated commands or environment variables.
    ps = capture(['ps', '-eo', 'pid,ppid,pgid,sid,etimes,args'])
    matching = [line for line in ps['stdout'].splitlines()
                if run.name in line and ('train_pretrain.py' in line or 'launch_flow_overnight.py' in line)]
    gpu = capture(['nvidia-smi', '-i', '4,5,6',
                   '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    apps = capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                    '--format=csv,noheader,nounits'])
    uuids = {line.split(',')[1].strip() for line in gpu['stdout'].splitlines() if ',' in line}
    matched_apps = [line for line in apps['stdout'].splitlines() if line.split(',')[0].strip() in uuids]
    apps['stdout'] = '\n'.join(matched_apps)+'\n' if matched_apps else ''
    gpu_rows = []
    for line in gpu['stdout'].splitlines():
        fields = [s.strip() for s in line.split(',')]
        if len(fields) == 5:
            gpu_rows.append(dict(index=int(fields[0]), uuid=fields[1], used_mib=int(fields[2]),
                                 free_mib=int(fields[3]), utilization_percent=int(fields[4])))
    evidence = [line for line in log.splitlines() if any(s in line for s in
                ('Received Signals.SIGINT', 'closing signal SIGINT', 'KeyboardInterrupt', 'got signal: 2'))]
    return dict(observed_at=timestamp(time.time()), known_pids=alive,
                matching_live_training_processes=matching,
                owned_training_stopped=not matching and not any(p['matches_this_run'] for p in alive),
                gpu_query=gpu, compute_apps_query=apps, gpu_rows=gpu_rows,
                gpus456_released=bool(len(gpu_rows) == 3 and not matched_apps and
                                      all(r['used_mib'] <= 8 and r['utilization_percent'] == 0 for r in gpu_rows)),
                sigint_log_evidence=evidence,
                reported_stop='Owner relayed Hermes SIGINT at 2026-10-22 12:29 Asia/Shanghai; signal sender is not independently identified by the training log',
                trainer_process_status=process['status'],
                training_stage=[s for s in process['stages'] if s['name'] == 'training'][0],
                original_process_file_mtime=timestamp((run/'process.json').stat().st_mtime),
                original_progress_status=read(run/'training/progress.json')['status'],
                note='Read-only observation. No signal, STOP file, restart or original manifest edit was performed.')


def inventory(run, related_roots, torch, owner_step):
    paths = set(p for p in run.rglob('*') if p.suffix in WEIGHT_SUFFIXES or p.name.endswith('.pt.part'))
    related = []
    for root in related_roots:
        if not root.exists():
            related.append(dict(root=str(root), exists=False)); continue
        checked = 0
        for p in root.rglob('*'):
            if p.suffix not in WEIGHT_SUFFIXES: continue
            checked += 1
            # Associate names, symlinks resolving into this run, or the owner's
            # exact checkpoint step. Other WAM/RL models are outside this audit.
            resolved = p.resolve()
            if (run == resolved or run in resolved.parents or 'flow' in p.name.lower()
                    or re.search(r'(?:step0*|\D)'+str(owner_step)+r'(?:\D|$)', p.name)):
                paths.add(p)
        related.append(dict(root=str(root), exists=True, checkpoint_paths_inspected=checked))
    payload_cache = {}
    rows = []
    for path in sorted(paths, key=str):
        stat = path.lstat()
        row = dict(path=str(path), is_symlink=path.is_symlink(),
                   link_target=os.readlink(path) if path.is_symlink() else None,
                   link_or_file_mtime=timestamp(stat.st_mtime), link_or_file_size=stat.st_size,
                   resolved=str(path.resolve()), exists=path.exists(),
                   within_original_run=run == path or run in path.parents)
        if path.exists():
            target_stat = path.stat()
            key = (target_stat.st_dev, target_stat.st_ino)
            row.update(bytes=target_stat.st_size, target_mtime=timestamp(target_stat.st_mtime),
                       device=target_stat.st_dev, inode=target_stat.st_ino)
            if key not in payload_cache:
                try:
                    payload_cache[key] = metadata(path, torch)
                except Exception as error:
                    payload_cache[key] = dict(read_error=repr(error))
            row['payload'] = payload_cache[key]
        rows.append(row)
    return rows, related


def curves(run):
    logs = [json.loads(line) for line in (run/'training/train.jsonl').read_text().splitlines() if line.strip()]
    dev = []
    for path in sorted((run/'training').glob('dev-step*.json')):
        packet = read(path)
        grab = packet.get('sources', {}).get('1', {})
        metrics = grab.get('metrics', {})
        hand_prefix = [metrics.get('hand/h%02d'%h, {}).get('mean') for h in range(1, 9)]
        object_prefix = [metrics.get('scene/h%02d'%h, {}).get('mean') for h in range(1, 9)]
        step = int(path.stem.split('step')[1])
        nearest = min(logs, key=lambda x: abs(x['step']-step))
        dev.append(dict(step=step, cumulative_hours=nearest['elapsed_s']/3600,
                        nearest_train_step=nearest['step'], path=str(path),
                        mtime=timestamp(path.stat().st_mtime),
                        source_macro_objective=packet.get('source_macro_objective'),
                        grab_hand_prefix8_mm=sum(hand_prefix)/8 if all(v is not None for v in hand_prefix) else None,
                        grab_object_prefix8_mm=sum(object_prefix)/8 if all(v is not None for v in object_prefix) else None,
                        grab_geometric_contact_mae=metrics.get('geometry_gt/contact_mae', {}).get('mean'),
                        grab_contact_component=grab.get('components', {}).get('contact')))
    by_step = {x['step']: x for x in logs}
    anchors = {str(s): by_step[s] for s in (9000, 11000, 11020, logs[-1]['step']) if s in by_step}
    windows = {}
    for hour in (8.7, 10.7):
        selected = [x for x in logs if abs(x['elapsed_s']/3600-hour) <= .25]
        keys = ['loss', 'contact', 'geometry_contact', 'flow_hand', 'flow_scene']
        windows[str(hour)] = dict(center_cumulative_hours=hour, half_width_hours=.25,
            logged_batches=len(selected), step_min=min(x['step'] for x in selected),
            step_max=max(x['step'] for x in selected),
            means={k: sum(x['loss'] if k == 'loss' else x['terms'][k] for x in selected)/len(selected) for k in keys})
    return logs, dev, anchors, windows


def plot(out, logs, dev):
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array([r['elapsed_s']/3600 for r in logs])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    axes[0, 0].plot(x, [r['loss'] for r in logs], alpha=.4, linewidth=.7, label='logged TRAIN loss')
    smooth = np.convolve([r['loss'] for r in logs], np.ones(15)/15, mode='valid')
    axes[0, 0].plot(x[14:], smooth, label='15 logged-batch trailing mean')
    for step, label in [(9000, '9000: 0.6890'), (11000, '11000: 0.6613'), (11380, '11380: 0.6610')]:
        row = next(r for r in logs if r['step'] == step)
        axes[0, 0].scatter(row['elapsed_s']/3600, row['loss'], s=30)
        offset = {9000: (-125, 35), 11000: (-125, -28), 11380: (-120, 18)}[step]
        axes[0, 0].annotate(label, (row['elapsed_s']/3600, row['loss']), xytext=offset,
                           textcoords='offset points', fontsize=8,
                           arrowprops=dict(arrowstyle='-', color='0.4', linewidth=.6))
    axes[0, 0].set_title('Training loss: not task success')
    for key in ['contact', 'geometry_contact']:
        y = [r['terms'][key] for r in logs]
        axes[0, 1].plot(x[14:], np.convolve(y, np.ones(15)/15, mode='valid'), label=key)
    axes[0, 1].set_title('TRAIN contact losses (15 logged-batch mean)')
    axes[1, 0].plot([r['cumulative_hours'] for r in dev], [r['grab_geometric_contact_mae'] for r in dev], 'o-', label='GRAB geometric contact MAE')
    axes[1, 0].set_title('Fixed DEV GRAB contact metric')
    for key in ['grab_hand_prefix8_mm', 'grab_object_prefix8_mm']:
        axes[1, 1].plot([r['cumulative_hours'] for r in dev], [r[key] for r in dev], 'o-', label=key)
    axes[1, 1].set_title('Fixed DEV GRAB prefix8 EPE (mm)')
    for ax in axes.flat:
        ax.set_xlabel('Cumulative training elapsed hours (includes 128-step trial)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('FM run stopped by SIGINT; latest retained main checkpoint = step11000')
    for extension in ('png', 'svg'):
        fig.savefig(out/('training-curves.'+extension), dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--related-root', type=Path, action='append', default=[])
    p.add_argument('--owner-observed-step', type=int, default=11020)
    a = p.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    if out.exists(): raise FileExistsError('Preserve prior audit: choose a fresh output')
    if run == out or run in out.parents: raise ValueError('audit output must be outside original run')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', MPLCONFIGDIR=str(out/'mplconfig'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    process = read(run/'process.json')
    receipt = inspect_processes(run, process)
    write(out/'stop-receipt.json', receipt)
    import torch
    torch.set_num_threads(2)
    rows, scopes = inventory(run, [r.resolve() for r in a.related_root], torch, a.owner_observed_step)
    write(out/'checkpoint-inventory.json', dict(records=rows, related_search_scopes=scopes,
        external_scope='Original run recursively and explicit related roots only; no remote machine, credential or unlisted external mount was searched.'))
    main_rows = [r for r in rows if Path(r['path']).parent == run/'training' and not r['is_symlink'] and 'step' in r.get('payload', {})]
    latest = max(main_rows, key=lambda r: r['payload']['step'])
    last = read(run/'training/progress.json')
    weights = sentinels(Path(latest['path']), torch)
    trial = sentinels(run/'flow-trial/step000128.pt', torch)
    sentinel_audit = {k: dict(all_finite=bool(torch.isfinite(v).all()),
        shape=list(v.shape), latest_rms=float(v.square().mean().sqrt()),
        trial128_rms=float(trial[k].square().mean().sqrt()),
        exact_equal_to_trial128=bool(torch.equal(v, trial[k])),
        rms_difference_to_trial128=float((v-trial[k]).square().mean().sqrt())) for k, v in weights.items()}
    logs, dev, anchors, windows = curves(run)
    for name, data in [('train-curve.csv', logs), ('dev-curve.csv', dev)]:
        fields = ['step', 'elapsed_s', 'loss', 'contact', 'geometry_contact', 'flow_hand', 'flow_scene'] if name.startswith('train') else ['step', 'cumulative_hours', 'source_macro_objective', 'grab_hand_prefix8_mm', 'grab_object_prefix8_mm', 'grab_geometric_contact_mae', 'grab_contact_component']
        with (out/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k, row.get('terms', {}).get(k)) for k in fields})
    retained = latest['payload']['step']
    gap = last['step']-retained
    summary = dict(schema='ref2dex.flow-stop-checkpoint-audit.v1', run=str(run),
        observed_at=receipt['observed_at'], python=sys.executable, torch=torch.__version__,
        audit_device='CPU only; no model forward, optimizer update or CUDA context',
        frozen_training_commit=process['git_commit'],
        source_process_status=process['status'], owner_observed_step=a.owner_observed_step,
        owner_step_weight_paths=[r['path'] for r in rows if r.get('payload', {}).get('step') == a.owner_observed_step],
        checkpoint_records=len(rows), unique_readable_payloads=len({(r.get('device'),r.get('inode')) for r in rows if 'step' in r.get('payload', {})}),
        latest_main_checkpoint=latest, last_logged_training=last,
        minimum_unsaved_completed_updates=gap,
        minimum_unsaved_sample_exposures=gap*last['global_batch'],
        retained_checkpoint_to_last_log_elapsed_s=last['elapsed_s']-latest['payload']['elapsed_s'],
        latest_weights_vs_trial128=sentinel_audit,
        training_curve_logged_batches=len(logs), dev_checkpoints=len(dev),
        exact_train_anchors=anchors, quarter_hour_windows=windows,
        dev_anchors=[r for r in dev if r['step'] in (500, 8500, 9000, 11000)],
        save_path_assessment=dict(periodic_save='Observed main checkpoints10500 and11000 match configured periodic retention; no evidence of all-main-stage save failure',
            sigint_final_save='Raw KeyboardInterrupt exits before loop epilogue line366; no handler/finally saves weights. Actual main final result.json is absent.',
            inherited_selection='best.pt points to trial20, trajectory-best.pt to strict-recovery9, inherited selectors; they are not latest checkpoint aliases',
            recommendation='For a later authorized repair, coordinated signal handling must request a safe distributed update-boundary save; use explicit latest/last checkpoint identity and record STOPPED_BY_SIGNAL separately. Do not save partially applied optimizer state from arbitrary exception context.'),
        limits='No full optimizer storage materialization or tensor-by-tensor optimizer audit. Missing11020 applies to declared search scopes, not unknown remote exports. Loss/contact curves are offline TRAIN/DEV metrics, not success rates or independent-seed inference.')
    write(out/'summary.json', summary)
    plot(out, logs, dev)
    print(json.dumps(dict(stopped=receipt['owned_training_stopped'], released=receipt['gpus456_released'],
        latest_main_step=retained, last_logged_step=last['step'], owner_step_weight_paths=summary['owner_step_weight_paths'],
        minimum_unsaved_updates=gap, output=str(out)), indent=2))


if __name__ == '__main__': main()
