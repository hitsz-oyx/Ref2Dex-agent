"""Compare fixed Oak-only and mixed checkpoints on an existing frozen HOCap Probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from evaluate_hocap_frame_probe import ROOT, TASK, sha, validate_frame_manifest, write


def check_identity(state, vendor):
    """Require current inference code; allow only the proven historical DDP trainer."""
    identity = state['identity']
    if identity['arm'] != 'action':
        raise ValueError('action checkpoint required')
    historical = {}
    for key, expected in identity['implementation_sources'].items():
        if sha(TASK/key) == expected:
            continue
        if key != 'tools/run/train_oakink2_pointworld_ddp.py':
            raise ValueError('inference implementation drift: '+key)
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                        identity['git_commit']+':'+str((TASK/key).relative_to(ROOT))])
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('historical trainer identity mismatch')
        historical[key] = dict(recorded_sha256=expected, current_sha256=sha(TASK/key),
                               verified_at_commit=identity['git_commit'], used_for_inference=False)
    for key, expected in identity['vendor_sources'].items():
        if sha(vendor/key) != expected:
            raise ValueError('vendor inference drift: '+key)
    return historical


def check_compatible(first, second):
    fields = ('schema', 'patch_size', 'fps', 'history', 'horizon', 'voxel_m',
              'max_scene_points', 'local_radius_m', 'object_points', 'temporal_pooling',
              'action_summary', 'motion_weighting', 'motion_tau_m', 'rotation_tau_rad',
              'motion_temperature', 'motion_floor')
    if any(first['config'][k] != second['config'][k] for k in fields):
        raise ValueError('different inference configuration')
    if (first['identity']['stats_sha256'] != second['identity']['stats_sha256']
            or first['identity']['stats'] != second['identity']['stats']):
        raise ValueError('different normalization')
    shapes = lambda state: {k: (tuple(v.shape), str(v.dtype)) for k, v in state['model'].items()}
    if shapes(first) != shapes(second):
        raise ValueError('different parameter/buffer contract')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--oak-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--allow-unverified-clock', action='store_true')
    args = parser.parse_args()
    if not args.allow_unverified_clock or not 1 <= args.seconds <= 600:
        parser.error('explicit clock acknowledgement and <=600s budget required')
    out = args.output.resolve()
    if ROOT/'outputs/cm-pointflow-effect-pretrain' not in out.parents or out.exists():
        parser.error('fresh task-owned output required')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    scratch = ROOT/'tmp/hocap-checkpoint-comparison'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(TASK/'src'))
    started = time.time()
    stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))
    out.mkdir(parents=True)
    manifest = dict(status='RUNNING', pid=os.getpid(), gpu=args.gpu, started_at=started,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    deadline=started+args.seconds, training_allowed=False, source_fps_verified=False,
                    test_ready=False, source_fps=None, horizon_unit='original array frame index',
                    nominal_feature_fps=30, frozen_inputs={}, weights_only_inference=True)
    signatures = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('original frozen input drift: '+str(path))
        manifest['frozen_inputs'][str(path)] = actual
        s = path.stat()
        signatures[str(path)] = (s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def guard(full=False):
        if stop[0] or time.time() >= manifest['deadline']:
            raise TimeoutError('bounded comparison stopped')
        if shutil.disk_usage(out).free < 20*2**30:
            raise RuntimeError('20GiB reserve')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 100*2**20:
            raise RuntimeError('100MiB output cap')
        for path, signature in signatures.items():
            s = Path(path).stat()
            if ((s.st_size, s.st_mtime_ns, s.st_ctime_ns) != signature
                    or (full and sha(path) != manifest['frozen_inputs'][path])):
                raise ValueError('input drift: '+path)
        pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
               '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip().splitlines()
        if any(int(p) != os.getpid() for p in pids if p.strip()):
            raise RuntimeError('foreign GPU process appeared')

    try:
        import numpy as np
        import torch
        from oakink_wm.data import Windows
        from oakink_wm.pointworld_temporal import model_from_config, VENDOR
        from oakink_wm.pointworld_performance import install_fused_hilbert
        import train_oakink2_pointworld_temporal as base
        base.configure_numerics()
        install_fused_hilbert()
        reference = args.reference_run.resolve()
        pin(reference/'result.json')
        original = json.loads((reference/'result.json').read_text())
        if original['status'] != 'COMPLETED' or original['source_fps_verified'] is not False:
            raise ValueError('completed unverified-clock reference required')
        for path, expected in original['frozen_inputs'].items():
            pin(path, expected)
        for name in ('moving', 'natural'):
            pin(reference/(name+'_metrics.json'))
        pin(args.oak_checkpoint)
        pin(Path(__file__))
        seed = original['seed']
        manifest.update(seed=seed, reference_run=str(reference), reference_git_commit=original['git_commit'])
        meta = json.loads((reference/'processed/manifest.json').read_text())
        validate_frame_manifest(meta, args.allow_unverified_clock)
        rows = np.load(reference/'processed/index_test.npy')
        panels = {name: np.load(reference/(name+'_panel.npy')) for name in ('moving', 'natural')}
        if any(not len(p) or len(np.unique(p)) != len(p) for p in panels.values()):
            raise ValueError('empty/duplicate frozen panel')
        if np.intersect1d(panels['moving'], panels['natural']).size:
            raise ValueError('overlapping panels')

        class FrameWindows(Windows):
            def __init__(self):
                self.root = reference
                self.meta = meta
                self.rows = rows
                self.sequences = meta['sequences']
                self.groups = [np.flatnonzero(rows[:, 3] == k) for k in range(3)]
                self.calls = 0

            def __getitem__(self, item):
                self.calls += 1
                if self.calls % 16 == 1:
                    guard()
                return super().__getitem__(item)

            def sequence(self, seq):
                value = dict(super().sequence(seq))
                # Same internal Oak120Hz index assertion as the original Probe; not timestamps.
                value['frame_ids'] = 4*value['frame_ids']
                return value

        dataset = FrameWindows()
        states = {'oak_only': torch.load(args.oak_checkpoint, map_location='cpu', weights_only=False),
                  'mixed': torch.load(original['checkpoint'], map_location='cpu', weights_only=False)}
        check_compatible(states['oak_only'], states['mixed'])
        manifest['checkpoints'] = {}
        for name, state in states.items():
            manifest['checkpoints'][name] = dict(step=state['step'], git_commit=state['identity']['git_commit'],
                stats_sha256=state['identity']['stats_sha256'], historical_trainer=check_identity(state, VENDOR))
        manifest['checkpoints']['oak_only']['path'] = str(args.oak_checkpoint.resolve())
        manifest['checkpoints']['mixed']['path'] = original['checkpoint']
        manifest['panels'] = {name: dict(windows=len(p), sequences=len(np.unique(rows[p, 0])),
             category_counts={str(k): int((rows[p, 3] == k).sum()) for k in (0, 1)}) for name, p in panels.items()}
        write(out/'input_manifest.json', manifest)
        torch.cuda.reset_peak_memory_stats()
        results = {}
        for name, state in states.items():
            guard()
            model = model_from_config(state['identity']['stats'], state['config']).cuda()
            model.load_state_dict(state['model'], strict=True)
            results[name] = {}
            for panel, indices in panels.items():
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                values = base.evaluate(model, dataset, indices, 'action', 2, True)
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError('nonfinite metrics')
                results[name][panel] = values
                write(out/(name+'_'+panel+'_metrics.json'), values)
                print(json.dumps(dict(model=name, panel=panel, elapsed=time.time()-started,
                    h24_mm=1000*values['model/anchor/cat-1/h24/point_epe'])), flush=True)
            del model
            torch.cuda.empty_cache()
        replay = {}
        for panel in panels:
            old = json.loads((reference/(panel+'_metrics.json')).read_text())
            new = results['mixed'][panel]
            if set(new) != set(old):
                raise ValueError('reference metric coverage changed')
            replay[panel] = max(abs(new[k]-old[k]) for k in old)
            if replay[panel] > 1e-7:
                raise ValueError('mixed model reference replay mismatch')
            for key, value in results['oak_only'][panel].items():
                if key.startswith('static/') and abs(value-new[key]) > 1e-12:
                    raise ValueError('control/target mismatch')
        comparison = {}
        for panel in panels:
            comparison[panel] = {}
            for cat in (-1, 0, 1):
                for h in (1, 4, 8, 12, 24):
                    key = 'anchor/cat'+str(cat)+'/h'+str(h)+'/point_epe'
                    if 'model/'+key not in results['mixed'][panel]:
                        continue
                    oak = results['oak_only'][panel]['model/'+key]*1000
                    mixed = results['mixed'][panel]['model/'+key]*1000
                    comparison[panel]['cat'+str(cat)+'/h'+str(h)] = dict(oak_only_mm=oak,
                        mixed_mm=mixed, static_mm=results['mixed'][panel]['static/'+key]*1000,
                        error_reduction_percent=100*(1-mixed/oak) if oak else None)
        guard(full=True)
        manifest.update(status='COMPLETED', elapsed_seconds=time.time()-started,
                        comparison=comparison, reference_replay_max_abs_error=replay,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, verdict='UNCLEAR',
                        limitation='Endpoint comparison confounds mixed data, additional updates and loss recipe; '
                        'unverified source clock, nominal30Hz features, observed future human hands; no fitting.')
        write(out/'result.json', manifest)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_seconds=time.time()-started)
        raise
    finally:
        write(out/'input_manifest.json', manifest)


if __name__ == '__main__':
    main()
