#!/usr/bin/env python3
"""Resume an interrupted frozen oracle run; retain all evidence and total caps."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import time

import torch
import run_rolling_gt_y as original


def immutable_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('cached decision/control mismatch: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')


class Resume(original.Campaign):
    def __init__(self, args):
        self.args = args
        self.root = args.run_dir.resolve()
        saved = json.loads((self.root / 'progress.json').read_text())
        if saved['status'] in ('COMPLETED', 'FAILED'):
            raise ValueError('only externally interrupted runs may resume')
        try:
            os.kill(saved['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('original coordinator is still alive')
        self.inputs = saved['inputs']
        for path, digest in self.inputs.items():
            if original.sha(Path(path)) != digest:
                raise ValueError('frozen input changed: ' + path)
        if args.gpus != saved['physical_gpus']:
            raise ValueError('resume must retain original GPU allocation')
        self.commit = saved['git_commit']
        self.resume_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=original.ROOT, text=True).strip()
        # Charge time after the last checkpoint, including interrupted workers.
        prior_manifest = self.root / 'resume_manifest.json'
        if prior_manifest.exists():
            previous = json.loads((self.root/'resume_progress.json').read_text())
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('previous recovery coordinator is still alive')
            if previous['status'] == 'COMPLETED':
                raise ValueError('completed run cannot resume')
            if previous['status'] == 'FAILED' and previous['events'][-1].get('error') != 'KeyboardInterrupt()':
                raise ValueError('failed scientific/resource checks cannot be bypassed')
            if (self.root/'resume_manifest-002.json').exists():
                raise ValueError('second recovery already exists')
            prior = json.loads(prior_manifest.read_text())
            # Recover the last ORIGINAL worker timestamp, before first recovery.
            # Exclude recovery logs so merely opening a new log cannot charge downtime.
            original_latest = max(p.stat().st_mtime for p in self.root.rglob('*')
                                  if p.is_file() and not p.name.startswith('resume')
                                  and p.stat().st_mtime < prior_manifest.stat().st_mtime)
            original_tail = max(0., original_latest-(self.root/'progress.json').stat().st_mtime)
            original_active = saved['elapsed_seconds'] + original_tail
            first_recovery_active = previous['elapsed_seconds']-prior['charged_seconds']
            self.charged_seconds = original_active + first_recovery_active
            immutable_write(self.root/'resume_progress-001.json', previous)
            accounting = dict(original_active_seconds=original_active,
                              first_recovery_active_seconds=first_recovery_active,
                              excluded_downtime_seconds=prior['charged_seconds']-original_active,
                              old_conservative_elapsed=previous['elapsed_seconds'])
            saved = previous
            manifest_path = self.root/'resume_manifest-002.json'
        else:
            latest = max(p.stat().st_mtime for p in self.root.rglob('*')
                         if p.is_file() and not p.name.startswith('resume'))
            tail = max(0., latest - (self.root / 'progress.json').stat().st_mtime)
            self.charged_seconds = saved['elapsed_seconds'] + tail
            accounting = dict(excluded_downtime_seconds=0.)
            manifest_path = prior_manifest
        self.begin = time.monotonic() - self.charged_seconds
        self.events = list(saved['events'])
        self.processes = {}
        self.summaries = []
        self.gpu_slots = queue.Queue()
        for gpu in args.gpus:
            self.gpu_slots.put(gpu)
            self.gpu_slots.put(gpu)
        self.checked_inputs = set()
        self.check()
        immutable_write(manifest_path, dict(
            original_commit=self.commit, resume_commit=self.resume_commit,
            resume_code_sha256=original.sha(Path(__file__)),
            original_progress_sha256=original.sha(self.root / 'progress.json'),
            charged_seconds=self.charged_seconds, wall_cap=args.wall_seconds,
            storage_gib=args.storage_gib, protocol_unchanged=True, accounting=accounting))
        self.progress('RESUMED')

    def progress(self, stage, **kwargs):
        event = dict(stage=stage, elapsed_seconds=time.monotonic()-self.begin, **kwargs)
        self.events.append(event)
        value = dict(status=stage, run_id=self.root.name, git_commit=self.commit,
                     resume_commit=self.resume_commit, pid=os.getpid(),
                     physical_gpus=self.args.gpus, inputs=self.inputs,
                     elapsed_seconds=event['elapsed_seconds'], events=self.events)
        path = self.root / 'resume_progress.json'
        pending = self.root / 'resume_progress.pending'
        pending.write_text(json.dumps(value, indent=2)+'\n')
        pending.replace(path)
        print(json.dumps(event), flush=True)

    def repeated_control(self, base, seed, group):
        control = json.loads((self.root / f's{seed}-g{group}-control.json').read_text())
        if (control['base'], control['seed'], control['group']) != (base, seed, group):
            raise ValueError('control identity mismatch')
        # Validate both baseline archives before reusing preflight evidence.
        reference = original.OUTPUT / (base+'-sync-reference')
        schedule = Path(control['schedule'])
        for suffix, record in (('baseline', True), ('repeat', False)):
            self.worker(f's{seed}-g{group}-{suffix}', reference, schedule,
                        group, seed, 0, window=90, record=record)
        return control

    def _worker(self, name, reference, schedule, group, seed, offset,
                candidate=0, window=32, record=False, plan=None, physical_gpu=6):
        folder = self.root / name
        if (folder / 'result.json').exists():
            report = json.loads((folder / 'result.json').read_text())
            manifest = json.loads((folder / 'run_manifest.json').read_text())
            if manifest['status'] != 'COMPLETED' or not manifest['inputs_unchanged']:
                raise ValueError('cached worker incomplete')
            command = manifest['command']
            expected = {'--reference': str(reference.resolve()),
                        '--anchor-schedule': str(schedule.resolve()),
                        '--group-id': str(group), '--seed': str(seed),
                        '--rolling-offset': str(offset), '--candidate': str(candidate),
                        '--post-window': str(window)}
            for flag, value in expected.items():
                actual = command[command.index(flag)+1]
                if flag in ('--reference', '--anchor-schedule'):
                    actual = str(Path(actual).resolve())
                if actual != value:
                    raise ValueError('cached worker argument mismatch: '+flag)
            if ('--record-rolling-trace' in command) != record:
                raise ValueError('cached trace contract mismatch')
            if ('--rolling-plan' in command) != (plan is not None):
                raise ValueError('cached plan contract mismatch')
            if plan is not None and Path(command[command.index('--rolling-plan')+1]).resolve() != plan.resolve():
                raise ValueError('cached plan path mismatch')
            for path, digest in manifest['input_sha256'].items():
                key = (path, digest)
                if key not in self.checked_inputs:
                    if original.sha(Path(path)) != digest:
                        raise ValueError('cached input drift: '+path)
                    self.checked_inputs.add(key)
            if original.sha(folder/'panel.pt') != report['output_sha256']:
                raise ValueError('cached panel changed')
            if record and not (folder/'trace.pt').exists():
                raise ValueError('cached actual trace missing')
            return folder, original.load(folder/'panel.pt'), report
        # Keep the interrupted partial attempt at an explicit immutable location.
        log = self.root / (name+'.log')
        if folder.exists() or log.exists():
            archive = self.root / 'interrupted-attempts'
            archive.mkdir(exist_ok=True)
            if folder.exists():
                target = archive/name
                if target.exists():
                    target = archive/(name+'-recovery1')
                folder.rename(target)
            if log.exists():
                target = archive/log.name
                if target.exists():
                    target = archive/(name+'-recovery1.log')
                log.rename(target)
        return super()._worker(name, reference, schedule, group, seed, offset,
                               candidate, window, record, plan, physical_gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--workers', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--wall-seconds', type=int, default=7200)
    parser.add_argument('--storage-gib', type=float, default=4)
    args = parser.parse_args()
    if args.wall_seconds > 7200 or args.storage_gib > 4:
        raise ValueError('resume cannot expand frozen resource caps')
    torch.set_num_threads(2)
    original.write = immutable_write
    campaign = Resume(args)
    try:
        specs = [('oracle-y-utility-s263', 263, 0), ('oracle-y-utility-s263', 263, 1),
                 ('oracle-y-utility-extra-s264', 264, 0), ('oracle-y-utility-extra-s264', 264, 1)]
        controls = [campaign.repeated_control(*spec) for spec in specs]
        for control in controls:
            campaign.run_group(control)
        campaign.finish()
    except BaseException as error:
        for process in list(campaign.processes.values()):
            process.terminate()
        campaign.progress('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
