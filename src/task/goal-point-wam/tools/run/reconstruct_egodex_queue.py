#!/usr/bin/env python3
"""Persistent GPU consumer of CRC-verified TRAIN records from full acquisition."""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

import h5py
import torch


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.json.part'); part.write_text(json.dumps(value, indent=2)+'\n'); part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--template-plan', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=604800)
    p.add_argument('--processed-cap-bytes', type=int, default=8_000_000_000_000)
    a = p.parse_args()
    task = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(task/'tools/audit'))
    import probe_long_video_depth as depth_module
    import probe_window_video_tracks as track_module
    template = json.loads(a.template_plan.read_text())
    assets = Path(template['depth_checkpoint']).parents[1]
    known = {r['session'].removesuffix('.mov'):r['split'] for r in template['records']}
    root = a.output/'reconstruction'; root.mkdir(exist_ok=True)
    state = dict(status='WAITING_FOR_DATA', pid=os.getpid(), started_at=time.time(), completed=0,
                 admitted_windows=0, total_windows=0, pending_long_context=0, failed=0,
                 processed_cap_bytes=a.processed_cap_bytes, processed_bytes=0, offline_pretraining_only=True)
    status_path = a.output/'reconstruction_status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        for key in ('completed','admitted_windows','total_windows','pending_long_context','failed','processed_bytes'):
            state[key] = old.get(key,0)
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    jobs = queue.Queue(maxsize=128)
    def feed():
        path = a.output/'ready.jsonl'; offset = 0
        while not stop[0]:
            if path.exists():
                with path.open() as f:
                    f.seek(offset)
                    for line in f:
                        if not line.endswith('\n'): break
                        row = json.loads(line)
                        while not stop[0]:
                            try: jobs.put(row,timeout=1); break
                            except queue.Full: pass
                        offset += len(line.encode())
            try: meta = json.loads((a.output/'acquisition_status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError): meta = {}
            if meta.get('status') in ('COMPLETED','INCOMPLETE','STOPPED','FAILED'):
                jobs.put(None); return
            time.sleep(2)
    threading.Thread(target=feed,daemon=True).start()
    atomic(status_path,state)
    try:
        while not stop[0] and time.time()-state['started_at'] < a.seconds:
            if state['processed_bytes']+2_000_000_000 > a.processed_cap_bytes or shutil.disk_usage(root).free < 10_000_000_000_000:
                state['status']='PAUSED_STORAGE'; break
            try: record = jobs.get(timeout=10)
            except queue.Empty:
                state['checked_at']=time.time(); atomic(status_path,state); continue
            if record is None:
                state['status']='DRAINED'; break
            if record['release_split'] != 'official_train': continue
            folder = root/record['key']; receipt = folder/'receipt.json'
            if receipt.exists(): continue
            folder.mkdir(parents=True,exist_ok=True)
            state.update(status='RECONSTRUCTING',current=record['member'],checked_at=time.time())
            atomic(status_path,state)
            report = dict(member=record['member'],key=record['key'],status='RUNNING',started_at=time.time(),
                          source_manifest=record['source_manifest'],training_allowed=False)
            try:
                source = Path(record['raw_hdf5'])
                with h5py.File(source,'r') as raw:
                    raw_frames = len(raw['transforms/camera'])
                    session = str(raw.attrs['session_name']).removesuffix('.mov')
                sampled = (raw_frames+2)//3
                if sampled > 450:
                    report.update(status='PENDING_LONG_CONTEXT',raw_frames=raw_frames,sampled_frames=sampled)
                    state['pending_long_context'] += 1
                elif sampled < 12:
                    report.update(status='TOO_SHORT',raw_frames=raw_frames)
                    state['failed'] += 1
                else:
                    split = known.get(session,'val' if int(hashlib.sha256(('283:'+session).encode()).hexdigest()[:8],16)%5==0 else 'train')
                    plan = dict(template, records=[dict(id=0,member=record['member'],session=session,split=split,
                                raw_hdf5=str(source),source_manifest=record['source_manifest'])])
                    atomic(folder/'plan.json',plan)
                    geometry = folder/'geometry'
                    if not (geometry/'result.json').exists():
                        if geometry.exists():
                            # Failed evidence is preserved; retry has a fresh owned directory.
                            geometry = folder/('geometry-retry-%d'%time.time())
                        sys.argv=['depth','--plan',str(folder/'plan.json'),'--member',record['member'],
                                  '--output',str(geometry),'--seconds','1200','--reuse-model',
                                  '--lossless-compressed','--discard-decoded-images']
                        with (folder/'depth.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                            depth_module.main()
                    dm = json.loads((geometry/'result.json').read_text())
                    if not dm['calibration_screen_pass']:
                        report.update(status='REJECTED_SCALE',calibration=dm['calibration']); state['failed'] += 1
                    else:
                        windows = folder/'windows'
                        if windows.exists(): windows = folder/('windows-retry-%d'%time.time())
                        sys.argv=['tracks','--geometry',str(geometry/'depth.npz'),'--tracker-vendor',str(assets/'vendor/co-tracker'),
                                  '--tracker-checkpoint',str(assets/'cotracker3-direct/scaled_offline.pth'),
                                  '--output',str(windows),'--all-windows','--max-windows','100','--seconds','900',
                                  '--compact-teachers','--reuse-model']
                        with (folder/'windows.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                            track_module.main()
                        wm = json.loads((windows/'result.json').read_text())
                        report.update(status='COMPLETED',split=split,session=session,geometry=str(geometry/'depth.npz'),
                                      windows=str(windows),admitted_windows=wm['admission_pass'],total_windows=wm['loader_windows'])
                        state['completed'] += 1; state['admitted_windows'] += wm['admission_pass']; state['total_windows'] += wm['loader_windows']
            except Exception as exc:
                report.update(status='FAILED',error=repr(exc)); state['failed'] += 1
            finally:
                gc.collect(); torch.cuda.empty_cache()
                report['ended_at']=time.time(); atomic(receipt,report)
                state['processed_bytes'] += sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                atomic(status_path,state)
        if stop[0] or time.time()-state['started_at'] >= a.seconds: state['status']='STOPPED'
    finally:
        stop[0]=True
        state['ended_at']=time.time(); atomic(status_path,state)


if __name__ == '__main__':
    main()
