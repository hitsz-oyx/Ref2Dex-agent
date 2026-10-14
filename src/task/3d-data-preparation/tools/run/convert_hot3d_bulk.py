"""Convert each fully extracted native sequence as acquisition progresses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
from convert_hot3d_pilot import mano_models, convert, geometry, SCHEMA

STOP = False


def write(path, value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--mano-root',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=86400)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'sequences').mkdir(exist_ok=True);(root/'geometry').mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(4);models=mano_models(a.mano_root)
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else dict(
        schema=SCHEMA,units='m',fps=30,training_allowed=False,objects=[],sequences=[],
        limitations=['Nearest30Hz alignment≤16ms; native timestamps/residuals retained; repeated native frames invalid.',
                     'Target is first metadata object for engineering, not admitted manipulation semantics.',
                     'Participant-disjoint engineering split; no efficacy validation.'])
    known={r['key'] for r in manifest['objects']}
    ledger=json.loads((root/'conversion-ledger.json').read_text()) if (root/'conversion-ledger.json').exists() else {}
    started=time.monotonic()
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGUSR1,stop)
    required=['metadata.json','dynamic_objects.csv','mano_hand_pose_trajectory.jsonl',
              'masks/mask_qa_pass.csv','masks/mask_hand_pose_available.csv','masks/mask_object_pose_available.csv']
    while not STOP and time.monotonic()-started<a.seconds:
        if shutil.disk_usage(root).free<20*2**30:break
        for family in ('aria','quest'):
            folder=a.input/'native'/family
            if not folder.exists():continue
            for path in sorted(folder.iterdir()):
                identity=family+'/'+path.name
                if identity in ledger or not all((path/file).is_file() for file in required):continue
                try:
                    sequence,meta,audit=convert(path,models,100000,alignment_limit_ms=16)
                    for uid in meta['object_uids']:
                        if uid not in known:
                            values=geometry(a.assets/(uid+'.glb'))
                            np.savez_compressed(root/'geometry'/(uid+'.npz'),**values)
                            manifest['objects'].append(dict(key=uid,path='geometry/'+uid+'.npz'));known.add(uid)
                    filename=family+'-'+path.name+'.npz'
                    temporary=root/'sequences'/('pending-'+filename)
                    np.savez_compressed(temporary,**sequence);temporary.replace(root/'sequences'/filename)
                    participant=meta['participant_id']
                    manifest['sequences'].append(dict(path='sequences/'+filename,format='canonical',objects=meta['object_uids'],
                        target_slot=0,target_selection='first metadata object; engineering only',source_id=6,
                        profile='human-mano-native11',split='dev' if participant in {'P0002','P0007','P0012','P0017'} else 'train',
                        group_id='HOT3D/'+participant))
                    ledger[identity]=dict(status='CONVERTED',**audit)
                    print('CONVERTED',identity,audit['frames'],audit['qa_passing_frames'],flush=True)
                except ValueError as error:
                    ledger[identity]=dict(status='SKIPPED_NATIVE_CONTRACT',reason=str(error))
                    print('SKIPPED',identity,type(error).__name__,flush=True)
                except Exception as error:
                    ledger[identity]=dict(status='FAILED',error_type=type(error).__name__)
                    print('FAILED',identity,type(error).__name__,flush=True)
                write(root/'manifest.json',manifest);write(root/'conversion-ledger.json',ledger)
                write(root/'progress.json',dict(status='RUNNING',pid=os.getpid(),converted_sequences=len(manifest['sequences']),
                    converted_frames=sum(r.get('frames',0) for r in ledger.values() if r['status']=='CONVERTED'),
                    skipped_sequences=sum(r['status']=='SKIPPED_NATIVE_CONTRACT' for r in ledger.values()),
                    failed_sequences=sum(r['status']=='FAILED' for r in ledger.values()),elapsed_s=time.monotonic()-started,
                    checked_at=time.time(),training_allowed=False))
        status=a.input/'acquisition-status.json'
        terminal=json.loads(status.read_text()).get('status') if status.exists() else None
        if terminal in ('COMPLETE','PARTIAL','STOPPED'):
            # Acquisition can publish the last sequence while this pass is
            # scanning an earlier directory snapshot. Drain all ready inputs.
            pending_ready=[path for family in ('aria','quest')
                for path in (a.input/'native'/family).iterdir()
                if family+'/'+path.name not in ledger and all((path/file).is_file() for file in required)]
            if not pending_ready:break
            continue
        time.sleep(10)
    write(root/'progress.json',dict(status='STOPPED' if STOP else 'FINISHED',pid=os.getpid(),
        converted_sequences=len(manifest['sequences']),converted_frames=sum(r.get('frames',0) for r in ledger.values() if r['status']=='CONVERTED'),
        skipped_sequences=sum(r['status']=='SKIPPED_NATIVE_CONTRACT' for r in ledger.values()),
        failed_sequences=sum(r['status']=='FAILED' for r in ledger.values()),elapsed_s=time.monotonic()-started,
        checked_at=time.time(),training_allowed=False))


if __name__=='__main__':main()
