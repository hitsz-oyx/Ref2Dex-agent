"""Complete a stopped, bound contact cache with the identical offline teacher."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.storage import read_json,load_array
from ptv3_shared_wam_v5.cache_contract import metadata_identity,array_identity,identity,validate_binding
from ptv3_shared_wam_v5.contracts import LABEL_SEMANTICS
from prepare_surface_contact import prepare,write

CONTEXT=None


def work(key):
    hands,output,geometries,contract=CONTEXT
    folder=hands/'sequences'/key
    return prepare(folder,read_json(folder/'meta.json'),geometries,output,contract)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','hands','geometry','output','control'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seconds',type=int,default=7200)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    if not 1<=a.workers<=16 or not 1<=a.seconds<=21600:p.error('bounded workers/runtime required')
    if Path('/proc/%d'%a.previous_pid).exists():raise ValueError('previous writer must exit before parallel takeover')
    if not (a.output/'STOP').exists():raise ValueError('explicit stopped-stage takeover required')
    a.control.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    jobs=read_json(a.plan)['jobs'];hc=read_json(a.hands/'cache-contract.json');cc=read_json(a.output/'cache-contract.json')
    if read_json(a.hands/'progress.json')['status']!='COMPLETED':raise ValueError('complete audited hands required')
    geo=read_json(a.geometry/'manifest.json')
    if geo['status']!='COMPLETED' or geo.get('failures'):raise ValueError('complete CAD required')
    geometries={}
    for row in geo['objects']:
        with load_array(a.geometry/row['path'],allow_pickle=False) as z:values={k:z[k] for k in z.files}
        values['identity']=array_identity(values);geometries[row['key']]=values
    expected=dict(protocol=LABEL_SEMANTICS,hands=str(a.hands.resolve()),hand_cache=hc['identity'],
        geometry=str(a.geometry.resolve()),geometry_manifest=metadata_identity(a.geometry/'manifest.json'),
        geometry_payloads=identity({k:v['identity'] for k,v in geometries.items()}))
    if cc['inputs']!=expected:raise ValueError('contact takeover upstream identity mismatch')
    old_hands=hc['inputs'].get('inherited_human_root');old_contacts=Path(old_hands).parent/'contacts' if old_hands else None
    available=set(os.listdir(a.output));legacy=set(os.listdir(old_contacts)) if old_contacts else set()
    pending=[];done=0;valid_frames=0;by_source=Counter()
    write(a.control/'run-manifest.json',dict(pid=os.getpid(),git_commit=a.git_commit,
        work_version='v5.identical-teacher-parallel-completion.v1',run_id=a.control.name,
        inputs=expected,workers=a.workers,seconds=a.seconds,min_free_gib=20,
        max_control_mib=64,python=sys.version,device='CPU offline KD-tree; unchanged from serial teacher',
        old_writer_pid=a.previous_pid,old_STOP_retained=True))
    for job in jobs:
        key=job['key'];folder=a.hands/'sequences'/key;meta=read_json(folder/'meta.json')
        validate_binding(folder,meta,hc,Path(old_hands)/'sequences' if old_hands else None)
        if key not in available and old_contacts and job['source_id']!=4 and key in legacy:
            (a.output/key).symlink_to((old_contacts/key).resolve(),target_is_directory=True);available.add(key)
        if key in available:
            row=read_json(a.output/key/'meta.json');validate_binding(a.output/key,row,cc,old_contacts)
            if row.get('hand_identity') and row['hand_identity']!=metadata_identity(folder/'meta.json'):
                raise ValueError('stale hand-dependent contact: '+key)
            if row['label_semantics']!=LABEL_SEMANTICS:raise ValueError('legacy teacher differs')
            done+=1;valid_frames+=row['valid_object_frames'];by_source[job['source_id']]+=1
        else:pending.append(key)
    # Large human sequences start early; remaining100-frame robot episodes
    # naturally fill short gaps. Fork workers share read-only CAD arrays.
    pending.sort(key=lambda key:(key.startswith('s4-'),key))
    global CONTEXT
    CONTEXT=(a.hands,a.output,geometries,cc)
    status='RUNNING';reason=None

    def progress():
        row=dict(status=status,pid=os.getpid(),sequences=done,total=len(jobs),valid_object_frames=valid_frames,
            source_sequences=dict(by_source),workers=a.workers,elapsed_s=time.monotonic()-started,
            checked_at=time.time(),reason=reason,control=str(a.control))
        write(a.output/'progress.json',row);write(a.control/'progress.json',row)
        print(json.dumps(row),flush=True)

    progress();pool=mp.get_context('fork').Pool(a.workers)
    try:
        iterator=pool.imap_unordered(work,pending,chunksize=1);remaining=len(pending);last=time.monotonic()
        while remaining:
            if time.monotonic()-started>a.seconds or (a.control/'STOP').exists() or shutil.disk_usage(a.output).free<20*2**30:
                status='STOPPED';reason='deadline/control STOP/free-space boundary';break
            try:row=iterator.next(timeout=10)
            except mp.TimeoutError:
                if time.monotonic()-last>=20:progress();last=time.monotonic()
                continue
            remaining-=1;done+=1;valid_frames+=row['valid_object_frames'];by_source[row['source_id']]+=1
            if time.monotonic()-last>=20:progress();last=time.monotonic()
        if not remaining:status='COMPLETED_FOR_AVAILABLE_HANDS'
    except BaseException as exc:
        status='FAILED';reason=repr(exc);raise
    finally:
        pool.terminate();pool.join();progress()


if __name__=='__main__':main()
