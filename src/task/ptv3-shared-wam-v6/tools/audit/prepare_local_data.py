"""Bounded local data inventory/preparation, without training or admission.

Original inputs stay read-only. Engineering windows are explicitly distinct
from the unavailable remote frozen window filter and normalization.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
GIB = 2**30


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def capacity(root, peak_bytes, floor_bytes=20*GIB):
    free = shutil.disk_usage(root).free
    if free-peak_bytes < floor_bytes:
        raise RuntimeError('Insufficient headroom: free=%d peak=%d floor=%d' % (free,peak_bytes,floor_bytes))
    return free


def npy_header(stream):
    magic = stream.read(8)
    if magic[:6] != b'\x93NUMPY': raise ValueError('invalid NPY magic')
    width = 2 if magic[6] == 1 else 4
    length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
    return ast.literal_eval(stream.read(length).decode('latin1'))


def local_job(job):
    job = dict(job)
    if job['source_id'] < 3:
        job['sequence_folder'] = str(ROOT/'outputs'/job['sequence_folder'].split('/outputs/',1)[1])
    else:
        job['payload'] = job['source_row']['source_path']
    return job


def inspect_job(job):
    job = local_job(job)
    if job['source_id'] < 3:
        folder = Path(job['sequence_folder'])
        with (folder/'hand.npy').open('rb') as stream: shape = npy_header(stream)['shape']
        with (folder/'poses.npy').open('rb') as stream: poses = npy_header(stream)['shape']
        n, objects = shape[0], poses[1]
        source = folder/'hand.npy'
    else:
        source = Path(job['payload'])
        with zipfile.ZipFile(source) as archive:
            with archive.open('hand.npy') as stream: shape = npy_header(stream)['shape']
            n = shape[0]
            if job['source_id'] == 4:
                objects = 1
                for name, expected in [('q.npy',(n,22)),('hand_root.npy',(n,13))]:
                    with archive.open(name) as stream: actual = npy_header(stream)['shape']
                    if actual != expected: raise ValueError('robot source shape changed: '+name)
                with archive.open('object.npy') as stream: object_shape = npy_header(stream)['shape']
                if object_shape not in ((n,7),(n,13)):raise ValueError('object pose7/state13 required')
            else:
                with archive.open('object_poses.npy') as stream: poses = npy_header(stream)['shape']
                if poses[0] != n: raise ValueError('hand/object frame mismatch')
                objects = poses[1]
    if objects != len(job['objects']) or n < 28: raise ValueError('source length/object identity mismatch')
    packet = None
    if job['source_id'] != 4:
        base = ROOT/'outputs/ptv3-shared-wam-v5/hand-packets-20261019-r1'
        record = json.loads((base/'records'/(job['key']+'.json')).read_text())
        if record['status'] != 'READY': raise ValueError('annotation not ready')
        packet = str(base/record['packet'])
        if not Path(packet).is_file(): raise FileNotFoundError(packet)
    # Includes headers/block padding and one atomic staging directory; rename
    # publishes it without making a second copy. Contact/event arrays follow.
    estimate = n*(2*778*3*2+2*21*3*4+2*21+5)+n*objects*(2048*3+65+24)+256*1024
    return dict(key=job['key'], source_id=job['source_id'], frames=n, objects=objects,
                source=str(source), packet=packet, estimated_bytes=estimate,
                split=sorted(job['windows']), source_sequence=job['source_sequence'])


def quota_snapshot():
    class D(ctypes.Structure):
        _fields_ = [(k,ctypes.c_uint64) for k in ('bhard','bsoft','space','ihard','isoft','inodes','btime','itime')]+[('valid',ctypes.c_uint32)]
    library=ctypes.CDLL(None,use_errno=True); results=[]
    for name,identifier,kind in [('user',os.getuid(),0),('group',os.getgid(),1),('project',0,2)]:
        value=D();ctypes.set_errno(0)
        result=library.quotactl(ctypes.c_int((0x800007<<8)|kind),ctypes.c_char_p(b'/dev/md0'),ctypes.c_int(identifier),ctypes.byref(value))
        error=ctypes.get_errno()
        results.append(dict(kind=name,id=identifier,result=result,errno=error,
            interpretation='no active quota' if result and error==3 else 'unverified: permission denied' if result and error==1 else os.strerror(error) if result else 'queried',
            values={k:getattr(value,k) for k,_ in value._fields_} if not result else None))
    return results


def inventory(args):
    out=args.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v6' not in out.parents:
        raise ValueError('fresh task-local output required')
    capacity(ROOT,256*1024**2);out.mkdir(parents=True)
    plan=json.loads(args.plan.read_text());start=time.monotonic();rows=[];failures=[]
    def checked(job):
        try:return inspect_job(job),None
        except Exception as exc:return None,dict(key=job['key'],source_id=job['source_id'],reason=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,failure) in enumerate(pool.map(checked,plan['jobs'])):
            if row:rows.append(row)
            else:failures.append(failure)
            if i%2000==0:
                save(out/'inventory-progress.json',dict(checked=i+1,total=len(plan['jobs']),failures=len(failures),elapsed_s=time.monotonic()-start))
                print(json.dumps(dict(checked=i+1,total=len(plan['jobs']),failures=len(failures))),flush=True)
            if time.monotonic()-start>900:raise TimeoutError('15-minute inventory bound')
    storage=shutil.disk_usage(ROOT);vfs=os.statvfs(ROOT)
    result=dict(status='COMPLETED' if not failures else 'INCOMPLETE',host=os.uname().nodename,
        target=str(out),filesystem=subprocess.check_output(['findmnt','-T',str(out),'-J'],text=True),
        uid=os.getuid(),free_bytes=storage.free,free_gib=storage.free/GIB,free_inodes=vfs.f_favail,
        quota=quota_snapshot(),checked_sequences=len(rows),failed_sequences=failures,
        estimated_full_cache_bytes=sum(r['estimated_bytes'] for r in rows),
        estimate='Fresh source NPY/ZIP headers for every planned sequence, plus writer dtypes and conservative per-sequence metadata padding; not yesterday census.',
        scope='Header/count/path inventory only; full CRC/content/geometry verification belongs to prepared sequences.',
        plan=str(args.plan.resolve()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        elapsed_s=time.monotonic()-start,rows=rows)
    save(out/'inventory.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','filesystem','failed_sequences')},default=str),flush=True)


def load_tool(name, filename):
    spec=importlib.util.spec_from_file_location(name,filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(args):
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(TASK/'src'))
    from ptv3_shared_wam_v6 import paths
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ptv3_shared_wam_v5.cache_contract import bind_cache,metadata_identity,array_identity
    from ptv3_shared_wam_v5.contracts import DATA_SCHEMA,INPUT_SEMANTICS,LABEL_SEMANTICS,SAMPLING_SEMANTICS
    from ptv3_shared_wam_v5.data import V5Windows,collate
    from ptv3_shared_wam_v5.window_contract import history_admitted
    from ptv3_shared_wam_v5.event_labels import EVENT_LABEL_SCHEMA,DEFAULT_PROTOCOL
    torch.set_num_threads(2)
    out=args.output.resolve();audit=json.loads((out/'inventory.json').read_text())
    if audit['status']!='COMPLETED':raise ValueError('complete current inventory required')
    plan=json.loads(args.plan.read_text());jobs={j['key']:local_job(j) for j in plan['jobs']}
    rows={r['key']:r for r in audit['rows']};selected={}
    # Full GRAB is directly useful for the preceding tau/contact question.
    # Other sources get one short complete sequence per available split.
    for key,row in rows.items():
        if row['source_id']==1:selected[key]=row
    for sid in (0,2,4,5,6,8,9,10):
        for split in ('train','dev'):
            candidates=[r for r in rows.values() if r['source_id']==sid and split in r['split']]
            if candidates:
                for chosen in sorted(candidates,key=lambda r:(r['estimated_bytes'],r['key']))[:3]:
                    selected[chosen['key']]=chosen
    expected=sum(r['estimated_bytes'] for r in selected.values())
    # No transport archive, no second expanded copy. Per-sequence staging
    # renames atomically; reserve the largest sequence once more for safety.
    peak=expected+4*max(r['estimated_bytes'] for r in selected.values())+512*1024**2
    if peak>8*GIB:raise ValueError('bounded necessary subset exceeds8GiB; do not start')
    capacity(out,peak+20*GIB)
    destination=out/'data'
    if destination.exists() and not args.resume_preparation:raise ValueError('fresh preparation directory or explicit resume required')
    destination.mkdir(exist_ok=args.resume_preparation);start=time.monotonic();deadline=start+3600
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(status='PREPARING',run_id=out.name,work_version='local-v6-data-subset.v1',
        git_commit=git_commit,python=sys.version,torch=torch.__version__,
        scope='all1132 planned GRAB sequences + up to3 complete sequences/split for other sources; not remote frozen admission',
        selected_sequences=len(selected),estimated_final_bytes=expected,estimated_peak_bytes=peak,
        floor_bytes=20*GIB,user_hard_floor_bytes=15*GIB,budget_bytes=8*GIB,
        device='CPU deterministic MANO/FK and geometry preprocessing; four bounded workers, torch2threads; no neural training',
        source_read_only=True,training_allowed=False,training_launched=False,
        missing_for_original_resume=['remote frozen v6/v5 manifests and binding metadata','original admitted window/stratum indices','original TRAIN normalizer JSON'])
    if (out/'preparation.json').exists():
        old=json.loads((out/'preparation.json').read_text())
        receipt['previous_preparation']=dict(git_commit=old['git_commit'],status=old['status'],
            reason=old.get('stop_reason','explicit resume of owned preparation'))
    save(out/'preparation.json',receipt)
    for name in ('hands','contacts','geometry','sources','packets','windows','events/sequences','recipe'):
        (destination/name).mkdir(parents=True,exist_ok=args.resume_preparation)
    hands=destination/'hands';contacts=destination/'contacts';geometry=destination/'geometry';recipe=destination/'recipe'
    selected_jobs=[];main3=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data'
    mainmeta=json.loads((main3/'processed/manifest.json').read_text());indices={}
    for sid in (0,1,2):
        for split,source_split in [('train','train'),('dev','val')]:
            recorded=Path(mainmeta['sources'][sid]['indices'][source_split]['path'])
            indices[(sid,split)]=np.load(main3/'processed'/recorded.name,mmap_mode='r')
    for key,row in selected.items():
        job=jobs[key];sid=job['source_id'];n=row['frames'];new_windows={}
        source=Path(job['sequence_folder']) if sid<3 else Path(job['payload'])
        link=destination/'sources'/key
        if link.is_symlink():
            if link.resolve()!=source.resolve():raise ValueError('existing source link changed')
        else:link.symlink_to(source,target_is_directory=source.is_dir())
        if row['packet']:
            link=destination/'packets'/(key+'.npz')
            if link.is_symlink():
                if link.resolve()!=Path(row['packet']).resolve():raise ValueError('existing annotation link changed')
            else:link.symlink_to(row['packet'])
        for split in row['split']:
            if sid<3:
                original=indices[(sid,split)];take=original[original[:,0]==job['sequence_id']]
                window=take[:,[1,2]].astype(np.int32)
                categories=take[:,3].astype(np.int8)
                if mainmeta['sources'][sid]['kind']=='native':categories-=1
            else:
                ticks=np.arange(3,n-24,8,dtype=np.int32)
                window=np.column_stack((np.full(len(ticks),job['source_row']['target_slot'],np.int32),ticks))
                categories=np.full(len(ticks),-1,np.int8)
            # Keep bounded explicit windows; full sequence geometry is retained.
            if len(window)>32:
                take=np.linspace(0,len(window)-1,32,dtype=int);window=window[take];categories=categories[take]
            target=destination/'windows'/(key+'-'+split+'.npy')
            category_path=target.with_name(target.stem+'-strata.npy')
            for path,value in [(target,window),(category_path,categories)]:
                if path.exists():np.testing.assert_array_equal(np.load(path),value)
                else:np.save(path,value)
            new_windows[split]=str(target.resolve())
        job=dict(job,windows=new_windows,window_origin='local baseline/safe stride8 engineering windows, not remote frozen filter')
        selected_jobs.append(job)
    new_plan=dict(jobs=selected_jobs,original_plan=str(args.plan.resolve()),engineering_subset=True,
        training_allowed=False,source_probabilities=plan['source_probabilities'])
    if (destination/'plan.json').exists():
        if json.loads((destination/'plan.json').read_text())!=new_plan:raise ValueError('resume plan changed')
    else:save(destination/'plan.json',new_plan)
    geo_origin=ROOT/'outputs/ptv3-shared-wam-v5/object-surfaces-20261019-r2'
    registry=json.loads((geo_origin/'manifest.json').read_text());needed=set()
    for job in selected_jobs:
        needed.update('s%d:%s'%(job['source_id'],name) if job['source_id']<3 else name for name in job['objects'])
    registry['objects']=[r for r in registry['objects'] if r['key'] in needed]
    if {r['key'] for r in registry['objects']}!=needed:raise ValueError('required original CAD cache missing')
    geometries={};(geometry/'geometry').mkdir(exist_ok=args.resume_preparation)
    for row in registry['objects']:
        original=geo_origin/row['path'];target=geometry/row['path'];capacity(out,original.stat().st_size)
        if target.exists():
            if target.read_bytes()!=original.read_bytes():raise ValueError('existing geometry changed')
        else:shutil.copy2(original,target)
        with np.load(target,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        if values['surface'].shape!=(2048,3) or not all(np.isfinite(v).all() for v in values.values()):raise ValueError('invalid object geometry')
        values['identity']=array_identity(values);geometries[row['key']]=values
    registry.update(total=len(registry['objects']),engineering_subset=True)
    save(geometry/'manifest.json',registry)
    hc=bind_cache(hands,'hand',dict(protocol='local reconstruction with original v5 implementation',
        plan=metadata_identity(destination/'plan.json'),original_plan=str(args.plan.resolve())))
    (hands/'sequences').mkdir(exist_ok=args.resume_preparation)
    cc=bind_cache(contacts,'contact',dict(protocol=LABEL_SEMANTICS,hands=str(hands.resolve()),hand_cache=hc['identity'],
        geometry_manifest=metadata_identity(geometry/'manifest.json')))
    tools=TASK.parent/'ptv3-shared-wam-v5/tools/run'
    hand_tool=load_tool('local_hand_preparation',tools/'prepare_hand_overlay.py')
    contact_tool=load_tool('local_contact_preparation',tools/'prepare_surface_contact.py')
    event_tool=load_tool('local_event_preparation',tools/'prepare_events.py')
    options=SimpleNamespace(output=hands,mano_root=Path('/home2/wyy/oyx_ws/Ref2Dex/dataset/arctic/data/body_models/mano/models'),
        shadow_asset=ROOT/'third_party/UniGraspTransformer/dexgrasp/hand_assets/open_ai_assets/hand/robot.xml',
        device='cpu',cache_contract=hc)
    sequences=[];all_indices={'train':[],'dev':[]};all_strata={'train':[],'dev':[]};failures=[];source_counts={};minimum_free=shutil.disk_usage(out).free
    def process_job(job):
        key=job['key'];row=selected[key]
        if time.monotonic()>deadline:raise TimeoutError('one-hour preparation limit')
        if (out/'STOP').exists():raise InterruptedError('preparation STOP requested')
        free=capacity(out,row['estimated_bytes']*8+256*1024**2)
        try:
            folder=hands/'sequences'/key;cf=contacts/key
            if (folder/'meta.json').exists():
                hm=json.loads((folder/'meta.json').read_text())
                if hm['cache_contract']!=hc['identity']:raise ValueError('existing hand cache binding changed')
            else:
                if row['packet']:
                    with np.load(row['packet'],allow_pickle=False) as z:packet={k:z[k] for k in z.files}
                else:
                    with np.load(job['payload'],allow_pickle=False) as z:packet=dict(q=z['q'],hand_root=z['hand_root'],robot=np.array('shadow-official-replay'))
                hm=hand_tool.prepare(job,packet,options)
            if (cf/'meta.json').exists():
                cm=json.loads((cf/'meta.json').read_text())
                if cm['cache_contract']!=cc['identity'] or cm['hand_identity']!=metadata_identity(folder/'meta.json'):
                    raise ValueError('existing contact cache binding changed')
            else:cm=contact_tool.prepare(folder,hm,geometries,contacts,cc)
            jvalid=np.load(folder/'joint_valid.npy',mmap_mode='r')
            known=np.load(folder/'object_pose_valid.npy',mmap_mode='r')
            selected_windows={}
            for split,path in job['windows'].items():
                window=np.load(path);admit=history_admitted(jvalid,window[:,1])
                admit &= np.array([known[t-3:t+25,int(slot)].all() for slot,t in window])
                window=window[admit]
                categories=np.load(Path(path).with_name(Path(path).stem+'-strata.npy'))[admit]
                if len(window):selected_windows[split]=(window,categories)
            if not selected_windows:raise ValueError('no usable engineering H4/K24 windows')
            return (dict(key=key,source_id=job['source_id'],path=str(folder.resolve()),contact_path=str(cf.resolve()),objects=job['objects'],
                hand_meta_identity=metadata_identity(folder/'meta.json'),contact_meta_identity=metadata_identity(cf/'meta.json')),selected_windows,free,None)
        except (OSError,MemoryError):
            (out/'STOP').touch();raise
        except Exception as exc:return None,None,free,dict(key=key,reason=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
      for result in pool.map(process_job,selected_jobs):
        row,selected_windows,free,failure=result;minimum_free=min(minimum_free,free)
        if failure:failures.append(failure)
        else:
            sid=row['source_id'];index=len(sequences);sequences.append(row)
            source_counts[str(sid)]=source_counts.get(str(sid),0)+1
            for split,(window,categories) in selected_windows.items():
                all_indices[split].append(np.column_stack((np.full(len(window),index),window)).astype(np.int32));all_strata[split].append(categories)
        if (len(sequences)+len(failures))%25==0 or failures:
            save(out/'preparation-progress.json',dict(completed=len(sequences),failed=failures,total=len(selected_jobs),free_bytes=shutil.disk_usage(out).free,elapsed_s=time.monotonic()-start))
            print(json.dumps(dict(completed=len(sequences),failures=len(failures),total=len(selected_jobs))),flush=True)
    index_paths={};stratum_paths={}
    for split in ('train','dev'):
        path=recipe/(split+'.npy');values=np.concatenate(all_indices[split]) if all_indices[split] else np.empty((0,3),np.int32)
        np.save(path,values);index_paths[split]=str(path.resolve())
        path=recipe/(split+'-strata.npy');np.save(path,np.concatenate(all_strata[split]) if all_strata[split] else np.empty(0,np.int8));stratum_paths[split]=str(path.resolve())
    manifest=dict(schema=DATA_SCHEMA,status='FROZEN',input_semantics=INPUT_SEMANTICS,label_semantics=LABEL_SEMANTICS,
        training_allowed=False,engineering_subset=True,sequences=sequences,indices=index_paths,strata=stratum_paths,
        geometry_manifest=str((geometry/'manifest.json').resolve()),source_ids=sorted(map(int,source_counts)),
        source_probabilities={sid:plan['source_probabilities'][sid] for sid in source_counts},sampling_semantics=SAMPLING_SEMANTICS,
        pending_sequences=len(plan['jobs'])-len(sequences),original_frozen_windows_recovered=False,
        cache_bindings=dict(hands=str(hands/'cache-contract.json'),hand_identity=hc['identity'],contacts=str(contacts/'cache-contract.json'),
            contact_identity=cc['identity'],geometry_identity=metadata_identity(geometry/'manifest.json')))
    save(recipe/'manifest.json',manifest)
    events=[];ds=V5Windows(recipe/'manifest.json','train');dev=V5Windows(recipe/'manifest.json','dev');checks=[]
    for row in sequences:
        capacity(out,selected[row['key']]['estimated_bytes']+64*1024**2)
        event=event_tool.prepare_one((row,destination/'events',DEFAULT_PROTOCOL,time.time()+300));events.append(event)
        cf=Path(row['contact_path']);confidence=np.load(cf/'contact.npy',mmap_mode='r');parts=np.load(cf/'part.npy',mmap_mode='r')
        if confidence.dtype!=np.float16 or not np.isfinite(confidence).all() or confidence.min()<0 or confidence.max()>1 or parts.max()>15:
            raise ValueError('invalid contact/part contents: '+row['key'])
    save(destination/'events/manifest.json',dict(schema=EVENT_LABEL_SCHEMA,status='COMPLETED',
        parent_manifest_identity=metadata_identity(recipe/'manifest.json'),sequences=events,training_allowed=False,engineering_subset=True))
    for split,dataset in [('train',ds),('dev',dev)]:
        for sid,pool in dataset.pools.items():
            for i in pool[:min(4,len(pool))]:
                sample=dataset[int(i)]
                if any(np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all() for v in sample.values()):raise ValueError('nonfinite reader output')
                checks.append(dict(split=split,source_id=sid,index=int(i),shape=list(sample['trajectory_target'].shape)))
    if sorted(map(int,source_counts))!=[0,1,2,4,5,6,8,9,10]:raise ValueError('engineering coverage missing a source')
    receipt.update(status='PREPARED_SUBSET',completed_sequences=len(sequences),failed_sequences=failures,
        source_counts=source_counts,windows=dict(train=len(ds),dev=len(dev)),reader_checks=checks,
        actual_allocated_bytes=sum(p.stat().st_blocks*512 for p in destination.rglob('*') if p.is_file() and not p.is_symlink()),
        min_free_observed_bytes=min(minimum_free,shutil.disk_usage(out).free),free_bytes=shutil.disk_usage(out).free,
        source_payload_bytes_copied=0,geometry_files_copied=len(geometries),elapsed_s=time.monotonic()-start,
        original_v6_resume_ready=False,original_stats_replaced=False)
    if receipt['actual_allocated_bytes']>8*GIB:raise RuntimeError('subset exceeded8GiB budget')
    save(out/'preparation.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('reader_checks','failed_sequences')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('inventory','prepare'),required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'tmp/wam-v5/plan-compact.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-preparation',action='store_true',help='Reuse only unchanged, completed owned cache bindings.')
    args=parser.parse_args()
    if args.stage=='inventory':inventory(args)
    else:prepare(args)


if __name__=='__main__':main()
