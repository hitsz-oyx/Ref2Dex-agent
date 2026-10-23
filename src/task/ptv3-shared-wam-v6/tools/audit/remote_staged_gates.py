"""Independently reconcile a NAS half/full census before training admission."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run'),str(TASK/'tools/audit')]
from remote_staged_data import read,write,sha,NAS,PROBE,GIB
from ptv3_shared_wam_v6.prepared_reader import PreparedWindows
from ptv3_shared_wam_v5.cache_contract import metadata_identity,array_identity

REGISTRY=NAS/'ptv3-shared-wam-v5-20261019-r1/data/manifest.json'


def native_clock(row, tuples):
    source=row['source_id'];job=row['job'];n=row['frames']
    if source==0:
        f=Path(job['sequence_folder'])/'frame_ids.npy'
        frames=np.load(f,allow_pickle=False)
        assert frames.shape==(n,) and np.isfinite(frames).all()
        bad=np.diff(frames)!=4
        native=dict(uri=str(f),fields=['frame_ids'],complete_certificate_fields=False)
    elif source in (1,2):
        return np.ones(len(tuples),bool),dict(fields=['native frame grid'],
            complete_certificate_fields=False,scope='30Hz converted frame grid; no absolute timestamps asserted')
    else:
        with np.load(job['payload'],allow_pickle=False) as z:
            clock=z['time'];assert clock.shape==(n,) and np.isfinite(clock).all()
            bad=~np.isclose(np.diff(clock),1/30,atol=1e-5,rtol=0)
            if 'segment' in z:bad|=np.diff(z['segment'])!=0
            fields=set(z.files)
            required={'task_spec','source_role','available_before_reset','reset_timestamp_ns',
                'source_available_timestamp_ns','source_artifact_sha256'}
            assert not required.issubset(fields),'certificate candidate requires explicit adjudication'
            native=dict(uri=job['payload'],fields=sorted(fields),complete_certificate_fields=False)
    prefix=np.r_[0,np.cumsum(bad,dtype=np.int64)];ticks=tuples[:,2]
    return prefix[ticks+24]==prefix[ticks-3],native


def publish(out,stage):
    plan=read(out/'plan.json');chosen=set(plan['half_keys'])
    assert sha(REGISTRY)==plan['registry_sha256'],'geometry registry drift'
    selected=[r for r in plan['rows'] if stage=='full' or r['key'] in chosen]
    rows=[]
    for r in selected:
        dest=out/('half' if r['key'] in chosen else 'remaining')
        with (dest/'receipts'/(r['key']+'.json')).open() as f:receipt=json.loads(f.readline())
        assert receipt['status']=='ADMITTED' and receipt['sequence_id']==r['sequence_id']
        rows.append((r,receipt))
    directory=out/(stage+'-census');directory.mkdir(exist_ok=False)
    lengths={s:sum(r['admitted_windows'] for _,r in rows if r['split']==s) for s in ('train','dev')}
    arrays={s:np.lib.format.open_memmap(directory/(s+'-all.npy'),mode='w+',dtype='i4',shape=(n,3)) for s,n in lengths.items()}
    clocks={s:np.lib.format.open_memmap(directory/(s+'-clock-valid.npy'),mode='w+',dtype=bool,shape=(n,)) for s,n in lengths.items()}
    phases={s:np.lib.format.open_memmap(directory/(s+'-phase.npy'),mode='w+',dtype='u1',shape=(n,)) for s,n in lengths.items()}
    positions=Counter();sources=defaultdict(Counter);coverage=Counter();bytes_measured=0
    reference={r['key']:r for r in json.loads(gzip.decompress((out/'tuple-reference.json.gz').read_bytes()))}
    seen=defaultdict(dict);raw_events=Counter();source_certs=defaultdict(Counter)
    with (directory/'sequences.jsonl').open('x') as ledger,(directory/'file-stat-ledger.jsonl').open('x') as files:
        for number,(r,receipt) in enumerate(sorted(rows,key=lambda x:x[0]['sequence_id'])):
            assert sha(receipt['window_shard'])==receipt['window_shard_sha256']
            with np.load(receipt['window_shard'],allow_pickle=False) as z:
                tuples=z['tuples'];phase=z['phase'];geom=z['geometry_class']
            assert len(tuples)==receipt['admitted_windows']==reference[r['key']]['windows']
            assert hashlib.sha256(tuples.tobytes()).hexdigest()==reference[r['key']]['tuple_payload_sha256']
            assert (tuples[:,0]==r['sequence_id']).all() and len(np.unique(tuples,axis=0))==len(tuples)
            split=r['split'];source=r['source_id'];offset=positions[split]
            valid,native=native_clock(r,tuples);assert valid.all(),r['key']+' native boundary crossing'
            actual_bytes=0;metadata={}
            for field,identity in [('path','hand_meta_identity'),('contact_path','contact_meta_identity'),('event_path','event_meta_identity')]:
                folder=Path(receipt[field]);assert metadata_identity(folder/'meta.json')==receipt[identity]
                metadata[field]=read(folder/'meta.json')
                for f in folder.iterdir():
                    if f.is_file():
                        n=f.stat().st_size;actual_bytes+=n
                        files.write(json.dumps(dict(path=str(f),bytes=n,key=r['key']))+'\n')
            assert metadata['contact_path']['hand_identity']==receipt['hand_meta_identity']
            assert metadata['event_path']['contact_meta_identity']==receipt['contact_meta_identity']
            assert actual_bytes==receipt['conversion_expected_bytes']==receipt['conversion_actual_bytes']==r['conversion_bytes']
            bytes_measured+=actual_bytes
            for kind,key in [('mother',r['mother_identity']),('payload',metadata['path']['payload_identity']),
                    ('hand_path',str(Path(receipt['path']).resolve()))]:
                previous=seen[kind].setdefault(key,split)
                assert previous==split,'cross-split '+kind
            arrays[split][offset:offset+len(tuples)]=tuples
            clocks[split][offset:offset+len(tuples)]=valid
            phases[split][offset:offset+len(tuples)]=phase
            assert np.array_equal(arrays[split][offset:offset+len(tuples)],tuples)
            positions[split]+=len(tuples);sources[source][split]+=len(tuples);sources[source][split+'_mothers']+=1
            values,counts=np.unique(np.stack([phase,geom],1),axis=0,return_counts=True)
            for (p,g),count in zip(values,counts):coverage[(split,source,int(p),int(g))]+=int(count)
            event=Path(receipt['event_path']);types=np.load(event/'event_type.npy',mmap_mode='r');ev=np.load(event/'event_valid.npy',mmap_mode='r')
            for slot in np.unique(tuples[:,1]):
                ticks=tuples[tuples[:,1]==slot,2];kinds=types[ticks,slot];known=ev[ticks,slot]
                for kind in range(3):raw_events[(source,split,kind)]+=int(((kinds==kind)&known).sum())
                raw_events[(source,split,3)]+=int((~known).sum())
            source_certs[source][split+'_uncertifiable_goal']+=len(tuples)
            receipt.update(normalized_offset=offset,native_evidence=native,clock_valid_windows=len(tuples),mother_identity=r['mother_identity'])
            ledger.write(json.dumps(receipt,separators=(',',':'))+'\n')
            if (number+1)%2000==0:print(json.dumps(dict(stage='census',mothers=number+1,rows=dict(positions))),flush=True)
    for array in list(arrays.values())+list(clocks.values())+list(phases.values()):array.flush()
    for split,n in lengths.items():assert positions[split]==n
    expected=sum(r['conversion_bytes'] for r in selected);assert bytes_measured==expected
    with (directory/'phase-geometry-source.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['split','source_id','phase','geometry_class','windows'])
        for split in lengths:
            for source in sorted(sources):
                for p in range(5):
                    for g in range(4):writer.writerow([split,source,p,g,coverage[(split,source,p,g)]])
    for f in directory.iterdir():
        if f.is_file():f.chmod(0o664)
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in directory.iterdir() if f.is_file()}
    manifest=dict(schema='ref2dex.nine-source-readiness.v1',expected_mothers=len(rows),files=files,
        sources=[dict(source_id=s,**{split:dict(admitted_windows=sources[s][split],mothers=sources[s][split+'_mothers']) for split in lengths}) for s in sorted(sources)],
        stage=stage,plan_sha256=sha(out/'plan.json'),geometry_sha256=sha(REGISTRY),
        conversion_bytes=bytes_measured,training_allowed=False,source8_dev_missing=True,
        raw_events={str(k):v for k,v in raw_events.items()},
        certification_by_source={str(s):dict(v,certified_goal=0,certified_event=0) for s,v in source_certs.items()},
        claim_scope='nine-source TRAIN, eight-source DEV; fixed TRAIN panel is not generalization',
        tuple_identity='all actual ordered tuples equal corresponding frozen local mother tuple payloads',
        input_version='NAS native packs, separately identified; original file byte identity not asserted')
    write(directory/'manifest.json',manifest,immutable=True)
    return directory


def audit(out,stage):
    directory=publish(out,stage);dest=out/(stage+'-gates');dest.mkdir(exist_ok=False)
    manifest=directory/'manifest.json';pinned=sha(manifest)
    def datasets():return {s:PreparedWindows(manifest,s,REGISTRY,pinned) for s in ('train','dev')}
    # Reuse independently written SciPy/recorded-label checks on this new
    # manifest. No optimizer admission follows from this audit-only reader.
    import verify_full_data_gates as checks
    checks.datasets=datasets
    sampled=checks.sample(dest);write(dest/'sample.json',sampled,immutable=True)
    geometry=checks.geometry(dest);write(dest/'geometry.json',geometry,immutable=True)
    ds=datasets();rng=np.random.default_rng(20261023);hashed=Counter()
    with (dest/'hashes.jsonl').open('x') as ledger:
        all_rows=list(ds['train'].rows.values())
        for source in sorted({r['source_id'] for r in all_rows}):
            pool=[r for r in all_rows if r['source_id']==source]
            chosen=rng.choice(len(pool),max(1,int(np.ceil(len(pool)*.015))),replace=False)
            for idx in chosen:
                r=pool[int(idx)];seq,contact,event=ds[r['split']].sequence(r['sequence_id'])
                for field,values in [('path',seq),('contact_path',contact)]:
                    assert array_identity(values)==read(Path(r[field])/'meta.json')['payload_identity']
                enter,leave=checks.proximity_counts(contact['contact'],contact['valid'])
                regenerated=checks.annotate_events(enter,leave,contact['valid'],
                    observation_stride=checks.OBSERVATION_STRIDE_BY_SOURCE.get(source,1))
                regenerated.pop('transitions');regenerated.update(enter_count=enter,leave_count=leave,raw_valid=np.asarray(contact['valid']))
                assert set(regenerated)==set(event)
                for name,value in regenerated.items():assert np.array_equal(value,event[name]),(r['key'],name)
                hashed['events_recomputed']+=1
                for field in ('path','contact_path','event_path'):
                    for f in Path(r[field]).iterdir():
                        if f.is_file():
                            ledger.write(json.dumps(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size))+'\n')
                            hashed['files']+=1;hashed['bytes']+=f.stat().st_size
    summary=dict(status='DATA_CHECKS_PASS_E2E_REQUIRED',manifest=str(manifest),manifest_sha256=pinned,
        sample=sampled,geometry=geometry,hash_sample=dict(hashed),conversion_bytes=read(manifest)['conversion_bytes'],
        floor_bytes=30*GIB,free_bytes=shutil.disk_usage(out).free,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gates=dict(reader='PASS',split_content='PASS',identity_bytes='PASS',disk='PASS',
            coverage='RESTRICTED_NINE_TRAIN_EIGHT_DEV',certification_counts='PASS_UNKNOWN_INPUTS',
            local_remote_tuples='PASS',e2e='NOT_RUN'),automatic_restart=False)
    assert summary['free_bytes']>30*GIB
    write(dest/'data-checks.json',summary,immutable=True);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('half','full'),required=True);a=p.parse_args();out=a.output.resolve()
    assert NAS in out.parents
    audit(out,a.stage)


if __name__=='__main__':main()
