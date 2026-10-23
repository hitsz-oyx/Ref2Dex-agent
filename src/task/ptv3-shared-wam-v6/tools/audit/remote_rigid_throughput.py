"""NAS-only input census and 32-parent timing of unchanged production conversion.

The only adapters replace RLG absolute paths with verified NAS input paths and
asset locations. No label, reconstruction, admission or precision rule changes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[5]
NAS=Path('/mnt/ugreen_nas/storage/oyx_storage')
BASE=NAS/'ptv3-shared-wam-v5-20261019-r1'
DATA=BASE/'data'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def allocation(counts,total=32):
    ids=sorted(counts);extra=total-len(ids);n=sum(counts.values())
    quotas={s:extra*counts[s]/n for s in ids}
    out={s:1+int(quotas[s]) for s in ids}
    for s in sorted(ids,key=lambda s:(-(quotas[s]-int(quotas[s])),s))[:total-sum(out.values())]:out[s]+=1
    assert sum(out.values())==total
    return out

def production():
    task=ROOT/'src/task/ptv3-shared-wam-v6'
    sys.path[:0]=[str(task/'tools/audit'),str(task/'src'),str(ROOT/'src/task/ptv3-shared-wam-v5/src')]
    spec=importlib.util.spec_from_file_location('production_full_rigid',task/'tools/audit/prepare_full_rigid_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def audit(out):
    started=time.monotonic();meta=json.loads(gzip.decompress((out/'frozen-census.json.gz').read_bytes()))
    old_plan=read(DATA/'plan/plan.json');remote={j['key']:j for j in old_plan['jobs']}
    registry=read(DATA/'manifest.json');assert digest(DATA/'manifest.json')==meta['local_registry_sha256'],'NAS geometry registry differs from frozen RLG registry'
    native={r['key']:r for r in meta['rows']};p=production()
    source_roots={j['source_id']:j.get('source_root') for j in old_plan['jobs'] if j['source_id']<3}
    def inspect(job):
        row=native[job['key']];job=dict(job);sid=job['source_id'];prior=remote.get(job['key']);missing=[];files=[]
        if sid<3:
            folder=Path(prior['sequence_folder']) if prior else Path(source_roots[sid])/'processed/sequences'/job['source_sequence']
            job['sequence_folder']=str(folder)
            files.extend(folder/x for x in ('hand.npy','hand_valid.npy','poses.npy','pose_valid.npy'))
        else:
            payload=Path(prior['payload']) if prior else Path(job['payload'])
            job['payload']=str(payload);files.append(payload)
        packet=None
        if sid!=4:
            # Prefer the exact local packet basename, not an arbitrary older variant.
            if row.get('packet'):candidate=DATA/'packets/packets'/Path(row['packet']).name
            else:
                record=DATA/'packets/records'/(job['key']+'.json')
                info=read(record) if record.is_file() else {}
                candidate=DATA/'packets'/info['packet'] if info.get('status')=='READY' else record.with_name('MISSING-'+job['key'])
            packet=str(candidate);files.append(candidate)
        sizes=[]
        for path in files:
            if NAS not in path.parents:raise ValueError('input must remain on NAS: '+str(path))
            try:st=path.stat();sizes.append(dict(path=str(path),bytes=st.st_size));assert path.is_file()
            except (FileNotFoundError,PermissionError):missing.append(str(path))
        return dict(key=job['key'],source_id=sid,frames=row['frames'],objects=row['objects'],
                    baseline_status=row['status'],sequence_id=row['sequence_id'],split=next(iter(job['windows'])),
                    job=job,packet=packet,files=sizes,missing=missing,
                    conversion_bytes=p.fixed_conversion_sizes(row['frames'],row['objects']))
    with ThreadPoolExecutor(16) as pool:rows=list(pool.map(inspect,meta['jobs']))
    geometry_missing=[];geometry_bytes=0
    for g in registry['objects']:
        f=DATA/g['path']
        if not f.is_file():geometry_missing.append(str(f))
        else:geometry_bytes+=f.stat().st_size
    assets=[BASE/'assets-shadow/robot.xml',BASE/'assets-mano/MANO_LEFT.pkl',BASE/'assets-mano/MANO_RIGHT.pkl']
    asset_missing=[str(f) for f in assets if not f.is_file()]
    counts=Counter(r['source_id'] for r in rows);quotas=allocation(counts);selected=[]
    for sid,n in quotas.items():
        available=sorted((r for r in rows if r['source_id']==sid and not r['missing']),key=lambda r:(r['frames'],r['key']))
        if len(available)<n:raise ValueError('not enough complete inputs in source '+str(sid))
        selected.extend(available[min(len(available)-1,int((i+.5)*len(available)/n))]['key'] for i in range(n))
    assert len(selected)==len(set(selected))==32
    totals={}
    for sid in sorted(counts):
        rs=[r for r in rows if r['source_id']==sid];unique={x['path']:x['bytes'] for r in rs for x in r['files']}
        totals[str(sid)]=dict(parents=len(rs),complete=sum(not r['missing'] for r in rs),missing=sum(bool(r['missing']) for r in rs),
                              unique_available_input_bytes=sum(unique.values()),smoke=quotas[sid],frames=sum(r['frames'] for r in rs),conversion_bytes=sum(r['conversion_bytes'] for r in rs))
    expected=sum(r['conversion_bytes'] for r in rows if r['key'] in selected)
    assert expected<4*2**30,'registered smoke output too large'
    report=dict(status='INPUTS_COMPLETE' if not any(r['missing'] for r in rows) and not geometry_missing and not asset_missing else 'INPUTS_INCOMPLETE',
                elapsed_s=time.monotonic()-started,total=len(rows),by_source=totals,geometry_missing=geometry_missing,geometry_bytes=geometry_bytes,
                asset_missing=asset_missing,assets=[dict(path=str(f),bytes=f.stat().st_size) for f in assets if f.is_file()],
                frozen_plan_sha256=meta['plan_sha256'],registry_exact_match=True,registry_sha256=meta['local_registry_sha256'],
                missing_parents=[dict(key=r['key'],source_id=r['source_id'],paths=r['missing']) for r in rows if r['missing']],
                selected=selected,selected_conversion_exact_bytes=expected,
                full_fresh_conversion_exact_bytes=sum(r['conversion_bytes'] for r in rows),baseline_conversion_bytes=meta['local_conversion_bytes'],
                nas_free_after_audit=shutil.disk_usage(out).free,full_payload_numerical_integrity_checked=False)
    write(out/'input-audit.json',report)
    with (out/'input-rows.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)

def configure_worker(out,device):
    global PROD
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    PROD=production();PROD.REGISTRY=DATA
    PROD.local_job=lambda job:dict(job)  # Inputs already resolved to NAS.
    def options(**kw):
        kw.update(mano_root=BASE/'assets-mano',shadow_asset=BASE/'assets-shadow/robot.xml')
        return SimpleNamespace(**kw)
    PROD.SimpleNamespace=options
    PROD.init_worker(device,out)

def convert(args):
    out,row=args;before=resource.getrusage(resource.RUSAGE_SELF);start=time.monotonic()
    inventory=dict(key=row['key'],source_id=row['source_id'],sequence_id=row['sequence_id'],source_sequence=row['job']['source_sequence'],
                   split=row['split'],frames=row['frames'],objects=row['objects'],status='CONVERT',conversion_bytes=row['conversion_bytes'],packet=row['packet'])
    result=PROD.process_one((out,inventory,row['job'],None,None))
    after=resource.getrusage(resource.RUSAGE_SELF)
    return dict(key=row['key'],source_id=row['source_id'],frames=row['frames'],objects=row['objects'],
                wall_s=time.monotonic()-start,cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                started_monotonic=start,finished_monotonic=time.monotonic(),
                process_peak_rss_bytes=after.ru_maxrss*1024,pid=os.getpid(),result=result)

def mount_counters():
    # Linux /proc/self/mountstats per-op columns: ops,xmit,timeouts,sent,recv,...
    active=False;out={}
    for line in Path('/proc/self/mountstats').read_text().splitlines():
        if line.startswith('device '):active=' mounted on /mnt/ugreen_nas ' in line
        if active:
            t=line.split()
            if t and t[0] in ('READ:','WRITE:'):
                values=[int(x) for x in t[1:]];out[t[0][:-1]]=dict(ops=values[0],sent=values[3],received=values[4])
    return out

def process_tree(pid):
    children=Path('/proc/%d/task/%d/children'%(pid,pid));out=[pid]
    try:
        for child in children.read_text().split():out.extend(process_tree(int(child)))
    except FileNotFoundError:pass
    return out

def sample():
    rss=0;details=[]
    for pid in process_tree(os.getpid()):
        try:
            fields={line.split(':')[0]:line.split(':',1)[1].strip() for line in Path('/proc/%d/status'%pid).read_text().splitlines() if ':' in line}
            mem=int(fields.get('VmRSS','0 kB').split()[0])*1024;rss+=mem;details.append(dict(pid=pid,rss=mem))
        except FileNotFoundError:pass
    cpu=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return dict(monotonic=time.monotonic(),rss_tree_sum=rss,processes=details,cpu=cpu,load=Path('/proc/loadavg').read_text().strip(),nfs=mount_counters())

def smoke(out,workers,device):
    audit=read(out/'input-audit.json');assert audit['registry_exact_match'] and not audit['geometry_missing'] and not audit['asset_missing']
    rows=[json.loads(line) for line in (out/'input-rows.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['key'] in audit['selected']];assert len(selected)==32 and not any(r['missing'] for r in selected)
    dest=out/'smoke';dest.mkdir(exist_ok=False)
    for name in ('hands/sequences','contacts','events','receipts','windows','batches'):(dest/name).mkdir(parents=True,exist_ok=True)
    write(dest/'inventory.json',dict(plan_sha256=audit['frozen_plan_sha256']))
    write(dest/'contract.json',dict(identity=hashlib.sha256((audit['frozen_plan_sha256']+'full-rigid-v1').encode()).hexdigest(),training_allowed=False))
    import subprocess
    manifest=dict(status='RUNNING',host=os.uname().nodename,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  workers=workers,device=device,python=sys.version,converter='prepare_full_rigid_data.process_one',parents=32,
                  path_adapters=['already-resolved NAS local_job','NAS MANO root','NAS Shadow asset','identical NAS geometry registry'],
                  numerical_semantics_changed=False,training_updates=0,root=str(ROOT),output=str(dest),start_time=time.time(),source_read_only=True)
    write(out/'smoke-manifest.json',manifest)
    first=sample();start=time.monotonic();stopped=threading.Event();samples=[]
    def monitor():
        with (out/'telemetry.jsonl').open('x') as f:
            while not stopped.wait(1):
                s=sample();samples.append(s);f.write(json.dumps(s)+'\n');f.flush()
    thread=threading.Thread(target=monitor,daemon=True);thread.start();results=[]
    before=resource.getrusage(resource.RUSAGE_CHILDREN);own=resource.getrusage(resource.RUSAGE_SELF)
    pool=ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=configure_worker,initargs=(str(dest),device))
    try:
        futures=[pool.submit(convert,(str(dest),r)) for r in selected]
        with (out/'sequence-times.jsonl').open('x') as f:
            for future in as_completed(futures,timeout=1200):
                result=future.result();results.append(result);f.write(json.dumps(result)+'\n');f.flush()
                print(json.dumps(dict({k:v for k,v in result.items() if k!='result'},status=result['result']['status'])),flush=True)
        pool.shutdown(wait=True)
        manifest['status']='COMPLETED' if all(r['result']['status']=='ADMITTED' for r in results) else 'CONTRACT_REJECTED'
    except BaseException as exc:
        manifest.update(status='FAILED',error=repr(exc))
        for child in list(pool._processes.values()):child.terminate()  # Only this registered pool.
        pool.shutdown(wait=True);raise
    finally:
        stopped.set();thread.join(timeout=5);last=sample();elapsed=time.monotonic()-start
        after=resource.getrusage(resource.RUSAGE_CHILDREN);me=resource.getrusage(resource.RUSAGE_SELF)
        cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+me.ru_utime+me.ru_stime-own.ru_utime-own.ru_stime
        delta_cpu=[b-a for a,b in zip(first['cpu'],last['cpu'])];total_cpu=sum(delta_cpu[:8]);idle=delta_cpu[3]+delta_cpu[4]
        nfs={kind:{k:last['nfs'][kind][k]-first['nfs'][kind][k] for k in first['nfs'][kind]} for kind in first['nfs']}
        files=[p for p in dest.rglob('*') if p.is_file()];output_bytes=sum(p.stat().st_size for p in files)
        manifest.update(wall_s=elapsed,completed=len(results),admitted=sum(r['result']['status']=='ADMITTED' for r in results),
                        output_bytes=output_bytes,conversion_bytes=sum(r['result'].get('conversion_actual_bytes',0) for r in results),
                        cpu_s=cpu_s,average_cpu_cores=cpu_s/elapsed,system_cpu_busy_fraction=1-idle/max(1,total_cpu),
                        peak_sampled_tree_rss_sum_bytes=max([first['rss_tree_sum'],last['rss_tree_sum']]+[s['rss_tree_sum'] for s in samples]),
                        nfs_delta=nfs,nfs_read_rpc_MB_s=nfs.get('READ',{}).get('received',0)/elapsed/1e6,
                        nfs_write_rpc_MB_s=nfs.get('WRITE',{}).get('sent',0)/elapsed/1e6,
                        nas_free_after=shutil.disk_usage(out).free,root_free_after=shutil.disk_usage('/').free)
        write(out/'smoke-manifest.json',manifest);print(json.dumps(manifest),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['audit','smoke'],required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--device',default='cpu');a=p.parse_args()
    out=a.output.resolve();assert NAS in out.parents and out.is_dir();assert 1<=a.workers<=12
    if shutil.disk_usage(out).free<30*2**30+4*2**30+256*2**20:raise RuntimeError('NAS space budget')
    short=ROOT/'tmp/r49';assert short.resolve()==out,'short TMPDIR must point to the NAS run'
    os.environ.update(TMPDIR=str(short/'scratch'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(out/'cache'),TORCH_HOME=str(out/'cache/torch'))
    sys.dont_write_bytecode=True
    if a.mode=='audit':audit(out)
    else:smoke(out,a.workers,a.device)

if __name__=='__main__':main()
