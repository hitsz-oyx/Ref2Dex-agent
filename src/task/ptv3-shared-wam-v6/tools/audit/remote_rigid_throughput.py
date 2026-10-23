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

def summarize(out):
    import numpy as np
    m=read(out/'smoke-manifest.json');a=read(out/'input-audit.json');coverage=read(out/'input-coverage-after-repair.json')
    rows=[json.loads(x) for x in (out/'sequence-times.jsonl').read_text().splitlines()]
    assert m['status']=='COMPLETED' and len(rows)==32 and m['admitted']==32
    source=[];weighted=0.;frame_weighted=0.;weighted_cpu=0.
    for sid,pop in sorted(a['by_source'].items(),key=lambda x:int(x[0])):
        rs=[r for r in rows if r['source_id']==int(sid)];wall=sum(r['wall_s'] for r in rs);cpu=sum(r['cpu_s'] for r in rs)
        frames=sum(r['frames'] for r in rs);service=pop['parents']*wall/len(rs);frame_service=pop['frames']*wall/frames
        weighted+=service;frame_weighted+=frame_service;weighted_cpu+=pop['parents']*cpu/len(rs)
        source.append(dict(source_id=int(sid),population=pop['parents'],n=len(rs),frames=frames,
                           task_wall_sum_s=wall,task_cpu_sum_s=cpu,p50_s=float(np.median([r['wall_s'] for r in rs])),
                           extrapolated_service_s=service,frame_extrapolated_service_s=frame_service))
    telemetry=[json.loads(x) for x in (out/'telemetry.jsonl').read_text().splitlines()];rates=[]
    for first,last in zip(telemetry,telemetry[1:]):
        dt=last['monotonic']-first['monotonic']
        rates.append({k:(last['nfs'][op][field]-first['nfs'][op][field])/dt/1e6 for k,op,field in [('read','READ','received'),('write','WRITE','sent')]})
    input_rows=[json.loads(x) for x in (out/'input-rows.jsonl').read_text().splitlines()]
    p=production();block=os.statvfs(out).f_frsize
    allocated=sum(sum(((size+block-1)//block)*block for size in p.fixed_conversion_file_sizes(r['frames'],r['objects'])) for r in input_rows)
    candidate=sum(max(0,r['frames']-27)*(r['objects'] if r['source_id']<3 else 1) for r in input_rows)
    manifest_upper=candidate*2048+len(input_rows)*(65536+8192)+512*2**20
    staging=max(r['conversion_bytes'] for r in input_rows)*12
    peak=allocated+manifest_upper+staging+2*2**30
    quantiles=np.quantile([r['wall_s'] for r in rows],[0,.5,.9,.95,1]).tolist()
    result=dict(status='SMOKE_PASS_ONE_HOUR_GATE_NOT_DEMONSTRATED',run_commit=m['commit'],parents=32,admitted=32,wall_s=m['wall_s'],
        single_sequence_quantiles_s=dict(zip(['min','p50','p90','p95','max'],quantiles)),output_bytes=m['output_bytes'],conversion_bytes=m['conversion_bytes'],
        per_source=source,average_cpu_cores=m['average_cpu_cores'],system_cpu_busy_fraction=m['system_cpu_busy_fraction'],
        peak_sampled_rss_sum_bytes=m['peak_sampled_tree_rss_sum_bytes'],maximum_worker_highwater_rss_bytes=max(r['process_peak_rss_bytes'] for r in rows),
        nfs_read_rpc_MB_s=m['nfs_read_rpc_MB_s'],nfs_write_rpc_MB_s=m['nfs_write_rpc_MB_s'],
        nfs_peak_read_MB_s=max(r['read'] for r in rates),nfs_peak_write_MB_s=max(r['write'] for r in rates),
        nfs_scope='host NFS mount including concurrent input QA; read is not exclusive conversion IO',
        naive_batch_extrapolation_h=m['wall_s']/32*39527/3600,
        source_weighted_12_workers_h=weighted/12/3600,source_frame_weighted_12_workers_h=frame_weighted/12/3600,
        cpu_service_only_12_workers_h=weighted_cpu/12/3600,
        ideal_48_workers_h=[weighted/48/3600,frame_weighted/48/3600],
        write_only_floor_s=a['full_fresh_conversion_exact_bytes']/116e6,
        baseline_write_only_floor_s=182331193978/116e6,
        at_sample_write_peak_s=a['full_fresh_conversion_exact_bytes']/max(r['write'] for r in rates)/1e6,
        planning_48_workers_minutes=[35,60],planning_scope='unmeasured scaling scenario, engineering allowance, not statistical interval or <=1h qualification',
        recommendation='4 disjoint shards x 12 workers = 48 total, one GPU per shard; leave 720/768 logical CPUs; validate bounded concurrency before full launch',
        input_coverage=coverage,
        storage=dict(block_bytes=block,fresh_fixed_logical_bytes=a['full_fresh_conversion_exact_bytes'],local_new_only_bytes=182331193978,
                     allocation_upper_bytes=allocated,candidate_windows_upper=candidate,manifest_upper_bytes=manifest_upper,
                     staging_12worker_upper_bytes=staging,filesystem_reserve_bytes=2*2**30,peak_upper_bytes=peak,
                     safety_floor_bytes=30*2**30,nas_free_now=shutil.disk_usage(out).free,
                     free_after_full_peak=shutil.disk_usage(out).free-peak))
    write(out/'summary.json',result)
    lines=['\n## 实测结算（原始预注册在上，未改判据）\n',
        '**裁决：32条转换PASS；≤1小时全量门尚未证明。** 输入原缺1个67,200B packet，已只复制到本run独立NAS目录，用显式alias补齐；未修改共享原件。全量39,527条必需文件可达，但只32条做了新转换/数值准入，不能称全量重新验证。\n',
        '代码实际运行commit `%s`；短checkout `%s`；原生产 `prepare_full_rigid_data.process_one` / `init_worker`，只改路径适配。远端graspenv Python3.8.20/Torch2.0.1+cu118/numpy1.24.4/SciPy1.10.1/trimesh4.12.2；RLG参考trimesh4.10.1，合同相同不冒称跨环境逐位相等。\n'%(m['commit'],m['root']),
        '|指标|实测|\n|---|---:|\n',
        '|母序列/准入|32/32|\n|wall s|%.6f|\n|输出/其中固定转换 B|%d / %d|\n|单条min/P50/P90/P95/max s|%s|\n|CPU核当量/系统忙比例|%.4f / %.4f%%|\n|采样进程树RSS峰值 GiB|%.4f|\n|单worker历史RSS高水位 MiB|%.3f|\n|NFS READ/WRITE RPC均速 MB/s|%.4f / %.4f|\n|NFS READ/WRITE约1秒峰值 MB/s|%.4f / %.4f|\n'%(m['wall_s'],m['output_bytes'],m['conversion_bytes'],' / '.join('%.6f'%v for v in quantiles),m['average_cpu_cores'],100*m['system_cpu_busy_fraction'],m['peak_sampled_tree_rss_sum_bytes']/2**30,result['maximum_worker_highwater_rss_bytes']/2**20,m['nfs_read_rpc_MB_s'],m['nfs_write_rpc_MB_s'],result['nfs_peak_read_MB_s'],result['nfs_peak_write_MB_s']),
        'CPU核当量来自本任务parent+已回收worker的rusage，不包括其它进程；RSS为1秒采样的进程RSS相加，含共享页重复计数，不是物理内存独占峰值；每worker另有ru_maxrss。NFS为整机同一mount的RPC字节，期间有本任务并行输入QA，尤其READ不能归为转换专属读量。无额外大读写benchmark。写入量接近实际输出，但不声称计数完全独占。\n',
        '## 全量公式与限制\n',
        '朴素 `24.5625/32*39527` = %.3fh，错误地把分层过采样与启动/尾部每批重复，不作主估计。分层服务估计 `sum_s N_s mean(task_wall_s)/12` = **%.3fh**；帧数估计 `sum_s F_s/F_sample_s sum(task_wall_s)/12` = **%.3fh**。二者为冷启动小样本情景，并非置信区间；CPU服务单独约%.3fh，剩余等待不能在GPU冷启动与NAS延迟间唯一拆分。\n'%(result['naive_batch_extrapolation_h'],result['source_weighted_12_workers_h'],result['source_frame_weighted_12_workers_h'],result['cpu_service_only_12_workers_h']),
        '原基线182,331,193,978B是新转换38,350条，另1,177条只读复用；NAS没有对应v6复用根，本次外推全39,527条fresh，固定输出**%dB**，多%dB；不把两者称完全相同工作量。\n'%(a['full_fresh_conversion_exact_bytes'],a['full_fresh_conversion_exact_bytes']-182331193978),
        '瓶颈判定：不是768核总CPU容量不足，也没有顺序NAS带宽持续打满的证据。32条有很强任务尾部：source6一条21.918s，其中CPU12.319s；其余人手任务多5.7–11.3s/CPU0.56–5.38s，含每worker首次MANO/CUDA代价。单条重建/接触计算与off-CPU等待混合，函数内部未剖析，不能把全部等待说成NAS瓶颈。source内少至n=1，冷缓存/长尾及扩worker曲线未验证。\n',
        '建议**4个互斥分片×12workers=48**，每GPU一组、数值库线程1，留720逻辑核；不是直接开768workers。48的理想线性服务项%.1f–%.1fmin；NAS写下界%.1fmin（引用已核116MB/s，不是本轮重测），若维持本轮写峰约%.1fmin。部署预算**35–60min**仅是未实测扩展情景，不是≤1h已通过。下一步最小动作是有上限的48并发吞吐验证，再排全量；本轮不自行启动。\n'%(result['ideal_48_workers_h'][0]*60,result['ideal_48_workers_h'][1]*60,result['write_only_floor_s']/60,result['at_sample_write_peak_s']/60),
        '## 输入与空间\n',
        '生产路径别名：母序列s0/1/2→NAS goal-point-wam.../repo/outputs/.../processed/sequences；s4/5/6/8/9/10→NAS ptv3-shared-wam-v3-nine-source-20261015-r1/data/native/payloads；packet→NAS v5/data/packets；geometry→NAS v5/data（manifest及2747几何文件均在，manifest与RLG参考SHA相同）；MANO→NAS v5/assets-mano；Shadow→NAS v5/assets-shadow/robot.xml。精确逐文件路径/字节在input-rows.jsonl，原始缺失在input-audit.json，补齐在input-overrides.json与input-coverage-after-repair.json。\n',
        '初始52,005个必需文件引用中52,004存在：51,994与RLG只读stat大小一致，10个HOT3D仅segment数组不同，其余数组逐字节身份相同；segment不被本次生产hand/contact/event/准入链读取（preparation.load_base仅透传、prepare_hand_overlay.prepare只使用hand/valid/object_pose/observed），因此保持NAS原件并显式保留差异，不把整个压缩包说成完全相等。未全量hash51,994个同大小文件，文件齐备不等于全量内容完整性PASS。\n',
        '唯一原缺：s1-local-extra-00000的MANO packet，RLG full-rigid.../remaining-legacy-parent/packet.npz→本run/input-patches/extra-grab-packet.npz，**67,200B**，SSH加密流已复制，远端独立核125帧/67200B；没有写入他人旧data目录。RLG↔4090直连只有**0.3–0.45MB/s**，182.331GB需约**4.69–7.03天**；本轮只传小metadata及67KB packet，没有迁移/删除本地182GB。\n',
        '空间固定逻辑转换%dB；按NAS %dB分配粒度估算上界%dB；manifest/逐窗证据预算上界%dB，12workers staging上界%dB，文件系统metadata预留%dB，总峰值上界**%dB**，另守30GiB安全底线。当前NAS可用%dB，扣全量峰值仍%dB。此预算含可能的full summary/provenance上界，和转换实际字节分开，未启动该全量。\n'%(a['full_fresh_conversion_exact_bytes'],block,allocated,manifest_upper,staging,2*2**30,peak,result['storage']['nas_free_now'],result['storage']['free_after_full_peak']),
        '## 逐source与逐母序列\n|source|全量N|样本n|样本帧|task wall合计s|task CPU合计s|\n|---|---:|---:|---:|---:|---:|\n']
    for s in source:lines.append('|%d|%d|%d|%d|%.6f|%.6f|\n'%(s['source_id'],s['population'],s['n'],s['frames'],s['task_wall_sum_s'],s['task_cpu_sum_s']))
    lines.append('\n|母序列|source|帧|秒|CPU秒|固定输出B|\n|---|---:|---:|---:|---:|---:|\n')
    for r in sorted(rows,key=lambda r:(r['source_id'],r['key'])):lines.append('|%s|%d|%d|%.6f|%.6f|%d|\n'%(r['key'],r['source_id'],r['frames'],r['wall_s'],r['cpu_s'],r['result']['conversion_actual_bytes']))
    lines.append('\n## 实际命令与边界\n\n```bash\ncd /home/wbcd/workspace/oyx_ws/Ref2Dex-agent\ngit fetch origin main\ngit pull --ff-only --no-stat origin main\nPYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/audit/remote_rigid_throughput.py --output /mnt/ugreen_nas/storage/oyx_storage/remote-rigid-throughput-20261023-r1 --mode audit\n# 已完成，禁止覆盖重跑：\nPYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/audit/remote_rigid_throughput.py --output /mnt/ugreen_nas/storage/oyx_storage/remote-rigid-throughput-20261023-r1 --mode smoke --workers 12 --device cuda:0\n# 仅离线重算摘要，可复跑，不做转换：\nPYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/remote_rigid_throughput.py --output /mnt/ugreen_nas/storage/oyx_storage/remote-rigid-throughput-20261023-r1 --mode summarize\n```\n\n原始input-audit保留补齐前状态；不要把再次audit覆盖它当“最初就齐”。metadata参考frozen-census.json.gz来自RLG冻结计划+recovery+唯一新增母序列；仅输入metadata，不含182GB转换数组。tmp/r49是指向NAS的短链接，scratch实际落NAS。home-cache-audit.json核到本次之后CUDA/torch/torch_extensions默认home缓存新增文件0。允许的系统盘写入是Git代码；没有系统盘数据产物。远端只fetch/pull、0次push；工具在RLG修改提交推送。未动他人进程/目录、未删除历史、0训练更新。\n')
    prefix=(out/'report.md').read_text().split('\n## 实测结算')[0]
    (out/'report.md').write_text(prefix+''.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_source','input_coverage')},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['audit','smoke','summarize'],required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--device',default='cpu');a=p.parse_args()
    out=a.output.resolve();assert NAS in out.parents and out.is_dir();assert 1<=a.workers<=12
    if shutil.disk_usage(out).free<30*2**30+4*2**30+256*2**20:raise RuntimeError('NAS space budget')
    short=ROOT/'tmp/r49';assert short.resolve()==out,'short TMPDIR must point to the NAS run'
    os.environ.update(TMPDIR=str(short/'scratch'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(out/'cache'),TORCH_HOME=str(out/'cache/torch'),CUDA_CACHE_PATH=str(out/'cache/cuda'),MPLCONFIGDIR=str(out/'cache/mpl'))
    sys.dont_write_bytecode=True
    if a.mode=='audit':
        if (out/'input-audit.json').exists():raise FileExistsError('immutable initial input audit already exists')
        audit(out)
    elif a.mode=='smoke':smoke(out,a.workers,a.device)
    else:summarize(out)

if __name__=='__main__':main()
