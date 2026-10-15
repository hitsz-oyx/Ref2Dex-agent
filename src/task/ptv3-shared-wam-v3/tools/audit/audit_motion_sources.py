"""Bounded all-source recorded-frame motion scan; never changes training admission."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip,json,math,shutil,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.review_data import Catalog,NAMES
from pose_mesh_data.motion_diagnostics import transition_diagnostics,window_coverage

BINS=np.r_[0.,np.geomspace(1e-8,1e6,2048),np.inf]


def write(path,value):
    pending=path.with_suffix('.part.json');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs'/TASK.name not in out.parents or not 1<=a.workers<=2:
        raise ValueError('fresh bounded v3 output and one/two workers required')
    catalog=Catalog(a.recipe);out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<20*2**30:raise RuntimeError('20GiB floor')
    records=sorted([r for sid in NAMES for r in catalog.select(sid)],key=lambda r:(r.source_id==4,r.source_id,r.split,r.row))
    out.mkdir();started=time.monotonic();load_lock=threading.Lock()
    progress=dict(status='RUNNING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        recipe=str(a.recipe.resolve()),max_seconds=1200,artifact_budget_bytes=200_000_000,
        device='CPU: file/array diagnostics only',workers=a.workers,total=len(records),processed=0,
        policy='Read-only ranking; no threshold tuning, filtering or training-statistic changes.')
    groups={};top=defaultdict(list);done_sources=Counter();processed=0
    def inspect(record):
        with load_lock:sequence=catalog.load(record)
        data=sequence.data;covered=window_coverage(sequence.ticks,sequence.frames)
        base=dict(source_id=record.source_id,source=NAMES[record.source_id],split=record.split,row=record.row,
            sequence=record.sequence,frames=sequence.frames,retained_windows=record.windows,payload=sequence.origin,
            existing_quarantine=sequence.motion_quality)
        results=[];record_metrics={}
        for name,values,mask,start,end in transition_diagnostics(data,sequence.queries):
            known=values[mask]
            if not np.isfinite(known).all():raise ValueError('nonfinite valid metric '+record.key+' '+name)
            context=covered[start]&covered[end]
            selected=mask&context.reshape((-1,)+(1,)*(mask.ndim-1))
            entry={}
            for scope,valid in [('full_capture',mask),('retained_window_context',selected)]:
                x=values[valid]
                if not len(x):continue
                flat=np.flatnonzero(valid)
                coordinate=np.unravel_index(int(flat[int(np.argmax(x))]),values.shape)
                peak=dict(frame_start=int(start[coordinate[0]]),frame_end=int(end[coordinate[0]]),
                    entity=list(map(int,coordinate[1:])),value=float(x.max()),
                    covered_by_retained_windows=bool(context[coordinate[0]]))
                entry[scope]=dict(count=len(x),maximum=peak)
                results.append((name,scope,len(x),float(x.sum()),float(np.square(x).sum()),
                    np.histogram(x,bins=BINS)[0],int((x==0).sum()),peak))
            record_metrics[name]=entry
        return base,record_metrics,results
    def checkpoint():
        progress.update(processed=processed,by_source_split=dict(done_sources),elapsed_s=time.monotonic()-started)
        write(out/'progress.json',progress)
        if (out/'STOP').exists() or progress['elapsed_s']>1200 or shutil.disk_usage(out).free<20*2**30:
            raise RuntimeError('audit STOP/deadline/free-space boundary')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>200_000_000:
            raise RuntimeError('audit artifact budget')
    try:
        checkpoint()
        with gzip.open(out/'records.jsonl.gz','xt',compresslevel=1) as f,ThreadPoolExecutor(max_workers=a.workers) as pool:
            # Keep bounded futures; do not let a fast reader retain the full corpus in memory.
            for offset in range(0,len(records),32):
                for base,metrics,results in pool.map(inspect,records[offset:offset+32]):
                    f.write(json.dumps(dict(**base,metrics=metrics),separators=(',',':'))+'\n')
                    sid,split=base['source_id'],base['split'];key=f'{sid}:{split}'
                    done_sources[key]+=1;processed+=1
                    for name,scope,count,total,square,hist,zeros,peak in results:
                        g=groups.setdefault((key,name,scope),dict(count=0,sum=0.,squares=0.,zeros=0,hist=np.zeros(len(BINS)-1,np.int64),maximum=0.))
                        g['count']+=count;g['sum']+=total;g['squares']+=square;g['zeros']+=zeros;g['hist']+=hist
                        g['maximum']=max(g['maximum'],peak['value'])
                        if scope=='retained_window_context':
                            slots=top[(key,name)];slots.append(dict(**base,metric=name,peak=peak))
                            slots.sort(key=lambda x:x['peak']['value'],reverse=True);del slots[10:]
                f.flush();checkpoint()
                if processed%512==0:print(json.dumps(progress),flush=True)
        summaries={}
        for (key,name,scope),g in groups.items():
            quantiles={}
            cumulative=np.cumsum(g['hist'])
            for q in (.5,.9,.95,.99,.999):
                rank=max(1,math.ceil(q*g['count']));index=int(np.searchsorted(cumulative,rank))
                lo,hi=BINS[index:index+2]
                quantiles[str(q)]=dict(lower=float(lo),upper=float(hi) if np.isfinite(hi) else None,
                    estimate=0. if rank<=g['zeros'] else float(math.sqrt(lo*hi)) if np.isfinite(hi) else float(lo))
            summaries.setdefault(key,{}).setdefault(scope,{})[name]=dict(count=g['count'],mean=g['sum']/g['count'],
                rms=math.sqrt(g['squares']/g['count']),maximum=g['maximum'],quantiles=quantiles)
        result=dict(status='COMPLETED',processed=processed,by_source_split=dict(done_sources),
            metrics=summaries,histogram_bins=BINS[:-1].tolist(),
            interpretation='Recorded transitions only. Quantiles are histogram-bounded; maximum/mean/RMS exact. Surface metric is maximum over32 fixed CAD queries; hand metrics pool valid landmarks. Context means both transition endpoints lie inside retained H4+K24 coverage; not every entity necessarily participates in every overlapping window.',
            limitation='Numerical motion ranking is not physical adjudication. TRAIN/DEV summaries separate; no data, normalization, recipe or running training modified.')
        write(out/'summary.json',result)
        write(out/'top-candidates.json',dict(status='REVIEW_CANDIDATES_ONLY',groups={key+' / '+name:rows for (key,name),rows in top.items()}))
        progress['status']='COMPLETED';checkpoint();print(json.dumps(progress),flush=True)
    except BaseException as error:
        progress.update(status='FAILED',reason=repr(error));write(out/'progress.json',progress);raise


if __name__=='__main__':main()
