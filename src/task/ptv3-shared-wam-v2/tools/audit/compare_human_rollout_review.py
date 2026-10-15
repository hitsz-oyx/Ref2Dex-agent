"""Compare immutable human review intervals with existing engineering/admission gates.

Descriptive CPU file analysis; never changes labels, membership or source data.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np


def spans(ticks):
    ticks=np.asarray(ticks,dtype=np.int64)
    if not len(ticks):return []
    return [[int(c[0]),int(c[-1])] for c in np.split(ticks,np.flatnonzero(np.diff(ticks)>1)+1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--subject',default='s1');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-cache',type=Path)
    a=p.parse_args();source=a.corpus.resolve();out=a.output.resolve()
    if out.exists() or source==out or source in out.parents:raise ValueError('fresh separate report output required')
    m=json.loads((source/'manifest.json').read_text());records={r['sequence']:r for r in m['records']};contract=m['observation_contract']
    engineering=json.loads((source/'current_observation_audit.json').read_text());annotations=[]
    for f in sorted((a.review/'annotations').glob('*.json')):
        r=json.loads(f.read_text())
        if not r['sequence'].startswith(a.subject+'_'):continue
        if Path(r['source_root']).resolve()!=source:raise ValueError('annotation source differs from reviewed corpus')
        annotations.append(dict(r,annotation_file=str(f.resolve())))
    if not annotations:raise ValueError('no saved annotations for subject')
    retained={}
    if a.fit_cache:
        cache=json.loads((a.fit_cache/'manifest.json').read_text())
        if Path(cache['source']).resolve()!=source:raise ValueError('fitted cache belongs to a different rollout')
        for split in ['train','val']:
            seq=np.load(a.fit_cache/(split+'-sequence.npy'));tick=np.load(a.fit_cache/(split+'-tick.npy'))
            for i,r in enumerate(cache['metadata'][split]['records']):retained[r['sequence']]=tick[seq==i]
    rows=[]
    for human in sorted(annotations,key=lambda x:(x['sequence'],x['created_at'])):
        record=records[human['sequence']];start,end=human['start_frame'],human['end_frame_exclusive'];length=record['frames']
        if not 0<=start<end<=length:raise ValueError('annotation interval outside sequence')
        with np.load(source/record['path']/'trajectory.npz') as data:
            ref=data['reference_current'];hand=np.linalg.norm(data['body_state'][:,contract['key_body_ids'],:3]-ref[:,119:167].reshape(-1,16,3),axis=-1).mean(-1)
            error=np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
            count=length-24;peak=np.maximum.reduce([error[i:i+count] for i in range(25)])
            ticks=np.flatnonzero(peak<=m.get('max_reference_error_m',.1))
            np.testing.assert_array_equal(ticks,record['admitted_ticks'])
            contact=(abs(data['body_contact_forces'][:,contract['contact_body_ids']])>.1).any(-1).any(-1)
            expected=(ref[:,168:184][:,[3,6,9,12,15]]>0).any(-1)
            # H4 observations and future24: overlap, not just annotated starts.
            touched=(np.maximum(ticks-3,0)<end)&(ticks+24>=start)
            fit=retained.get(human['sequence'])
            row=dict(sequence=human['sequence'],annotation_id=human['annotation_id'],start=start,end_exclusive=end,
                verdict=human['verdict'],reason=human['reason'],note=human['note'],split=record['split'],frames=length,
                engineering_pass=human['sequence'] in engineering['records'],possible_windows=count,
                position_admitted_windows=len(ticks),position_admitted_fraction=len(ticks)/count,
                position_admitted_ranges_inclusive=spans(ticks),windows_touching_annotation=int(touched.sum()),
                fit_admitted_windows=len(fit) if fit is not None else None,
                hand_mean_mm=float(hand[start:end].mean()*1000),hand_p95_mm=float(np.percentile(hand[start:end],95)*1000),
                hand_max_mm=float(hand[start:end].max()*1000),object_mean_mm=float(error[start:end].mean()*1000),
                object_max_mm=float(error[start:end].max()*1000),expected_contact_frames=int(expected[start:end].sum()),
                expected_without_measured_contact=int((expected[start:end]&~contact[start:end]).sum()),
                first_object_error_over100mm=next(iter(np.flatnonzero(error>.1).tolist()),None),
                retained_max_horizon_hand_mm=float(hand[ticks[:,None]+np.arange(25)].max()*1000) if len(ticks) else None,
                object_z_last_m=float(data['obj'][end-1,2,3]),reference_z_last_m=float(data['reference_obj'][end-1,2,3]))
            rows.append(row)
    out.mkdir(parents=True)
    report=dict(status='COMPLETED',checked_at=time.time(),subject=a.subject,annotations=len(rows),unique_sequences=len({r['sequence'] for r in rows}),
        total_subject_sequences=sum(name.startswith(a.subject+'_') for name in records),
        scope='human notes versus engineering and existing position/window admission, not a binary grasp classifier; unannotated is not automatically usable',records=rows)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (out/'annotation-snapshot.json').write_text(json.dumps(annotations,indent=2,ensure_ascii=False)+'\n')
    fields=[k for k in rows[0] if k!='position_admitted_ranges_inclusive']
    with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    lines=['| 序列 | 人工备注 | 原位置窗口保留 | 手误差 mean mm | 物体误差 max mm |','|---|---|---:|---:|---:|']
    for r in rows:lines.append('| %s | %s | %d/%d (%.1f%%) | %.1f | %.1f |'%(r['sequence'],r['note'].replace('|','/').replace('\n',' '),r['position_admitted_windows'],r['possible_windows'],100*r['position_admitted_fraction'],r['hand_mean_mm'],r['object_max_mm']))
    lines+=['','所有保存的判定字段照原样保留；备注的异常不自动转换成unusable。',
        '工程PASS不等同成功抓取。部分失败序列保留可用前缀可以合理，整条标注不足以指定失败起点。',
        '参考接触与实测接触不匹配是诊断线索；实测力也可能来自桌面，不保证手物抓持。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
