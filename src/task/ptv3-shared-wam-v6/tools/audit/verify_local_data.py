"""Read every prepared sequence and check geometry/contact independently."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();start=time.monotonic()
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    from ptv3_shared_wam_v6 import paths
    import numpy as np
    from ptv3_shared_wam_v5.data import V5Windows
    from ptv3_shared_wam_v5.cache_contract import metadata_identity
    data=out/'data';manifest=json.loads((data/'recipe/manifest.json').read_text())
    receipt=json.loads((out/'preparation.json').read_text())
    if receipt['status']!='PREPARED_SUBSET' or manifest['training_allowed'] or not manifest['engineering_subset']:
        raise ValueError('explicit prepared non-training subset required')
    events=json.loads((data/'events/manifest.json').read_text())
    if events['parent_manifest_identity']!=metadata_identity(data/'recipe/manifest.json'):raise ValueError('events bound to different parent')
    event_rows={r['key']:r for r in events['sequences']}
    registry=json.loads((data/'geometry/manifest.json').read_text());geometry={}
    for row in registry['objects']:
        with np.load(data/'geometry'/row['path']) as archive:geometry[row['key']]=archive['surface'].copy()
    source_counts={};maximum_alignment=0.;contact_difference=0.;geometry_checks=0;reader_windows=0
    arrays_checked=0;frames_checked=0;masked_nonfinite_poses=0;event_arrays_checked=0
    for i,row in enumerate(manifest['sequences']):
        if time.monotonic()-start>600:raise TimeoutError('ten-minute verification bound')
        folder=Path(row['path']);cf=Path(row['contact_path']);meta=json.loads((folder/'meta.json').read_text())
        if metadata_identity(folder/'meta.json')!=row['hand_meta_identity'] or metadata_identity(cf/'meta.json')!=row['contact_meta_identity']:
            raise ValueError('published sequence metadata changed')
        arrays={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy')}
        c=np.load(cf/'contact.npy',mmap_mode='r');parts=np.load(cf/'part.npy',mmap_mode='r');valid=np.load(cf/'valid.npy',mmap_mode='r')
        n=len(arrays['joints']);objects=len(row['objects'])
        if arrays['joints'].shape!=(n,2,21,3) or arrays['surface_offsets'].shape!=(n,2,778,3):raise ValueError('hand geometry shape')
        if c.shape!=(n,objects,2048) or parts.shape!=c.shape or valid.shape!=(n,objects):raise ValueError('contact shape')
        if c.dtype!=np.float16 or parts.dtype!=np.uint8:raise ValueError('contact dtype')
        for name,value in arrays.items():
            if name=='object_poses':
                known=arrays['object_pose_valid'];finite=np.isfinite(value).all((-1,-2))
                masked_nonfinite_poses+=int((~known&~finite).sum());meaningful=value[known]
            else:meaningful=value
            if np.issubdtype(value.dtype,np.floating) and not np.isfinite(meaningful).all():raise ValueError('nonfinite valid sequence array: '+name)
            arrays_checked+=1
        if not np.isfinite(c).all() or c.min()<0 or c.max()>1 or parts.max()>15:raise ValueError('contact range')
        if np.any(c[~valid]) or np.any(valid[~arrays['observed_frame']]):raise ValueError('masked/synthetic contact leaked')
        maximum_alignment=max(maximum_alignment,meta['alignment_max_error_m'])
        if meta['alignment_max_error_m']>5e-5 or meta['surface_quantization_max_error_m']>1e-4:raise ValueError('reconstruction tolerance')
        erow=event_rows[row['key']];ep=Path(erow['path'])
        if metadata_identity(ep/'meta.json')!=erow['event_meta_identity']:raise ValueError('event metadata changed')
        e={name:np.load(ep/(name+'.npy'),mmap_mode='r') for name in ('event_type','event_delta_steps','event_valid')}
        if any(a.shape!=(n,objects) for a in e.values()) or not np.isin(e['event_type'],[0,1,2]).all():raise ValueError('event encoding')
        if not np.isfinite(e['event_delta_steps']).all() or np.any(e['event_delta_steps'][e['event_valid']]<0):raise ValueError('event time')
        for path in ep.glob('*.npy'):
            values=np.load(path,mmap_mode='r')
            if values.shape!=(n,objects) or not np.isfinite(values).all():raise ValueError('event auxiliary array')
            event_arrays_checked+=1
        # Brute-force world-space distances, independent of the preparation
        # KD tree and its bounding-box pruning. Check strongest contacts too.
        candidates=np.flatnonzero(valid.any(1))
        if len(candidates):
            scores=np.asarray(c).max((1,2))
            ticks=sorted({int(candidates[0]),int(candidates[-1]),int(np.argmax(scores))})
            for tick in ticks:
                sides=arrays['surface_valid'][tick]
                vertices=(arrays['surface_offsets'][tick].astype(np.float32)+arrays['joints'][tick,:,None,0])[sides].reshape(-1,3)
                vertex_parts=arrays['surface_parts'][sides].reshape(-1)
                for slot,name in enumerate(row['objects']):
                    if not valid[tick,slot]:continue
                    key='s%d:%s'%(row['source_id'],name) if row['source_id']<3 else name
                    ids=np.unique(np.r_[np.linspace(0,2047,10,dtype=int),np.argsort(c[tick,slot])[-6:]])
                    pose=arrays['object_poses'][tick,slot]
                    query=geometry[key][ids]@pose[:3,:3].T+pose[:3,3]
                    distance=np.linalg.norm(query[:,None].astype(float)-vertices[None].astype(float),axis=-1)
                    which=distance.argmin(1);nearest=distance[np.arange(len(ids)),which]
                    expected=np.exp(-np.maximum(nearest-.003,0)/.010);expected[nearest>.033]=0
                    difference=float(np.max(abs(expected-c[tick,slot,ids])))
                    contact_difference=max(contact_difference,difference)
                    if difference>.001:raise ValueError('independent contact geometry differs: '+row['key'])
                    second=np.partition(distance,1,axis=1)[:,1];unique=(second-nearest>1e-7)&(expected>0)
                    if not np.array_equal(vertex_parts[which[unique]],parts[tick,slot,ids[unique]]):raise ValueError('contact part identity')
                    geometry_checks+=len(ids)
        source_counts[str(row['source_id'])]=source_counts.get(str(row['source_id']),0)+1;frames_checked+=n
        if (i+1)%100==0:print(json.dumps(dict(checked_sequences=i+1,geometry_queries=geometry_checks)),flush=True)
    # Every sequence that has windows gets both its first and last window,
    # rather than repeatedly testing only the first source example.
    for split in ('train','dev'):
        ds=V5Windows(data/'recipe/manifest.json',split,cache_size=1)
        for seq_id in np.unique(ds.index[:,0]):
            ids=np.flatnonzero(ds.index[:,0]==seq_id)
            for index in set([int(ids[0]),int(ids[-1])]):
                sample=ds[index]
                if any(np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all() for v in sample.values()):raise ValueError('reader nonfinite')
                if sample['trajectory_target'].shape!=(24,2,21,3):raise ValueError('reader horizon')
                reader_windows+=1
    result=dict(status='PASS',checked_sequences=len(manifest['sequences']),source_counts=source_counts,
        frames_checked=frames_checked,hand_arrays_fully_read=arrays_checked,reader_windows=reader_windows,
        event_arrays_fully_read=event_arrays_checked,masked_nonfinite_object_poses=masked_nonfinite_poses,
        independent_bruteforce_contact_queries=geometry_checks,maximum_contact_confidence_difference=contact_difference,
        maximum_source_alignment_error_m=maximum_alignment,masked_contact_and_event_encoding_valid=True,
        training_allowed=False,original_v6_resume_ready=False,normalizer_replaced=False,
        scope='all published hand/contact/event arrays checked; independent contact subsets; every sequence reader endpoints; no original frozen-window equivalence claim',
        elapsed_s=time.monotonic()-start)
    (out/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
