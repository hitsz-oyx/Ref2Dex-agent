"""Full independently reconstructed SciPy/NumPy mesh geometry and all decision rows."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read_vertices(path):
    return np.asarray([list(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')],np.float32).astype(np.float64)


def independent_clearance(objects,tables,vertices,table_vertices):
    rotation=Rotation.from_quat(objects[:,3:7]).as_matrix();table_rotation=Rotation.from_quat(tables[:,3:7]).as_matrix()
    axis=int(np.argmin(np.ptp(table_vertices,axis=0)));assert axis==1
    assert np.all(np.abs(table_rotation[:,2,axis])>.999)
    normal=np.zeros((len(objects),3));normal[:,axis]=np.sign(table_rotation[:,2,axis])
    assert np.max(np.abs(np.einsum('bij,bj->bi',table_rotation,normal)-np.asarray([0,0,1])))<1e-10
    relative=np.einsum('bji,bjk->bik',table_rotation,rotation)
    direction=np.einsum('bi,bij->bj',normal,relative)
    displacement=np.einsum('bji,bj->bi',table_rotation,objects[:,:3]-tables[:,:3])
    result=[]
    for start in range(0,len(objects),256):
        stop=start+256
        heights=direction[start:stop]@vertices.T
        top=np.max(normal[start:stop]@table_vertices.T,axis=1)
        result.append(heights.min(1)+(normal[start:stop]*displacement[start:stop]).sum(1)-top)
    return np.concatenate(result)


def independent_report(rows,mask):
    if not mask.any():return dict(windows=0,episodes=0,episode_flip_rate=None,weighted_rotation_p95_m=None,unresolved_windows=0)
    ids=np.unique(rows['env'][mask]);rates=[];weights=np.zeros(int(mask.sum()),np.float64)
    env=rows['env'][mask]
    for e in ids:
        subset=env==e;weights[subset]=1/subset.sum();rates.append(float(rows['resolved_flip'][mask][subset].mean()))
    rotation=np.abs(rows['rotation_delta_m'][mask]);order=np.lexsort((np.arange(len(rotation)),rotation))
    cumulative=np.cumsum(weights[order]);chosen=np.flatnonzero(cumulative>=.95*cumulative[-1])[0]
    return dict(windows=int(mask.sum()),episodes=len(ids),episode_flip_rate=float(sum(rates)/len(rates)),
                weighted_rotation_p95_m=float(rotation[order[chosen]]),unresolved_windows=int(rows['unresolved'][mask].sum()))


def compare(report,saved):
    maximum=0.
    for key,value in report.items():
        if value is None:assert saved[key] is None
        elif key in ('windows','episodes','unresolved_windows'):assert value==saved[key],(key,value,saved[key])
        else:maximum=max(maximum,abs(value-saved[key]));assert abs(value-saved[key])<1e-9,(key,value,saved[key])
    return maximum


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    parent=json.loads((a.execution_source/'run_manifest.json').read_text());source=Path(parent['source_native'])/'s655'
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(source/'trace.pt',map_location='cpu',weights_only=False)
    history=np.concatenate((initial['object_root'][None].numpy(),trace['object_root'].numpy()),0).astype(np.float64)
    assert np.array_equal(trace['progress'].numpy(),np.broadcast_to(np.arange(1,203)[:,None],(202,768)))
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects';vertices=read_vertices(assets/'airplane/airplane.obj');table_vertices=read_vertices(assets/'table/table.obj');assert len(vertices)==25002
    tables=np.broadcast_to(initial['table_root'].numpy()[None],history.shape).reshape(-1,13).astype(np.float64)
    clear=independent_clearance(history.reshape(-1,13),tables,vertices,table_vertices).reshape(203,768)
    out=a.root/'geometry';stored=np.load(out/'pose_clearance.npz')['clearance'];maximum=float(np.abs(clear-stored).max());assert maximum<1e-10
    native_max=float(np.abs(clear[1:]-trace['clearance'].numpy()).max());assert native_max<2e-6
    pre=history[:-1];post=history[1:];dz=post[...,2]-pre[...,2];current=clear[:-1];following=clear[1:]
    rotation=following-current-dz;translation=current+dz;rise=pre[...,2]-initial['initial_height'].numpy().astype(np.float64);next_rise=post[...,2]-initial['initial_height'].numpy().astype(np.float64)
    q=history[...,3:7];q=q/np.linalg.norm(q,axis=-1,keepdims=True);angle=2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(-1)),0,1))
    eligible=(rise>=.03)&(current>=.015)&(current<=.025);unresolved=(np.abs(following-.02)<=1e-6)|(np.abs(translation-.02)<=1e-6);flip=((following>=.02)!=(translation>=.02))&~unresolved
    rng=np.random.default_rng(4001);train=[];held=[];motion=initial['motion'].numpy();arm=initial['arm_assignment'].numpy()
    for m in range(3):
        for g in range(4):
            ids=np.flatnonzero((motion==m)&(arm==g));assert len(ids)==64;ids=rng.permutation(ids);train.extend(ids[:32]);held.extend(ids[32:])
    assert not set(train)&set(held)
    hflags=np.isin(np.arange(768),held);tick=np.broadcast_to(np.arange(202)[:,None],(202,768));env=np.broadcast_to(np.arange(768)[None],(202,768));progress=tick+1
    stop=initial['phase_stop'].numpy()[motion];lift=initial['lift_start'].numpy()[motion];phase=np.where(progress<lift[None],0,np.where(progress<stop[None]-74,1,np.where(progress<=stop[None],2,3)))
    arrays=dict(env=env.ravel(),tick=tick.ravel(),held=np.broadcast_to(hflags[None],(202,768)).ravel(),motion=np.broadcast_to(motion[None],(202,768)).ravel(),arm=np.broadcast_to(arm[None],(202,768)).ravel(),phase=phase.ravel(),current_clearance_m=current.ravel(),next_clearance_m=following.ravel(),translation_oracle_clearance_m=translation.ravel(),rotation_only_clearance_m=(following-dz).ravel(),translation_delta_m=dz.ravel(),rotation_delta_m=rotation.ravel(),current_rise_m=rise.ravel(),next_rise_m=next_rise.ravel(),rotation_angle_rad=angle.ravel(),eligible=eligible.ravel(),unresolved=unresolved.ravel(),resolved_flip=flip.ravel())
    saved=np.load(out/'rows.npz');derived_max=0.
    for key,value in arrays.items():
        if value.dtype.kind in 'biu':assert np.array_equal(value,saved[key]),key
        else:
            error=float(np.abs(value-saved[key]).max());derived_max=max(derived_max,error);assert error<1e-9,(key,error)
    source_current=trace['context'].numpy()[...,36:49]
    for sl in (slice(0,3),slice(7,10)):assert np.array_equal(pre[...,sl].astype(np.float32),source_current[...,sl])
    # Independently reproduce original float32 units/rounding, then promote for geometry.
    target=np.concatenate((post[...,:3].astype(np.float32)-pre[...,:3].astype(np.float32),post[...,7:10].astype(np.float32)-pre[...,7:10].astype(np.float32)),-1)/np.asarray([.005]*3+[.05]*3,np.float32)
    assert np.array_equal(target,trace['physical_transition'].numpy())
    for group,ids in [('train',train),('held',held)]:
        with np.load(a.execution_source/'qualified'/(group+'_rows.npz')) as f:old={k:f[k] for k in f.files}
        assert np.array_equal(old['env'],np.repeat(np.sort(ids),16))
        assert np.array_equal(pre[old['tick'],old['env']].astype(np.float32),old['current_obj'])
        assert np.array_equal(post[old['tick'],old['env']].astype(np.float32),old['next_obj'])
    r=json.loads((out/'results.json').read_text());reports={};metric_max=0.
    for g,mask in [('all',np.ones(155136,bool)),('train',~arrays['held']),('held',arrays['held'])]:
        reports[g]=independent_report(arrays,arrays['eligible']&mask);metric_max=max(metric_max,compare(reports[g],r['summaries'][g]))
    for name,key,count in [('motion','motion',3),('arm','arm',4),('phase','phase',4)]:
        for i in range(count):metric_max=max(metric_max,compare(independent_report(arrays,arrays['eligible']&arrays['held']&(arrays[key]==i)),r['diagnostics'][name+'_'+str(i)]))
    mask=(progress>=stop[None]-74)&(progress<=stop[None]+30);assert np.all(mask.sum(0)==105)
    actual=np.all(np.where(mask,(following>=.02)&(next_rise>=.03),True),axis=0);linear=np.all(np.where(mask,(translation>=.02)&(next_rise>=.03),True),axis=0)
    for g,selection in [('all',np.ones(768,bool)),('train',~hflags),('held',hflags)]:
        expected=dict(episodes=int(selection.sum()),actual_success=int(actual[selection].sum()),perstep_translation_oracle_success=int(linear[selection].sum()),discordant=int((actual[selection]!=linear[selection]).sum()))
        assert expected==r['task'][g]
    held_report=reports['held'];enough=held_report['windows']>=128 and held_report['episodes']>=32
    gates=dict(coverage=enough,rotation_changes_decision=held_report['episode_flip_rate'] is not None and held_report['episode_flip_rate']>=.10,rotation_magnitude=held_report['weighted_rotation_p95_m'] is not None and held_report['weighted_rotation_p95_m']>=.002)
    label='UNCLEAR' if not enough else ('PROMISING' if all(gates.values()) else 'UNPROMISING');assert gates==r['gates'] and label==r['label']
    audit=dict(run_status='COMPLETED',label=label,independent_fullmesh_poses=155904,all_transition_rows=155136,
               geometry_max_m=maximum,native_clearance_max_m=native_max,derived_field_max=derived_max,metric_max=metric_max,
               split_and_sparse_ancestor_rows_verified=True,old6d_target_exact=True,all_task105_labels_verified=True,
               oracle_not_deployable_or_causal=True)
    (a.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
