"""Existing-weight, overlap-controlled GRAB identifiability/prefix/geometry audit.

No fitting, optimizer, input mutation or training control. One explicitly
selected GPU0..3 only. Future-contact refinements are oracle diagnostics.
"""
import argparse
from collections import Counter,defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.geometry_consistency import skin_attachment,skin_future,affine_query_weights,BONES,gather_vertices
from ptv3_shared_wam_v6.geometry_gt_contact import nearest_indices,proximity_bce
from ptv3_shared_wam_v5.data import transform

VERSION='v6.existing-weight-identifiability-prefix-self-consistency.v1'
PHASES=('pre_contact','contact','post_release','unknown')
START=time.monotonic()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    pending=Path(path).with_suffix('.json.part');pending.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');pending.replace(path)
def guard(out):
    if time.monotonic()-START>3600 or (out/'STOP').exists():raise RuntimeError('owned audit time/STOP bound')
    if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB disk floor')
    if torch.cuda.is_initialized() and torch.cuda.mem_get_info()[0]<512*2**20:raise RuntimeError('audit yields below512MiB device headroom')


def phases(folder,slot,tick):
    state=np.load(folder/'state.npy',mmap_mode='r')[:,slot]
    transitions=np.load(folder/'transitions.npy',mmap_mode='r')
    transitions=transitions[transitions[:,1]==slot]
    result=np.full(24,3,np.int64)
    for k,t in enumerate(range(tick+1,tick+25)):
        previous=transitions[transitions[:,0]<=t]
        if state[t]==1:result[k]=1
        elif state[t]==0:
            result[k]=2 if len(previous) and previous[-1,2]==2 else 0
    return result


def disjoint_pick(rows,count,rng,max_per_capture=3):
    groups=defaultdict(list)
    for row in rows:groups[row['phase_group']].append(row)
    for values in groups.values():rng.shuffle(values)
    selected=[];taken=defaultdict(list);capture_counts=Counter()
    keys=sorted(groups)
    while len(selected)<count and any(groups.values()):
        for key in keys:
            if not groups[key]:continue
            row=groups[key].pop();cap=row['capture'];t=row['tick']
            if capture_counts[cap]>=max_per_capture or any(abs(t-other)<28 for other in taken[cap]):continue
            selected.append(row);taken[cap].append(t);capture_counts[cap]+=1
            if len(selected)==count:break
    return selected


def prepare(a):
    out=a.output;guard(out);recipe=read(a.manifest);meta=read(recipe['parent_manifest']);events=read(recipe['event_manifest'])
    event_rows={r['key']:r for r in events['sequences']}
    indices={k:np.load(meta['indices'][k],mmap_mode='r') for k in ('train','dev')}
    positions={}
    for split,ix in indices.items():
        positions[split]=defaultdict(list)
        for i,(seq,slot,tick) in enumerate(ix):positions[split][int(seq)].append(i)
    rows={};capture_to_ids=defaultdict(list)
    for sid,row in enumerate(meta['sequences']):
        if row['source_id']!=1:continue
        m=read(Path(row['path'])/'meta.json');cap=m['source_sequence']
        rows[sid]=dict(row,capture=cap,frames=m['frames'],event_path=event_rows[row['key']]['path'])
        capture_to_ids[cap].append(sid)
    train_caps={rows[i]['capture'] for i in positions['train'] if i in rows}
    dev_caps={rows[i]['capture'] for i in positions['dev'] if i in rows}
    if train_caps&dev_caps:raise ValueError('raw GRAB capture TRAIN/DEV overlap')
    pools={'train_prefix':[],'heldout_sequence':[],'heldout_prefix_local_recipe':[]}
    bank=[];rng=np.random.default_rng(a.seed);eligible_prefix=0;prefix_sequences=0
    for sid,info in rows.items():
        guard(out)
        for split,label in (('train','train_prefix'),('dev','heldout_sequence')):
            ids=positions[split].get(sid,[])
            for i in ids[::max(1,len(ids)//24)]:
                seq,slot,tick=map(int,indices[split][i]);phase=phases(Path(info['event_path']),slot,tick)
                group=int(np.bincount(phase,minlength=4)[:3].argmax()) if (phase<3).any() else 3
                row=dict(seq_id=sid,slot=slot,tick=tick,index=int(i),split=split,capture=info['capture'],
                         object=info['objects'][slot],phase_group=group,phase=phase.tolist())
                pools[label].append(row)
            if split=='train':
                possible=[dict(seq_id=sid,slot=int(indices[split][i,1]),tick=int(indices[split][i,2]),index=int(i),
                    split=split,capture=info['capture'],object=info['objects'][int(indices[split][i,1])],phase_group=0) for i in ids]
                bank.extend(disjoint_pick(possible,8,rng,max_per_capture=8))
        if sid in positions['train']:
            # Exclude the complete H4+K24 support of EVERY local TRAIN window,
            # across all slots and all segments of the same raw recording.
            covered=np.zeros(info['frames'],bool)
            for alias in capture_to_ids[info['capture']]:
                for i in positions['train'].get(alias,[]):
                    t=int(indices['train'][i,2]);covered[max(0,t-3):min(len(covered),t+25)]=True
            candidate=np.flatnonzero(np.convolve((~covered).astype(np.int32),np.ones(28,np.int32),'valid')==28)+3
            valid=np.load(Path(info['path'])/'joint_valid.npy',mmap_mode='r')
            pose_valid=np.load(Path(info['path'])/'object_pose_valid.npy',mmap_mode='r')
            accepted=[]
            for t in candidate:
                if not valid[t-3:t+25].all((0,1)).any():continue
                for slot in range(len(info['objects'])):
                    if not pose_valid[t-3:t+25,slot].all():continue
                    phase=phases(Path(info['event_path']),slot,int(t));group=int(np.bincount(phase,minlength=4)[:3].argmax()) if (phase<3).any() else 3
                    accepted.append(dict(seq_id=sid,slot=slot,tick=int(t),index=None,split='train',capture=info['capture'],
                        object=info['objects'][slot],phase_group=group,phase=phase.tolist()))
            eligible_prefix+=len(accepted);prefix_sequences+=bool(accepted)
            pools['heldout_prefix_local_recipe'].extend(accepted[::max(1,len(accepted)//8)])
    rng.shuffle(bank);bank=bank[:a.bank_samples]
    panels={key:disjoint_pick(value,a.panel_samples,rng) for key,value in pools.items()}
    # Pairwise support must also be disjoint across panel rows of each capture.
    for label,panel in panels.items():
        for i,row in enumerate(panel):
            for other in panel[:i]:
                if row['capture']==other['capture'] and abs(row['tick']-other['tick'])<28:raise AssertionError('panel overlap')
    result=dict(work_version=VERSION,seed=a.seed,source='GRAB1 only;99.8% current optimizer draws',
        scope='existing cached observations only; not new training data',
        manifest=str(a.manifest.resolve()),train_windows=len(indices['train']),dev_windows=len(indices['dev']),
        grab_train_captures=len(train_caps),grab_dev_captures=len(dev_caps),raw_capture_overlap=0,
        frozen_split_sequence_overlap=len(set(positions['train'])&set(positions['dev'])),
        local_prefix_uncovered_candidates=eligible_prefix,local_prefix_uncovered_sequences=prefix_sequences,
        heldout_prefix_caveat='Local-recipe frame-disjoint only. Remote ancestor admitted windows unavailable, so inherited pretraining exposure is UNKNOWN; do not call this strict lifetime heldout.',
        ancestor_split_caveat='Sequence holdout verified against local frozen TRAIN and native GRAB split provenance; complete remote ancestor window manifest unavailable.',
        interval='inclusive[t-3,t+24],28frames; capture-level cross-slot exclusion; sampled windows are disjoint within each capture',
        panels=panels,bank=bank,sequence_info={str(k):v for k,v in rows.items()})
    write(out/'panels.json',result)
    print(json.dumps(dict(stage='prepare',panels={k:len(v) for k,v in panels.items()},bank=len(bank),prefix_candidates=eligible_prefix)),flush=True)


def window(dataset,row):
    # Explicit cached-triple reader permits evaluation of unadmitted prefixes
    # without writing or expanding any TRAIN manifest.
    previous=dataset.index;dataset.index=np.asarray([[row['seq_id'],row['slot'],row['tick']]],np.int64)
    try:return dataset[0]
    finally:dataset.index=previous


def contact_descriptor(sample,all24=True):
    c=sample['contact_target'][0];valid=sample['interaction_valid'][0]
    if not all24:c=c[:1];valid=valid[:1]
    weight=np.where(valid,c,0.).sum(0);total=float(weight.sum())
    if total<1e-4:return -1,-1,0,False
    p=sample['interaction_points'][0];center=p.mean(0);scale=np.ptp(p,axis=0).max().clip(.01)
    centroid=((p-center)*weight[:,None]).sum(0)/total
    # Eight CAD-octants; coarse physical centroid recorded as a sensitivity.
    region=int(((centroid>0)*np.asarray([1,2,4])).sum())
    parts=sample['part_target'][0][:len(c)]
    part_valid=valid&(parts>=0)&(parts<16)
    score=np.bincount(parts[part_valid].flatten(),weights=np.where(valid,c,0.)[part_valid].flatten(),minlength=16)
    part=int(score.argmax());finger=0 if part==0 else 1+(part-1)//3
    return region,finger,part,True


def reconstruct_history(start,features):
    velocity,acceleration,displacement=np.split(features,3,axis=-1)
    previous=start-velocity/30
    return np.stack((start-displacement,acceleration/900+2*previous-start,previous,start))


def extract_bank(a):
    panel=read(a.output/'panels.json');dataset=V6Windows(a.manifest,'train',geometry_gt_object=False)
    arrays=defaultdict(list);kept=[];skipped=Counter()
    for number,row in enumerate(panel['bank']):
        guard(a.output);s=window(dataset,row)
        hv=s['start_hand_valid'];present=hv.all(-1)
        if not present.any() or np.any(hv.any(-1)!=present) or not s['trajectory_valid'][:,present].all():skipped['incomplete_hand']+=1;continue
        if not s['event_valid']:skipped['unknown_event']+=1;continue
        _,contact=dataset._sequence(row['seq_id']);tick=row['tick'];slot=row['slot']
        observed=dict(s,contact_target=np.asarray(contact['contact'][tick:tick+1,slot],np.float32)[None],
            part_target=np.asarray(contact['part'][tick:tick+1,slot],np.int64)[None],
            interaction_valid=np.broadcast_to(contact['valid'][tick:tick+1,slot,None],(1,1,2048)))
        obs=contact_descriptor(observed,False);oracle=contact_descriptor(s,True)
        obj=row['object'];name=obj.removeprefix('grab_') if hasattr(obj,'removeprefix') else obj[5:]
        cap=row['capture'];body=cap.split('_',2)[-1];action=body[len(name)+1:].rsplit('_',1)[0] if body.startswith(name+'_') else body
        arrays['hand'].append(s['trajectory_target']);arrays['scene'].append(s['scene_target'][0]);arrays['start'].append(s['start_hand'])
        arrays['goal'].append(s['goal_object_delta']);arrays['hand_history'].append(reconstruct_history(s['start_hand'],s['start_hand_history']))
        ids=np.linspace(0,2047,64,dtype=int)
        arrays['object_history'].append(reconstruct_history(s['xyz'][ids],s['features'][ids,3:12]))
        arrays['present'].append(present)
        # Coarse original condition groups: same CAD, same presence and E_next.
        kept.append(dict(row,event_type=int(s['event_type']),event_delta=float(s['event_delta_steps']),
            presence=present.tolist(),region=oracle[0],finger=oracle[1],part=oracle[2],contact_available=oracle[3],
            observed_region=obs[0],observed_finger=obs[1],interaction_type=action,ref_available=False))
        if number%256==0:print(json.dumps(dict(stage='bank',done=number,kept=len(kept))),flush=True)
    np.savez_compressed(a.output/'bank.npz',**{k:np.asarray(v) for k,v in arrays.items()})
    write(a.output/'bank.json',dict(samples=kept,skipped=dict(skipped),seed=a.seed,notes='No neural inference or fitting; current object frame; all added future-contact labels are post-hoc oracle descriptors.'))


def pair_variances(hand,scene,present,pairs):
    values=[]
    for first in range(0,len(pairs),256):
        ij=pairs[first:first+256];i,j=ij[:,0],ij[:,1]
        h=(hand[i]-hand[j]).square().sum(-1);mask=present[i][:,None,:,None].expand_as(h)
        s=(scene[i]-scene[j]).square().sum(-1)
        values.append(torch.stack([(h[:,:8]*mask[:,:8]).sum((1,2,3))/mask[:,:8].sum((1,2,3))/2,
            (h*mask).sum((1,2,3))/mask.sum((1,2,3))/2,s[:,:8].mean((1,2))/2,s.mean((1,2))/2],-1).cpu().numpy()*1e6)
    return np.concatenate(values) if values else np.empty((0,4))


def analyze_bank(a):
    meta=read(a.output/'bank.json');samples=meta['samples'];z=np.load(a.output/'bank.npz');n=len(samples)
    device='cuda:0';start=torch.as_tensor(z['start'],device=device);present=torch.as_tensor(z['present'],device=device)
    goal=torch.as_tensor(z['goal'],device=device);hand=torch.as_tensor(z['hand'],device=device);scene=torch.as_tensor(z['scene'],device=device)
    hh=torch.as_tensor(z['hand_history'],device=device);oh=torch.as_tensor(z['object_history'],device=device)
    groups=defaultdict(list)
    for i,s in enumerate(samples):groups[(s['object'],tuple(s['presence']),s['event_type'])].append(i)
    candidates=[]
    for ids in groups.values():
        guard(a.output)
        if len(ids)<2:continue
        ix=torch.tensor(ids,device=device);h=hh[ix][:,:,present[ix[0]]].flatten(1);g=goal[ix].flatten(1);o=oh[ix].flatten(1)
        hd=torch.cdist(h,h)/math.sqrt(h.shape[1]/3);gd=torch.cdist(g,g)/math.sqrt(32);od=torch.cdist(o,o)/math.sqrt(o.shape[1]/3)
        keep=torch.triu((hd<=.1)&(gd<=.1)&(od<=.1),diagonal=1).nonzero().cpu().numpy()
        hd=hd.cpu().numpy();gd=gd.cpu().numpy();od=od.cpu().numpy()
        for i,j in keep:
            left,right=ids[i],ids[j]
            if samples[left]['capture']==samples[right]['capture']:continue
            candidates.append((left,right,float(hd[i,j]),float(gd[i,j]),float(od[i,j])))
    rng=np.random.default_rng(a.seed+2);reports={};names=('hand_prefix8','hand_all24','object_prefix8','object_all24')
    for gate in (.02,.05,.1):
        best={}
        for i,j,h,g,o in candidates:
            if max(h,g,o)>gate:continue
            key=tuple(sorted((samples[i]['capture'],samples[j]['capture'])))
            if key not in best or h*h+g*g+o*o<sum(v*v for v in best[key][2:]):best[key]=(i,j,h,g,o)
        picked=list(best.values());pairs=np.asarray([[v[0],v[1]] for v in picked],np.int64).reshape(-1,2)
        variance=pair_variances(hand,scene,present,pairs)
        np.savez_compressed(a.output/('pairs-%dmm.npz'%round(gate*1000)),pairs=pairs,variance_mm2=variance,
                            state_goal_distance_m=np.asarray([v[2:] for v in picked]))
        if not len(pairs):reports[str(round(gate*1000))]=dict(pairs=0,status='INSUFFICIENT_MATCHED_CONDITIONS');continue
        i,j=pairs.T
        labels={k:np.asarray([s[k] for s in samples]) for k in ('region','finger','part','interaction_type','observed_region','observed_finger')}
        stages=[('base',np.ones(len(pairs),bool)),('oracle_object_region',(labels['region'][i]==labels['region'][j])&(labels['region'][i]>=0)),
            ('oracle_object_and_finger',(labels['region'][i]==labels['region'][j])&(labels['region'][i]>=0)&(labels['finger'][i]==labels['finger'][j])),
            ('oracle_object_finger_and_interaction',(labels['region'][i]==labels['region'][j])&(labels['region'][i]>=0)&(labels['finger'][i]==labels['finger'][j])&(labels['interaction_type'][i]==labels['interaction_type'][j])),
            ('observed_object_and_finger',(labels['observed_region'][i]==labels['observed_region'][j])&(labels['observed_region'][i]>=0)&(labels['observed_finger'][i]==labels['observed_finger'][j]))]
        capture_ids={s['capture']:k for k,s in enumerate({s['capture']:s for s in samples}.values())}
        # Compact capture index for two-endpoint cluster bootstrap.
        unique=sorted(set(s['capture'] for s in samples));lookup={c:k for k,c in enumerate(unique)}
        ci=np.asarray([lookup[samples[k]['capture']] for k in i]);cj=np.asarray([lookup[samples[k]['capture']] for k in j])
        boot=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=200)
        bw=boot[:,ci]*boot[:,cj]
        report=dict(pairs=len(pairs),captures=len(set(ci)|set(cj)),gate_mm=gate*1000,
            state_H4_rms_mm=float(np.mean([v[2] for v in picked]))*1000,goal_rms_mm=float(np.mean([v[3] for v in picked]))*1000,
            object_H4_rms_mm=float(np.mean([v[4] for v in picked]))*1000,stages={})
        previous=np.ones(len(pairs),bool)
        for name,mask in stages:
            count=int(mask.sum());values=variance[mask].mean(0) if count else None
            row=dict(pairs=count,captures=len(set(ci[mask])|set(cj[mask])),variance_mm2=dict(zip(names,map(float,values))) if count else None,
                rms_spread_mm=dict(zip(names,map(float,np.sqrt(values)))) if count else None)
            if count and name!='base':
                reference=np.ones(len(pairs),bool) if name=='observed_object_and_finger' else previous
                ref=variance[reference].mean(0);row['drop_vs_previous_percent']=dict(zip(names,map(float,100*(1-values/ref))))
                denominator=bw[:,mask].sum(1);refden=bw[:,reference].sum(1);ok=(denominator>0)&(refden>0)
                stat=100*(1-(bw[ok][:,mask]@variance[mask]/denominator[ok,None])/(bw[ok][:,reference]@variance[reference]/refden[ok,None]))
                row['drop_capture_bootstrap95']=dict(zip(names,np.percentile(stat,[2.5,97.5],axis=0).T.tolist()))
            report['stages'][name]=row
            if name!='observed_object_and_finger':previous=mask
        # E_next timing is already a v6 input. Control it before interpreting
        # omitted contact intent as the source of remaining dispersion.
        time_match=np.asarray([abs(samples[x]['event_delta']-samples[y]['event_delta'])<=3 for x,y in pairs])
        report['event_time3_control']={}
        for name,mask in stages[:4]:
            mask=mask&time_match;v=variance[mask].mean(0) if mask.any() else None
            report['event_time3_control'][name]=dict(pairs=int(mask.sum()),variance_mm2=dict(zip(names,map(float,v))) if v is not None else None)
        # Permute full contact/intent descriptor tuples within original CAD/event
        # groups; preserves descriptor correlations and base matching graph.
        permutation=[]
        for repeat in range(100):
            shuffled=np.arange(n)
            for ids in groups.values():shuffled[ids]=rng.permutation(ids)
            si,sj=shuffled[i],shuffled[j];m=(labels['region'][si]==labels['region'][sj])&(labels['region'][si]>=0)
            stage=[]
            for index in range(3):
                if index==1:m&=labels['finger'][si]==labels['finger'][sj]
                if index==2:m&=labels['interaction_type'][si]==labels['interaction_type'][sj]
                stage.append(variance[m].mean(0).tolist() if m.any() else [float('nan')]*4)
            permutation.append(stage)
        permutation=np.asarray(permutation)
        report['permutation_variance_mm2_median']=dict(zip([s[0] for s in stages[1:4]],np.nanmedian(permutation,axis=0).tolist()))
        report['permutation_variance_mm2_95']=dict(zip([s[0] for s in stages[1:4]],np.nanpercentile(permutation,[2.5,97.5],axis=0).transpose(1,2,0).tolist()))
        reports[str(round(gate*1000))]=report
    result=dict(status='COMPLETED',samples=n,seed=a.seed,metrics=list(names),gates=reports,
        variance_definition='0.5*mean(||GT_tau_i-GT_tau_j||^2),averaged valid joints/points/time,mm^2; sqrt is RMS spread,not mean EPE',
        matching='Same CAD,hand-presence,E_next; full hand H4,object64 H4 and goal32 vector RMS gates20/50/100mm; one minimum-condition-distance pair per distinct capture-pair; no GT tau used in matching.',
        limitations=['finite-radius descriptive conditional dispersion,not exact-input irreducibility','event_time3_control restricts an additional condition already available to v6','oracle refinements use realized future contact,unavailable to causal inference inputs','permutation and capture bootstrap assess sampling effects,not causal identification','no Ref annotation supplied; metadata interaction name is an oracle intent proxy'])
    write(a.output/'identifiability.json',result)
    print(json.dumps(dict(stage='identifiability',pairs={k:v['pairs'] for k,v in reports.items()})),flush=True)


def update_metric(store,name,value,valid):
    value=value.detach();valid=valid.expand_as(value)
    number=int(valid.sum());total=float(torch.where(valid,value,0.).double().sum())
    item=store.setdefault(name,dict(sum=0.,count=0));item['sum']+=total;item['count']+=number


def summarize(store):return {k:dict(v,mean=v['sum']/v['count'] if v['count'] else None) for k,v in store.items()}


def geometry_metrics(pred,batch,config):
    start=batch['start_hand'].float();attachment=skin_attachment(start,batch['dense_hand'].float(),batch['start_hand_valid'],batch['dense_hand_valid'])
    hv=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
    hand=skin_future(start[:,None]+pred['trajectory'],attachment).flatten(2,3)
    gt_hand=skin_future(start[:,None]+batch['trajectory_target'],attachment).flatten(2,3)
    aa=torch.tensor([x for x,_ in BONES],device=start.device);bb=torch.tensor([y for _,y in BONES],device=start.device)
    known=hv[...,aa]&hv[...,bb]&hv[...,[0,5,17]].all(-1)[...,None]
    known=(known.gather(-1,attachment[0][:,None].expand(-1,24,-1,-1))&attachment[3][:,None]).flatten(2,3)
    points=batch['interaction_points'][:,0];current_normals=batch['interaction_normals'][:,0]
    w=affine_query_weights(batch['scene_points'],points[:,None,None])[:,0,0]
    objects=points[:,None]+w[:,None]@pred['scene'][:,0]
    target=batch['geometry_object_future_gt'][:,0]
    # Recover affine matrix from the predicted32 branch, for signed normal proxy.
    x=torch.cat((batch['scene_points'][:,0],torch.ones_like(batch['scene_points'][:,0,:,:1])),-1)
    mapping=torch.linalg.pinv(x)[:,None]@(batch['scene_points'][:,0,None]+pred['scene'][:,0])
    normal_pred=F.normalize(current_normals[:,None]@torch.linalg.pinv(mapping[:,:,:3]).transpose(-1,-2),dim=-1)
    # GT normals from the actual recorded rigid transform; recover it from dense
    # matched CAD query correspondences, without using predicted32 points.
    sx=torch.cat((points,torch.ones_like(points[...,:1])),-1)
    gt_mapping=torch.linalg.pinv(sx)[:,None]@target
    normal_gt=F.normalize(current_normals[:,None]@gt_mapping[:,:,:3],dim=-1)
    valid=batch['interaction_valid'][:,0]&known.any(-1)[...,None]
    confidence=batch['contact_target'][:,0];pos=valid&(confidence>.25);neg=valid&(confidence<.01)
    gt_ids=nearest_indices(target,gt_hand,known,64);gt_gap=target-gather_vertices(gt_hand,gt_ids)
    margin=(gt_gap.norm(dim=-1).clamp(max=.033)-.002).clamp_min(0.)
    result={}
    for label,obj,normals in [('pred_hand_GT_object',target,normal_gt),('pred_hand_pred_object',objects,normal_pred),('GT_hand_GT_object',target,normal_gt)]:
        skin=gt_hand if label=='GT_hand_GT_object' else hand
        ids=nearest_indices(obj,skin,known,64);gap=obj-gather_vertices(skin,ids);distance=gap.norm(dim=-1)
        _,prob=proximity_bce(distance,confidence,dict(config,geometry_contact_radius_m=.003,geometry_contact_falloff_m=.010,geometry_contact_smoothing_m=.001))
        clear=torch.where(distance<margin,(margin-(gap*F.normalize(gt_gap,dim=-1,eps=1e-6)).sum(-1)).clamp_min(0.),0.)
        # Signed nearest-oriented-point proxy on a fixed subset of observed hand
        # surface vertex IDs. It is not a closed-mesh inside test or exact SDF.
        vertices=torch.arange(0,skin.shape[-2],12,device=skin.device)
        q=skin.index_select(2,vertices);known_q=known.index_select(2,vertices)
        dn=torch.cdist(q.flatten(0,1),obj.flatten(0,1));index=dn.argmin(-1).reshape(*q.shape[:-1])
        nearest=obj.gather(-2,index[...,None].expand(*index.shape,3));nn=normals.gather(-2,index[...,None].expand(*index.shape,3))
        signed=((q-nearest)*nn).sum(-1)
        result[label]=dict(distance=distance*1000,positive=pos,negative=neg,valid=valid,mae=(prob-confidence).abs(),
            clearance=clear*1000,contact_recall=(distance<=.003).float(),signed_depth=(-signed).clamp_min(0.)*1000,
            signed_inside=(signed<-.001).float(),signed_valid=known_q)
    result['branch_object_dense_error_mm']=(objects-target).norm(dim=-1)*1000
    fitted=batch['scene_points'][:,0,None]+pred['scene'][:,0]
    fit=x[:,None]@mapping
    result['affine_fit_error_mm']=(fitted-fit).norm(dim=-1)*1000
    return result


def evaluate(a):
    panel=read(a.output/'panels.json');dataset=V6Windows(a.manifest,'train',geometry_gt_object=True);stats=read(a.stats)
    reports={}
    models=[('healthy4371',a.baseline),('flow8500',a.flow)]
    for label,path in models:
        guard(a.output);saved=torch.load(path,map_location='cpu',weights_only=False);config=dict(saved['config'],strict_fp32=True,amp=False,tf32=False,fp32_attention_backend='efficient')
        torch.manual_seed(a.seed);model=SharedWAMV6(stats,config).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);step=int(saved['step']);del saved;gc.collect()
        model_report=dict(checkpoint=str(path.resolve()),step=step,precision=model.precision_runtime,panels={})
        for name,rows in panel['panels'].items():
            aggregates=defaultdict(dict);records=[]
            for first in range(0,len(rows),a.batch_size):
                guard(a.output);selected=rows[first:first+a.batch_size];samples=[window(dataset,row) for row in selected]
                batch={k:v.cuda() for k,v in collate(samples).items()}
                with torch.inference_mode():
                    prediction=model(batch);geometry=geometry_metrics(prediction,batch,config)
                    for bi,row in enumerate(selected):
                        sample_metrics={};hv=batch['trajectory_valid'][bi]&batch['start_hand_valid'][bi,None];sv=batch['scene_valid'][bi,0]
                        he=(prediction['trajectory'][bi]-batch['trajectory_target'][bi]).norm(dim=-1)*1000
                        se=(prediction['scene'][bi,0]-batch['scene_target'][bi,0]).norm(dim=-1)*1000
                        cp=(prediction['interaction'][bi,0,...,0]-batch['contact_target'][bi,0]).abs()
                        phase=torch.tensor(row['phase'],device='cuda');times=torch.arange(24,device='cuda')
                        masks={'all24':torch.ones(24,dtype=torch.bool,device='cuda')}
                        masks.update({part:(times>=lo)&(times<hi) for part,lo,hi in [('prefix8',0,8),('middle8',8,16),('last8',16,24)]})
                        masks.update({PHASES[k]:phase==k for k in range(4)})
                        masks.update({part+'/'+PHASES[k]:mask&(phase==k) for part,mask in list(masks.items()) if part in ('prefix8','middle8','last8') for k in range(4)})
                        for part,tm in masks.items():
                            store=sample_metrics.setdefault(part,{})
                            update_metric(store,'hand_epe_mm',he,hv&tm[:,None,None]);update_metric(store,'object_epe_mm',se,sv&tm[:,None])
                            update_metric(store,'C_head_mae',cp,batch['interaction_valid'][bi,0]&tm[:,None])
                            for geom,values in geometry.items():
                                if not isinstance(values,dict):continue
                                for metric,key,mask in [('contact_distance_mm','distance','positive'),('geometric_contact_mae','mae','valid'),
                                    ('clearance_violation_mm','clearance','negative'),('contact_recall3mm','contact_recall','positive'),
                                    ('oriented_penetration_proxy_mm','signed_depth','signed_valid'),('oriented_inside_proxy_fraction','signed_inside','signed_valid')]:
                                    update_metric(store,geom+'/'+metric,values[key][bi],values[mask][bi]&tm[:,None])
                            update_metric(store,'object_dense_affine_epe_mm',geometry['branch_object_dense_error_mm'][bi],tm[:,None])
                            update_metric(store,'affine_fit_error_mm',geometry['affine_fit_error_mm'][bi],tm[:,None])
                        records.append(dict(row=row,metrics={k:summarize(v) for k,v in sample_metrics.items()}))
                        for part,store in sample_metrics.items():
                            for key,value in store.items():
                                item=aggregates[part].setdefault(key,dict(sum=0.,count=0));item['sum']+=value['sum'];item['count']+=value['count']
                if first%16==0:print(json.dumps(dict(stage='evaluate',model=label,panel=name,done=min(first+a.batch_size,len(rows)))),flush=True)
            report=dict(windows=len(rows),captures=len(set(r['capture'] for r in rows)),metrics={k:summarize(v) for k,v in aggregates.items()},records=records)
            write(a.output/(label+'-'+name+'.json'),report);model_report['panels'][name]={k:v for k,v in report.items() if k!='records'}
        reports[label]=model_report;del model;gc.collect();torch.cuda.empty_cache()
    write(a.output/'evaluation.json',dict(status='COMPLETED',models=reports,
        limitations=['GRAB only; fixed disjoint windows, no new fitting','heldout_prefix_local_recipe does not certify unexposed remote-ancestor frames',
        'dense predicted object uses observed-CAD affine extension of model32 points,not GT poses; affine_fit_error quantifies reconstruction residual',
        'oriented penetration is a nearest-oriented-point proxy on every12th hand vertex,not a watertight-mesh inside test; GT-hand/GT-object calibrates proxy bias',
        'all contact masks/phases share the same realized GT frame and confidence labels; future labels only affect diagnostics','single fixed flow noise seed172,no best-of-N selection']))


def geometry_smoke(a):
    panel=read(a.output/'panels.json');dataset=V6Windows(a.manifest,'train',geometry_gt_object=True)
    sample=window(dataset,panel['panels']['train_prefix'][0]);batch={k:v.cuda() for k,v in collate([sample]).items()}
    config=dict(geometry_contact_radius_m=.003,geometry_contact_falloff_m=.010,geometry_contact_smoothing_m=.001)
    pred=dict(trajectory=batch['trajectory_target'],scene=batch['scene_target'])
    with torch.inference_mode():
        result=geometry_metrics(pred,batch,config)
        error=float(result['branch_object_dense_error_mm'].max())
        matched=(result['pred_hand_GT_object']['distance']-result['pred_hand_pred_object']['distance']).abs().max()
        shifted=geometry_metrics(dict(pred,scene=pred['scene']+.01),batch,config)
        shifted_error=float(shifted['branch_object_dense_error_mm'].mean())
    if error>.01 or matched>.01 or abs(shifted_error-math.sqrt(3)*10)>.01:raise ValueError('geometry identity/independent-object smoke failed')
    write(a.output/'geometry-smoke.json',dict(status='PASS',GT_affine_max_error_mm=error,
        GT_anchor_vs_pred_anchor_max_gap_mm=float(matched),translated_object_error_mm=shifted_error,
        expected_translation_mm=math.sqrt(3)*10,GT_anchor_distance_unchanged=bool(torch.equal(result['pred_hand_GT_object']['distance'],shifted['pred_hand_GT_object']['distance']))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','stats','baseline','flow','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','bank','analyze','evaluate','smoke','all'),default='all')
    p.add_argument('--memory-gib',type=float,default=8.)
    p.add_argument('--seed',type=int,default=178);p.add_argument('--bank-samples',type=int,default=4096);p.add_argument('--panel-samples',type=int,default=96);p.add_argument('--batch-size',type=int,default=2)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3'):raise ValueError('one owner-authorized physical GPU0..3 required')
    if not 1<=a.memory_gib<=8:raise ValueError('bounded1..8GiB allocator required')
    a.output.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(a.memory_gib*2**30/total,0)
    write(a.output/('environment-'+a.phase+'.json'),dict(work_version=VERSION,command=sys.argv,seed=a.seed,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),training=False,seconds_bound=3600,artifact_bound_gib=3,memory_limit_gib=a.memory_gib))
    if a.phase in ('prepare','all'):prepare(a)
    if a.phase in ('bank','all'):extract_bank(a)
    if a.phase in ('analyze','all'):analyze_bank(a)
    if a.phase in ('evaluate','all'):evaluate(a)
    if a.phase=='smoke':geometry_smoke(a)
    write(a.output/('complete-'+a.phase+'.json'),dict(status='COMPLETED',elapsed_s=time.monotonic()-START,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
