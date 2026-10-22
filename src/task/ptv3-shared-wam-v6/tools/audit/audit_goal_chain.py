"""Bounded, resumable existing-checkpoint goal intervention audit; no fitting.

Planner-G is the unmodified deployment OrderedGoalScheduler, replayed once per
recording/slot with measured history. The known instruction path is explicitly
the recorded object path (oracle reference), NOT an independently inferred task.
It isolates the spatial-waypoint versus realized-endpoint contract mismatch.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_identifiability as old
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.geometry_consistency import skin_attachment, skin_future, affine_query_weights, gather_vertices, BONES
from ptv3_shared_wam_v6.flow_matching import pack, unpack, current_mask, euler_sample

ROOT = next(p for p in HERE.parents if (p / 'AGENTS.md').exists())
sys.path.insert(0, str(ROOT / 'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler, rigid_goal_delta

VERSION = 'v6.goal-contract-intervention.large-panels.event-alignment.v1'
MANIFEST = ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
OLDOUT = ROOT/'outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1'
NOISE_SEEDS = (172,181,182,183,184,185,186,187)
GOALS = ('oracle','planner','perturbed')
PHASES = ('all','pre_contact','contact_onset','hold','post_release','unknown')
BANDS = ('all24','prefix8','middle8','last8')
METRICS = ('hand_epe_mm','object_epe_mm','C_tp','C_fp','C_fn','C_location_mm',
           'GTobj_contact_distance_mm','ownobj_contact_distance_mm',
           'GTobj_geom_tp','GTobj_geom_fp','GTobj_geom_fn',
           'ownobj_geom_tp','ownobj_geom_fp','ownobj_geom_fn',
           'GTobj_geom_location_mm','ownobj_geom_location_mm',
           'GTobj_oriented_depth_proxy_mm','ownobj_oriented_depth_proxy_mm',
           'GTobj_clearance_proxy_mm','ownobj_clearance_proxy_mm',
           'object_affine_residual_mm','object_rigid_residual_mm',
           'hand_object_common_translation_mm','C_missed_contact_frame')


def dump(path, obj): old.write(Path(path), obj)
def load(path): return old.read(path)


def check(out, started, seconds):
    if time.monotonic()-started > seconds or (out/'STOP').exists():
        raise RuntimeError('diagnostic wall-time/owned STOP boundary; completed shards preserved')
    if shutil.disk_usage(out).free < 20*2**30: raise RuntimeError('20GiB disk floor')


def phase_ids(state, transitions, slot, t):
    tr = transitions[transitions[:,1] == slot]
    result = []
    for k in range(t+1,t+25):
        prev = tr[tr[:,0]<=k]
        if state[k,slot] == 1:
            # First 0.2s after recorded entry is onset; later contact is hold.
            entry = prev[-1,0] if len(prev) and prev[-1,2] == 1 else -100000
            result.append(2 if k-entry < 6 else 3)
        elif state[k,slot] == 0:
            result.append(4 if len(prev) and prev[-1,2] == 2 else 1)
        else: result.append(5)
    return result


def select_spaced(ids, index, rng, maximum):
    chosen=[]
    for i in rng.permutation(ids):
        tick=int(index[i,2])
        if all(abs(tick-int(index[j,2]))>=28 for j in chosen): chosen.append(int(i))
        if len(chosen)>=maximum: break
    return sorted(chosen, key=lambda i:int(index[i,2]))


def sequence_job(job):
    """CPU-only persistent causal observation replay; no subprocess GPU use."""
    sid, meta, ixrows, prefixrows, seed = job
    seqpath=Path(meta['path']); seqmeta=load(seqpath/'meta.json')
    poses=np.load(seqpath/'object_poses.npy',mmap_mode='r')
    valid=np.load(seqpath/'object_pose_valid.npy',mmap_mode='r')
    joints=np.load(seqpath/'joints.npy',mmap_mode='r')
    jvalid=np.load(seqpath/'joint_valid.npy',mmap_mode='r')
    ep=Path(meta['event_path']);state=np.load(ep/'state.npy',mmap_mode='r')
    transitions=np.load(ep/'transitions.npy',mmap_mode='r')
    geometry=load(meta['geometry_manifest']); key='s%d:%s'%(meta['source_id'],meta['objects'][0]) if meta['source_id']<3 else meta['objects'][0]
    grows={r['key']:r for r in geometry['objects']}
    allrows=ixrows+prefixrows; labels={}; reports=[]
    for slot in sorted(set(r['slot'] for r in allrows)):
        name=meta['objects'][slot]; key='s%d:%s'%(meta['source_id'],name) if meta['source_id']<3 else name
        with np.load(Path(meta['geometry_manifest']).parent/grows[key]['path']) as z: query=z['query'];surface=z['surface']
        wanted={r['tick'] for r in allrows if r['slot']==slot}
        good=np.flatnonzero(valid[:,slot]);reference=poses[good,slot].astype(np.float64)
        scheduler=OrderedGoalScheduler(reference,good/30.,query,interaction_points=surface)
        mode=Counter();invalid=0
        for t in range(max(wanted)+1):
            if not valid[t,slot]: invalid+=1;continue
            hv=jvalid[t].all(-1);hand=None
            if hv.any():
                h=1 if hv[1] else 0
                hand=np.asarray(joints[t,h,[0,1,5,9,13,17,4,8,12,16,20]],np.float64)
            result=scheduler.update(poses[t,slot],t/30.,hand_points=hand)
            mode[result['mode']]+=1
            if result['goal_semantics']!='spatial_path_waypoint' or result['reference_endpoint_time_s'] is not None: raise AssertionError('scheduler contract drift')
            if t in wanted:
                labels[(slot,t)]=dict(delta=result['goal_delta'].astype(np.float32).tolist(),
                    pose=result['goal_pose'].tolist(),mode=result['mode'],reason=result['reason'],
                    target_reference_index=float(result['target_reference_index']),
                    lookahead_m=float(result['lookahead_m']),timed_out=result['timed_out'],
                    reference_endpoint_time_s=None)
        reports.append(dict(slot=slot,replayed_until=max(wanted),valid_observations=sum(mode.values()),
            missing_observations=invalid,modes=dict(mode),reference_frames=len(reference)))
    for r in allrows:
        r.update(phase=phase_ids(state,transitions,r['slot'],r['tick']),planner=labels[(r['slot'],r['tick'])])
    return dict(seq_id=sid,rows=allrows,replays=reports,capture=seqmeta.get('source_sequence',meta['key']))


def prepare(a):
    started=time.monotonic();out=a.output;dataset=V6Windows(MANIFEST,'train',geometry_gt_object=False)
    parent=dataset.meta;recipe=load(MANIFEST);geom=parent['geometry_manifest']; events=dataset.event_rows
    ix={s:np.load(parent['indices'][s]) for s in ('train','dev')}; by={s:defaultdict(list) for s in ix}
    for s,index in ix.items():
        for i,row in enumerate(index): by[s][int(row[0])].append(i)
    if set(by['train'])&set(by['dev']): raise AssertionError('TRAIN/DEV sequence overlap')
    oldpan=load(OLDOUT/'panels.json'); prefixby=defaultdict(list)
    # Reuse previous verified uncovered-candidate selection, then independently
    # re-verify exclusion against ALL local TRAIN support below.
    for r in oldpan['panels']['heldout_prefix_local_recipe']: prefixby[r['seq_id']].append(r)
    rng=np.random.default_rng(a.seed);jobs=[];bank=[];captures={};supports=defaultdict(list)
    # Select up to16 disjoint TRAIN windows/capture for bank;8 for evaluation.
    for sid,meta in enumerate(parent['sequences']):
        ids=by['train'].get(sid,[]) or by['dev'].get(sid,[])
        if not ids: continue
        split='train' if sid in by['train'] else 'dev';seqmeta=load(Path(meta['path'])/'meta.json')
        cap=seqmeta.get('source_sequence',meta['key']);captures[sid]=str(meta['source_id'])+':'+cap
        if split=='train':
            picked=select_spaced(ids,ix[split],rng,16);evalset=set(rng.permutation(picked)[:8].tolist());raw=picked
        else:
            raw=ids;evalset=set(ids)
        records=[]
        for i in raw:
            seq,slot,t=map(int,ix[split][i]);r=dict(seq_id=sid,slot=slot,tick=t,index=int(i),split=split,
                source_id=meta['source_id'],object=meta['objects'][slot],capture=captures[sid],
                panel='train_prefix' if split=='train' else 'heldout_sequence',evaluate=i in evalset,bank=split=='train')
            records.append(r)
        # Find NEW cached observation prefixes without creating training data.
        prefixes=[]
        if split=='train':
            covered=np.zeros(seqmeta['frames'],bool)
            for i in ids:
                t=int(ix[split][i,2]);covered[max(0,t-3):min(len(covered),t+25)]=True
            candidates=np.flatnonzero(np.convolve((~covered).astype(np.int16),np.ones(28,np.int16),'valid')==28)+3
            if len(candidates):
                seq,_=dataset._sequence(sid)
                for t in rng.permutation(candidates):
                    if not seq['joint_valid'][t-3:t+25].all((0,1)).any():continue
                    slots=[j for j in range(len(meta['objects'])) if seq['object_pose_valid'][t-3:t+25,j].all()]
                    if not slots:continue
                    prefixes.append(dict(seq_id=sid,slot=slots[0],tick=int(t),index=None,split='train',source_id=meta['source_id'],
                        object=meta['objects'][slots[0]],capture=captures[sid],panel='heldout_prefix_local_recipe',evaluate=True,bank=False));break
        jobmeta=dict(meta,event_path=events[meta['key']]['path'],geometry_manifest=geom)
        jobs.append((sid,jobmeta,records,prefixes,a.seed))
    dump(out/'selection.json',dict(seed=a.seed,jobs=len(jobs),scope='existing recipe only; no new training data',
        train_sequences=len(by['train']),dev_sequences=len(by['dev']),full_dev_windows=len(ix['dev']),
        support='inclusive[t-3,t+24];fullDEV may overlap, separate greedy nonoverlap core;capture cluster inference',
        reference='recorded object path supplied as known task instruction;oracle reference, causal observation replay;not task inference',
        full_task_scheduler='NOT REPLAYED: TaskScheduler requires measured grasp/release/support feedback absent from these caches; this audits its unmodified OrderedGoalScheduler submodule',
        peak_new_disk_gib=8,available_gib=shutil.disk_usage(out).free/2**30))
    rows=[];replays=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n,result in enumerate(pool.map(sequence_job,jobs)):
            rows.extend(result['rows']);replays.append({k:v for k,v in result.items() if k!='rows'})
            if n%100==0:print(json.dumps(dict(stage='replay',sequences=n,windows=len(rows),elapsed_s=time.monotonic()-started)),flush=True)
    # Strict complete support exclusion between TRAIN and local-prefix panels.
    for r in rows:
        if r['panel']=='train_prefix':supports[r['capture']].append(r['tick'])
    cores=defaultdict(list)
    for r in rows:
        r['nonoverlap_core']=False
        key=(r['panel'],r['capture'])
        if all(abs(r['tick']-t)>=28 for t in cores[key]):
            r['nonoverlap_core']=True;cores[key].append(r['tick'])
    # Distribution of planner errors on TRAIN only. Draw a whole paired SE(3)
    # residual in the oracle endpoint frame, preserving translation/rotation
    # correlation. Applying this to legal Planner-G stays within scheduler bounds.
    residuals=[];sourceids=[]
    for r in rows:
        if not r['bank']:continue
        seq,_=dataset._sequence(r['seq_id']);t=r['tick'];slot=r['slot'];cur=np.asarray(seq['object_poses'][t,slot],float)
        oracle=np.linalg.inv(cur)@seq['object_poses'][t+24,slot];plan=np.linalg.inv(cur)@np.asarray(r['planner']['pose'])
        err=np.linalg.inv(oracle)@plan
        residuals.append(np.r_[err[:3,3],Rotation.from_matrix(err[:3,:3]).as_rotvec()]);sourceids.append(r['seq_id'])
    np.savez_compressed(out/'planner-error-distribution.npz',residuals=np.array(residuals),seq_ids=np.array(sourceids))
    dump(out/'panels.json',dict(work_version=VERSION,seed=a.seed,rows=rows,replays=replays,
        panels={k:sum(r['panel']==k and r['evaluate'] for r in rows) for k in sorted(set(r['panel'] for r in rows))},
        train_sequences=len(by['train']),dev_sequences=len(by['dev']),
        full_dev_windows=len(ix['dev']),nonoverlap_core={k:sum(r['panel']==k and r['evaluate'] and r['nonoverlap_core'] for r in rows) for k in sorted(set(r['panel'] for r in rows))}))
    print(json.dumps(dict(stage='prepare_complete',rows=len(rows),elapsed_s=time.monotonic()-started)),flush=True)


def bank(a):
    out=a.output;pan=load(out/'panels.json');a.manifest=MANIFEST
    rows=[r for r in pan['rows'] if r['bank']]
    # Existing bank extractor has tested H4 reconstruction/contact descriptors.
    bankout=out/'bank';bankout.mkdir(exist_ok=True)
    dump(bankout/'panels.json',dict(bank=rows));a.output=bankout
    old.extract_bank(a)


def condition(batch, rows, name, errors, seed):
    if name=='oracle':return batch
    result=dict(batch);delta=[]
    for r in rows:
        d=np.asarray(r['planner']['delta'],np.float32)
        if name=='perturbed':
            rng=np.random.default_rng(np.random.SeedSequence([seed,r['seq_id'],r['slot'],r['tick']]))
            allowed=np.flatnonzero(errors['seq_ids']!=r['seq_id'])
            e=errors['residuals'][rng.choice(allowed)]
            # Start at the legal scheduler waypoint, apply empirically sampled
            # residual, then use the exact scheduler bound projection.
            p0=batch['scene_points'][len(delta),0].detach().cpu().numpy()
            # Recover local rigid plan from its same-ID displacement.
            x=p0-p0.mean(0);y=p0+d;u,_,v=np.linalg.svd(x.T@(y-y.mean(0)))
            fix=np.eye(3);fix[-1,-1]=np.sign(np.linalg.det(u@v));rot=u@fix@v
            plan=np.eye(4);plan[:3,:3]=rot.T;plan[:3,3]=y.mean(0)-p0.mean(0)@rot
            perturb=np.eye(4);perturb[:3,:3]=Rotation.from_rotvec(e[3:]).as_matrix();perturb[:3,3]=e[:3]
            sched=OrderedGoalScheduler([np.eye(4),plan],[0.,.8],p0)
            legal=sched._bounded(np.eye(4),plan@perturb)
            d=rigid_goal_delta(np.eye(4),legal,p0).astype(np.float32)
        delta.append(d)
    result['goal_object_delta']=torch.as_tensor(np.asarray(delta),device='cuda')
    result['goal_object_points']=result['start_object_points']+result['goal_object_delta']
    return result


def samples_from_base(model, batch, seeds):
    # Identical existing model.forward computation with the observation backbone
    # evaluated once; only the already-present residual ODE is repeated for K.
    enabled=model.config.get('flow_matching',False);model.config['flow_matching']=False
    base=model(batch,return_shared=True);model.config['flow_matching']=enabled
    if not enabled:return [base]
    flow=model.flow_future;mask=current_mask(batch)
    packed=torch.where(mask,pack(base['trajectory_normalized'],base['scene_normalized']),0.)
    context=flow.context(base['shared_tokens']);result=[];original=flow.config['flow_eval_seed']
    for seed in seeds:
        flow.config['flow_eval_seed']=seed;noise=torch.where(mask,flow.noise(packed,batch),0.)
        field=lambda value,t:flow.velocity(value,t,packed,context,base['shared_valid'],mask)
        hz,sz=unpack(torch.where(mask,packed+euler_sample(field,noise,flow.config['flow_steps']),0.))
        pred=dict(base,trajectory=torch.where(batch['start_hand_valid'][:,None,...,None],model.normalizer.denormalize('hand',hz),0.),
            scene=torch.where(batch['object_valid'][:,:,None,None,None],model.normalizer.denormalize('scene',sz),0.))
        result.append(pred)
    flow.config['flow_eval_seed']=original
    return result


def reduce_metrics(pred,batch,rows,geometry=True):
    """Sum/count arrays, all horizon/phase strata. Geometry only prefix8, Q2048."""
    b=len(rows);device=batch['xyz'].device
    vals=torch.zeros((b,4,6,len(METRICS),2),device=device,dtype=torch.float64)
    phase=torch.tensor([r['phase'] for r in rows],device=device);t=torch.arange(24,device=device)
    time_masks=torch.stack((t>=0,t<8,(t>=8)&(t<16),t>=16))
    phase_masks=torch.stack((torch.ones_like(phase,dtype=torch.bool),*(phase==p for p in range(1,6))),1)
    masks=time_masks[None,:,None,:]&phase_masks[:,None]
    def add(name,value,valid,prefix=False):
        # value/valid B,T,(point or joint dimensions).
        if value.ndim==2:value=value[...,None];valid=valid[...,None]
        value=value.flatten(2);valid=valid.flatten(2)
        if prefix:
            # Do not mislabel 8-frame geometry as all24.
            m=masks[:,:,:,0:8].clone();m[:,0]=False
        else:m=masks
        vv=torch.where(valid,value,0.).sum(-1).double();nn=valid.sum(-1).double()
        mi=METRICS.index(name);vals[...,mi,0]=torch.einsum('bapt,bt->bap',m.double(),vv)
        vals[...,mi,1]=torch.einsum('bapt,bt->bap',m.double(),nn)
    hv=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
    add('hand_epe_mm',(pred['trajectory']-batch['trajectory_target']).norm(dim=-1)*1000,hv)
    add('object_epe_mm',(pred['scene'][:,0]-batch['scene_target'][:,0]).norm(dim=-1)*1000,batch['scene_valid'][:,0])
    gt=batch['contact_target'][:,0];cv=batch['interaction_valid'][:,0];truth=gt>=.8
    cp=pred['interaction'][:,0,...,0];positive=cp>=.8
    for key,value in [('C_tp',positive&truth),('C_fp',positive&~truth),('C_fn',~positive&truth)]:add(key,value.float(),cv)
    target=batch['geometry_object_future_gt'][:,0]
    def centroid_error(prob,obj,gtobj,valid):
        weights=torch.where(valid,prob,0.);gw=torch.where(valid,gt,0.)
        center=(weights[...,None]*obj).sum(-2)/weights.sum(-1)[...,None].clamp_min(1e-8)
        gc=(gw[...,None]*gtobj).sum(-2)/gw.sum(-1)[...,None].clamp_min(1e-8)
        known=truth.logical_and(valid).any(-1)&(weights.sum(-1)>1e-8)
        return (center-gc).norm(dim=-1)*1000,known
    ce,known=centroid_error(cp,target,target,cv);add('C_location_mm',ce,known)
    add('C_missed_contact_frame',(~positive.any(-1)).float(),(truth&cv).any(-1))
    p0=batch['scene_points'][:,0];proposed=p0[:,None]+pred['scene'][:,0];x=p0-p0.mean(-2,keepdim=True)
    y=proposed-proposed.mean(-2,keepdim=True);u,_,vh=torch.linalg.svd(x[:,None].transpose(-1,-2)@y)
    fix=torch.eye(3,device=device).expand(b,24,3,3).clone();fix[:,:,-1,-1]=torch.det(u@vh).sign()
    rigid=x[:,None]@(u@fix@vh)+proposed.mean(-2,keepdim=True)
    add('object_rigid_residual_mm',(proposed-rigid).norm(dim=-1)*1000,batch['scene_valid'][:,0])
    he=torch.where(hv[...,None],pred['trajectory']-batch['trajectory_target'],0.).sum((2,3))/hv.sum((2,3))[...,None].clamp_min(1)
    oe=(pred['scene'][:,0]-batch['scene_target'][:,0]).mean(-2)
    # Difference between mean hand and mean object error; small values do not
    # prove accuracy. Absolute EPE must always be read alongside this statistic.
    add('hand_object_common_translation_mm',(he-oe).norm(dim=-1)*1000,hv.any((2,3)))
    if not geometry:return vals.cpu().numpy()
    k=8;att=skin_attachment(batch['start_hand'],batch['dense_hand'],batch['start_hand_valid'],batch['dense_hand_valid'])
    hand=skin_future(batch['start_hand'][:,None]+pred['trajectory'][:,:k],att)
    gt_hand=skin_future(batch['start_hand'][:,None]+batch['trajectory_target'][:,:k],att)
    hand=hand.flatten(2,3);gt_hand=gt_hand.flatten(2,3)
    aa=torch.tensor([x for x,_ in BONES],device=device);bb=torch.tensor([y for _,y in BONES],device=device)
    hvalid=hv[:,:k];known=hvalid[...,aa]&hvalid[...,bb]&hvalid[...,[0,5,17]].all(-1)[...,None]
    known=(known.gather(-1,att[0][:,None].expand(-1,k,-1,-1))&att[3][:,None]).flatten(2,3)
    points=batch['interaction_points'][:,0];normals=batch['interaction_normals'][:,0];target=target[:,:k]
    w=affine_query_weights(batch['scene_points'],points[:,None,None])[:,0,0]
    obj=points[:,None]+w[:,None]@pred['scene'][:,0,:k]
    x32=torch.cat((p0,torch.ones_like(p0[...,:1])),-1);mapping=torch.linalg.pinv(x32)[:,None]@proposed[:,:k]
    fitted=x32[:,None]@mapping
    add('object_affine_residual_mm',(proposed[:,:k]-fitted).norm(dim=-1)*1000,batch['scene_valid'][:,0,:k],True)
    sx=torch.cat((points,torch.ones_like(points[...,:1])),-1);gtmap=torch.linalg.pinv(sx)[:,None]@target
    ng=torch.nn.functional.normalize(normals[:,None]@torch.linalg.pinv(gtmap[:,:,:3]).transpose(-1,-2),dim=-1)
    no=torch.nn.functional.normalize(normals[:,None]@torch.linalg.pinv(mapping[:,:,:3]).transpose(-1,-2),dim=-1)
    valid=cv[:,:k]&known.any(-1)[...,None];g=gt[:,:k];pos=valid&(g>.25);neg=valid&(g<.01)
    gtid=old.nearest_indices(target,gt_hand,known,128);gtgap=target-gather_vertices(gt_hand,gtid)
    margin=(gtgap.norm(dim=-1).clamp(max=.033)-.002).clamp_min(0.)
    for anchor,objects,ns in [('GTobj',target,ng),('ownobj',obj,no)]:
        ids=old.nearest_indices(objects,hand,known,128);gap=objects-gather_vertices(hand,ids);distance=gap.norm(dim=-1)
        add(anchor+'_contact_distance_mm',distance*1000,pos,True)
        probability=((.003-distance)/.010).exp().clamp(max=1.)
        pred_contact=distance<=.0052314355 # equivalent exp-prob>=.8
        for suffix,value in [('tp',pred_contact&truth[:,:k]),('fp',pred_contact&~truth[:,:k]),('fn',~pred_contact&truth[:,:k])]:add(anchor+'_geom_'+suffix,value.float(),valid,True)
        weights=torch.where(valid,probability,0.);gw=torch.where(valid,g,0.)
        center=(weights[...,None]*objects).sum(-2)/weights.sum(-1)[...,None].clamp_min(1e-8)
        gc=(gw[...,None]*target).sum(-2)/gw.sum(-1)[...,None].clamp_min(1e-8)
        add(anchor+'_geom_location_mm',(center-gc).norm(dim=-1)*1000,(truth[:,:k]&valid).any(-1)&(weights.sum(-1)>1e-8),True)
        clear=torch.where(distance<margin,(margin-(gap*torch.nn.functional.normalize(gtgap,dim=-1,eps=1e-6)).sum(-1)).clamp_min(0.),0.)
        add(anchor+'_clearance_proxy_mm',clear*1000,neg,True)
        vertices=torch.arange(0,1556,12,device=device);q=hand.index_select(2,vertices);qknown=known.index_select(2,vertices)
        nearest=torch.cdist(q.flatten(0,1),objects.flatten(0,1)).argmin(-1).reshape(b,k,-1)
        p=objects.gather(-2,nearest[...,None].expand(-1,-1,-1,3));n=ns.gather(-2,nearest[...,None].expand(-1,-1,-1,3))
        add(anchor+'_oriented_depth_proxy_mm',(-(q-p)*n).sum(-1).clamp_min(0.)*1000,qknown,True)
    return vals.cpu().numpy()


def evaluate(a):
    out=a.output;started=time.monotonic();physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in ('0','1','2','3'):raise ValueError('one authorized GPU0..3 only')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(a.memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    rows=[r for r in load(out/'panels.json')['rows'] if r['evaluate']]
    # Sequence-contiguous shards avoid repeatedly mmap-loading the same recording.
    seqs=sorted(set(r['seq_id'] for r in rows));owned=set(seqs[a.shard::a.shards]);rows=[r for r in rows if r['seq_id'] in owned]
    if a.limit:rows=rows[:a.limit]
    errors=np.load(out/'planner-error-distribution.npz');errors={k:errors[k] for k in errors.files}
    dataset=V6Windows(MANIFEST,'train',geometry_gt_object=True);stats=load(MANIFEST.parent/'stats.json')
    models=[('healthy4371',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt'),('flow8500',OLDOUT/'flow-step008500.pt')]
    dump(out/('environment-worker%d.json'%a.shard),dict(command=sys.argv,physical_gpu=physical,work_version=VERSION,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        training=False,max_seconds=a.max_seconds,allocator_gib=a.memory_gib,seeds=list(NOISE_SEEDS[:a.k]),windows=len(rows)))
    for label,path in models:
        check(out,started,a.max_seconds);saved=torch.load(path,map_location='cpu',weights_only=False)
        config=dict(saved['config'],strict_fp32=True,amp=False,tf32=False,fp32_attention_backend='efficient')
        model=SharedWAMV6(stats,config).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);del saved;gc.collect()
        folder=out/('worker%d'%a.shard)/label;folder.mkdir(parents=True,exist_ok=True)
        for first in range(0,len(rows),a.batch_size):
            check(out,started,a.max_seconds);target=folder/('%06d.npz'%first)
            if target.exists():continue
            if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('GPU headroom below2GiB; preserve shards and yield')
            selected=rows[first:first+a.batch_size];batch={k:v.cuda() for k,v in collate([old.window(dataset,r) for r in selected]).items()}
            values=[];draws=[]
            with torch.inference_mode():
                for goal in GOALS:
                    conditioned=condition(batch,selected,goal,errors,a.seed)
                    predictions=samples_from_base(model,conditioned,NOISE_SEEDS[:a.k])
                    if a.smoke and config.get('flow_matching') and goal=='oracle':
                        native=model(conditioned);delta=float((native['trajectory']-predictions[0]['trajectory']).abs().max())
                        if delta>1e-6:raise AssertionError('cached-backbone FM differs from native forward: '+str(delta))
                        dump(out/('flow-native-control-worker%d.json'%a.shard),dict(max_abs_m=delta,status='PASS'))
                    values.append(np.stack([reduce_metrics(p,batch,selected,geometry=not a.no_geometry) for p in predictions],1))
                    draws.append(np.stack([pack(p['trajectory'],p['scene']).cpu().numpy() for p in predictions],1))
                # K diversity is forecast spread, not best-of-K accuracy.
                d=np.stack(draws,1);diversity=np.std(d,axis=2,ddof=1) if d.shape[2]>1 else np.zeros_like(d[:,:,0])
            with open(str(target)+'.part','wb') as f:np.savez_compressed(f,metrics=np.stack(values,1),diversity=diversity,
                ids=np.array([[r['seq_id'],r['slot'],r['tick']] for r in selected]),metric_names=np.array(METRICS))
            Path(str(target)+'.part').replace(target)
            dump(out/('progress-worker%d.json'%a.shard),dict(status='RUNNING',model=label,done=min(first+a.batch_size,len(rows)),total=len(rows),
                elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,physical_gpu=physical,
                goal_conditions=3,flow_K=a.k,geometry_prefix8=not a.no_geometry))
            if first%(a.batch_size*16)==0:print(json.dumps(load(out/('progress-worker%d.json'%a.shard))),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    dump(out/('complete-worker%d.json'%a.shard),dict(status='COMPLETED',elapsed_s=time.monotonic()-started,windows=len(rows)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','bank','evaluate'),required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=181)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--k',type=int,default=8);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--memory-gib',type=float,default=18);p.add_argument('--max-seconds',type=int,default=10800)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--no-geometry',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 1<=a.k<=8:raise ValueError('fixed K1..8; no hidden best-of-K')
    if a.phase=='prepare':prepare(a)
    elif a.phase=='bank':bank(a)
    else:evaluate(a)


if __name__=='__main__':main()
