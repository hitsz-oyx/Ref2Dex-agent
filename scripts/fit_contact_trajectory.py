#!/usr/bin/env python3
"""GPU adaptation of recorded physical trajectories; no recycled utility test."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (str(args.gpu),admission['uuid']):
        raise ValueError('explicit admitted GPU required')
    import torch
    from src.task.CmResidual.contact_ranker import physical_history,shuffled_numeric_actions
    from src.task.CmResidual.contact_trajectory import trajectory_targets,TrajectoryNetwork,all_trajectories,decode_trajectories,retained_choice
    from src.task.CmResidual.physical_value_contract import private_initialization
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');started=time.monotonic()
    if sha(args.warm)!='a38be701e1d32fbe66ea4c4ecc4e78178d205f5d8649beb921818ace364e808d':raise ValueError('warm input drift')
    paths=[];source=[]
    for name,directory in enumerate([args.uniform,args.targeted]):
        m=json.loads((directory/'run_manifest.json').read_text())
        if m['run_status']!='COMPLETED':raise ValueError('incomplete source')
        for phase in m['phases']:
            path=Path(phase['directory'])/'records.pt'
            if sha(path)!=phase['record_sha256']:raise ValueError('source record drift')
            paths.append(path);source.append(name)
    code=[Path(__file__),ROOT/'src/task/CmResidual/contact_trajectory.py',ROOT/'src/task/CmResidual/contact_ranker.py',
          ROOT/'src/task/CmResidual/physical_value_contract.py']
    hashes={str(p.resolve()):sha(p) for p in paths+[args.warm]+code}
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if args.output.resolve().parent!=base:raise ValueError('owned output required')
    args.output.mkdir(parents=True,exist_ok=False)
    prior=[]
    if args.prior:
        old=json.loads((args.prior/'run_manifest.json').read_text())
        if old['experiment_id']!='P-20261001-contact-trajectory-model' or old['run_status'] not in ('FAILED','STOPPED'):raise ValueError('invalid retry')
        prior=old.get('prior_attempts',[])+[dict(path=str(args.prior.resolve()),seconds=old['elapsed_seconds'],
            bytes=sum(p.stat().st_size for p in args.prior.rglob('*') if p.is_file()))]
    prior_seconds=sum(p['seconds'] for p in prior);prior_bytes=sum(p['bytes'] for p in prior)
    manifest=dict(experiment_id='P-20261001-contact-trajectory-model',family='HF10',probe_index_in_family=1,
                  run_status='RUNNING',pid=os.getpid(),command=sys.argv,gpu=admission,input_sha256=hashes,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  prior_attempts=prior,actor_training=False,value_training=False,cm_training=True,
                  updates_per_member=1000,ensemble_size=3,variants=['cm','state_only','action_shuffled'])
    def save(): (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-started>3540:raise TimeoutError('trajectory fit budget')
        if prior_bytes+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>8<<30:raise ValueError('storage budget')
        if any(sha(p)!=hashes[str(p.resolve())] for p in code):raise ValueError('fitting source drift')
    save()
    try:
        warm=torch.load(args.warm,map_location='cpu',weights_only=False)
        payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        data={k:torch.cat([p[k] for p in payloads]) for k in ['history','state','candidate_actions','assignment','rest_z','motion_id','start_frame','trigger','future_state','future_contact']}
        source=torch.cat([torch.full((len(p['assignment']),),s,dtype=torch.long) for p,s in zip(payloads,source)])
        episodes=[e for p in payloads for e in p['episode_id']]
        groups=[f'{int(m)}/{int(f)}' for m,f in zip(data['motion_id'],data['start_frame'])]
        mapping=dict(warm['group_split']);generator=torch.Generator().manual_seed(9831)
        for motion in sorted(set(data['motion_id'].tolist())):
            extra=sorted({g for g in groups if g.startswith(str(motion)+'/') and g not in mapping})
            order=torch.randperm(len(extra),generator=generator).tolist();nf=max(1,int(.6*len(extra)))
            nc=max(1,(len(extra)-nf)//2)
            for i,index in enumerate(order):mapping[extra[index]]=0 if i<nf else 1 if i<nf+nc else 2
        split=torch.tensor([mapping[g] for g in groups]);fit,cal,oldhold=[(split==i).nonzero().flatten() for i in range(3)]
        episode_split={}
        for e,s in zip(episodes,split.tolist()):
            if e in episode_split and episode_split[e]!=s:raise ValueError('episode split leak')
            episode_split[e]=s
        (args.output/'split.json').write_text(json.dumps(dict(group_split=mapping,row_split=split.tolist(),episode_ids=episodes,
            boundary='HF09 historical held outcomes already used; fresh prospective controls still required'),indent=2)+'\n')
        targets=trajectory_targets(data['future_state'],data['future_contact'].all(-1),data['state'][:,38],data['rest_z'])
        stats={}
        for name,rows in [('fit',fit),('calibration',cal),('old_held_engineering_audit',oldhold)]:
            stats[name]=dict(rows=len(rows),episodes=len({episodes[i] for i in rows.tolist()}),
                initially_lifted=int(targets['initially_lifted'][rows].sum()),release_events=int(targets['release'][rows].sum()),
                source_counts=torch.bincount(source[rows],minlength=2).tolist(),arms=torch.bincount(data['assignment'][rows],minlength=6).tolist())
            if stats[name]['episodes']<24:raise ValueError('insufficient grouped data')
        release_supported=stats['fit']['release_events']>=20 and stats['calibration']['release_events']>=5
        print(json.dumps(dict(split_stats=stats,release_supported=release_supported)),flush=True)
        normalizers={k:warm[k].to(device) for k in ['history_mean','history_scale','action_mean','action_scale','context_mean','context_scale']}
        h=physical_history(data['history'],data['rest_z']).to(device)
        h=(h-normalizers['history_mean'])/normalizers['history_scale']
        candidate=(data['candidate_actions'].to(device)-normalizers['action_mean'])/normalizers['action_scale']
        context=torch.cat((data['candidate_actions'][:,4],torch.nn.functional.one_hot(data['motion_id'].long(),3).float(),
                          ((data['start_frame']+data['trigger']).float()/600)[:,None]),-1).to(device)
        context=(context-normalizers['context_mean'])/normalizers['context_scale']
        height_mean=targets['height_mm'][fit].mean(0).to(device)
        height_scale=targets['height_mm'][fit].std(0,unbiased=False).clamp_min(1).to(device)
        target={k:v.to(device) for k,v in targets.items()};assignment=data['assignment'].long().to(device)
        fits=fit.to(device)
        corrupted=shuffled_numeric_actions(candidate[fit],assignment[fit],seed=9931)
        fg=sorted({groups[i] for i in fit.tolist()});gi=torch.tensor([fg.index(groups[i]) for i in fit.tolist()],device=device)
        source_fit=source[fit].to(device)
        release_prior=float(target['release'][fits].mean())
        class_prior=torch.cat((torch.zeros(10,device=device),target['contact'][fits].mean(0).clamp(.001,.999).logit(),
                               target['joint_contact'][fits].mean().clamp(.001,.999).logit()[None],
                               target['release'][fits].mean().clamp(.001,.999).logit()[None]))
        states={};calibration={};audits={};updates=[]
        input_norm=dict(history_abs_max=float(h.abs().max()),candidate_abs_max=float(candidate.abs().max()),context_abs_max=float(context.abs().max()))
        def predict(pool,indices):
            members={}
            with torch.no_grad():
                for member,model in enumerate(pool):
                    chunks={}
                    for offset in range(0,len(indices),128):
                        rows=indices[offset:offset+128].to(device)
                        decoded=decode_trajectories(all_trajectories(model,h[rows],candidate[rows],context[rows]),height_mean,height_scale)
                        for key,value in decoded.items():chunks.setdefault(key,[]).append(value)
                    members[member]={k:torch.cat(v) for k,v in chunks.items()}
            return {k:torch.stack([members[i][k] for i in range(len(pool))]) for k in members[0]}
        def factual_metrics(pred,indices):
            means={k:v.mean(0) for k,v in pred.items()};ar=assignment[indices];row=torch.arange(len(indices),device=device)
            factual={k:v[row,ar] for k,v in means.items()}
            retained_error=(factual['retained_score_mm']-target['retained_lift_mm'][indices]).abs()
            return dict(height_rmse_mm=float((factual['height_mm']-target['height_mm'][indices]).square().mean().sqrt()),
                retained_mae_mm=float(retained_error.mean()),joint_contact_mae=float((factual['joint_contact']-target['joint_contact'][indices]).abs().mean()),
                release_brier=float((factual['release']-target['release'][indices]).square().mean()),
                constant_release_brier=float((release_prior-target['release'][indices]).square().mean())),retained_error
        for variant in ['cm','state_only','action_shuffled']:
            pool=[]
            for member in range(3):
                check()
                with private_initialization(9861+member):
                    model=TrajectoryNetwork(state_only=variant=='state_only').to(device)
                    pretrained={k:v for k,v in warm['models'][variant][member].items() if k.startswith(('history.','context.','action.'))}
                    model.load_state_dict(pretrained,strict=False)
                    last=model.slot_head[-1] if variant=='state_only' else model.base_head[-1]
                    torch.nn.init.zeros_(last.weight)
                    with torch.no_grad():last.bias.copy_(class_prior.repeat(6) if variant=='state_only' else class_prior)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                gen=torch.Generator(device=device).manual_seed(9961+member)
                sampled=torch.randint(len(fg),(len(fg),),generator=gen,device=device)
                weight=torch.bincount(sampled,minlength=len(fg)).float()[gi]
                for s in [0,1]:
                    mask=source_fit==s
                    if not mask.any() or weight[mask].sum()==0:raise ValueError('bootstrap source support')
                    weight[mask]*=.5/weight[mask].sum()
                for update in range(1000):
                    local=torch.multinomial(weight,64,replacement=True,generator=gen);rows=fits[local]
                    if variant=='state_only':raw=model(h[rows],context=context[rows])[torch.arange(len(rows),device=device),assignment[rows]]
                    else:raw=model(h[rows],corrupted[local] if variant=='action_shuffled' else candidate[rows,assignment[rows]],candidate[rows,4],context[rows])
                    loss=torch.nn.functional.smooth_l1_loss(raw[:,:10],(target['height_mm'][rows]-height_mean)/height_scale)
                    loss+=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,10:20],target['contact'][rows])
                    loss+=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,20],target['joint_contact'][rows])
                    loss+=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,21],target['release'][rows])
                    if not torch.isfinite(loss):raise ValueError('nonfinite physical fitting')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    if update%250==0:check()
                model.eval().requires_grad_(False);pool.append(model)
                detail=dict(variant=variant,member=member,updates=1000,last_loss=float(loss),encoder_keys_warmed=len(pretrained))
                updates.append(detail);print(json.dumps(detail),flush=True)
            states[variant]=[{k:v.detach().cpu() for k,v in m.state_dict().items()} for m in pool]
            cp=predict(pool,cal);hp=predict(pool,oldhold)
            cm,error=factual_metrics(cp,cal);hm,_=factual_metrics(hp,oldhold)
            margin=max(.5,float(error.median()));q90=float(torch.quantile(error,.9))
            choice,_=retained_choice(cp,margin,release_supported)
            calibration[variant]=dict(**cm,margin_mm=margin,retained_error_q90_mm=q90,release_supported=release_supported,
                proposal_fraction=float((choice!=4).float().mean()),choice_counts=torch.bincount(choice,minlength=6).tolist(),
                scope='factual calibration; relative-effect ensemble uncertainty heuristic')
            audits[variant]=hm
        uniform_fit=fit[source[fit]==0];fixed_means=[]
        for arm in range(6):fixed_means.append(float(targets['retained_lift_mm'][uniform_fit[data['assignment'][uniform_fit]==arm]].mean()))
        best_fixed=max(range(6),key=lambda a:fixed_means[a])
        c=calibration['cm'];passed=release_supported and c['joint_contact_mae']<=.20 and c['release_brier']<=c['constant_release_brier'] and c['proposal_fraction']>0
        saved=dict(schema='ref2dex.contact_trajectory.v1',models=states,**{k:v.cpu() for k,v in normalizers.items()},
            height_mean=height_mean.cpu(),height_scale=height_scale.cpu(),calibration=calibration,
            release_supported=release_supported,group_split=mapping,best_fixed=best_fixed,
            source_record_sha256={str(p):sha(p) for p in paths},warm_checkpoint_sha256=sha(args.warm),
            execution='cached current candidate2 + own base8;6step cooldown',labels='height10/contact10/joint-last3/acquired-or-existing-release')
        torch.save(saved,args.output/'trajectory.pt')
        check()
        if any(sha(Path(p))!=v for p,v in hashes.items()):raise ValueError('training input drift')
        result=dict(run_status='COMPLETED',label='UNCLEAR',physical_preparation_gate=passed,
            decision='PROSPECTIVE_SELECTOR_REPLANNING' if passed else 'REVIEW_PHYSICAL_REPRESENTATION_OR_CALIBRATION',
            split_stats=stats,calibration=calibration,old_held_engineering_audit=audits,best_fixed=best_fixed,
            uniform_fit_fixed_retained_means_mm=fixed_means,updates=updates,input_norm=input_norm,
            source_record_sha256=saved['source_record_sha256'],trajectory_sha256=sha(args.output/'trajectory.pt'),
            input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
            cumulative_seconds=prior_seconds+time.monotonic()-started,actor_training=False,value_training=False,
            utility_boundary='historical held targets previously used; no new utility or safe-grasp result from this adaptation')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',label='UNCLEAR',physical_preparation_gate=passed,input_hashes_unchanged=True)
        print(json.dumps(result),flush=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uniform',type=Path,required=True);p.add_argument('--targeted',type=Path,required=True)
    p.add_argument('--warm',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--prior',type=Path)
    run(p.parse_args())
