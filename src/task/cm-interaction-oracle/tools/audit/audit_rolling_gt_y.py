#!/usr/bin/env python3
"""Ref13_1 saved-path rolling GT-Y, with explicit horizon/causal limitations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from consequence_sufficiency import readouts
from oracle_y_utility import CANDIDATES,utility,stable_grasp_z
from rolling_y_audit import OFFSETS,DENSE_OFFSETS,EPSILON,rolling_y,failure_events,pair_signal


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    inputs={};sources=[Path(__file__),ROOT/'src/task/cm-interaction-oracle/src/rolling_y_audit.py',
        ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',ROOT/'src/task/cm-interaction-oracle/src/consequence_sufficiency.py',
        ROOT/'src/task/cm-interaction-oracle/src/intervention.py']
    code={str(p.resolve()):sha(p) for p in sources}
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    manifest=dict(run_status='STARTED',created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        command=sys.argv,code_sha256=code,input_sha256=inputs,device='cpu',
        cpu_reason='Recorded height/forcepair label arithmetic and statistics only; no neural model or simulation.',
        primary_offsets=OFFSETS,dense_diagnostic_offsets=DENSE_OFFSETS,epsilon=EPSILON,
        seed=267,new_fits=0,new_simulation=0,new_policy_training=0)
    save('manifest.json',manifest)
    def record(path,expected=None):
        path=Path(path);h=sha(path)
        if expected is not None and h!=expected:raise ValueError('source input drift:'+str(path))
        inputs[str(path.resolve())]=h
    try:
        protocol=args.run_dir/'frozen_protocol.md'
        protocol.write_text((ROOT/'src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-gt-y.md').read_text())
        record(protocol)
        source_result=args.source_run/'result.json';record(source_result)
        original=json.loads(source_result.read_text());record(args.source_run/'result.npz')
        source_arrays=np.load(args.source_run/'result.npz')
        heights=[];pairs=[];rests=[];befores=[];before_pairs=[];labels=[];zs=[];qualifications=[];identities=[]
        for b,entry in enumerate(original['batches']):
            report_path=Path(entry['path']);record(report_path,entry['sha256']);record(report_path.with_suffix('.npz'),entry['npz_sha256'])
            report=json.loads(report_path.read_text());rows=torch.tensor(report['screen']['indices'],dtype=torch.long)
            for p,h in report['input_sha256'].items():record(p,h)
            for p,h in report['screen']['inputs'].items():record(p,h)
            paths=[Path(p) for p in report['input_sha256'] if p.endswith('panel.pt')]
            panels=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
            if len(panels)!=7:raise ValueError('complete7candidatepanel required')
            for k,p in enumerate(panels):
                if p['candidate']!=k or not p['valid_steps'][rows].all() or p['prefix_errors'][rows].abs().max()!=0:
                    raise ValueError('assigned prefix/window drift')
                for key in ('before','history','actor_obs','hand_root','before_fingertip_positions','before_hand_base_pose'):
                    if not torch.equal(p[key][rows],panels[0][key][rows]):raise ValueError('originalsameH drift')
                if not torch.equal(p['height'][rows,:32],p['trajectory'][rows,:,2]) or not torch.equal(p['pair'][rows,:32],p['trajectory'][rows,:,71]>.5):
                    raise ValueError('recorded sufficient fields disagree')
                for path,h in p.get('group_source_sha256',{}).items():record(path,h)
            heights.append(torch.stack([p['height'][rows] for p in panels],1))
            pairs.append(torch.stack([p['pair'][rows] for p in panels],1))
            rests.append(torch.stack([p['rest_height'][rows] for p in panels],1))
            befores.append(torch.stack([p['before'][rows,2] for p in panels],1))
            before_pairs.append(torch.stack([p['before'][rows,71]>.5 for p in panels],1))
            batch_y=[];batch_z=[];batch_q=[]
            for p in panels:
                selected={key:v[rows] if isinstance(v,torch.Tensor) and v.ndim and len(v)==len(p['triggers']) else v for key,v in p.items()}
                batch_y.append(readouts(selected)[3])
                z,d=stable_grasp_z(p['height'][rows],p['pair'][rows],p['rest_height'][rows]);batch_z.append(z);batch_q.append(d['first_stable_step'])
            labels.append(torch.stack(batch_y,1));zs.append(torch.stack(batch_z,1));qualifications.append(torch.stack(batch_q,1))
            identities.extend(dict(batch=b,env=int(row),clock=int(panels[0]['triggers'][row]),motion_id=int(panels[0]['motion_id'][row])) for row in rows)
        height,pair,rest,before_h,before_p,original_y,z,qualification=[torch.cat(v) for v in (heights,pairs,rests,befores,before_pairs,labels,zs,qualifications)]
        if not np.array_equal(original_y.numpy(),source_arrays['y']) or not np.array_equal(z.numpy(),source_arrays['z']):raise ValueError('originalY/Z replay failed')
        dense_y,dense_risk=rolling_y(height,pair,rest,before_h,before_p,DENSE_OFFSETS)
        if not torch.equal(dense_y[:,:,0],original_y):raise ValueError('tau0must reproduce originalY8 EXACTLY')
        indices=np.asarray(OFFSETS);y=dense_y[:,:,indices];risk=dense_risk[:,:,indices]
        score=utility(y.numpy());dense_score=utility(dense_y.numpy());zz=z.numpy();rr=risk.numpy();dr=dense_risk.numpy()
        events=[[failure_events(height[a,k].numpy(),pair[a,k].numpy(),float(rest[a,k]),int(qualification[a,k])) for k in range(7)] for a in range(len(z))]
        pair_reports=[];delayed_anchor=np.zeros(len(z),dtype=bool);dense_delayed=np.zeros_like(delayed_anchor)
        for a in range(len(z)):
            for good in np.flatnonzero(zz[a]):
                for bad in np.flatnonzero(~zz[a]):
                    gap=score[a,good]-score[a,bad];dgap=dense_score[a,good]-dense_score[a,bad]
                    event=events[a][bad]['first_raw_failure_step'];initial_tie=bool(gap[0]==0)
                    primary=pair_signal(gap,rr[a,good]&rr[a,bad],OFFSETS,event)
                    dense=pair_signal(dgap,dr[a,good]&dr[a,bad],DENSE_OFFSETS,event)
                    delayed_anchor[a]|=initial_tie and primary['delayed_signal'] is not None
                    dense_delayed[a]|=initial_tie and dense['delayed_signal'] is not None
                    pair_reports.append(dict(anchor=a,**identities[a],good=int(good),bad=int(bad),initial_tie=initial_tie,
                        bad_event=events[a][bad],primary=primary,dense_diagnostic=dense,gap=gap.tolist()))
        selections=np.argmax(score,axis=1);lookup=[]
        for t,offset in enumerate(OFFSETS):
            chosen=selections[:,t];outcome=zz[np.arange(len(z)),chosen]
            lookup.append(dict(offset=offset,branch_lookup_z_count=int(outcome.sum()),selection_counts=np.bincount(chosen,minlength=7).tolist(),
                all_utility_tied_anchors=int((score[:,:,t]==score[:,0:1,t]).all(1).sum()),
                current_risk_anchors_per_candidate=rr[:,:,t].sum(0).tolist(),
                warning='Different later H; counts are non-executable retrospective branch lookups, not rolling controller outcomes.'))
        opportunity=[]
        for a in np.flatnonzero(zz.max(1)&~zz[:,0]):
            event=events[a][0]['first_raw_failure_step'];times=[];dtimes=[]
            for offsets,scores,risks,dest in ((OFFSETS,score,rr,times),(DENSE_OFFSETS,dense_score,dr,dtimes)):
                for t,offset in enumerate(offsets):
                    k=int(np.argmax(scores[a,:,t]))
                    if event is not None and offset<event and zz[a,k] and risks[a,0,t] and risks[a,k,t]:
                        dest.append(dict(offset=int(offset),picked=k,lead_steps=int(event-offset),gap_over_baseline=float(scores[a,k,t]-scores[a,0,t])))
            opportunity.append(dict(anchor=int(a),**identities[a],baseline_event=events[a][0],primary_supported_branch_lookup=times,dense_supported_branch_lookup=dtimes))
        bootstrap=delayed_anchor[np.random.default_rng(267).integers(len(z),size=(2000,len(z)))].mean(1)
        result=dict(status='PROMISING' if delayed_anchor.any() else 'UNCLEAR',
            recorded_path_signal_status='PROMISING' if delayed_anchor.any() else 'UNPROMISING',rolling_control_utility='UNCLEAR_NOT_EXECUTED',
            anchors=len(z),candidate_windows=int(z.numel()),primary_offsets=OFFSETS,dense_diagnostic_offsets=DENSE_OFFSETS,
            supported_latest_fullwindow_offset=58,unsupported_next_primary_offset=64,epsilon=EPSILON,
            engineering_replay=dict(original_y_z_exact=True,tau0_y_exact=True,all_prefix_and_sameH_exact=True,complete90=True),
            baseline_z=int(zz[:,0].sum()),one_shot_y_z=int(np.asarray(original['oracle_y_success'])*len(z)),gt_z_original_branch_ceiling=int(zz.max(1).sum()),
            discordant_pairs=len(pair_reports),initial_tied_discordant_pairs=sum(p['initial_tie'] for p in pair_reports),
            primary_correct_separated_pairs=sum(p['primary']['first_correct_separation'] is not None for p in pair_reports),
            primary_initial_tied_delayed_signal_pairs=sum(p['initial_tie'] and p['primary']['delayed_signal'] is not None for p in pair_reports),
            primary_initial_tied_delayed_signal_anchors=int(delayed_anchor.sum()),
            dense_initial_tied_delayed_signal_anchors=int(dense_delayed.sum()),
            original_allutility_tied_anchors=int((score[:,:,0]==score[:,0:1,0]).all(1).sum()),
            delayed_signal_coverage=dict(mean=float(delayed_anchor.mean()),lower95=float(np.quantile(bootstrap,.025)),upper95=float(np.quantile(bootstrap,.975)),resampling_unit='originalanchor',seed=267,repeats=2000),
            event_type_counts={kind:sum(events[a][k]['event_kind']==kind for a in range(len(z)) for k in np.flatnonzero(~zz[a])) for kind in ('qualified_then_drop','unqualified_with_threshold_event','unqualified_no_observed_threshold_event')},
            pair_reports=pair_reports,events=events,identities=identities,branch_lookups=lookup,baseline_rescue_opportunities=opportunity,
            decision='Prioritize new same-current-state rolling candidate evidence if further control work proceeds; no saved-data rolling policy claim.' if delayed_anchor.any() else 'No qualified sampled delayed-tie signal; retain right-censoring and unknown rolling-control utility. No predictor fits.',
            limits=['Original candidate worlds diverge afterinitial K8; later scores are factual trajectories, not newly applied candidates atcommon H.',
                   'No executable rolling-intervention upperbound identifiable. Original25/32 is only finitebranch-access ceiling.',
                   'Primaryfuture endpoint88 missesstep89; queries64+ need longer data. Dense offsets are boundary diagnostics only.',
                   'Fourco-treated sharedsolver groups/29s3+3s7, forcepair proxy and boundedZ; anchor bootstrap is descriptive.',
                   'No claiming Y waswrong attau0 or refuting all Y/Cm; no newfits/simulation/policy.'])
        save('result.json',result)
        np.savez_compressed(args.run_dir/'rolling.npz',y=y.numpy(),u=score,risk=rr,dense_y=dense_y.numpy(),dense_u=dense_score,dense_risk=dr,
                            offsets=indices,dense_offsets=np.asarray(DENSE_OFFSETS),height=height.numpy(),pair=pair.numpy(),rest=rest.numpy(),z=zz,before_height=before_h.numpy(),before_pair=before_p.numpy())
        with (args.run_dir/'rolling.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['anchor','batch','env','clock','candidate','offset','current_risk','utility',*[f'Y{j}' for j in range(8)]])
            for a in range(len(z)):
                for k in range(7):
                    for t,offset in enumerate(OFFSETS):writer.writerow([a,identities[a]['batch'],identities[a]['env'],identities[a]['clock'],CANDIDATES[k],offset,int(rr[a,k,t]),score[a,k,t],*y[a,k,t].tolist()])
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,env in zip(axes,(33,36)):
            a=next(j for j,v in enumerate(identities) if v['batch']==0 and v['env']==env)
            for k in range(7):ax.plot(DENSE_OFFSETS,dense_score[a,k],label=CANDIDATES[k],alpha=.8)
            ax.axvline(56,color='black',linestyle=':',alpha=.5);ax.scatter(OFFSETS,score[a,0],s=20,color='black',zorder=5)
            ax.set_title(f'seed263/env{env}: factual branch Y evolution');ax.set_xlabel('Observed offset (full32future steps required)');ax.set_ylabel('Unchanged short-Y utility');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle('Ref13_1 • dense boundary diagnostic; black dots =8step baseline cadence\nLater candidate states differ; no rolling controller was executed')
        fig.tight_layout();fig.savefig(args.run_dir/'rolling_examples.png',dpi=170);plt.close(fig)
        for path,h in {**inputs,**code}.items():
            if sha(path)!=h:raise ValueError('input/code changed duringaudit')
        if time.monotonic()-started>120:raise TimeoutError('registered audit wall budget')
        if sum(p.stat().st_size for p in args.run_dir.rglob('*') if p.is_file())>30*1024**2:raise ValueError('outputbudget')
        manifest['run_status']='COMPLETED';print(json.dumps({k:result[k] for k in ('status','recorded_path_signal_status','anchors','discordant_pairs','initial_tied_discordant_pairs','primary_initial_tied_delayed_signal_anchors','dense_initial_tied_delayed_signal_anchors','baseline_rescue_opportunities')}))
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}:{error}');raise
    finally:
        manifest.update(input_sha256=inputs,elapsed_seconds=time.monotonic()-started,completed_at=datetime.now(timezone.utc).isoformat());save('manifest.json',manifest)
if __name__=='__main__':main()
