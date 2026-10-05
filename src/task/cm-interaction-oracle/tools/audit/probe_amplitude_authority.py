#!/usr/bin/env python3
"""Fixed-K8 nonlinear amplitude authority and conditional useful-direction Probe."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from intervention import ARM_NAMES, residuals, physical_targets, continuation_outcomes, surface_force_projection
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits, dexplore_action_to_native_targets, dexplore_root_pose
from probe_duration_response import current_design, residual_fit, randomization, sha

FAMILIES = dict(short_contact=[0], late_contact=[1], late_height_failure=[2],
    late_combined_failure=[3], all32_height_failure=[4], I8=list(range(5,19)),
    signed_force=list(range(19,37)), surface_projection=list(range(37,47)),
    projection_ratio=list(range(47,52)))


def targets(p):
    tr = p['trajectory']
    y, details = continuation_outcomes(p['before'], tr, p['rest_height'])
    task = torch.stack((details['short_contact_fraction'], y[:,3],
        details['late_height_failure'].float(), y[:,5], details['height_failure32'].float()), -1)
    i8 = physical_targets(p['before'], tr[:,7])[:,12:]
    force = tr[:,:,48:63].reshape(-1,32,5,3)
    fn, ft = surface_force_projection(force, p['surface_normals'])
    ratio = ft/(fn.abs()+1.)
    signed = torch.asinh(tr[:,7,48:66])
    target = torch.cat((task, i8, signed, torch.asinh(fn[:,7]), torch.log1p(ft[:,7]),
        torch.log1p(ratio[:,7])), -1).numpy().astype(float)
    height = tr[:,:,2]-p['rest_height'][:,None]
    # Descriptive physical units; keep separate from registered tests.
    curves = torch.stack((fn.abs().mean(-1), ft.mean(-1), tr[:,:,66:71].mean(-1),
        tr[:,:,71], (height < .02).float(), ratio.mean(-1)), -1).numpy().astype(float)
    plot_target = np.stack((curves[:,:8,0].mean(1), curves[:,:8,1].mean(1),
        curves[:,:8,2].mean(1), target[:,0], target[:,1], target[:,2]), -1)
    return target, details, curves, plot_target


def aligned_late(beta, tails, halves, row, orientation):
    passed = []
    for axis, threshold, sign, name in ((1,.05,orientation,'late_contact'),
                                       (2,.10,-orientation,'late_height_failure')):
        repeated = [float(b[row,axis]*sign) for b in halves]
        if tails[axis] <= .10 and beta[row,axis]*sign >= threshold and min(repeated)>0:
            passed.append(dict(quantity=name, signed_effect=float(beta[row,axis]*sign), half_signed_effects=repeated))
    return passed


def authority_candidates(beta, tails, halves):
    candidates = []
    for level, alpha in ((1,2.),(2,4.)):
        for arm in range(1,7):
            row, initial = level*6+arm-1, arm-1
            effect = beta[row,0]; sign = np.sign(effect)
            repeated = [float(b[row,0]*sign) for b in halves]
            late = aligned_late(beta,tails,halves,row,sign)
            if (tails[0]<=.10 and abs(effect)>=.15 and (effect-beta[initial,0])*sign>=.10
                    and min(repeated)>=.03 and late):
                candidates.append(dict(alpha=alpha,arm=arm,gate='contact',orientation=int(sign),
                    short_effect=float(effect), growth_from_alpha1=float((effect-beta[initial,0])*sign),
                    half_signed_short=repeated,alignments=late))
            for axis in range(13,18):  # old I8 five surface-distance axes
                change = beta[row,axis]-beta[initial,axis]
                direction = np.sign(beta[row,axis])
                repeated_distance = [float(b[row,axis]*direction) for b in halves]
                orientation = -direction  # reduced distance paired with better task outcome
                late = aligned_late(beta,tails,halves,row,orientation)
                if (tails[axis]<=.05 and abs(change)>=.003 and change*direction>0
                        and min(repeated_distance)>0 and late):
                    candidates.append(dict(alpha=alpha,arm=arm,gate='distance',axis=axis,
                        orientation=int(orientation),distance_change_from_alpha1=float(change),
                        half_signed_distance=repeated_distance,alignments=late))
    return candidates


def useful_candidates(beta, tails, halves, families):
    interaction = any(families[name]['max_tail'] is not None and families[name]['max_tail']<=.10
        for name in ('I8','signed_force','surface_projection','projection_ratio'))
    rows=[]
    for arm in range(1,7):
        row=arm-1
        if (interaction and tails[1]<=.10 and tails[2]<=.10 and beta[row,1]>=.10
                and beta[row,2]<=-.10 and all(b[row,1]>0 and b[row,2]<0 for b in halves)):
            rows.append(dict(arm=arm,late_contact_gain=float(beta[row,1]),
                late_height_failure_reduction=float(-beta[row,2]),
                half_effects=[[float(b[row,1]),float(b[row,2])] for b in halves]))
    return rows


def audit_packet(p, dataset, expected_delta=None):
    assert p['schema']==('ref2dex.randomized_intervention.v3' if expected_delta is None else 'ref2dex.randomized_intervention.v4')
    assert p['duration_levels']==[8] and (p['duration']==8).all()
    assert p['decision_region']=='early-hold' and torch.equal(p['delta'],residuals() if expected_delta is None else expected_delta)
    for key,value in p.items():
        if isinstance(value,torch.Tensor): assert torch.isfinite(value).all(),key
    assert p['valid_steps'].all() and (p['pre_hold_steps']>=6).all()
    assert ((p['before'][:,2]-p['rest_height']>=.03)&(p['before'][:,71]>.5)).all()
    assert torch.equal(p['history'][:,-1,36:39],p['before'][:,:3])
    assert torch.equal(p['history'][:,-1,86:139],p['before'][:,13:66])
    assert torch.equal(p['base_action'],p['base_actions'][:,0])
    active=torch.arange(32)[None,:,None]<8
    delta=p['amplitude'][:,None,None]*p['delta'][p['arm']][:,None]
    expected=torch.where(active,(p['base_actions']+delta).clamp(-1,1),p['base_actions'])
    assert torch.equal(expected,p['actions'])
    ad=p['actions']-p['base_actions']; pd=p['pd_targets']-p['pd_base_targets']
    assert (ad[:,8:]==0).all() and (pd[:,8:]==0).all()
    assert (ad[p['arm']==0]==0).all() and (pd[p['arm']==0]==0).all()
    lo,hi=native_joint_limits(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf','cpu')
    a=p['actions'].flatten(0,1); b=p['base_actions'].flatten(0,1); q=torch.zeros_like(a)
    rec=dexplore_action_to_native_targets(a,q,lo,hi)-dexplore_action_to_native_targets(b,q,lo,hi)
    error=float((rec.reshape_as(pd)-pd).abs().max()); assert error<5e-7
    for key in ('before_surface_normals','surface_normals'):
        assert torch.allclose(p[key].norm(dim=-1),torch.ones_like(p[key][...,0]),atol=1e-5)
    # Replay saved local sample using the exact measured poses and body centers.
    geom=json.loads((dataset.parent/'geometry_provenance.json').read_text())
    sample_path=dataset.parent/'object_surface_sample.pt'
    assert sha(sample_path)==geom['sample_sha256']
    sample=torch.load(sample_path,weights_only=False)
    points=sample['points'][::sample['stride']]; normals=sample['normals'][::sample['stride']]
    maximum=0.; distance_error=0.; tied_normal_rows=0; exact_errors=[]
    for phys,norm in ((p['before'],p['before_surface_normals']),
                      (p['trajectory'].flatten(0,1),p['surface_normals'].flatten(0,1))):
        for start in range(0,len(phys),256):
            v=phys[start:start+256]; poses=dexplore_root_pose(v[:,:13])
            pos=points[None]@poses[:,:3,:3].transpose(-1,-2)+poses[:,None,:3,3]
            ns=normals[None]@poses[:,:3,:3].transpose(-1,-2)
            body=v[:,13:48].reshape(-1,5,7)[:,:,:3]
            all_dist=torch.cdist(body,pos)
            dist,nearest=all_dist.min(-1)
            exact=torch.cdist(body,pos,compute_mode='donot_use_mm_for_euclid_dist').min(-1)[0]
            exact_errors.append((exact-v[:,66:71]).abs().flatten())
            # cdist's float32 MM path subtracts nearly equal world-coordinate
            # squared norms. Bound squared-distance error before sqrt, rather
            # than impose an arbitrary absolute distance threshold near zero.
            roundoff=16*torch.finfo(pos.dtype).eps*(body.square().sum(-1)+pos.square().sum(-1).amax(-1)[:,None])
            assert ((exact.square()-v[:,66:71].square()).abs()<=roundoff).all()
            ns=torch.nn.functional.normalize(ns,dim=-1)
            matched=torch.gather(ns,1,nearest[:,:,None].expand(-1,-1,3))
            maximum=max(maximum,float((matched-norm[start:start+256]).abs().max()))
            distance_error=max(distance_error,float((dist-v[:,66:71]).abs().max()))
            stored=norm[start:start+256]
            mismatch=(matched-stored).abs().amax(-1)>1e-4
            tied_normal_rows+=int(mismatch.sum())
            # Every recorded normal must match a sampled normal at a nearest
            # point within GPU/CPU distance roundoff, even when argmin ties.
            compatible=(ns[:,None]-stored[:,:,None]).abs().amax(-1)<1e-4
            near=all_dist<=dist[:,:,None]+1e-4
            assert ((compatible & near).any(-1)).all(), 'stored normal is not from nearest surface'
    exact_error=torch.cat(exact_errors)
    cells=[]
    for alpha in p['amplitude_levels']:
        for arm in range(len(p['arm_names'])):
            keep=(p['amplitude']==alpha)&(p['arm']==arm); ratio=None
            if arm and keep.any():
                d=p['delta'][arm]
                ratio=float((ad[keep,:8]*d.sign()).sum()/(keep.sum()*8*alpha*d.abs().sum()))
            cells.append(dict(alpha=alpha,arm=arm,count=int(keep.sum()),dose_ratio=ratio))
    return dict(status='PASS',trials=len(p['arm']),native_pd_max_error=error,
        normal_cpu_replay_max_error=maximum,distance_cpu_replay_max_error=distance_error,
        distance_direct_max_error_m=float(exact_error.max()),distance_direct_q99_error_m=float(exact_error.quantile(.99)),
        normal_tie_compatible_rows=tied_normal_rows,normal_nearest_compatibility_pass=True,cells=cells)


def plots(out, levels, labels, plot_y, beta, curves, curve_beta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('normal force projection |proxy| (N)','tangential projection proxy (N)',
        'mean body-to-surface distance (m)','contact fraction steps1..8',
        'contact fraction steps9..32','height failure steps9..32')
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    zero=plot_y[labels==0].mean(0)
    for axis,ax in enumerate(axs.flat):
        for arm in range(1,7):
            rows=[j*6+arm-1 for j in range(len(levels))]
            ax.plot(levels,beta[rows,axis],marker='o',label=ARM_NAMES[arm])
        ax.axhline(0,color='black',linewidth=.7); ax.set_title(names[axis]); ax.set_xlabel('amplitude'); ax.set_ylabel('adjusted arm minus pooled zero')
    axs[0,0].legend(fontsize=7)
    fig.suptitle('K=8 amplitude Probe; aggregate geometric force proxies, not paired friction/slip')
    fig.tight_layout(); fig.savefig(out/'amplitude_adjusted.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for axis,ax in enumerate(axs.flat):
        for arm in range(1,7):
            means=[plot_y[labels==j*6+arm,axis].mean() for j in range(len(levels))]
            ax.plot(levels,means,marker='o',label=ARM_NAMES[arm])
        ax.axhline(zero[axis],color='black',linestyle='--',label='pooled zero')
        ax.set_title(names[axis]); ax.set_xlabel('amplitude'); ax.set_ylabel('raw cohort mean')
    axs[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'amplitude_raw.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    curve_names=names[:3]+('contact proxy','physical height loss (<2cm)','tangent/(|normal|+1N) proxy')
    for axis,ax in enumerate(axs.flat):
        for row in range(len(curve_beta)):
            arm=row%6+1; level=row//6
            ax.plot(np.arange(1,33),curve_beta[row,:,axis],alpha=.65,
                label=f'{ARM_NAMES[arm]} a{levels[level]}')
        ax.axvline(8,color='black',linestyle='--'); ax.axhline(0,color='black',linewidth=.7)
        ax.set_title(curve_names[axis]); ax.set_xlabel('post-step'); ax.set_ylabel('adjusted arm minus zero')
    axs[0,0].legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(out/'time_response_adjusted.png',dpi=150); plt.close(fig)


def signed_force_curves(out, p, design, labels, n_cells, levels):
    import matplotlib.pyplot as plt
    tr=p['trajectory']
    vector=torch.cat((tr[:,:,48:63].reshape(-1,32,5,3).sum(2),tr[:,:,63:66]),-1).numpy().astype(float)
    beta=residual_fit(design,labels,vector.reshape(len(vector),-1),n_cells)[0].reshape(n_cells,32,6)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for axis,ax in enumerate(axs.flat):
        for row in range(n_cells):
            ax.plot(np.arange(1,33),beta[row,:,axis],alpha=.65,
                label=f'{ARM_NAMES[row%6+1]} a{levels[row//6]}')
        ax.axvline(8,color='black',linestyle='--'); ax.axhline(0,color='black',linewidth=.7)
        ax.set_title(('hand net-force sum ' if axis<3 else 'object net-force ')+('x','y','z')[axis%3])
        ax.set_xlabel('post-step'); ax.set_ylabel('adjusted arm minus zero (N)')
    axs[0,0].legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(out/'signed_force_response.png',dpi=150); plt.close(fig)
    return beta.tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=223)
    parser.add_argument('--stage',type=int,choices=(1,2),default=1)
    args=parser.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.run_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    manifest=dict(run_status='STARTED',command=sys.argv,dataset=str(args.dataset.resolve()),
        dataset_sha256=sha(args.dataset),collection_manifest_sha256=sha(args.dataset.parent/'manifest.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in (SCRIPT,SCRIPT.with_name('probe_duration_response.py'),ROOT/'src/task/cm-interaction-oracle/src/intervention.py',ROOT/'src/task/CmResidual/dexplore_cm_geometry.py')},
        seed=args.seed,stage=args.stage,created_at=datetime.now(timezone.utc).isoformat(),
        device='CPU statistics and file analysis; no neural model')
    save('manifest.json',manifest)
    try:
        p=torch.load(args.dataset,map_location='cpu',weights_only=False)
        levels=p['amplitude_levels']; assert levels==[1.,2.,4.] if args.stage==1 else len(levels)==1
        collection=json.loads((args.dataset.parent/'manifest.json').read_text()); assert collection['run_status']=='COMPLETED'
        audit=audit_packet(p,args.dataset); save('engineering_audit.json',audit)
        env_count=int(collection['command'][collection['command'].index('--num_envs')+1])
        waves=p['episode_id'].numpy()//env_count; half=waves>=collection['waves']//2
        arms=p['arm'].numpy(); alpha=p['amplitude'].numpy(); n_cells=18 if args.stage==1 else 6
        labels=np.where(arms==0,0,np.searchsorted(levels,alpha)*6+arms)
        half_counts=[[int(((labels==cell)&(half==side)).sum()) for cell in range(n_cells+1)] for side in (False,True)]
        min_cell,min_zero,min_half=(20,60,8) if args.stage==1 else (40,40,15)
        support=all(row['count']>=min_cell for row in audit['cells'] if row['arm']) and int((arms==0).sum())>=min_zero
        support &= min(min(rows[1:] if args.stage==1 else rows) for rows in half_counts)>=min_half
        dose=all(row['dose_ratio'] is not None and row['dose_ratio']>=.90 for row in audit['cells'] if row['arm'])
        target,details,curves,plot_y=targets(p); design,groups=current_design(p,waves)
        beta,tails,families,statistics=randomization(design,labels,target,groups,args.seed,FAMILIES,n_cells)
        half_fits=[residual_fit(design[half==side],labels[half==side],target[half==side],n_cells) for side in (False,True)]
        halves=[fit[0] for fit in half_fits]
        valid=support and dose and statistics['treatment_rank']==n_cells and all(fit[3]==n_cells for fit in half_fits)
        candidates=authority_candidates(beta,tails,halves) if args.stage==1 else useful_candidates(beta,tails,halves,families)
        plot_beta=residual_fit(design,labels,plot_y,n_cells)[0]
        curve_beta=residual_fit(design,labels,curves.reshape(len(curves),-1),n_cells)[0].reshape(n_cells,32,6)
        plots(args.run_dir,levels,labels,plot_y,plot_beta,curves,curve_beta)
        signed_curves=signed_force_curves(args.run_dir,p,design,labels,n_cells,levels)
        measured=p['trajectory'][:,:,13:48].reshape(-1,32,5,7)[:,:,:,:3].mean(2)
        measured-=p['before'][:,13:48].reshape(-1,5,7)[:,:,:3].mean(1)[:,None]
        object_shift=p['trajectory'][:,:,:3]-p['before'][:,None,:3]
        displacement=torch.cat((measured,object_shift),-1).numpy().astype(float)
        displacement_beta=residual_fit(design,labels,displacement.reshape(len(displacement),-1),n_cells)[0].reshape(n_cells,32,6)
        save('mechanical_response.json',dict(metric_names=['hand_dx_m','hand_dy_m','hand_dz_m','object_dx_m','object_dy_m','object_dz_m'],
            adjusted_arm_minus_zero=displacement_beta.tolist(),note='Measured body centroid and object root displacement; descriptive mechanical authority, not an interaction gate.'))
        rows=[dict(alpha=levels[j],arm=a,count=int((labels==j*6+a).sum()),
            adjusted_arm_minus_zero=beta[j*6+a-1].tolist(),
            half_adjusted_arm_minus_zero=[b[j*6+a-1].tolist() for b in halves]) for j in range(len(levels)) for a in range(1,7)]
        result=dict(status='PROMISING' if valid and candidates else ('UNPROMISING' if valid else 'UNCLEAR'),
            stage=args.stage,selected_alpha=min(c['alpha'] for c in candidates) if args.stage==1 and valid and candidates else None,
            support_pass=bool(support),dose_pass=bool(dose),candidates=candidates,families=families,
            statistics=statistics,half_treatment_ranks=[fit[3] for fit in half_fits],half_counts=half_counts,
            adjusted_cells=rows,audit=audit,event_counts={k:int(v.sum()) for k,v in details.items() if v.dtype==torch.bool},
            elapsed_seconds=time.monotonic()-started,neural_models_executed=False,
            limits='Single-cohort aggregate force/proximity Probe; no certified friction/slip, universal threshold, same-state ranking or trained-policy Cm utility.')
        save('result.json',result)
        save('response_curves.json',dict(time_steps=list(range(1,33)),metric_names=['normal_abs_proxy_N','tangent_proxy_N','distance_m','contact_proxy','height_loss','ratio_proxy'],
            adjusted_arm_minus_zero=curve_beta.tolist(),raw_cells=[dict(alpha=levels[j],arm=a,
                mean=curves[labels==j*6+a].mean(0).tolist()) for j in range(len(levels)) for a in range(1,7)],
            signed_force_world_adjusted=signed_curves,
            pooled_zero_mean=curves[labels==0].mean(0).tolist(),
            note='Post-treatment descriptive physical-unit curves; not friction/slip measurements or additional gates.'))
        torch.save(dict(target=target,details=details,labels=labels,groups=groups,design=design,wave=waves,half=half,
            coefficient=beta,half_coefficient=halves,family_tails=tails,plot_target=plot_y,plot_coefficient=plot_beta),args.run_dir/'diagnostic.pt')
        print(json.dumps({k:result[k] for k in ('status','stage','selected_alpha','support_pass','dose_pass','candidates','families','event_counts','elapsed_seconds')},indent=2))
        manifest['run_status']='COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}'); raise
    finally:
        manifest['completed_at']=datetime.now(timezone.utc).isoformat(); save('manifest.json',manifest)


if __name__=='__main__': main()
