#!/usr/bin/env python3
"""Ref10: matched nominal point-flow and physics-innovation Decision Probe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/cm-interaction-oracle'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK/'src'))
sys.path.insert(0, str(TASK/'tools/audit'))
from geometric_consequence import (NominalSurfaceActions, physics_baseline, persistence_baseline,
    standardize_fit, normalize, pca_fit, pca_apply, pad, action_design, ridge_fit, ridge_predict, RIDGE)
from conditional_consequence import environment_folds, permute_within, contrast_scores, oracle_retention
from consequence_sufficiency import readouts, cluster_gain, PRIMARY
from probe_conditional_consequence import preprocess, cpu_state
from probe_interventions import sha, standardized
from probe_gt_consequence_sufficiency import EXPECTED_SHA
from probe_duration_response import current_design, residual_fit
from audit_finger_amplitudes import finger_audit, markdown

VARIANTS = ('State', 'Arm', 'Joint', 'Flow', 'FlowShuffled', 'RawState', 'RawArm')


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--oracle-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    def save(name, obj):
        (args.run_dir/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    paths = [Path(__file__).resolve(), TASK/'src/geometric_consequence.py', TASK/'src/conditional_consequence.py',
        TASK/'src/consequence_sufficiency.py', TASK/'src/intervention.py',
        TASK/'tools/run/probe_conditional_consequence.py', TASK/'tools/run/probe_interventions.py',
        TASK/'tools/audit/probe_duration_response.py', TASK/'tools/audit/audit_finger_amplitudes.py',
        ROOT/'src/task/CmResidual/dexplore_cm_geometry.py', ROOT/'src/task/CmResidual/v118_planner.py',
        ROOT/'src/task/CmResidual/surface_execution.py', urdf]
    from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
    mesh_paths = sorted({_surface_geometry_class()._mesh_path(urdf, node.get('filename'))
                         for node in ET.parse(urdf).getroot().findall('./link/visual/geometry/mesh')})
    manifest = dict(run_status='STARTED', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=sys.argv, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device='cuda:0', seeds=[241,242],
        dataset_sha256=sha(args.dataset), oracle_diagnostic_sha256=sha(args.oracle_run/'diagnostic.pt'),
        hand_visual_sha256={str(p):sha(p) for p in mesh_paths},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}, ridge=RIDGE,
        geometry_points=120, action_pca=32, state_geometry_pca=16, state_action_products=512,
        created_at=datetime.now(timezone.utc).isoformat())
    save('manifest.json', manifest)
    try:
        assert manifest['dataset_sha256']==EXPECTED_SHA
        assert manifest['oracle_diagnostic_sha256']=='16b6b0961c467e2836ca90cf0dc7285269570ac8fb1ed8129619e84091c63884'
        p = torch.load(args.dataset, map_location='cpu', weights_only=False)
        old = torch.load(args.oracle_run/'diagnostic.pt', map_location='cpu', weights_only=False)
        train_np, test_np = old['train'], old['test']
        clusters, motion, waves = old['clusters'], old['motion'], old['waves']
        arms = p['arm'].numpy()
        assert not set(clusters[train_np]) & set(clusters[test_np])
        assert (p['duration']==8).all() and (p['amplitude']==1).all()
        e, i, _, y, details = readouts(p)
        assert details['risk'].all() and np.array_equal(y.numpy(), old['target'])
        z = torch.cat((e, i), -1)
        device = torch.device('cuda:0')
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        tr = torch.as_tensor(train_np, device=device)
        te = torch.as_tensor(test_np, device=device)
        bridge = NominalSurfaceActions(urdf, device)
        geometry = bridge.build(p)
        zg, yg = z.to(device), y.to(device)
        base = physics_baseline(p['before'].to(device), float(p['control_dt'])*8)
        persistence = persistence_baseline(p['before'].to(device))
        zn, zm, zs = standardized(zg, tr)
        yn, ym, ys = standardized(yg, tr)
        folds = environment_folds(train_np, motion, clusters, seed=241)
        design, groups = current_design(p, waves)
        audit = finger_audit(p, design, arms)
        save('finger_amplitudes.json', audit)
        (args.run_dir/'finger_amplitudes.md').write_text(markdown(audit))
        onehot = torch.nn.functional.one_hot(p['arm'].to(device), 15)[:,1:].float()
        rows = torch.arange(len(arms), device=device)
        arm_gpu = p['arm'].to(device)
        full, oof, candidates, records, states, norms, permutations = {}, {}, {}, {}, {}, {}, {}
        oof = {name:torch.full_like(zg, float('nan')) for name in VARIANTS}
        normal_keys = ('h_mean','h_scale','projection','hp_mean','hp_scale','before_mean','before_scale','a_mean','a_scale')

        def run(prefix, fit_np, hold_np, full_fit=False):
            fit_ids = torch.as_tensor(fit_np, device=device)
            hold_ids = torch.as_tensor(hold_np, device=device)
            saved = {k:old['normalizers'][k] for k in normal_keys} if full_fit else None
            h, _, hn = preprocess(p, fit_ids, device, saved)
            gn = pca_fit(geometry['geometry'], fit_ids, 16)
            h = torch.cat((h, pca_apply(geometry['geometry'], gn)), -1)
            fn = pca_fit(geometry['flow'][rows, arm_gpu], fit_ids, 32)
            an = standardize_fit(onehot, fit_ids)
            jn = standardize_fit(geometry['joint'][rows, arm_gpu], fit_ids)
            zscale = standardize_fit(zg, fit_ids)
            perm = np.arange(len(arms))
            for ids in (fit_np, hold_np): perm[ids] = permute_within(groups, ids, 242)[ids]
            perm_arms = torch.as_tensor(arms[perm], device=device)
            # Regenerate intended flow at SAME state with permuted arm. Never
            # move geometry/state-dependent flows between environments.
            action = dict(State=torch.zeros(len(arms),32,device=device),
                Arm=pad(normalize(onehot, an)), Joint=pad(normalize(geometry['joint'][rows,arm_gpu], jn)),
                Flow=pca_apply(geometry['flow'][rows,arm_gpu], fn),
                FlowShuffled=pca_apply(geometry['flow'][rows,perm_arms], fn))
            action['RawState'], action['RawArm'] = action['State'], action['Arm']
            norms[prefix] = cpu_tree(dict(h=hn, geometry=gn, flow=fn, arm=an, joint=jn, target=zscale,
                fit=fit_np, hold=hold_np))
            permutations[prefix] = dict(permutation=perm, fit_changed=float((arms[fit_np]!=arms[perm[fit_np]]).mean()),
                hold_changed=float((arms[hold_np]!=arms[perm[hold_np]]).mean()))
            for name in VARIANTS:
                baseline = torch.zeros_like(base) if name.startswith('Raw') else base
                target = (zg-baseline)/zscale['scale']
                x = action_design(h, action[name])
                fitted = ridge_fit(x, target, fit_ids)
                pred = ridge_predict(x, fitted)*zscale['scale']+baseline
                states[prefix+'_'+name] = cpu_tree(fitted)
                records[prefix+'_'+name] = dict(train_joint_mse=float(((pred[fit_ids]-zg[fit_ids])/zscale['scale']).square().mean()),
                    feature_width=x.shape[1], coefficient_slots=fitted['weight'].numel(), ridge=RIDGE, optimization='exact centered float64 solve')
                oof[name][hold_ids] = pred[hold_ids]
                if full_fit:
                    full[name] = pred.cpu().numpy()
                    arrays = []
                    for arm in range(15):
                        if name in ('State','RawState'): ap = torch.zeros(len(hold_ids),32,device=device)
                        elif name in ('Arm','RawArm'):
                            oh = torch.zeros(len(hold_ids),14,device=device)
                            if arm: oh[:,arm-1]=1
                            ap = pad(normalize(oh, an))
                        elif name=='Joint': ap = pad(normalize(geometry['joint'][hold_ids,arm], jn))
                        else: ap = pca_apply(geometry['flow'][hold_ids,arm], fn)
                        arrays.append((ridge_predict(action_design(h[hold_ids], ap), fitted)*zscale['scale']+baseline[hold_ids]).cpu().numpy())
                    candidates[name] = np.stack(arrays, 1)
                    if name=='Flow':
                        full['Flow_test_shuffled'] = (ridge_predict(action_design(h, action['FlowShuffled']), fitted)*zscale['scale']+base).cpu().numpy()
                print(json.dumps(dict(fit=prefix+'_'+name, **records[prefix+'_'+name])), flush=True)
            return h, action

        h, action = run('full', train_np, test_np, True)
        for fold in range(3):
            run('fold'+str(fold), train_np[folds[train_np]!=fold], train_np[folds[train_np]==fold])
        assert all(torch.isfinite(value).all() for value in oof.values())
        full['TrainMean'] = zm.expand_as(zg).cpu().numpy()
        full['Persistence'] = persistence.cpu().numpy()
        full['SimpleDynamics'] = base.cpu().numpy()
        errors = {name:((value[test_np]-z.numpy()[test_np])/zs.cpu().numpy())**2 for name,value in full.items()}
        metrics = {name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean()),joint_mse=float(err.mean())) for name,err in errors.items()}
        comparisons = {}
        for new, control in (('Flow','State'),('Flow','Arm'),('Flow','Joint'),('Flow','FlowShuffled'),
            ('Flow','SimpleDynamics'),('Flow','TrainMean'),('Flow','Persistence'),('Flow','RawArm'),
            ('Arm','RawArm'),('State','RawState'),('Flow_test_shuffled','Flow')):
            comparisons[new+'_vs_'+control] = {key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),clusters[test_np],242)
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)),('EI',np.arange(26)))}
        beta, _, _, rank = residual_fit(design[test_np], arms[test_np], z.numpy()[test_np], 14)
        contrasts = {}
        for name, values in candidates.items():
            effect = (values[:,1:]-values[:,:1]).mean(0)
            centered_gt, centered_effect = beta-beta.mean(0), effect-effect.mean(0)
            paired_gt = beta[::2]-beta[1::2]
            paired_effect = effect[::2]-effect[1::2]
            contrasts[name] = {key:dict(raw=contrast_scores(beta,effect,zs.cpu().numpy(),axes),
                arm_centered=contrast_scores(centered_gt,centered_effect,zs.cpu().numpy(),axes),
                plus_minus=contrast_scores(paired_gt,paired_effect,zs.cpu().numpy(),axes))
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}

        # Current physical baselines are known at decision and available to ALL
        # downstream controls, to avoid credit for merely passing current I.
        task_h = torch.cat((h, (base-zm)/zs), -1)
        blank = torch.zeros_like(zg)
        task_specs = dict(H=(torch.zeros_like(action['State']),blank),
            Arm=(action['Arm'],blank), Joint=(action['Joint'],blank), Flow=(action['Flow'],blank),
            GT=(torch.zeros_like(action['State']),zn),
            P_State=(torch.zeros_like(action['State']),(oof['State']-zm)/zs),
            P_Flow=(torch.zeros_like(action['State']),(oof['Flow']-zm)/zs),
            Flow_P_Flow=(action['Flow'],(oof['Flow']-zm)/zs))
        task_predictions, task_states, task_errors, task_metrics = {}, {}, {}, {}
        for name,(a,physical) in task_specs.items():
            x = torch.cat((action_design(task_h,a),physical),-1)
            model = ridge_fit(x,yn,tr)
            pred = ridge_predict(x,model)*ys+ym
            error = ((pred-yg)/ys).square().cpu().numpy()
            task_predictions[name],task_states[name] = pred.cpu(),cpu_tree(model)
            task_errors[name] = error[test_np]
            task_metrics[name] = dict(primary_mse=float(error[test_np][:,PRIMARY].mean()),
                physical_failure_mse=float(error[test_np,6].mean()),train_primary_mse=float(error[train_np][:,PRIMARY].mean()))
        shuffled_z = (torch.as_tensor(full['Flow_test_shuffled'],device=device)-zm)/zs
        shuffled_x = torch.cat((action_design(task_h,torch.zeros_like(action['State'])),shuffled_z),-1)
        shuffled_prediction = ridge_predict(shuffled_x, {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in task_states['P_Flow'].items()})*ys+ym
        task_errors['P_Flow_test_shuffled'] = (((shuffled_prediction-yg)/ys).square().cpu().numpy())[test_np]
        task_comparisons = {}
        for new,control in (('GT','H'),('P_Flow','H'),('P_Flow','Flow'),('Flow_P_Flow','Flow'),
            ('P_Flow','P_State'),('P_Flow_test_shuffled','P_Flow')):
            task_comparisons[new+'_vs_'+control] = {key:cluster_gain(task_errors[control][:,axes].mean(1),task_errors[new][:,axes].mean(1),clusters[test_np],242)
                for key,axes in (('primary',PRIMARY),('physical_failure',(6,)))}
        retained = oracle_retention(task_errors['H'][:,PRIMARY].mean(1),task_errors['GT'][:,PRIMARY].mean(1),task_errors['P_Flow'][:,PRIMARY].mean(1),clusters[test_np],242)
        cm = contrasts['Flow']['I']
        A = (comparisons['Flow_vs_State']['I']['gain']>=.05 and comparisons['Flow_vs_State']['I']['lower95']>0
            and comparisons['Flow_vs_FlowShuffled']['I']['gain']>=.03
            and -comparisons['Flow_test_shuffled_vs_Flow']['I']['gain']>=.03)
        B = cm['raw']['correlation'] is not None and cm['raw']['correlation']>=.5 and cm['raw']['signed_agreement']>=.65 and cm['raw']['gain_vs_zero']>=.1 and cm['arm_centered']['gain_vs_zero']>0
        geometry_gain = comparisons['Flow_vs_Arm']['I']['gain']>=.03 and comparisons['Flow_vs_Joint']['I']['gain']>=.03
        C = any(task_comparisons[name+'_vs_Flow']['primary']['gain']>=.05
            and task_comparisons[name+'_vs_Flow']['primary']['lower95']>0
            and task_comparisons[name+'_vs_Flow']['physical_failure']['gain']>=-.02
            for name in ('P_Flow','Flow_P_Flow'))
        status = 'PROMISING' if rank==14 and A and B and geometry_gain else ('UNPROMISING' if rank==14 else 'UNCLEAR')
        # Overall does not upgrade to selector without unique transfer gate C.
        nominal_amplitudes = []
        for arm in range(1,15):
            mm = geometry['incremental_flow'][:,arm].reshape(len(arms),5,24,3).norm(dim=-1).mean(-1)*1000
            nominal_amplitudes.append(dict(arm=str(p['arm_names'][arm]),mean_finger_surface_shift_mm=mm.mean(0).cpu().tolist(),
                maximum_finger_surface_shift_mm=mm.max(0).values.cpu().tolist(),
                max_direct_pd_offset_rad=float(geometry['joint'][:,arm,6:].abs().max())))
        result = dict(status=status,gates=dict(A_action_information=bool(A),B_contrast=bool(B),geometry_specific=bool(geometry_gain),C_unique_transfer=bool(C),GT_contrast_rank=rank),
            predictor_metrics=metrics,predictor_comparisons=comparisons,contrasts=contrasts,
            downstream_metrics=task_metrics,downstream_comparisons=task_comparisons,oracle_retention=retained,
            fingertip_error_max_m=geometry['fingertip_error_max_m'],handbase_matrix_error_max=geometry['handbase_matrix_error_max'],
            nominal_finger_amplitudes=nominal_amplitudes,
            baseline_nominal_surface_shift_mean_mm=float(geometry['nominal_flow'][:,0].norm(dim=-1).mean()*1000),
            split=dict(train=len(train_np),test=len(test_np),train_environments=len(set(clusters[train_np])),test_environments=len(set(clusters[test_np]))),
            elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
            limitations='Nominal endpoint surface action + PCA/ridge, not actual OI-CmV2 spatial network or executed K8 flow; compact E12, single Probe split; no policy utility.',policy_executed=False)
        save('result.json',result);save('fit_records.json',records)
        torch.save(dict(train=train_np,test=test_np,clusters=clusters,folds=folds,arms=arms,z=z,y=y,
            base=base.cpu(),target_scale=zs.cpu(),target_mean=zm.cpu(),geometry=cpu_tree(geometry),
            preprocess=norms,permutations=permutations,predictor_states=states,
            full_predictions=full,oof_predictions=cpu_tree(oof),candidate_arrays=candidates,GT_contrasts=beta,
            design=design,downstream_states=task_states,task_predictions=task_predictions,task_target_mean=ym.cpu(),task_target_scale=ys.cpu()),args.run_dir/'diagnostic.pt')
        assert sha(args.dataset)==EXPECTED_SHA and all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        assert all(sha(Path(k))==v for k,v in manifest['hand_visual_sha256'].items())
        manifest['run_status']='COMPLETED'
        print(json.dumps(dict(status=status,gates=result['gates'],predictor_metrics=metrics,downstream_metrics=task_metrics),indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save('manifest.json',manifest)


if __name__=='__main__': main()
