#!/usr/bin/env python3
"""No-fit replay of execution forecasts, spatial weights and projection effects."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from probe_execution_geometry import source_state,to_device
from execution_geometry import execution_inputs,endpoint_flows,spatial_variant,joint_units,joint_slots
from geometric_consequence import NominalSurfaceActions,ridge_predict
from execution_projection import continuous_mask,project_execution
from spatial_consequence import SpatialConsequence,evaluate
from probe_interventions import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();torch.set_num_threads(2)
    out=args.run_dir/'engineering_replay.json'
    if out.exists():raise FileExistsError(out)
    m=json.loads((args.run_dir/'manifest.json').read_text());assert m['run_status']=='COMPLETED' and not m['smoke']
    assert sha(args.dataset)==m['dataset_sha256']
    assert all(sha(ROOT/k)==v for k,v in m['code_sha256'].items())
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    repaired=m.get('projection')=='bounded_only_preserve_continuous'
    if repaired:
        origin=Path(d['original_run'])
        assert sha(origin/'diagnostic.pt')==m['original_diagnostic_sha256']
        assert sha(origin/'manifest.json')==m['original_manifest_sha256']
        original=torch.load(origin/'diagnostic.pt',map_location='cpu',weights_only=False)
        for key in ('source_geometry','execution_states','execution_norms','target_q','realized_flow'):
            d[key]=original[key]
        d['states']={**original['states'],**d['states']}
    old=torch.load(Path(d['source_run'])/'diagnostic.pt',map_location='cpu',weights_only=False)
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',dev)
    g=bridge.build(p);g['obj_points']=old['geometry']['obj_points'].to(dev);g['obj_normals']=old['geometry']['obj_normals'].to(dev)
    assert all(torch.equal(g[k].cpu(),d['source_geometry'][k]) for k in ('geometry','points','nominal_flow','joint'))
    units=joint_units(dev);q0=p['history'][:,-1,:18].to(dev);q8=p['native_q'][:,7].to(dev)
    assert torch.equal(q8.cpu(),d['target_q'])
    actual=torch.as_tensor(d['arms'],device=dev);rows=torch.arange(len(actual),device=dev)
    zero=torch.zeros_like(actual);test=torch.as_tensor(d['test'],device=dev)
    oof={name:torch.full_like(d['oof_q'][name],float('nan'),device=dev) for name in ('H','Ha')}
    replay,clip,normal_equations={}, {}, {}
    for prefix,norm in old['preprocess'].items():
        ids=torch.as_tensor(norm['fit'],device=dev);hold=torch.as_tensor(norm['hold'],device=dev)
        assert not set(d['clusters'][norm['fit']])&set(d['clusters'][norm['hold']])
        h=source_state(p,old,g,prefix,dev)
        saved=to_device(d['execution_norms'][prefix],dev)
        state,action,fresh=execution_inputs(p,h,g,ids)
        assert max(float((fresh[k]-saved[k]).abs().max()) for k in fresh)==0
        for name in ('H','Ha'):
            model=to_device(d['execution_states'][prefix+'_'+name],dev)
            values=[]
            for arm in range(15):
                a=action[:,arm] if name=='Ha' else torch.zeros_like(action[:,0])
                values.append(ridge_predict(torch.cat((state,a),-1),model))
            raw=q0[:,None]+torch.stack(values,1)*units
            bounded=project_execution(raw,bridge.lower,bridge.upper,continuous_mask(
                ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',dev)) if repaired else torch.maximum(torch.minimum(raw,bridge.upper),bridge.lower)
            if repaired:assert torch.equal(bounded[...,3:6],raw[...,3:6])
            oof[name][hold]=bounded[hold]
            expected=d['full_q'][name] if prefix=='full' else d['oof_q'][name]
            select=rows if prefix=='full' else hold
            replay[prefix+'_'+name]=float((bounded[select].cpu()-expected[select.cpu()]).abs().max())
            changed=(raw[hold]!=bounded[hold]).float()
            factual_raw=raw[hold,actual[hold]];factual_q=bounded[hold,actual[hold]]
            clip[prefix+'_'+name]=dict(hold_rows=len(hold),all_candidates_per_joint_fraction=changed.mean((0,1)).cpu().tolist(),
                factual_per_joint_fraction=(factual_raw!=factual_q).float().mean(0).cpu().tolist(),
                factual_clip_mae_raw_units=(factual_raw-factual_q).abs().mean(0).cpu().tolist(),
                factual_before_projection_scaled_mse=float(((factual_raw-q8[hold])/units).square().mean()),
                factual_after_projection_scaled_mse=float(((factual_q-q8[hold])/units).square().mean()))
            a=action[ids,actual[ids]] if name=='Ha' else torch.zeros_like(action[ids,0])
            x=torch.cat((state[ids],a),-1).double()-model['x_mean']
            y=((q8[ids]-q0[ids])/units).double()-model['y_mean']
            rhs=x.T@y
            residual=(x.T@x+model['alpha']*torch.eye(x.shape[1],device=dev,dtype=torch.float64))@model['weight']-rhs
            normal_equations[prefix+'_'+name]=float(residual.norm()/rhs.norm().clamp_min(1e-12))
    for name in oof:
        assert torch.isfinite(oof[name]).all()
        replay['OOF_'+name]=float((oof[name].cpu()-d['oof_q'][name]).abs().max())
    realized,_=endpoint_flows(p,bridge,g,q8[:,None]);predicted,_=endpoint_flows(p,bridge,g,oof['Ha'])
    replay['realized_flow']=float((realized.cpu()-d['realized_flow']).abs().max())
    shuffled=actual[torch.as_tensor(d['permutation'],device=dev)]
    for partition in (d['train'],d['test']):assert set(d['permutation'][partition])==set(partition)
    a_blank=torch.zeros(len(actual),32,device=dev);blank=torch.zeros(len(actual),26,device=dev)
    h=source_state(p,old,g,'full',dev);base=old['base'].to(dev);scale=old['target_scale'].to(dev)
    specs=dict(RealizedSurface=(a_blank,spatial_variant(g,realized)),RealizedJoint=(joint_slots(q8-q0),g),
        PredictedSurface=(a_blank,spatial_variant(g,predicted[rows,actual][:,None])),
        PredictedJoint=(joint_slots(oof['Ha'][rows,actual]-q0),g),
        PredictedSurfaceShuffled=(a_blank,spatial_variant(g,predicted[rows,shuffled][:,None])))
    for name,(a,gg) in specs.items():
        model=SpatialConsequence(h.shape[1],26).to(dev);model.load_state_dict(d['states'][name]);model.eval()
        raw=(evaluate(model,h,a,blank,gg,zero,rows)*scale+base).cpu().numpy()
        replay[name]=float(np.max(np.abs(raw-d['predictions'][name])))
        if name=='PredictedSurface':
            candidates=[]
            for arm in range(15):candidates.append((evaluate(model,h,a_blank,blank,spatial_variant(g,predicted[:,arm:arm+1]),zero,test)*scale+base[test]).cpu().numpy())
            replay['PredictedSurface_candidates']=float(np.max(np.abs(np.stack(candidates,1)-d['candidates'][name])))
            sf=(evaluate(model,h,a_blank,blank,specs['PredictedSurfaceShuffled'][1],zero,rows)*scale+base).cpu().numpy()
            replay['PredictedSurface_test_shuffled']=float(np.max(np.abs(sf-d['predictions']['PredictedSurface_test_shuffled'])))
    assert max(replay.values())<1e-5 and max(normal_equations.values())<1e-7
    finger_axes=[6,8,10,12,14,15]
    arm_table=[]
    for arm,name in enumerate(p['arm_names']):
        keep=actual==arm
        forecast=oof['Ha'][rows,actual]
        arm_table.append(dict(arm=name,n=int(keep.sum()),nominal_driver_delta_rad=(g['targets'][keep,arm]-q0[keep])[:,finger_axes].mean(0).cpu().tolist(),
            realized_driver_delta_rad=(q8[keep]-q0[keep])[:,finger_axes].mean(0).cpu().tolist(),
            predicted_OOF_driver_delta_rad=(forecast[keep]-q0[keep])[:,finger_axes].mean(0).cpu().tolist()))
    report=dict(status='PASS',projection=m.get('projection','INVALID_absolute_all18_clamp'),replay_errors=replay,normal_equation_relative_residuals=normal_equations,
        clipping=clip,per_arm_factual_execution=arm_table,elapsed_seconds=time.monotonic()-started,
        limits='No additional fit. Per-arm deltas are raw factual travel with baseline evolution, not adjusted treatment effects; real motion remains post-treatment oracle.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    vectors=(d['candidates']['PredictedSurface'][:,1:]-d['candidates']['PredictedSurface'][:,:1]).mean(0)
    (args.run_dir/'per_arm_contrasts.json').write_text(json.dumps(dict(PredictedSurface=vectors.tolist(),GT_adjusted=old['GT_contrasts'].tolist()),indent=2)+'\n')
    lines=['# Execution and consequence diagnostics','', 'Driver deltas: radians, raw factual q(step8)-q(current), not adjusted arm-minus-zero effects.','', '| Arm | Nominal 6drivers | Realized 6drivers | Predicted OOF 6drivers |','| --- | --- | --- | --- |']
    for v in arm_table:lines.append('| '+v['arm']+' | '+' | '.join(', '.join(f'{x:+.4f}' for x in v[k]) for k in ('nominal_driver_delta_rad','realized_driver_delta_rad','predicted_OOF_driver_delta_rad'))+' |')
    lines+=['','## Same-state predicted-minus-zero consequences','', 'E12 units m/rad/m/s/rad/s; I14 log-force/proximity(m)/proxy. All inputs here are predicted execution, not measured future hand motion.','', '| Arm | E12 | I14 |','| --- | --- | --- |']
    for name,value in zip(p['arm_names'][1:],vectors):lines.append('| '+name+' | '+', '.join(f'{x:+.5g}' for x in value[:12])+' | '+', '.join(f'{x:+.5g}' for x in value[12:])+' |')
    (args.run_dir/'execution_diagnostics.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((args.run_dir/'result.json').read_text())
    names=list(r['predictor_metrics']);fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,key in zip(axes,('E_mse','I_mse')):
        values=[r['predictor_metrics'][n][key] for n in names]
        ax.barh(names,values);ax.set_xlabel(key+' / train scale');ax.invert_yaxis()
        for i,v in enumerate(values):ax.text(v,i,f' {v:.3f}',va='center',fontsize=8)
    fig.suptitle('Post-treatment realized oracles versus causal execution forecasts (fixed-fit Probe)')
    fig.tight_layout();fig.savefig(args.run_dir/'execution_geometry.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='PASS',replay_max=max(replay.values()),ridge_relative_residual_max=max(normal_equations.values()),elapsed=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
