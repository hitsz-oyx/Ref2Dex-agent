#!/usr/bin/env python3
"""Replay all spatial weights; report contrasts and nominal execution mismatch."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run'),str(TASK/'tools/audit')]
from spatial_consequence import SpatialConsequence, evaluate, action_slots
from geometric_consequence import NominalSurfaceActions, pca_apply, persistence_baseline
from probe_conditional_consequence import preprocess
from consequence_sufficiency import readouts, PRIMARY, cluster_gain
from probe_interventions import sha
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
from src.task.CmResidual.v118_planner import QUERY_LINKS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();torch.set_num_threads(2)
    output=args.run_dir/'engineering_replay.json'
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((args.run_dir/'manifest.json').read_text())
    assert manifest['run_status']=='COMPLETED' and not manifest['smoke']
    assert sha(args.dataset)==manifest['dataset_sha256']
    assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
    p=torch.load(args.dataset,weights_only=False,map_location='cpu')
    d=torch.load(args.run_dir/'diagnostic.pt',weights_only=False,map_location='cpu')
    result=json.loads((args.run_dir/'result.json').read_text())
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    bridge=NominalSurfaceActions(urdf,device);g=bridge.build(p)
    sample=torch.load(args.dataset.parent/'object_surface_sample.pt',weights_only=False)
    selected=torch.linspace(0,len(sample['points'])-1,64).round().long()
    g['obj_points']=sample['points'][selected].to(device)[None]
    g['obj_normals']=sample['normals'][selected].to(device)[None]
    geom_error=max(float((g[key].cpu()-d['geometry'][key]).abs().max()) for key in ('points','normals','nominal_flow','geometry','joint','targets'))
    e,i,_,y,_=readouts(p);z=torch.cat((e,i),-1)
    assert torch.equal(z,d['z']) and torch.equal(y,d['y'])
    assert not set(d['clusters'][d['train']])&set(d['clusters'][d['test']])
    actual=d['arms'];actual_t=torch.as_tensor(actual,device=device)
    rows=torch.arange(len(actual),device=device);test=torch.as_tensor(d['test'],device=device)
    onehot=torch.nn.functional.one_hot(actual_t,15)[:,1:].float()
    blank=torch.zeros(len(actual),26,device=device)
    baseline=persistence_baseline(p['before'].to(device))
    full={};oof={name:torch.full_like(baseline,float('nan')) for name in ('State','Arm','Joint','Flow','Shuffled')}
    replay={};models={};full_h=None
    for prefix,norm in d['preprocess'].items():
        ids=torch.as_tensor(norm['fit'],device=device);hold=torch.as_tensor(norm['hold'],device=device)
        assert not set(d['clusters'][norm['fit']])&set(d['clusters'][norm['hold']])
        h,_,_=preprocess(p,ids,device,norm['h'])
        gn={k:v.to(device) if isinstance(v,torch.Tensor) else {kk:vv.to(device) for kk,vv in v.items()} for k,v in norm['geometry'].items()}
        h=torch.cat((h,pca_apply(g['geometry'],gn)),-1)
        scale=norm['scale'].to(device)
        perm=d['permutations'][prefix]
        assert set(perm[norm['fit']])==set(norm['fit']) and set(perm[norm['hold']])==set(norm['hold'])
        for name in oof:
            intended=actual_t[torch.as_tensor(perm,device=device)] if name=='Shuffled' else actual_t
            a,sa=action_slots(name,g,intended,onehot)
            model=SpatialConsequence(h.shape[1],26).to(device)
            model.load_state_dict(d['states'][prefix+'_'+name]);model.eval()
            raw=evaluate(model,h,a,blank,g,sa,rows)*scale+baseline
            oof[name][hold]=raw[hold]
            expected=d['full_predictions'][name][norm['hold']] if prefix=='full' else d['oof_predictions'][name][norm['hold']].numpy()
            replay[prefix+'_'+name]=float(np.max(np.abs(raw[hold].cpu().numpy()-expected)))
            if prefix=='full':
                full[name]=raw.cpu().numpy();models[name]=model;full_h=h
                if name=='Flow':
                    cand=[]
                    for arm in range(15):
                        intended=torch.full_like(actual_t,arm)
                        a,sa=action_slots(name,g,intended,onehot)
                        cand.append((evaluate(model,h,a,blank,g,sa,test)*scale+baseline[test]).cpu().numpy())
                    candidate_error=float(np.max(np.abs(np.stack(cand,1)-d['candidate_arrays']['Flow'])))
                    a,sa=action_slots(name,g,actual_t[torch.as_tensor(perm,device=device)],onehot)
                    shuffled=(evaluate(model,h,a,blank,g,sa,rows)*scale+baseline).cpu().numpy()
                    replay['Flow_test_shuffled']=float(np.max(np.abs(shuffled-d['full_predictions']['Flow_test_shuffled'])))
    for name in oof:
        assert torch.isfinite(oof[name]).all()
        replay['OOF_'+name]=float((oof[name].cpu()-d['oof_predictions'][name]).abs().max())
    zm,zs=d['target_mean'].to(device),d['target_scale'].to(device)
    task_h=torch.cat((full_h,(baseline-zm)/zs),-1)
    task_recomputed={}
    for name,state in d['task_states'].items():
        variant=name if name in ('Arm','Joint','Flow') else ('Flow' if name=='Flow_P_Flow' else 'State')
        a,sa=action_slots(variant,g,actual_t,onehot)
        physical=blank if name not in ('GT','P_State','P_Flow','Flow_P_Flow') else ((z.to(device)-zm)/zs if name=='GT' else (oof['State' if name=='P_State' else 'Flow']-zm)/zs)
        model=SpatialConsequence(task_h.shape[1],8).to(device);model.load_state_dict(state);model.eval()
        raw=evaluate(model,task_h,a,physical,g,sa,rows)*d['task_scale'].to(device)+d['task_mean'].to(device)
        replay['task_'+name]=float((raw.cpu()-d['task_predictions'][name]).abs().max())
        error=((raw.cpu().numpy()[d['test']]-y.numpy()[d['test']])/d['task_scale'].numpy())**2
        task_recomputed[name]=dict(primary_mse=float(error[:,PRIMARY].mean()),physical_failure_mse=float(error[:,6].mean()))
    assert max(replay.values())<1e-5 and geom_error<1e-6 and candidate_error<1e-5
    for name,values in task_recomputed.items():
        for key,value in values.items(): assert abs(value-result['task_metrics'][name][key])<1e-6

    # Factual endpoint-vs-realized q and true tips in respective measured
    # hand-base frames. This is descriptive execution error, not treatment
    # effect: later source actions vary and there is no paired zero rollout.
    nominal_q=g['targets'][rows,actual_t]
    realized_q=p['native_q'][:,7].to(device)
    current_q=p['history'][:,-1,:18].to(device)
    current_fk=bridge.kinematics.forward(current_q[:,None])[:,0]
    target_fk=bridge.kinematics.forward(nominal_q[:,None])[:,0]
    tip_ids=[QUERY_LINKS.index(name+'_tip') for name in ('index','middle','pinky','ring','thumb')]
    def relative_fk(fk):
        base=fk[:,0]
        return torch.einsum('bnj,bji->bni',fk[:,tip_ids,:3,3]-base[:,None,:3,3],base[:,:3,:3])
    nominal_tip_delta=relative_fk(target_fk)-relative_fk(current_fk)
    before_base=dexplore_root_pose(torch.nn.functional.pad(p['before_hand_base_pose'].to(device),(0,6)))
    after_base=dexplore_root_pose(torch.nn.functional.pad(p['hand_base_pose'][:,7].to(device),(0,6)))
    before_tip=torch.einsum('bnj,bji->bni',p['before_fingertip_positions'].to(device)-before_base[:,None,:3,3],before_base[:,:3,:3])
    after_tip=torch.einsum('bnj,bji->bni',p['fingertip_positions'][:,7].to(device)-after_base[:,None,:3,3],after_base[:,:3,:3])
    tip_error=(nominal_tip_delta-(after_tip-before_tip)).norm(dim=-1)*1000
    nominal_shift=nominal_tip_delta.norm(dim=-1)*1000
    realized_shift=(after_tip-before_tip).norm(dim=-1)*1000
    execution=[]
    drivers=[6,8,10,12,14,15]
    for arm,name in enumerate(p['arm_names']):
        keep=actual_t==arm
        execution.append(dict(arm=name,n=int(keep.sum()),nominal_endpoint_q_error_mae_rad=(nominal_q[keep][:,drivers]-realized_q[keep][:,drivers]).abs().mean(0).cpu().tolist(),
            nominal_tip_shift_mm=nominal_shift[keep].mean(0).cpu().tolist(),realized_tip_shift_mm=realized_shift[keep].mean(0).cpu().tolist(),
            tip_endpoint_vector_error_mm=tip_error[keep].mean(0).cpu().tolist()))
    report=dict(status='PASS',geometry_max_error=geom_error,candidate_max_error=candidate_error,weight_replay_errors=replay,
        task_metrics=task_recomputed,elapsed_seconds=time.monotonic()-started,nominal_execution=execution,
        limitations='Inference replay, not additional fitting/Validation; nominal-vs-realized differences include baseline evolving feedback, not isolated treatment effect.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    vectors={name:(values[:,1:]-values[:,:1]).mean(0).tolist() for name,values in d['candidate_arrays'].items()}
    vectors['GT_adjusted']=d['GT_contrasts'].tolist()
    (args.run_dir/'per_arm_contrasts.json').write_text(json.dumps(vectors,indent=2,allow_nan=False)+'\n')
    lines=['# Signed candidate-minus-zero E12/I14','', 'Raw units: E translation metres, rotation radians, velocity m/s and rad/s; I log-force xyz magnitudes, proximity metres, contact proxy. GT is current-adjusted marginal arm contrast, not matched per-state truth.','']
    for name,values in vectors.items():
        lines+=['## '+name,'','| Arm | E12 | I14 |','| --- | --- | --- |']
        for arm,v in zip(p['arm_names'][1:],values):lines.append('| '+arm+' | '+', '.join(f'{x:+.5g}' for x in v[:12])+' | '+', '.join(f'{x:+.5g}' for x in v[12:])+' |')
        lines.append('')
    lines+=['## Nominal versus realized tip endpoint motion','', 'Factual travel in measured hand-base frames, mm; includes evolving source feedback. Not arm-minus-zero or causal displacement.','', '| Arm | Nominal 5tips | Realized 5tips | Vector error 5tips |','| --- | --- | --- | --- |']
    for v in execution:lines.append('| '+v['arm']+' | '+' | '.join(', '.join(f'{x:.2f}' for x in v[k]) for k in ('nominal_tip_shift_mm','realized_tip_shift_mm','tip_endpoint_vector_error_mm'))+' |')
    (args.run_dir/'per_arm_contrasts.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,name in zip(axes,('GT_adjusted','Flow','Joint')):
        value=np.array(vectors[name])[:,12:]/d['target_scale'].numpy()[12:]
        im=ax.imshow(value,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set_title(name+' I / train scale');ax.set_yticks(range(14));ax.set_yticklabels(p['arm_names'][1:],fontsize=8)
        ax.set_xlabel('I dimension');fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(args.run_dir/'spatial_contrasts.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='PASS',max_replay_error=max(replay.values()),candidate_error=candidate_error,elapsed=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
