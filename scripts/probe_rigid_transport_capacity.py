"""Fixed endpoint family and outcome-fitted oracle capacity, no neural fit."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_transport_capacity import VISUAL_IDS,transport_fields,optimal_segments,episode_report,classify
from src.task.CmResidual.v118_planner import QUERY_LINKS


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and ROOT in a.output.resolve().parents and not a.output.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    old=a.execution_source/'qualified'
    with np.load(old/'held_rows.npz') as f:rows={k:f[k] for k in f.files}
    with np.load(old/'geometry.npz') as f:object_local=f['object_local'];visual_ids=tuple(np.unique(f['links']).tolist())
    assert visual_ids==VISUAL_IDS
    with np.load(old/'joint_predictions.npz') as f:predicted=f['action_velocity']
    with np.load(old/'features.npz') as f:near=f['stationary'][...,15].min(-1)*.05<.02
    assert len(rows['env'])==6144 and np.array_equal(np.unique(rows['env'],return_counts=True)[1],np.full(384,16))
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    chunks={key:[] for key in ('anchor','target','causal','measured_hand','state_only')}
    details={mode:{key:[] for key in ('coefficients','segment_errors_m','winner','prediction')} for mode in ('causal','measured_hand','state_only')}
    sdk_max=0.;ancestor=json.loads((a.execution_source/'run_manifest.json').read_text())
    metadata=json.loads((Path(ancestor['source_native'])/'s655/physical_metadata.json').read_text())
    contact=[QUERY_LINKS.index(metadata['native_body_names'][i]) for i in metadata['contact_body_ids']]
    for start in range(0,6144,128):
        stop=min(start+128,6144);subset={k:v[start:stop] for k,v in rows.items()}
        fields=transport_fields(subset,predicted[start:stop],object_local,urdf,'cuda:0')
        sdk=float((fields['measured_next_hand_poses'][:,contact][:,:,:3,3]-torch.as_tensor(subset['sdk_next_positions'],device='cuda:0',dtype=torch.float64)).abs().max())
        sdk_max=max(sdk_max,sdk);assert sdk<2e-4
        endpoints=dict(fields['candidates']);endpoints['state_only']=endpoints['causal'][:,:2]
        for key in ('anchor','target'):chunks[key].append(fields[key].cpu().numpy())
        for mode,value in endpoints.items():
            chunks[mode].append(value.cpu().numpy())
            result=optimal_segments(fields['anchor'],value,fields['target'])
            for key,v in result.items():details[mode][key].append(v.cpu().numpy())
        if (start//128+1)%8==0:print(json.dumps(dict(windows=stop,total=6144)),flush=True)
    banks={key:np.concatenate(value) for key,value in chunks.items()}
    fits={mode:{key:np.concatenate(value) for key,value in values.items()} for mode,values in details.items()}
    reports={mode:episode_report(values['prediction'],banks['target'],rows['env']) for mode,values in fits.items()}
    baselines={name:episode_report(value,banks['target'],rows['env']) for name,value in
               (('persistence',banks['anchor']),('zero',np.zeros_like(banks['target'])),('rigid_inertia',banks['state_only'][:,1]))}
    diagnostics={}
    masks={**{'motion_'+str(i):rows['motion']==i for i in range(3)},**{'arm_'+str(i):rows['arm']==i for i in range(4)},'near':near,'far':~near}
    for name,mask in masks.items():
        diagnostics[name]={mode:episode_report(values['prediction'][mask],banks['target'][mask],rows['env'][mask]) for mode,values in fits.items()}
        diagnostics[name]['persistence']=episode_report(banks['anchor'][mask],banks['target'][mask],rows['env'][mask])
    gates,label=classify(reports,diagnostics['near'])
    names=['zero','rigid_inertia']+[QUERY_LINKS[i] for i in VISUAL_IDS]
    selection={mode:dict(winner_counts={names[i]:int((value['winner']==i).sum()) for i in range(value['coefficients'].shape[1])},
                         coefficient_quantiles=np.quantile(value['coefficients'][np.arange(6144),value['winner']],[0,.25,.5,.75,1]).tolist()) for mode,value in fits.items()}
    a.output.mkdir();np.savez(a.output/'fields.npz',**banks)
    for mode,value in fits.items():np.savez(a.output/(mode+'_oracle.npz'),**value)
    result=dict(run_status='COMPLETED',label=label,gates=gates,reports=reports,baselines=baselines,diagnostics=diagnostics,
                selection=selection,endpoint_names=names,held_episodes=384,held_windows=6144,oracle_segments_solved=6144*(15+15+2),
                sdk_origin_max_m=sdk_max,new_native_ticks=0,new_neural_optimizer_updates=0,
                outcome_fitted_oracle_weights_not_deployable=True,mixture_mean_not_necessarily_rigid_pose=True,
                unrestricted_hand_endpoint_choice_not_contact_validated=True,policy_utility_unmeasured=True,
                seen_same_seed_held_episode_distribution=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=label,gates=gates,errors={k:v['episode_epe_mm'] for k,v in reports.items()})),flush=True)


if __name__=='__main__':main()
