"""Diagnose saved states; historical scalar-only PPO failures are unrecoverable."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
from ptv3_shared_wam_v6.negative_result_invariants import voxel_first_failure,verified
from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,load_geometry,v4_online_samples
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM


def compare_gate_protocols(valid,ticks,prefix_controls,*,hz=30):
    """Censor the SAME saved state sequence; never predict post-reset dynamics."""
    valid=np.asarray(valid,dtype=bool);ticks=np.asarray(ticks,dtype=int);prefix=np.asarray(prefix_controls,dtype=int)
    if valid.ndim!=2 or len(ticks)!=len(valid) or len(prefix)!=valid.shape[1] or not len(ticks):raise ValueError('tick x environment validity required')
    if np.any(np.diff(ticks)!=1) or ticks[0]!=3:raise ValueError('contiguous first causal H4 at tick3 required')
    final=int(ticks[-1]);admitted=valid[0];admission_stop=np.where(admitted,final,ticks[0]);every_stop=admission_stop.copy()
    first=[]
    for i in range(valid.shape[1]):
        bad=np.flatnonzero(~valid[:,i]);tick=int(ticks[bad[0]]) if len(bad) else None;first.append(tick)
        if tick is not None:every_stop[i]=tick
    return dict(scope='observational censoring of existing state trajectory; no regenerated actions, reset, simulation or training',
        admission_tick=3,admission_definition='first complete causal H4, not a claim reset-only gate was checked',
        initial_accepted=int(admitted.sum()),environments=len(prefix),grid_checked_states=int(valid.size),
        first_invalid_tick=first,first_invalid_s=[None if t is None else t/hz for t in first],
        admission_only=dict(stop_tick=admission_stop.tolist(),retained_full_controls=int(admission_stop.sum()),
            retained_primary_controls=int(np.minimum(prefix,admission_stop).sum()),rejected_at_admission=int((~admitted).sum())),
        every_tick=dict(stop_tick=every_stop.tolist(),retained_full_controls=int(every_stop.sum()),
            retained_primary_controls=int(np.minimum(prefix,every_stop).sum()),terminated=int(sum(t is not None for t in first))),
        retained_full_control_difference=int((admission_stop-every_stop).sum()),
        retained_primary_control_difference=int((np.minimum(prefix,admission_stop)-np.minimum(prefix,every_stop)).sum()),
        not_identified='An admission-only training runner would still encounter invalid PTv3 inputs; safe fallback and future actions require a separately authorized protocol. Gate removal success, false-positive rate and recovered training steps are UNKNOWN.')


def protocol_replay(traces,panel,geometry,runtime,source):
    ticks=np.arange(3,len(traces['objects']));valid=np.zeros((len(ticks),len(geometry)),bool);details={}
    for time_index,tick in enumerate(ticks):
        samples=v4_online_samples(np.transpose(traces['objects'][tick-3:tick+1,:,:7],(1,0,2)),
            np.transpose(traces['hand'][tick-3:tick+1,:][:,:,NATIVE_TO_WAM],(1,0,2,3)),geometry,
            goal_delta=np.zeros((len(geometry),32,3)))
        valid[time_index]=runtime.workspace_valid(samples)
        for env in np.flatnonzero(~valid[time_index]):
            if int(env) in details:continue
            pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(traces['objects'][tick,env,3:7]).as_matrix();pose[:3,3]=traces['objects'][tick,env,:3]
            details[int(env)]=dict(sequence=panel['records'][env]['sequence'],env=int(env),tick=int(tick),seconds=tick/30,
                prefix_end_tick=int(panel['records'][env]['reference_controls']),**voxel_first_failure(samples[env],runtime.config,pose))
    prefixes=[int(row['reference_controls']) for row in panel['records']]
    comparison=compare_gate_protocols(valid,ticks,prefixes)
    comparison['first_invalid_details']=list(details.values())
    comparison['raw_invalid_state_count']=int((~valid).sum())
    expected=np.maximum.accumulate(~valid,axis=0)
    comparison['saved_v6_cumulative_flags_match_v4_gate']=bool(np.array_equal(expected,traces['workspace_failed'][3:]))
    comparison['flag_comparison_scope']='Agreement on this saved P0 panel; not a general proof v6 float64 and v4 float32 boundary arithmetic agree.'
    comparison['original_actual_physical_controls']=int((len(traces['objects'])-1)*len(geometry))
    comparison['same_trace_task_complete_admission_only']=int(sum(traces['complete'][:end+1,i].any() for i,end in enumerate(comparison['admission_only']['stop_tick'])))
    comparison['same_trace_task_complete_every_tick']=int(sum(traces['complete'][:end+1,i].any() for i,end in enumerate(comparison['every_tick']['stop_tick'])))
    comparison['source']=str(source)
    return comparison


def diagnose():
    source=ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1'
    panel=json.loads((source/'panel.json').read_text());objects=[dict(object_code='dexplore/'+r['object'],scale=1.) for r in panel['records']]
    geometry=load_geometry(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache',objects)
    packet=torch.load(ROOT/'outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt',map_location='cpu',weights_only=False)
    checkpoint_identity=dict(schema=packet['schema'],config_schema=packet['config'].get('schema'),step=packet.get('step'))
    rt=object.__new__(V4Runtime);rt.config=packet['config'];del packet
    rows=[];checks=[]
    with np.load(source/'flow/trajectory.npz') as z:
        traces={k:z[k] for k in ['objects','hand','workspace_failed','complete']}
    for i,row in enumerate(panel['records']):
        flags=np.flatnonzero(traces['workspace_failed'][:,i]);ticks=[3]
        if len(flags):ticks += [int(flags[0])-1,int(flags[0])]
        first=None
        for t in sorted(set(ticks)):
            sample=v4_online_samples(traces['objects'][t-3:t+1,i,:7][None],
                traces['hand'][t-3:t+1,i][:,NATIVE_TO_WAM][None],[geometry[i]],goal_delta=np.zeros((1,32,3)))[0]
            pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(traces['objects'][t,i,3:7]).as_matrix();pose[:3,3]=traces['objects'][t,i,:3]
            detail=voxel_first_failure(sample,rt.config,pose);valid=bool(rt.workspace_valid([sample])[0])
            assert valid==(detail is None)
            if detail is not None and first is None:first=dict(tick=t,seconds=t/30,**detail)
            rows.append(dict(sequence=row['sequence'],env=i,tick=t,recorded_v6_flag=bool(traces['workspace_failed'][t,i]),v4_grid_valid=valid,first_bad_point=detail))
        if first is not None:
            assert np.allclose(np.asarray(first['world_xyz']),traces['hand'][first['tick'],i][NATIVE_TO_WAM[first['point'][2]]] if first['point'][0]=='hand' else np.asarray(first['world_xyz']),atol=1e-6)
    # Mutate a real valid state: one valid hand coordinate just below the lower grid boundary.
    i=0;t=3
    sample=v4_online_samples(traces['objects'][0:4,i,:7][None],traces['hand'][0:4,i][:,NATIVE_TO_WAM][None],[geometry[i]],goal_delta=np.zeros((1,32,3)))[0]
    good=voxel_first_failure(sample,rt.config,np.eye(4));assert good is None
    sample['start_hand'][0,0,1]=rt.config['voxel_origin_m'][1]-.001
    bad=voxel_first_failure(sample,rt.config,np.eye(4))
    checks.append(verified('voxel gate rejects injected first hand lower-bound violation',bad['point']==('hand',0,0) and bad['axis']=='y' and not rt.workspace_valid([sample])[0],True,good is not None,'move one valid hand coordinate below voxel origin'))
    comparison=protocol_replay(traces,panel,geometry,rt,source/'flow/trajectory.npz')
    checks.append(verified('saved P0 cumulative gate flags equal actual V4 float32 replay',comparison['saved_v6_cumulative_flags_match_v4_gate'],True,False,'ignore first invalid state'))
    checks.append(verified('suffix gate exits do not censor the admitted primary prefixes',comparison['retained_primary_control_difference'],0,1,'count a post-prefix voxel exit as a primary-prefix failure'))
    return dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',checkpoint_identity=checkpoint_identity,config={k:rt.config[k] for k in ['voxel_origin_m','voxel_m','hand_in_ptv3']},
        historical_r3_first_point='UNAVAILABLE: historical PPO saved scalar counts, not terminal point clouds/hand states; do not attribute these separate P0 points to its 27792 episodes.',
        source=str(source/'flow/trajectory.npz'),scope='observational CPU replay of separate existing P0 trace with the actual V4 float32 gate; no simulation',
        cases=rows,protocol_comparison=comparison,checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    result=diagnose();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],states=len(result['cases']),failures=[c for c in result['cases'] if c['first_bad_point']],output=str(args.output))))
    raise SystemExit(int(result['status']!='PASS'))
