"""Audit corresponding environments across fresh ref13 whole-world replays."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_y_utility import stable_grasp_z, candidate_deltas
from intervention import all_arms_have_headroom
from consequence_sufficiency import readouts


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def current_masks(panel, trace, initial, *, s3_only=False, future=122):
    """Select opportunities using current history/geometry, never future Y/Z."""
    n = len(panel['triggers'])
    hold = torch.zeros(n,dtype=torch.long)
    terminal = torch.zeros(n,dtype=torch.bool)
    lengths = initial['tensors']['max_episode_length'][panel['motion_id']]
    masks = []
    for tick, physical in enumerate(trace['physical']):
        held = (physical[:,2]-panel['rest_height']>=.03)&(physical[:,71]>.5)
        hold = torch.where(held,hold+1,0)
        eligible = (tick>=9)&(hold>=6)&(hold<45)&~terminal
        eligible &= physical[:,66:71].amin(-1)<.06
        eligible &= lengths-panel['start_frame']-tick>future
        eligible &= initial['scalars']['rollout_length']-tick>future
        eligible &= all_arms_have_headroom(trace['action'][tick],candidate_deltas())
        if s3_only:
            eligible &= panel['motion_id']==0
        masks.append(eligible)
        terminal |= trace['done'][tick]
    return torch.stack(masks)


def compare(first, second, first_trace, second_trace):
    keys = ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint')
    if any(first[key]!=second[key] for key in keys):
        raise ValueError('paired provenance mismatch')
    selected = first['triggers']>=0
    if not torch.equal(first['triggers'],second['triggers']):
        raise ValueError('paired trigger mismatch')
    if not first['valid_steps'][selected].all() or not second['valid_steps'][selected].all():
        raise ValueError('incomplete outcome window')
    diffs = {}
    for key in ('before','history','actor_obs','hand_root','height','pair','actions','pd_targets'):
        a,b=first[key][selected],second[key][selected]
        diffs[key] = dict(exact=bool(torch.equal(a,b)),
                         max_abs=float((a.double()-b.double()).abs().max()) if a.numel() else 0.)
    traces = {}
    length = min(len(first_trace['action']),len(second_trace['action']))
    for key in ('physical','dof','root','action','done'):
        a,b=first_trace[key][:length],second_trace[key][:length]
        traces[key] = dict(exact=bool(torch.equal(a,b)),max_abs=float((a.double()-b.double()).abs().max()))
    geometry = {}
    for key in ('object_pose','hand_keypoints'):
        a=first_trace['progress_geometry'][key][:length+1]
        b=second_trace['progress_geometry'][key][:length+1]
        geometry[key] = dict(exact=bool(torch.equal(a,b)),max_abs=float((a.double()-b.double()).abs().max()))
    z0,_=stable_grasp_z(first['height'][selected],first['pair'][selected],first['rest_height'][selected])
    z1,_=stable_grasp_z(second['height'][selected],second['pair'][selected],second['rest_height'][selected])
    tolerance=torch.tensor([1e-4]*5+[1e-5,0.])
    mechanical = bool((second['prefix_errors'][selected]<=tolerance).all())
    ya=readouts(first)[3]; yb=readouts(second)[3]
    eligible=selected & (second['prefix_errors']<=tolerance).all(-1)
    eligible &= (ya-yb).abs().amax(-1)<=.05
    all_z0=stable_grasp_z(first['height'],first['pair'],first['rest_height'])[0]
    all_z1=stable_grasp_z(second['height'],second['pair'],second['rest_height'])[0]
    eligible &= all_z0==all_z1
    screen=dict(assigned=int(selected.sum()),accepted=int(eligible.sum()),
                accepted_rows=eligible.nonzero().flatten().tolist(),
                y_max_abs=float((ya-yb)[selected].abs().max()),
                z_disagreements=int((z0!=z1).sum()),
                passed=bool(selected.any() and eligible.sum()>=.8*selected.sum()))
    return dict(anchors=int(selected.sum()),baseline_z=int(z0.sum()),repeat_z=int(z1.sum()),
                z_exact=bool(torch.equal(z0,z1)),mechanical_prefix_pass=mechanical,
                paired_fields=diffs,full_world_trace=traces,progress_geometry=geometry,
                all_recorded_fields_exact=all(x['exact'] for group in (diffs,traces,geometry) for x in group.values()),
                historical_repeat_screen=screen,old_contract_pass=screen['passed'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.run_dir.resolve()
    panel=load(root/'baseline/panel.pt'); trace=load(root/'baseline/trace.pt')
    initial=load(root/'baseline/initial_state.pt')
    masks=current_masks(panel,trace,initial)
    s3=current_masks(panel,trace,initial,s3_only=True)
    clock=int(s3.sum(1).argmax()); rows=s3[clock].nonzero().flatten().tolist()
    lift=trace['progress_geometry']['object_pose'][:,:,2,3]
    result=dict(scope='engineering recovery, frozen reconstructed actor; no Y utility claim',
                initial_motion_counts=torch.bincount(panel['motion_id'],minlength=3).tolist(),
                initial_start_frames=panel['start_frame'].tolist(),
                max_lift_per_env=(lift-lift[0]).amax(0).tolist(),
                earliest_eligible_anchors=int((panel['triggers']>=0).sum()),
                current_eligible_total_envs=int(masks.any(0).sum()),
                best_current_only_s3_clock=clock,s3_group_rows=rows,s3_group_size=len(rows))
    repeat=root/'repeat/panel.pt'
    if repeat.exists():
        result['repeat']=compare(panel,load(repeat),trace,load(root/'repeat/trace.pt'))
    with (root/'engineering-audit.json').open('w') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
