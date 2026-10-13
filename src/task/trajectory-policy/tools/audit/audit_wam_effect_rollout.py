"""Independent saved native-command, measured-history and grasp audit for C."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'tools/audit'), str(ROOT/'src/task/consequence-evaluator/src')]
from audit_competent_pd import independent_pd, independent_features, ACTIVE
from consequence_evaluator.gate1 import episode_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    out = args.native.resolve()
    destination = out/'root_effect_rollout_audit.json'
    if destination.exists():raise FileExistsError(destination)
    manifest = json.loads((out/'manifest.json').read_text())
    result = json.loads((out/'result.json').read_text())
    if manifest['status'] != 'COMPLETED' or result['status'] != 'COMPLETED' or manifest['controls'] != 542:
        raise ValueError('completed full542 native rollout required')
    with np.load(out/'trajectory.npz') as stream:
        data = {key:stream[key].copy() for key in stream.files}
    errors = {}
    def check(key, actual, expected):
        if actual.shape != expected.shape:raise ValueError('shape differs: '+key)
        errors[key] = float(np.max(np.abs(actual.astype(float)-expected.astype(float))))
        if errors[key] > 1e-6:raise ValueError('native contract differs: '+key)
    if not all(np.isfinite(value).all() for value in data.values()):raise ValueError('nonfinite arrays')
    if len(data['q']) != 543 or len(data['applied']) != 542:raise ValueError('terminal boundary differs')
    check('actual_commands', data['action'], data['applied'])
    check('native_PD', data['pd_targets'], independent_pd(data['applied'],data['q'][:-1],manifest['native_controller']))
    check('first_chunk_only', data['applied'][:,1:3,ACTIVE], data['encoded_target'][:,:,0].clip(-1,1))
    passive = [i for i in range(18) if i not in ACTIVE]
    check('passive_zero', data['applied'][:,1:3,passive], np.zeros_like(data['applied'][:,1:3,passive]))
    check('clipping_flags', data['clipped'], np.abs(data['intended']-data['applied']).max(-1)>1e-6)
    ticks = np.repeat(np.arange(542),2); envs = np.tile([1,2],542)
    with np.load(ROOT/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2/trajectory.npz') as stream:
        oracle = stream['hand'][:,24].copy()
    future = oracle[np.minimum(np.arange(542)[:,None]+np.arange(1,25),542)]
    state, tau = independent_features(data,ticks,envs,np.repeat(future,2,axis=0))
    check('own_live_R_state', data['state_features'].reshape(-1,87),state)
    check('GT_own_frame_tau', data['tau_features'][:,0],tau.reshape(542,2,24,33)[:,0])
    past = np.maximum(np.arange(542)[:,None]+np.arange(-3,1),0)
    check('own_measured_hand_H4', data['wam_hand_history'],data['hand'][past,2])
    check('own_measured_object_H4', data['wam_object_history'],data['obj'][past,2])
    with np.load(ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz') as stream:
        goals=stream['obj'][np.minimum(np.arange(542)+24,542)]
    check('planned_goal_only', data['wam_goal'],goals)
    if manifest.get('control_identity',{}).get('applied'):
        with np.load(ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz') as stream:
            refs=stream['obj'][np.minimum(np.arange(542)[:,None]+np.array([8,16,24]),542)]
        previous=np.concatenate((np.zeros((1,18),np.float32),data['applied'][:-1,2]),axis=0)
        proprio=np.concatenate((data['q'][:-1,2],data['dq'][:-1,2],previous[:,ACTIVE]),axis=-1)
        check('own_q_dq_previous_applied',data['control_proprio'],proprio)
        check('three_planned_references',data['control_reference_poses'],refs)
        # Independent homogeneous inverse rather than the producer helper.
        relative=np.linalg.inv(data['obj'][:-1,2].astype(np.float64))[:,None]@refs.astype(np.float64)
        robot_goal=np.concatenate((relative[:,:,:3,3],relative[:,:,:3,0],relative[:,:,:3,1]),axis=-1).reshape(542,27)
        check('own_relative_translation_rotation6D',data['control_robot_goal'],robot_goal)
        if manifest['control_identity']!=manifest['server_identity']['control_identity']:
            raise ValueError('native/server robot adapter identity differs')
    outcomes = []
    for env, saved in enumerate(result['outcomes']):
        actual = episode_outcome(dict(object_pose=data['obj'][:,env],object_velocity=data['velocity'][:,env],
                                      surface_gap=data['surface_gap'][:,env],support_gap=data['support_gap'][:,env],
                                      table_footprint=data['table_footprint'][:,env]))
        if any(actual[key] != saved[key] for key in actual):raise ValueError('outcome differs')
        height=data['obj'][:,env,2,3]-data['obj'][0,env,2,3]
        held=((data['surface_gap'][:,env]<=.01)&(height>=.03)
              &~(data['table_footprint'][:,env].astype(bool)&(np.abs(data['support_gap'][:,env])<=.02)))
        held[0]=False
        run=0;maximum=0;first=None
        for tick,value in enumerate(held):
            run=run+1 if value else 0;maximum=max(maximum,run)
            if value and first is None:first=tick
        if maximum != actual['maximum_held_frames']:raise ValueError('independent consecutive hold differs')
        actual.update(role=saved['role'],clipping_rate=float(data['clipped'][:,env].mean()),
                      maximum_height_m=float(height.max()),first_held_tick=first,
                      held_frame_count=int(held.sum()),lift45=maximum>=45)
        outcomes.append(actual)
    calibration=all(value['lift45'] and value['clipping_rate']<.01 for value in outcomes[:2])
    report=dict(status='PASS',contract_max_difference=max(errors.values()),errors=errors,
                controls=542,measured_states=543,grasp_calibration=calibration,outcomes=outcomes,
                conclusion='PROMISING' if calibration and outcomes[2]['lift45'] else 'UNPROMISING' if calibration else 'UNCLEAR',
                scope='one simulated development reset; outcome and execution audit; no matched Cm-off attribution')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':main()
