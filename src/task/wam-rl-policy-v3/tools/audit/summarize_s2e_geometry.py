"""Recompute the paired S2e outcomes and all saved geometric absolute targets."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
from audit_first_tau_loss import load, physical_rows
from wam_rl_policy_v3.tau_replay_contract import bouts, pd_targets, require_saved_clock
from wam_rl_policy_v3.inspire_contract import expand_action, NATIVE_TO_WAM


def pose(value):
    result = np.eye(4)
    result[:3,:3] = Rotation.from_quat(value[3:7]).as_matrix()
    result[:3,3] = value[:3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    m = json.loads((args.run/'manifest.json').read_text())
    if m['status'] != 'COMPLETED' or len(m['runs']) != 10:
        raise ValueError('completed 2x5 comparison required')
    original = json.loads((Path(m['original_run'])/'manifest.json').read_text())
    points = load(Path(original['source_pd_reference'])/'canonical.npz')['points']
    bounds = load(Path(m['asset_root'])/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/m['source_sequence']/'admission.npz')['table_bounds']
    reference = load(Path(m['complete_reference']))
    hand, objects = reference['hand'][:,0], reference['objects'][:,0]
    # Independent homogeneous-matrix reconstruction, not geometry_tau helpers.
    local = np.stack([(np.linalg.inv(pose(o)) @ np.c_[h,np.ones(11)].T).T[:,:3]
                      for h,o in zip(hand,objects)])
    spec = importlib.util.spec_from_file_location('s2e_domain', args.run/'runtime-code/src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/execution_contract.py')
    domain = importlib.util.module_from_spec(spec);spec.loader.exec_module(domain)
    offset = np.asarray(m['native_initial_audit']['controller']['offset'])
    scale = np.asarray(m['native_initial_audit']['controller']['scale'])
    records = []
    for online in m['runs']:
        folder = args.run/online['label']
        d = load(folder/'trajectory.npz');require_saved_clock(d)
        infos = json.loads((folder/'consumer.json').read_text())
        rows = physical_rows(d,points,bounds,m['native_initial_audit']['physical_joint_limits'])
        spans = bouts([r['held'] for r in rows])
        longest = max(((hi-lo-1)/30 for lo,hi in spans), default=0.)
        ever = any(hi-lo>=31 for lo,hi in spans)
        final = bool(spans and spans[-1][1]==len(rows) and spans[-1][1]-spans[-1][0]>=31)
        if (ever,final)!=(online['ever_success'],online['final_success']) or abs(longest-online['longest_hold_elapsed_s'])>1e-9:
            raise ValueError('independent physical outcome differs')
        physical_failures = [v for row in rows for v in row['failed_predicates'] if v not in ('lift_below5cm','grasp_not_established','grasp_lost','force_invalid','fewer_than_two_force_contacts','table_supported')]
        first_abort = physical_failures[0] if physical_failures else None
        if first_abort != online['physical_safety_abort']:
            raise ValueError('independent safety verdict differs')
        n = len(d['action'])
        clip = int((np.abs(d['raw_action']-d['action'])>1e-6).any(axis=-1).sum())
        if clip != online['clipped_controls'] or not np.array_equal(np.clip(d['raw_action'],-1,1),d['action']):
            raise ValueError('raw/output clipping mismatch')
        if not np.array_equal(expand_action(d['action']),d['expanded_action']):
            raise ValueError('expanded action mismatch')
        pd_error = float(np.abs(pd_targets(d['action'],d['q'][:n,0],offset,scale)-d['pd_targets_actual']).max())
        if pd_error>2e-5 or not np.array_equal(d['episodes_input'],d['episodes'][:n,0]) or np.diff(d['episodes'][:,0]).any():
            raise ValueError('PD or episode ownership mismatch')
        if not np.array_equal(d['observation'][:,421:685],d['tau'].reshape(len(rows),-1)) or not np.array_equal(d['observation'][:,685],d['tau_valid']):
            raise ValueError('packed tau mismatch')
        target_error = delta_error = hand_counterfactual_error = 0.
        if online['chain']=='geometry':
            for t in range(len(rows)):
                live = pose(d['objects'][t,0])
                phase = 'approach' if t==0 else infos[t-1]['phase']
                valid = bool(t>=3 and t+24<len(local) and phase in ('approach','close') and domain.encoder_domain_available(points,d['hand'][t,0,NATIVE_TO_WAM],live))
                if valid != bool(d['forecast_valid'][t]) or valid != bool(d['tau_valid'][t]):
                    raise ValueError('geometric validity mismatch')
                if not valid:
                    if np.any(d['tau'][t]) or np.any(d['forecast_hand'][t]):
                        raise ValueError('invalid geometric target not zero')
                    continue
                expected = local[t+np.arange(1,25)] @ live[:3,:3].T + live[:3,3]
                target_error = max(target_error,float(np.abs(expected-d['forecast_hand'][t]).max()))
                encoded = (expected[:8]-d['hand'][t,0]).astype(np.float32)
                delta_error = max(delta_error,float(np.abs(encoded-d['tau'][t]).max()))
                changed_hand = d['hand'][t,0]+[.17,-.11,.09]
                hand_counterfactual_error = max(hand_counterfactual_error,float(np.abs((expected[:8]-changed_hand)+changed_hand-expected[:8]).max()))
            if target_error>1e-6 or delta_error>1e-6 or hand_counterfactual_error>1e-12 or online['wam_calls']:
                raise ValueError('geometry target/anti-alignment contract failed')
        heights = d['objects'][:,0,2]-d['objects'][0,0,2]
        reached = np.flatnonzero(heights>=.05)
        records.append(dict(**online,independent_longest_hold_s=longest,pd_error=pd_error,
            geometric_target_max_error=target_error,geometric_tau_max_error=delta_error,
            changed_hand_target_reconstruction_error=hand_counterfactual_error,
            first_lift5cm_tick=int(reached[0]) if len(reached) else None,
            hold_intervals=[dict(first_tick=lo,last_tick=hi-1,elapsed_s=(hi-lo-1)/30) for lo,hi in spans]))
    chains = {}
    for name in ('wam','geometry'):
        part = sorted([r for r in records if r['chain']==name],key=lambda r:r['block'])
        if [r['block'] for r in part] != list(range(5)):
            raise ValueError('five blocks required')
        chains[name] = dict(n=5,longest_hold_s=[r['independent_longest_hold_s'] for r in part],
            ever_successes=sum(r['ever_success'] for r in part),final_successes=sum(r['final_success'] for r in part),
            controls=sum(r['controls'] for r in part),clipped_controls=sum(r['clipped_controls'] for r in part),
            adapter_clipped_controls=sum(r['adapter_clipped_controls'] for r in part),
            joint_limit_episodes=sum(r['physical_safety_abort']=='physical_joint_limit' for r in part),
            other_abort_episodes=sum(r['physical_safety_abort'] not in (None,'physical_joint_limit') for r in part),
            valid_forecast_controls=sum(r['encoder_covered_controls'] for r in part),
            measured_grasp_handoffs=sum(r['phase_switch_time_s'] is not None for r in part),
            final_object_goal_errors_m=[r['final_object_goal_error_m'] for r in part])
    if sum(c['controls'] for c in chains.values()) != m['environment_controls']:
        raise ValueError('control ledger mismatch')
    result = dict(status='PASS',experiment_id=m['experiment_id'],run_id=m['run_id'],git_commit=m['git_commit'],
        protocol=m['protocol'],consumer=m['consumer'],environment=m['environment'],elapsed_s=m['elapsed_s'],
        learning_updates=m['learning_updates'],environment_controls=m['environment_controls'],
        maximum_actual_pd_error=max(r['pd_error'] for r in records),
        maximum_visible_initial_error=max(v for r in records for v in r['old_reference_initial_max_errors'].values()),
        wam_minus_geometry_successes=chains['wam']['ever_successes']-chains['geometry']['ever_successes'],
        chains=chains,records=records)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
