"""Recompute first narrow-hold loss against saved actual GT-tau masks, on CPU."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def audit(run):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.teacher_quality import table_support
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from ptv3_shared_wam_v6.tau_contract import first_loss_evidence, reference_coverage
    report=json.loads((run/'manifest.json').read_text())
    with np.load(Path(report['source_pd_reference'])/'canonical.npz') as z:points=z['points'].copy()
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/report['source_sequence']
    with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
    with np.load(run/'source-pd/trajectory.npz') as z:reference=z['hand'][:,0].copy()
    limits=report['native_initial_audit']['physical_joint_limits'];rows=[]
    for label in ['calibrated-gt-0','calibrated-gt-1']:
        with np.load(run/label/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
        observer=FeedbackObserver(points);held=[];feedback_lost=[]
        if not np.array_equal(data['tick'][:,0],np.arange(len(data['tick']))):raise ValueError('tick discontinuity')
        for t in range(len(data['q'])):
            obj=data['objects'][t,0];pose=np.eye(4)
            pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
            support,gap,footprint=table_support(pose[None],points,data['table_pose'][t,0],bounds,data['object_contact_force'][t])
            q=data['q'][t,0];mask=np.asarray(limits['has_limits'],bool)
            bad=mask&((q<np.asarray(limits['lower'])-.10)|(q>np.asarray(limits['upper'])+.10))
            abort=bool(bad.any() or (footprint[0] and gap[0]<-.02))
            forces=data['contact_forces'][t,0];valid=bool(data['contact_force_valid'][t,0])
            feedback=observer.update(pose,data['hand'][t,0,NATIVE_TO_WAM],obj[7:13],t/30,
                contact_forces=forces,force_valid=valid,supported=bool(support[0] and valid))
            held.append(bool(obj[2]-data['objects'][0,0,2]>=.05 and feedback.grasp_established
                and not feedback.grasp_lost and valid and not feedback.supported
                and (np.linalg.norm(forces,axis=-1)>=.1).sum()>=2 and not abort))
            feedback_lost.append(feedback.grasp_lost)
        row=first_loss_evidence(held,data['tau_valid'],len(reference))
        tick=row['first_loss_tick']
        if tick is not None:
            np.testing.assert_allclose(data['tau'][tick],reference[tick+np.arange(1,9)]-reference[tick],atol=1e-7)
        row.update(label=label, loss_predicate='first loss of lift5cm/force/coupled/unsupported/safe conjunction',
            feedback_grasp_lost_at_first_loss=None if tick is None else bool(feedback_lost[tick]),
            first_feedback_grasp_lost_tick=next((i for i,v in enumerate(feedback_lost) if v),None),
            valid_tau_controls=int(data['tau_valid'][:len(data['action'])].sum()),
            loss_has_following_control=bool(tick is not None and tick<len(data['action'])))
        rows.append(row)
    # Historical 66/64 were the inclusive LAST positive ticks, not first false.
    assert [r['last_held_tick_before_first_loss'] for r in rows]==[66,64], rows
    assert [r['first_loss_tick'] for r in rows]==[67,65], rows
    assert all(r['loss_while_gt_valid'] and r['loss_has_following_control'] for r in rows)
    return dict(status='PASS',reference=reference_coverage(len(reference),80),records=rows,
        python=sys.executable,numpy=np.__version__,gpu_allocated=False,optimizer_updates=0,
        limitation='FeedbackObserver reused; this verifies timing/channel attribution, not independent grasp truth or executor qualification.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.run.resolve());text=json.dumps(result,indent=2)+'\n'
    if a.output:
        if TASK not in a.output.resolve().parents:raise ValueError('Task-local output required')
        with a.output.open('x') as stream:stream.write(text)
    print(text)


if __name__=='__main__':main()
