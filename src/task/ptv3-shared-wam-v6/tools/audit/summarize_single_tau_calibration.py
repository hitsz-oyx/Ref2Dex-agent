"""Independently recompute the narrow physical score from saved trajectories."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.task_feedback import FeedbackObserver
from wam_rl_policy_v3.teacher_quality import table_support
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM


def spans(mask):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return [(int(a),int(b)) for a,b in zip(edges[::2],edges[1::2])]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.run/'manifest.json').read_text());source=Path(report['source_pd_reference'])
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/report['source_sequence']
    with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
    limits=report['native_initial_audit']['physical_joint_limits'];rows=[]
    for record in report['runs']:
        with np.load(a.run/record['label']/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
        observer=FeedbackObserver(points);held=[];violations=[]
        for t in range(len(data['q'])):
            obj=data['objects'][t,0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
            support,gap,footprint=table_support(pose[None],points,data['table_pose'][t,0],bounds,data['object_contact_force'][t])
            q=data['q'][t,0];mask=np.asarray(limits['has_limits'],bool)
            bad=np.flatnonzero(mask&((q<np.asarray(limits['lower'])-.10)|(q>np.asarray(limits['upper'])+.10)))
            physical_abort=bool(len(bad) or (footprint[0] and gap[0]<-.02))
            if physical_abort:violations.append(dict(tick=t,indices=bad.tolist(),positions=q[bad].tolist(),lower=np.asarray(limits['lower'])[bad].tolist(),upper=np.asarray(limits['upper'])[bad].tolist(),table_gap_m=float(gap[0])))
            forces=data['contact_forces'][t,0];valid=bool(data['contact_force_valid'][t,0]);supported=bool(support[0] and valid)
            feedback=observer.update(pose,data['hand'][t,0,NATIVE_TO_WAM],obj[7:13],t/30,contact_forces=forces,force_valid=valid,supported=supported)
            held.append(bool(obj[2]-data['objects'][0,0,2]>=.05 and feedback.grasp_established and not feedback.grasp_lost
                and valid and not supported and (np.linalg.norm(forces,axis=-1)>=.1).sum()>=2 and not physical_abort))
        bouts=spans(held);n=max((hi-lo for lo,hi in bouts),default=0);ever=any(hi-lo>=31 for lo,hi in bouts)
        final=bool(bouts and bouts[-1][1]==len(held) and bouts[-1][1]-bouts[-1][0]>=31)
        rows.append(dict(label=record['label'],ever_success=ever,final_success=final,longest_positive_states=n,longest_elapsed_s=max(0,n-1)/30,
            hold_intervals=[dict(first_tick=lo,last_tick=hi-1,elapsed_s=(hi-lo-1)/30) for lo,hi in bouts],
            matches_online=ever==record['ever_success'] and final==record['final_success'] and abs(max(0,n-1)/30-record['longest_hold_elapsed_s'])<1e-9,
            first_safety_violation=violations[0] if violations else None,valid_tau_controls=int(data['tau_valid'][:len(data['action'])].sum()),
            gt_reference_length_states=len(np.load(a.run/'source-pd/trajectory.npz')['hand']) if record['channel']=='gt' else None))
    summary=dict(status='PASS' if all(r['matches_online'] for r in rows) else 'FAIL',records=rows,
        scope='Independent integer-duration physical predicate; FeedbackObserver reused for causal coupling, not an independent contact detector.',
        limitations=['GT reference ends at source first success; later invalid tau uses declared fallback, limiting closed-loop ceiling attribution',
            'Repeated resets share a single sequence, and GPU physical suffixes differ despite equal visible initial arrays',
            'Raw runtime asserts checked episodes but this run did not save per-frame episode IDs; native manifests retain mismatch count',
            'No complete RELEASE or exact8cm endpoint success claim'])
    (a.run/'independent-score.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(status=summary['status'],records=rows)))
    return int(summary['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
