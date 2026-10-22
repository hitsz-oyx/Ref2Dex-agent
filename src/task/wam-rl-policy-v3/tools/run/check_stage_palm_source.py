"""Reproduce the wrist-origin/S2c mismatch on the unchanged S2b positive source."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/ptv3-shared-wam-v6/src'),str(TASK/'tools/run')]
from wam_rl_policy_v3.stage_evaluation import PalmSurface
from prepare_large_grasp_panel import qualify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1';original=json.loads((old/'manifest.json').read_text())
    with np.load(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz') as z:reference={k:z[k].copy() for k in z.files}
    with np.load(Path(original['source_pd_reference'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/original['source_sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
    limits=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r2/manifest.json').read_text())['native_initial_audit']['physical_joint_limits']
    class WristOriginMutant:
        def features(self,features,*args):return features
    table=reference['table_pose'][0,0]
    wrong=qualify(reference,points,table,bounds,WristOriginMutant(),limits)
    correct=qualify(reference,points,table,bounds,PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'),limits)
    passed=bool(not wrong['stage']['achieved']['approach'] and correct['stage']['achieved']['hold'] and abs(correct['longest_hold_s']-2.)<1e-8)
    report=dict(status='PASS' if passed else 'FAIL',wrong_origin=wrong,actual_cad_palm=correct,scope='same recorded physical source and unchanged S2c thresholds; no policy execution, not formal n40 results')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return int(not passed)


if __name__=='__main__':raise SystemExit(main())
