"""Create the immutable n40-per-arm-per-subset v9 assignment before execution."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('registration must be new')
    panel=json.loads(a.panel.read_text())
    if panel['status']!='READY' or len(panel['unseen'])!=8:raise ValueError('eight eligible fixed new objects required')
    seen=dict(sequence='s1_cylindersmall_pass_1',object='cylindersmall',reference=str(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz'))
    jobs=[];arms=['A1','A2','A3','A4']
    # Alternate seen/unseen to avoid assigning elapsed-time drift to one subset.
    for repeat in range(40):
        for subset,case in [('seen',seen),('unseen',panel['unseen'][repeat%8])]:
            i=len(jobs);rotation=(repeat if subset=='seen' else repeat//8+repeat%8+1)%4
            jobs.append(dict(id='p%03d'%i,subset=subset,sequence=case['sequence'],object=case['object'],reference=case['reference'],
                             initial_family=case['sequence'],replicate=repeat if subset=='seen' else repeat//8,native_seed=100+i,model_seed=180+i,
                             worker=i%3,order=arms[rotation:]+arms[:rotation]))
    reg=dict(protocol='grasp-intent.stagefinal.hold20s.precise15mm.4arms.n40x2.cadpalm.v1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        experiment_id='P-20261022-grasp-interface-stat40',panel=str(a.panel.resolve()),n_per_arm_per_subset=40,arms=arms,subsets=['seen','unseen'],jobs=jobs,
        primary='at600: S2c current.hold && goal_xyz_error<=15mm && no physical abort',horizon_controls=600,control_hz=30,worker_wall_s=21600,maximum_controls=192000,
        success_does_not_stop=True,release_applicable=False,task='same synthetic initial object pose+8cm position-only HOLD',
        main_rule='A1>A2 iff two-sided Fisher p<.05 AND Wilson95(A1).lower>Wilson95(A2).upper; separately in each subset',
        secondary='A1 vs A3 and all pairwise Fisher reported within subset; A4>=32/40 and A1<=8/40 flags WAM redesign/retraining recommendation, no automatic training',
        units='40 paired seed trials per subset, conditional fixed initial families; seen one family, unseen eight families x five; not 40 objects, not unseen WAM pretraining',
        missing='technical setup failure never labeled physical failure; preserve partial budget and resume same assigned job; all physical safety aborts count failures',
        gpu_assignment={'0':4,'1':5,'2':6},learning_updates=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(reg,indent=2)+'\n');print('REGISTERED: 2 subsets x 4 arms x 40 = 320 episodes; maximum192000 controls; fresh native each episode')


if __name__=='__main__':main()
