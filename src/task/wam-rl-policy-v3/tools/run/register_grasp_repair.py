"""Freeze v10 fixes and matched-budget verification before GPU execution."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering',action='store_true');p.add_argument('--engineering-trigger',action='append');a=p.parse_args()
    if a.engineering_trigger and not a.engineering:raise ValueError('engineering trigger selection cannot change formal registration')
    if a.output.exists():raise ValueError('fresh registration required')
    reg=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json').read_text())
    for j in reg['jobs']:
        j['order']=['A1','A2'] if int(j['id'][1:])//2%2==0 else ['A2','A1']
    if a.engineering:
        # One trigger per object plus two seen seeds, no efficacy claims or resampling.
        selected=a.engineering_trigger or ['p000','p006','p001','p005','p013']
        if len(selected)!=len(set(selected)) or not set(selected)<={j['id'] for j in reg['jobs']}:raise ValueError('known distinct engineering triggers required')
        reg['jobs']=[j for j in reg['jobs'] if j['id'] in selected]
        for i,j in enumerate(reg['jobs']):j['worker']=i%3;j['order']=['A1']
        reg['n_per_arm_per_subset']=None;reg['engineering_only']=True
    reg.update(protocol='grasp.sharedR.zero-tau.reachable-PI.stagefinal20s15mm.v1',experiment_id='P-20261022-grasp-interface-repair',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),arms=['A1'] if a.engineering else ['A1','A2'],
        maximum_controls=sum(len(j['order']) for j in reg['jobs'])*600,
        repair_targets=dict(seen_strict_success='>=32/40 A1',unseen_joint_aborts='<40/40 A1; target <=4/40',ablation='same R, observation non-tau fields, phase machine and PD; ONLY 24-frame displacement content zeroed; valid mask preserved at common states'),
        scope='v9 fixed panel reused for before/after repair; eight formerly unseen objects are now diagnostic/dev data, NOT a fresh untouched generalization test',
        development_budget='at most 24 engineering episodes, <=14400 controls, no efficacy claims; final frozen verification n40 per arm per subset, no resampling',
        secondary='A1 vs v9 A1 descriptive matched comparison; no A4 upper-bound/retraining claim in this repair batch',
        main_rule='strict >=32/40 A1 on seen; same-R tau contribution only if Fisher p<.05 AND Wilson95 lower(A1)>upper(A2), separately by subset')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(reg,indent=2)+'\n');print('REGISTERED',len(reg['jobs']),'initials',reg['maximum_controls'],'max controls',reg['arms'])


if __name__=='__main__':main()
