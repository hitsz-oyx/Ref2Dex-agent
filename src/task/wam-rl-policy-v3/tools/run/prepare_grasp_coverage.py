"""Freeze more object/sequence initial states by lexical geometry-only selection."""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'tools/run'))
from prepare_large_grasp_panel import load_raw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/registration.json').read_text())
    old_sequences={j['sequence'] for j in old['jobs']};old_objects={j['object'] for j in old['jobs']}
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    geometry=json.loads((ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry/surface-manifest.json').read_text());geometry_text=json.dumps(geometry)
    selected=[];objects=set()
    # First eight new objects, then four different sequences of existing objects.
    for subset,number in [('new_object',8),('new_sequence',4)]:
        n=0
        for record in sorted(catalog,key=lambda r:r['sequence']):
            if record['engineering']!='PASS' or record['sequence'] in old_sequences or 'dexplore/'+record['object'] not in geometry_text:continue
            if subset=='new_object' and (record['object'] in old_objects or record['object'] in objects):continue
            if subset=='new_sequence' and (record['object'] not in old_objects or record['object'] in objects):continue
            ref,points,table,bounds=load_raw(record);folder=out/record['sequence'];folder.mkdir();np.savez_compressed(folder/'reference.npz',**ref)
            selected.append(dict(sequence=record['sequence'],object=record['object'],subset=subset,reference=str(folder/'reference.npz')));objects.add(record['object']);n+=1
            if n==number:break
    if len(selected)!=12:raise ValueError('insufficient fixed coverage candidates')
    jobs=[]
    for i,case in enumerate(selected):
        arms=['A1','A2','A0'];rotation=i%3
        jobs.append(dict(case,id='p%03d'%i,native_seed=260+i,model_seed=280+i,worker=i%3,order=arms[rotation:]+arms[:rotation],replicate=0,initial_family=case['sequence']))
    reg=dict(protocol='grasp.geometry-relativeSE3.coverage12.3arms.v1',experiment_id='P-20261022-grasp-geometry-coverage',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        jobs=jobs,arms=['A1','A2','A0'],subsets=['new_object','new_sequence'],horizon_controls=600,control_hz=30,worker_wall_s=7200,maximum_controls=21600,learning_updates=0,release_applicable=False,
        selection='lexical engineering-PASS cached-geometry source records, 8 new object identities plus 4 new sequences of prior-panel objects; NO policy outcomes or constants used in selection',
        primary='full600, measured geometry S2c hold, object position error<=0.10*minimum convex-hull width, no physical abort',
        positive_control='A0 source-PD approach until measured grasp then SAME pose servo. Admission rejection and fallback counted; a failed positive control does not prove task impossible.',
        scope='one seed per initial family; coverage diagnostic, no significance/generalization/retraining claim',
        decision='A0 succeeds but A1/A2 fail => intent/approach bottleneck; A1/A2 both succeed => shared control helps, no tau attribution; all fail => source/servo/task qualification unresolved, cannot blame WAM')
    (out/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');print('REGISTERED',len(jobs),'initials',len({j['object'] for j in jobs}),'objects',reg['maximum_controls'],'controls')

if __name__=='__main__':main()
