"""CPU replay of v8 saved physical states, using each run's frozen observer."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
FIELDS=('q','dq','objects','hand','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose','tau_valid')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();frozen=run/'runtime-code'
    os.environ.update(REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen),PYTHONDONTWRITEBYTECODE='1')
    sys.path[:0]=[str(run),str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    from physical_scoring import physical_rows
    from wam_rl_policy_v3.evaluation_sources import continuous_hold
    m=json.loads((run/'manifest.json').read_text());cat=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];r=next(r for r in cat if r['sequence']==m['source_sequence'])
    with np.load(Path(r['source'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/m['source_sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
    result=[]
    for r in m['runs']:
        with np.load(run/r['label']/'trajectory.npz') as z:data={k:z[k].copy() for k in FIELDS}
        if not np.array_equal(data['tick'][:,0],np.arange(len(data['q']))):raise ValueError('trace frame tick gap')
        rows=physical_rows(data,points,bounds,m['native_initial_audit']['physical_joint_limits']);score=continuous_hold(data['tick'][:,0]/30,[v['held'] for v in rows])
        match=score['ever_success']==r['ever_success'] and score['final_success']==r['final_success'] and abs(score['longest_hold_s']-r['longest_hold_elapsed_s'])<1e-8
        result.append(dict(label=r['label'],match=match,**score))
    report=dict(status='PASS' if all(r['match'] for r in result) else 'FAIL',n=len(result),matched=sum(r['match'] for r in result),records=result,scope='new causal replay from raw physical states; shares frozen contact observer; no stage/reward accumulation')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
