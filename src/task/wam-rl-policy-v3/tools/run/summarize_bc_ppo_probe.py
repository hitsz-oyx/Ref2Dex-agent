"""Summarize only paired deterministic BC/PPO assessment; training stays separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_source_ceiling import death


def load(folder):
    m=json.loads((folder/'manifest.json').read_text())
    if m['status']!='COMPLETED':raise ValueError('incomplete run: '+str(folder))
    return m


def metrics(folder,m):
    rows=m['runs'];nominal=0;joint_events=0
    for r in rows:
        d=folder/r['id'];bounds=json.loads((d/'native-audit.json').read_text())['physical_joint_limits']
        with np.load(d/'trajectory.npz') as z:
            q=z['q'][1:,0];mask=np.asarray(bounds['has_limits'],bool)
            nominal+=int((mask&((q<np.asarray(bounds['lower'])-1e-6)|(q>np.asarray(bounds['upper'])+1e-6))).any(-1).sum())
        joint_events+=len(r['physical_safety_final']['joint_violation_indices'])
    return dict(episodes=len(rows),successes=sum(r['final_success'] for r in rows),success_rate=float(np.mean([r['final_success'] for r in rows])),joint_abort_episodes=sum(r['physical_abort']=='physical_joint_limit' for r in rows),joint_events=joint_events,nominal_joint_excursion_states=nominal,failures=dict(Counter(death(r) for r in rows)),controls=m['environment_controls'],wall_s=m['elapsed_s'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bc',type=Path,required=True);p.add_argument('--ppo',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bc=load(a.bc);ppo=load(a.ppo);training=load(a.training)
    if bc['ppo'] or ppo['ppo']:raise ValueError('assessment must use deterministic means without updates')
    if bc['runtime_source']!=ppo['runtime_source']:raise ValueError('mixed frozen runtimes')
    b={r['sequence']:r for r in bc['runs']};c={r['sequence']:r for r in ppo['runs']}
    if set(b)!=set(c) or len(b)!=len(bc['runs']) or len(c)!=len(ppo['runs']):raise ValueError('assessment must have one matched episode per initial')
    pairs=[]
    for name in b:
        x,y=b[name],c[name]
        if x['seed']!=y['seed'] or x['partial_budget_episode'] or y['partial_budget_episode']:raise ValueError('seed/complete horizon mismatch')
        with np.load(a.bc/x['id']/'initial.npz') as z,np.load(a.ppo/y['id']/'initial.npz') as w:
            if set(z.files)!=set(w.files):raise ValueError('initial fields differ')
            maximum=max(float(np.abs(z[k].astype(float)-w[k].astype(float)).max()) for k in z.files)
        if maximum>2e-6:raise ValueError('initial mismatch '+name)
        bs,cs=int(x['final_success']),int(y['final_success']);bj,cj=int(x['physical_abort']=='physical_joint_limit'),int(y['physical_abort']=='physical_joint_limit')
        improved=cs>bs or cj<bj;regressed=cs<bs or cj>bj
        pairs.append(dict(sequence=name,initial_max_abs_error=maximum,bc_success=bs,ppo_success=cs,bc_joint_abort=bj,ppo_joint_abort=cj,improved=improved,regressed=regressed,bc_failure=death(x),ppo_failure=death(y)))
    updates=[dict(episode=r['id'],sequence=r['sequence'],cumulative_controls=sum(t['controls'] for t in training['runs'][:i+1]),reward_sum=r['reward_sum'],**r['ppo_update']) for i,r in enumerate(training['runs'])]
    exact=[e['exact_kl'] for u in updates for e in u['epochs']];clips=[e['clip_fraction'] for u in updates for e in u['epochs']];entropies=[e['entropy'] for u in updates for e in u['epochs']]
    numerical_ok=all(np.isfinite(exact)) and all(0<=v<=.030001 for v in exact) and all(e['accepted'] for u in updates for e in u['epochs'])
    retries=sum(u['rollback_retries'] for u in updates);attempts=sum(u['optimizer_attempts'] for u in updates)
    # The preregistration does not quantify "frequent". Report the raw rate;
    # never quietly introduce a new efficacy threshold for rollback frequency.
    bm,cm=metrics(a.bc,bc),metrics(a.ppo,ppo)
    gain=cm['success_rate']>bm['success_rate'] or cm['joint_abort_episodes']<bm['joint_abort_episodes']
    positive=bool(numerical_ok and gain and sum(p['improved'] for p in pairs)>=3 and not any(p['regressed'] for p in pairs))
    result=dict(bc=bm,ppo=cm,paired_initials=pairs,improved_initials=sum(p['improved'] for p in pairs),regressed_initials=sum(p['regressed'] for p in pairs),positive=positive,numerical_gate_passed=numerical_ok,training_controls=training['environment_controls'],budget_reduced=training['budget_controls']<100000,kl_min_max=[min(exact),max(exact)],clip_min_max=[min(clips),max(clips)],entropy_min_max=[min(entropies),max(entropies)],rollback_retries=retries,optimizer_attempts=attempts,optimizer_updates=training['learning_updates'],update_sequence=updates,caveat='fixed exposed six-initial panel; budget-reduced no-gain cannot test the original full1e5 no-gain branch; frequent rollback has no preregistered numeric threshold')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('update_sequence','paired_initials')},indent=2))


if __name__=='__main__':main()
