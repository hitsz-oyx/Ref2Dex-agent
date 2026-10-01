"""Fixed final-only physical105 gates; no checkpoint selection or subgroup rescue."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory;begin=time.monotonic()
    m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COLLECTION_COMPLETED' or m['evaluation_seeds']!=[568,569] or m['training_seeds']!=list(range(547,567)):raise ValueError('fixed complete experiment')
    rows=[];training=[];audits=[]
    for seed in list(range(547,567))+[568,569]:
        d=root/f's{seed}';r=json.loads((d/'rows.json').read_text());audit=json.loads((d/'panel_audit.json').read_text())
        if len(r)!=768 or audit['run_status']!='COMPLETED':raise ValueError('every full panel audited')
        (rows if seed>567 else training).extend(r);audits.append(audit)
    if len(rows)!=1536 or len(training)!=15360:raise ValueError('full prospective coverage')
    names=['unchanged','cm','state_only','none'];pooled={};by_seed={};by_motion={}
    def count(data):
        out={}
        for arm,name in enumerate(names):
            group=[r for r in data if r['arm']==arm];out[name]=dict(success=sum(r['physical105'] for r in group),n=len(group))
            out[name]['rate']=out[name]['success']/out[name]['n']
        return out
    pooled=count(rows)
    for seed in (568,569):by_seed[str(seed)]=count([r for r in rows if r['seed']==seed])
    for motion in range(3):by_motion[str(motion)]=count([r for r in rows if r['motion']==motion])
    gates={}
    for control in ('state_only','none'):
        gates['pooled_cm_gain_5pp_over_'+control]=pooled['cm']['rate']-pooled[control]['rate']>=.05
        for seed in (568,569):gates[f's{seed}_cm_noninferior_'+control]=by_seed[str(seed)]['cm']['rate']>=by_seed[str(seed)][control]['rate']
    gates['pooled_cm_noninferior_unchanged']=pooled['cm']['rate']>=pooled['unchanged']['rate']
    gradient=[]
    for update in range(1,21):
        d=root/f'u{update:02d}';g=json.loads((d/'gradient_audit.json').read_text());r=json.loads((d/'results.json').read_text())
        if g['run_status']!='COMPLETED' or r['update']!=update:raise ValueError('all actual training updates audited')
        gradient.append(g)
    maximum={k:max(q['independent_numeric_maximum'][k] for q in audits) for k in audits[0]['independent_numeric_maximum']}
    gmaximum={k:max(q['maximum_error'][k] for q in gradient) for k in gradient[0]['maximum_error']}
    report=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',pooled=pooled,by_seed=by_seed,by_motion=by_motion,gates=gates,training_trajectories=15360,evaluation_trajectories=1536,training_episodes_per_variant=3840,training_seed=762,native_panels=22,actual_ppo_minibatches_per_variant=3040,total_actual_optimizer_steps=9120,independent_gradient_scope='60 predetermined first minibatches (3 per panel), all actor/critic gradients and first Adam steps; other steps not independently replayed',independent_rollout_numeric_maximum=maximum,independent_gradient_numeric_maximum=gmaximum,final_only=True,no_model_reward=True,no_official_actor_actions_or_initialization=True,synthetic_reference_task=True,one_optimization_seed_probe_only=True,wall_seconds=time.monotonic()-begin)
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
