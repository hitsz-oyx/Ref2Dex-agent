"""Read-only complete-coverage audit, using integer final gates from raw rows."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
from scripts.resume_continuous_critic_policy import bytes_in,SCIENTIFIC_FILES

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.run.resolve();out=a.output.resolve();m=read(root/'run_manifest.json');report=read(root/'results.json')
    if out.exists() or ROOT not in out.parents:raise ValueError('new own closeout only')
    if m['run_status']!='COMPLETED' or report['run_status']!='COMPLETED':raise ValueError('full authoritative terminal result required')
    assert m['training_seeds']==list(range(547,567)) and m['evaluation_seeds']==[568,569]
    assert m['scientific_git_commit'].startswith('e37eca3')
    for f,h in m['input_sha256'].items():assert sha(Path(f))==h,f
    physical_source=Path(m['original_source_run']);original_root=Path(read(physical_source/'run_manifest.json')['isolated_worktree'])
    assert all(sha(ROOT/f)==sha(original_root/f) for f in SCIENTIFIC_FILES)
    allrows=[];counts=Counter();wins=Counter();protected={}
    for seed in list(range(547,567))+[568,569]:
        d=root/f's{seed}';rows=read(d/'rows.json');audit=read(d/'panel_audit.json');native=read(d/'results.json')
        assert audit['run_status']=='COMPLETED' and audit['rows']==768 and audit['all_cohorts_included']
        assert native['run_status']=='COMPLETED' and native['physical_steps_each']==202
        assert native['deterministic']==(seed in [568,569])
        assert native['continuous_updates']==(seed-547 if seed<567 else 20)
        assert native['continuous_checkpoint_sha256']==m['panel_checkpoints'][str(seed)]['sha256']
        assert len(rows)==768 and {q['environment'] for q in rows}==set(range(768))
        balanced=Counter((q['motion'],q['arm']) for q in rows)
        assert balanced==Counter({(mo,arm):64 for mo in range(3) for arm in range(4)})
        for row in rows:
            assert row['seed']==seed and row['physical105'] in [True,False]
            key=(seed,row['motion'],row['arm']);counts[key]+=1;wins[key]+=int(row['physical105'])
        allrows+=rows
        for name in ['rows.json','panel_audit.json','results.json']:protected[str((d/name).resolve())]=sha(d/name)
    assert len(allrows)==16896
    arms=['unchanged','cm','state_only','none'];evalrows=[q for q in allrows if q['seed']>567]
    def summarize(selected):
        n=Counter(q['arm'] for q in selected);s=Counter(q['arm'] for q in selected if q['physical105'])
        return {name:{'success':s[arm],'n':n[arm],'rate':s[arm]/n[arm]} for arm,name in enumerate(arms)}
    pooled=summarize(evalrows);by_seed={str(seed):summarize([q for q in evalrows if q['seed']==seed]) for seed in [568,569]}
    by_motion={str(mo):summarize([q for q in evalrows if q['motion']==mo]) for mo in range(3)}
    assert all(v['n']==384 for v in pooled.values())
    gates={}
    for ctrl in ['state_only','none']:
        # Five percentage points over384trials requires at least20extra successes.
        gates['pooled_cm_gain_5pp_over_'+ctrl]=pooled['cm']['success']-pooled[ctrl]['success']>=20
        for seed in ['568','569']:
            assert all(v['n']==192 for v in by_seed[seed].values())
            gates['s'+seed+'_cm_noninferior_'+ctrl]=by_seed[seed]['cm']['success']>=by_seed[seed][ctrl]['success']
    gates['pooled_cm_noninferior_unchanged']=pooled['cm']['success']>=pooled['unchanged']['success']
    assert [pooled,by_seed,by_motion,gates]==[report['pooled'],report['by_seed'],report['by_motion'],report['gates']]
    label='PROMISING' if all(gates.values()) else 'UNPROMISING';assert report['label']==m['label']==label
    steps=0
    for update in range(1,21):
        d=root/f'u{update:02d}';result=read(d/'results.json');audit=read(d/'gradient_audit.json')
        assert audit['run_status']=='COMPLETED' and result['update']==audit['update']==update
        assert audit['all_actor_and_critic_first_step_gradients'] and audit['first_step_adam'] and audit['actual_physical_returns']
        assert result['checkpoint_sha256']==sha(d/'policy_heads.pt')
        for variant in ['cm','state_only','none']:
            v=result['variants'][variant];assert (v['episodes'],v['transitions'],v['epochs'],v['minibatches'],v['actor_changed'])==(192,38784,4,152,True)
            steps+=v['minibatches']
        for name in ['results.json','gradient_audit.json']:protected[str((d/name).resolve())]=sha(d/name)
    assert steps==9120 and report['total_actual_optimizer_steps']==steps
    assert sha(root/'u20/policy_heads.pt')==m['final_checkpoint_sha256']
    correction=m.get('audit_correction')
    if correction:
        corrected=root/'u17/gradient_audit.json';c=read(corrected)
        assert sha(corrected)==correction['sha256']
        assert c['original_audit_retained_failed'] and c['gpu_saved_gradients_exactly_reproduced']
        assert c['no_optimizer_or_physics_updates'] and len(c['branch_certificates'])==1
        assert c['scalar_tolerances_unchanged']==dict(input=0.,advantage=2e-5,forward=2e-5,loss=2e-5,gradient=2e-5,adam=2e-5,norm_relative=2e-6)
        assert all(c['maximum_error'][k]<=v for k,v in c['scalar_tolerances_unchanged'].items())
        assert not (Path(correction['original_failed_parent_retained'])/'u17/gradient_audit.json').exists()
    total=bytes_in(root)+m['previous_bytes'];assert total<=6<<30 and m['accounted_total_wall_seconds']<=3600
    for name in ['run_manifest.json','results.json']:protected[str(root/name)]=sha(root/name)
    result=dict(run_status='COMPLETED',label=label,pooled=pooled,by_seed=by_seed,by_motion=by_motion,gates=gates,
        native_panels=22,training_trajectories=15360,evaluation_trajectories=1536,actual_optimizer_steps=steps,
        first_minibatches_independently_audited=60,remaining_optimizer_steps_not_independently_replayed=9060,
        source_protected_paths_verified=len(m['input_sha256']),full_unique_cohort_coverage=True,
        fixed_final_only_integer_gates_independently_reconstructed=True,scientific_sources_unchanged=True,
        hardware_migration_recorded=m['runtime_migration'],all_prior_failure_records_retained=True,
        witnessed_relu_branch_audit_correction=correction,
        combined_bytes=total,accounted_execution_seconds=m['accounted_total_wall_seconds'],
        source_sha256=protected,closeout_script_sha256=sha(Path(__file__)),
        one_optimization_seed_probe_only=True,journal_ready=False)
    out.mkdir();(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
