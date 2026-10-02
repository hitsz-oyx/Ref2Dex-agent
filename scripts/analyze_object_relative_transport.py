"""Fixed all-arm/motion105qualification; no fitted model or checkpoint selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();m=json.loads((root/'run_manifest.json').read_text());rows=[];hashes={}
    for seed in (591,592):
        d=root/f's{seed}'
        for f in ('panel_audit.json','rows.json'):
            hashes[str(d/f)]=sha(d/f)
        audit=json.loads((d/'panel_audit.json').read_text());assert audit['run_status']=='COMPLETED' and audit['relative_transport_independently_reconstructed'] and audit['same_state_lift_rotation_finger_targets_preserved']
        new=json.loads((d/'rows.json').read_text());assert len(new)==192 and all(r['seed']==seed for r in new);rows+=new
    def summarize(rs):
        assert rs
        return dict(n=len(rs),successes=sum(r['physical105'] for r in rs),rate=sum(r['physical105'] for r in rs)/len(rs))
    arms={str(g):summarize([r for r in rows if r['arm']==g]) for g in range(4)}
    by_motion={str(g):{str(mo):summarize([r for r in rows if r['arm']==g and r['motion']==mo]) for mo in range(3)} for g in range(4)}
    by_seed={str(seed):{str(g):{str(mo):summarize([r for r in rows if r['seed']==seed and r['arm']==g and r['motion']==mo]) for mo in range(3)} for g in range(4)} for seed in (591,592)}
    assert len(rows)==384 and all(a['n']==96 for a in arms.values()) and all(q['n']==32 for x in by_motion.values() for q in x.values())
    gates=dict(dynamic_gain5pp_all_controls=all(arms['2']['rate']>=arms[str(control)]['rate']+.05 for control in (0,1,3)),each_seed_noninferiority=all(sum(by_seed[str(seed)]['2'][str(m)]['successes'] for m in range(3))>=sum(by_seed[str(seed)][str(control)][str(m)]['successes'] for m in range(3)) for seed in (591,592) for control in (0,1,3)),competent_motion1_noninferiority5pp=all(by_motion['2']['1']['rate']>=by_motion[str(control)]['1']['rate']-.05 for control in (0,1)))
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',arms=arms,arm_labels=['P0','duplicate_P0','dynamic_object_relative_transport','static_initial_offset'],by_motion=by_motion,by_seed=by_seed,gates=gates,source_sha256=hashes,not_cm_utility=True,not_formal_validation=True,no_model_or_actor_update=True,no_historical_success_transferred=True,no_source_actor_actions=True)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
