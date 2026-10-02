"""Fixed all-arm/motion105qualification; no fitted model or checkpoint selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();m=json.loads((root/'run_manifest.json').read_text());rows=[];hashes={}
    for seed in (587,588):
        d=root/f's{seed}'
        for f in ('panel_audit.json','teacher_replay.json','rows.json'):
            hashes[str(d/f)]=sha(d/f)
        audit=json.loads((d/'panel_audit.json').read_text());replay=json.loads((d/'teacher_replay.json').read_text());assert audit['run_status']==replay['run_status']=='COMPLETED' and replay['rows']==201*192
        new=json.loads((d/'rows.json').read_text());assert len(new)==192 and all(r['seed']==seed for r in new);rows+=new
    def summarize(rs):
        assert rs
        return dict(n=len(rs),successes=sum(r['physical105'] for r in rs),rate=sum(r['physical105'] for r in rs)/len(rs))
    arms={str(g):summarize([r for r in rows if r['arm']==g]) for g in range(4)}
    by_motion={str(g):{str(mo):summarize([r for r in rows if r['arm']==g and r['motion']==mo]) for mo in range(3)} for g in range(4)}
    by_seed={str(seed):{str(g):{str(mo):summarize([r for r in rows if r['seed']==seed and r['arm']==g and r['motion']==mo]) for mo in range(3)} for g in range(4)} for seed in (587,588)}
    assert len(rows)==384 and all(a['n']==96 for a in arms.values()) and all(q['n']==32 for x in by_motion.values() for q in x.values())
    gates=dict(each_expert_motion25=all(by_motion[str(g)][str(mo)]['rate']>=.25 for g in (2,3) for mo in range(3)),each_expert_seed_motion12_5=all(by_seed[str(s)][str(g)][str(mo)]['rate']>=.125 for s in (587,588) for g in (2,3) for mo in range(3)),each_expert_gain5pp_both_p0=all(arms[str(g)]['rate']>=arms[str(base)]['rate']+.05 for g in (2,3) for base in (0,1)))
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',arms=arms,arm_labels=['P0','duplicate_P0','source_e260','duplicate_source_e260'],by_motion=by_motion,by_seed=by_seed,gates=gates,source_sha256=hashes,not_cm_utility=True,not_formal_validation=True,no_model_or_actor_update=True,no_historical_success_transferred=True,first_tick_common_p0=True)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
