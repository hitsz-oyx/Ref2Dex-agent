"""Retain FIRST completed unchanged A3 per trial; every physical abort is eligible.

Input run order is chronological and fixed before examining outcomes. No A4
variant is mislabeled as full source GT. All old physical work stays in budget.
"""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]

def build_index(source_runs,jobs):
    selected={};runs=[];total=upper=0
    for run in source_runs:
        run=run.resolve();ack=extra=0;completed=[];partial=[]
        for manifest in sorted((run/'comparators').glob('g*/manifest.json')):
            m=json.loads(manifest.read_text());worker=int(m['worker'])
            for folder in sorted(manifest.parent.glob('p*-A*')):
                if not folder.is_dir():continue
                if (folder/'result.json').exists():
                    row=json.loads((folder/'result.json').read_text());ack+=row['controls'];completed.append(row)
                    if row['arm']=='A3' and row['trial'] not in selected:
                        j=jobs[row['trial']]
                        if row['seed']!=j['native_seed'] or row['model_seed']!=j['model_seed'] or row['sequence']!=j['sequence'] or worker!=j['worker']:raise ValueError('fixed panel carry mismatch')
                        selected[row['trial']]=dict(row=row,episode=str(folder),runtime_code=str(manifest.parent/'runtime-code'),source_manifest=str(manifest),worker=worker)
                elif (folder/'trajectory.npz').exists():
                    with np.load(folder/'trajectory.npz') as z:n=len(z['action'])
                    ack+=n;extra+=1;partial.append(dict(episode=str(folder),acknowledged_actions=n,inflight_upper=1))
                elif (folder/'initial.npz').exists():extra+=1;partial.append(dict(episode=str(folder),acknowledged_actions=0,inflight_upper=1))
        total+=ack;upper+=ack+extra;runs.append(dict(run=str(run),acknowledged_controls=ack,physical_controls_upper=ack+extra,completed_rows=completed,partials=partial))
    records=[selected[k] for k in sorted(selected)];carried=sum(e['row']['controls'] for e in records)
    index=dict(rule='chronological source-run order, first completed A3 per trial; no outcome filtering; physical aborts retained',records=records,n=len(records),carried_formal_controls=carried,source_runs=[str(r.resolve()) for r in source_runs])
    ledger=dict(acknowledged_controls=total-carried,physical_controls_upper=upper-carried,old_total_acknowledged=total,old_total_upper=upper,carried_A3_formal_controls=carried,source_runs=runs,scope='Engineering overhead excludes first completed A3 already counted in formal rows. Includes all non-carried duplicates, both old A4 variants, and interrupted partials; one in-flight step upper per partial.')
    return index,ledger

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,action='append',required=True);p.add_argument('--output-run',type=Path,required=True);a=p.parse_args();out=a.output_run.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs={j['id']:j for j in json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/registration.json').read_text())['jobs']}
    index,ledger=build_index(a.source_run,jobs)
    (out/'carry-A3-index.json').write_text(json.dumps(index,indent=2)+'\n');(out/'engineering-overhead.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(dict(carried_A3=index['n'],carried_controls=index['carried_formal_controls'],old_physical_lower=ledger['old_total_acknowledged'],old_physical_upper=ledger['old_total_upper'],extra_engineering_lower=ledger['acknowledged_controls'],extra_engineering_upper=ledger['physical_controls_upper'],remaining_formal_episodes=160-index['n'],whole_turn_upper=95620+ledger['old_total_upper']+(160-index['n'])*600),indent=2))
if __name__=='__main__':main()
