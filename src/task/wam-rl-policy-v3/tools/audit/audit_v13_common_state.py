"""Check the first common valid forecast state in saved A1/A2 pairs, without inference."""
import argparse,json
from pathlib import Path
import numpy as np

def audit(run):
    rows={}
    for gpu in (4,5,6):
        for row in json.loads((run/f'g{gpu}-resume2/manifest.json').read_text())['runs']:rows[row['id']]=(row,run/f'g{gpu}-resume2'/row['id'])
    records=[]
    for row,folder in rows.values():
        if row['subset']!='seen' or row['arm']!='A1':continue
        with np.load(folder/'trajectory.npz') as a,np.load(rows[row['trial']+'-A2'][1]/'trajectory.npz') as b:
            n=min(len(a['tau_valid']),len(b['tau_valid']));indices=np.flatnonzero(a['tau_valid'][:n]&b['tau_valid'][:n]);tick=int(indices[0]);error=float(np.abs(a['observation'][tick,:421]-b['observation'][tick,:421]).max());delta=a['adapter_raw_action'][tick]-b['adapter_raw_action'][tick]
            if error>2e-6:raise ValueError('not a common physical state')
            records.append(dict(id=row['id'],tick=tick,nontau_error=error,wrist_action_norm=float(np.linalg.norm(delta[:6])),finger_action_norm=float(np.linalg.norm(delta[6:])),wrist_delta=delta[:6].tolist(),finger_delta=delta[6:].tolist()))
    return dict(n=len(records),maximum_common_state_error=max(r['nontau_error'] for r in records),wrist_norm_quantiles=np.quantile([r['wrist_action_norm'] for r in records],[0,.5,1]).tolist(),finger_norm_quantiles=np.quantile([r['finger_action_norm'] for r in records],[0,.5,1]).tolist(),records=records,scope='First common valid forecast step; normalized action units differ from physical pose/finger units. Descriptive sensitivity, not sufficiency/necessity or success mechanism.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
