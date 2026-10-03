"""Build only corrected train causal geometry; retain the exact earlier held bank."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_calibration import causal_features
from scripts.run_contact_response_probe import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and not a.output.exists() and ROOT in a.output.resolve().parents
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    old=a.execution_source/'qualified'
    with np.load(old/'train_rows.npz') as f:rows={k:f[k] for k in f.files}
    with np.load(old/'geometry.npz') as f:geometry={k:f[k] for k in f.files}
    fitted=json.loads((old/'execution_fit.json').read_text());fitted['scale']=np.asarray(fitted['scale']);fitted['coefficients']={k:np.asarray(v) for k,v in fitted['coefficients'].items()}
    initial=torch.load(a.native_source/'s655/initial.pt',map_location='cpu',weights_only=False)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    result=causal_features(rows,geometry,fitted,initial['native_lower'].numpy(),initial['native_upper'].numpy(),urdf,'cuda:0',
                           lambda done,total:print(json.dumps(dict(train_windows=done,total=total)),flush=True))
    assert result['features'].shape==(6144,64,22) and result['target'].shape==(6144,64,3)
    a.output.mkdir();np.savez(a.output/'train_features.npz',**result)
    record=dict(run_status='COMPLETED',train_windows=6144,train_episodes=384,
                held_bank=str(old/'features.npz'),held_bank_sha256=sha(old/'features.npz'),
                train_bank_sha256=sha(a.output/'train_features.npz'),actuator_refit=False)
    (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
