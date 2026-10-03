"""Current near-hand query coverage; no prediction or model fitting."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
from scripts.prepare_surface_prior_data import file_stat


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists();start=time.monotonic();out.mkdir()
    provenance_path=a.source/'data/provenance.json';provenance=json.loads(provenance_path.read_text());ids=np.array(provenance['object_point_ids'])
    inputs={str(provenance_path.resolve()):sha(provenance_path),str(Path(__file__).resolve()):sha(Path(__file__)),str(ROOT/'docs/experiments/probes/P-20261003-surface-granularity-audit.md'):sha(ROOT/'docs/experiments/probes/P-20261003-surface-granularity-audit.md')}
    m=dict(experiment_id='P-20261003-surface-granularity-audit',run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=inputs,execution_device='cpu',model_calls=0,wall_limit_seconds=180,storage_limit_bytes=10<<20)
    def save():m['wall_seconds']=time.monotonic()-start;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    save()
    try:
        reports={};evidence={}
        for group,records in provenance['selection'].items():
            full=np.zeros(len(records),np.int64);sample=np.zeros_like(full);unique=np.zeros_like(full)
            for geometry in sorted(set(row['geometry'] for row in records)):
                if time.monotonic()-start>180:raise TimeoutError('coverage audit bound')
                paths=[Path(geometry)/name for name in ['obj_candidate_mask_2cm.npy','obj_knn_indices.npy']]
                for path in paths:assert file_stat(path)==provenance['source_stat'][str(path)]
                mask=np.load(paths[0],mmap_mode='r');neighbors=np.load(paths[1],mmap_mode='r')
                for i,row in enumerate(records):
                    if row['geometry']!=geometry:continue
                    t=row['current'];full[i]=int(mask[t].sum());sample[i]=int(mask[t,ids].sum())
                    unique[i]=len(np.unique(neighbors[t,ids,:4]))
                for path in paths:assert file_stat(path)==provenance['source_stat'][str(path)]
            assert np.all(full>0)
            missed=int(np.sum(sample==0))
            reports[group]=dict(windows=len(records),missing_queries=missed,missing_fraction=missed/len(records),full_near_hand_median=float(np.median(full)),sampled_near_hand_median=float(np.median(sample)),unique_hand_ids_median=float(np.median(unique)),unique_hand_ids_mean=float(np.mean(unique)),local_hand_id_upper_bound=256)
            evidence[group+'_full']=full;evidence[group+'_sampled']=sample;evidence[group+'_unique_hand']=unique
        np.savez(out/'counts.npz',**evidence)
        label='PROMISING' if any(r['missing_fraction']>=.1 for r in reports.values()) else 'UNCLEAR'
        result=dict(run_status='COMPLETED',label=label,reports=reports,near_hand_threshold_m=.02,object_pool=4096,object_queries=64,neighbors_each=4,hand_source_points=dict(mano=2048,inspire=10135),actual_physical_contact_not_measured=True,performance_cause_not_established=True,model_calls=0)
        for path,digest in inputs.items():assert sha(Path(path))==digest
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');m.update(run_status='COMPLETED',label=label,inputs_unchanged=True)
        print(json.dumps(result),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
