"""Read-only recomputation of completed fixed-panel means and paired cluster CIs."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];OUT=ROOT/'tmp/rh/diagnosis'
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',nargs='+',default=['C1','C2','C4-source1','C4-source10','C4-source0']);a=p.parse_args()
 with np.load(ROOT/'tmp/rh/half-training/panel-step000000/paired-panel.npz') as z:reference=z['sums'];counts=z['counts'];mothers=z['mothers']
 for arm in a.arm:
  panels=sorted((OUT/arm).glob('panel-step*/paired-panel.npz'))
  if not panels or panels[-1].parent.name=='panel-step000000':print(json.dumps(dict(arm=arm,status='NO_FINAL_PANEL')));continue
  with np.load(panels[-1]) as z:
   sums=z['sums'];assert np.array_equal(z['counts'],counts) and np.array_equal(z['mothers'],mothers)
  means=sums.sum(0)/counts.sum(0);delta=means-reference.sum(0)/counts.sum(0);ids=np.unique(mothers);s=np.array([sums[mothers==m].sum(0) for m in ids]);c=np.array([counts[mothers==m].sum(0) for m in ids]);r=np.array([reference[mothers==m].sum(0) for m in ids]);rng=np.random.default_rng(170);values=[]
  for _ in range(4000):
   chosen=rng.integers(0,len(ids),len(ids));den=c[chosen].sum(0);v=s[chosen].sum(0)/den;v-=r[chosen].sum(0)/den;values.append(v)
  ci=np.percentile(values,[2.5,97.5],axis=0).T;stored=json.load(open(panels[-1].parent/'summary.json'));assert np.allclose(means,stored['means_mm'],atol=1e-12,rtol=0);assert np.allclose(ci,stored['versus_local3000']['paired_mother_cluster_ci95_mm'],atol=1e-12,rtol=0)
  print(json.dumps(dict(arm=arm,means_mm=means.tolist(),delta_mm=delta.tolist(),paired_ci95_mm=ci.tolist(),stored_summary_verified=True)))
if __name__=='__main__':main()
