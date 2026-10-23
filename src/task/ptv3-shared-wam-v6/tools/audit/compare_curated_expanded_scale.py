"""Decision audit: replay actual C1/C2 source1 sampling, no model execution."""
import sys,json,time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/ptv3-shared-wam-v6';sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from remote_staged_data import read,write,sha
from diagnostic_components import DiagnosticWindows
OUT=ROOT/'tmp/rh/diagnosis'
def main():
 started=time.monotonic();stats=read(ROOT/'tmp/rh/half-recipe/stats.json');results={}
 for arm in ('C1','C2'):
  r=read(OUT/arm/'recipe-v2.json');ds=DiagnosticWindows(r['manifest'],'train',r['geometry_manifest'],r['manifest_sha256'],geometry_gt_object=False,probabilities=r['source_probabilities'],curated=r.get('curated_manifest'));positions=[]
  for step in range(1,10):
   positions.extend(i for i in ds.draw(96,170,step,0) if ds.rows[int(ds.index[i,0])]['source_id']==1)
   if len(positions)>=256:break
  positions=positions[:256];raw=[];normalized=[];vel=[];phases=Counter()
  for i in sorted(positions):
   a=ds[i];mask=a['trajectory_valid'][:8]&a['start_hand_valid'][None];v=a['trajectory_target'][:8];raw.append(v[mask]);z=(v-np.asarray(stats['hand']['mean'])[:8,None,None])/np.asarray(stats['hand']['std'])[:8,None,None];normalized.append(z[mask]);vel.append(a['start_hand_history'][a['start_hand_valid']][:,:3]);phases[int(ds.phase[i])]+=1
  def summarize(values):
   x=np.concatenate(values).astype(np.float64);return dict(point_frames=len(x),coordinate_rms=float(np.sqrt(np.mean(x*x))),norm_mean=float(np.linalg.norm(x,axis=-1).mean()),mean_xyz=x.mean(0).tolist(),std_xyz=x.std(0).tolist(),p1_xyz=np.percentile(x,1,axis=0).tolist(),p50_xyz=np.percentile(x,50,axis=0).tolist(),p99_xyz=np.percentile(x,99,axis=0).tolist())
  results[arm]=dict(windows=256,unique_windows=len(set(positions)),mothers=len(set(map(int,ds.index[positions,0]))),phase_counts=dict(phases),tau_first8_m=summarize(raw),tau_inherited_z=summarize(normalized),input_velocity_mps=summarize(vel),recipe_sha256=sha(OUT/arm/'recipe-v2.json'))
  print(arm,json.dumps(results[arm]),flush=True)
 ratio=results['C2']['tau_first8_m']['coordinate_rms']/results['C1']['tau_first8_m']['coordinate_rms'];write(OUT/'C1-C2-source1-scale-audit.json',dict(classification='Decision',question='Does expanded within-source1 supervision differ in scale despite identical source weighting?',sampling='first256 source1 draws from actual rank0 training sampler, seed170, steps1 onward; replayed on production reader; no model forward or optimizer',sources=results,C2_over_C1_tau_rms_ratio=ratio,elapsed_s=time.monotonic()-started),immutable=True)
 print('RMS ratio',ratio,flush=True)
if __name__=='__main__':main()
