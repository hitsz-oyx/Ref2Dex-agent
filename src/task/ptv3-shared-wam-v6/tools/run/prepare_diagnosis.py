"""Prepare immutable diagnosis recipes; C1 requires original curated metadata."""
import argparse,json,subprocess,sys
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write,sha
D=Path('/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending');OUT=D/'diagnosis'
COUNTS={0:96,1:5632,2:96,4:30,5:18,6:66,8:12,9:4,10:46}

def prepare(curated=None):
 manifest=OUT/'full-census/manifest.json';assert manifest.exists(); assert read(OUT/'source-supervision-stats.json')['manifest_sha256']==sha(manifest)
 for arm in ('C1','C2','C3','C4-source1','C4-source10','C4-source0'):
  folder=OUT/arm;folder.mkdir(exist_ok=True); path=folder/'recipe.json'
  if path.exists():continue
  if arm=='C1' and not curated:
   write(folder/'progress.json',dict(status='BLOCKED_MISSING_CURATED_SUPPORT',step=0,automatic_restart=False,reason='original curated 6000-window metadata absent on remote; weighted full data equals C2'))
   continue
  config=read(D/'half-recipe/config.json');config.update(updates=500,max_seconds=2400,execution_campaign='v6.pre-registered-source-diagnosis-20261023',save_initial_weights=False)
  write(folder/'config.json',config,immutable=True)
  r=dict(schema='ref2dex.diagnostic-training.v1',arm=arm,stage=arm,recipe_path=str(path),manifest=str(manifest),stats=str(D/'half-recipe/stats.json'),config=str(folder/'config.json'),panel_manifest=str(D/'fixed-panel/manifest.json'),geometry_manifest=str(D.parent/'ptv3-shared-wam-v5-20261019-r1/data/manifest.json'),initialization=str(D/'local3000-weights.pt'),initialization_sha256='3f903dea22402c3f8ac7e13131ae322793bcb8f03331b82373f676a126ff092d',local3000_panel=str(D/'half-training/panel-step000000/paired-panel.npz'),world_size=1 if arm.startswith('C4') else 4,local_batch=96,global_batch=96 if arm.startswith('C4') else 384,last_step=500,max_seconds=2400,flow_only=arm=='C3',automatic_restart=False,classification='Decision',sampling='declared source weights -> uniform mother -> uniform observed known phase -> uniform window, max8/mother/rank/batch',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_probabilities={str(s):n/6000 for s,n in COUNTS.items()} if arm in ('C1','C2') else {str(s):1/9 for s in COUNTS})
  if arm.startswith('C4'):r['source_probabilities']={arm.split('source')[1]:1.}
  if arm=='C1':r.update(curated_manifest=str(Path(curated).resolve()),curated_sha256=sha(curated))
  for field in ('manifest','stats','config','panel_manifest'):r[field+'_sha256']=sha(r[field])
  write(path,r,immutable=True);write(folder/'progress.json',dict(status='PREREGISTERED',step=0,automatic_restart=False))
 print('recipes prepared; C1 requires exact original support metadata')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--curated');a=p.parse_args();prepare(a.curated)
