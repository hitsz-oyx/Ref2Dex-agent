"""Versioned append-only recipes for restored C1 and C5 validity control."""
import sys,json,hashlib,subprocess
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write,sha
D=ROOT/'tmp/rh';OUT=D/'diagnosis';COUNTS={0:96,1:5632,2:96,4:30,5:18,6:66,8:12,9:4,10:46}
def main():
 for arm in ('C1','C2','C3','C4-source1','C4-source10','C4-source0','C5'):
  folder=OUT/arm;folder.mkdir(exist_ok=True);path=folder/'recipe-v2.json'
  if path.exists():continue
  match=read(OUT/'C1-support-match.json')
  if arm=='C1' and match['match_fraction']<.95:
   write(folder/'progress.json',dict(status='NOT_REPRODUCIBLE',step=0,match_fraction=match['match_fraction'],matched=match['matched'],automatic_restart=False));continue
  if arm=='C5' and not (OUT/'C5-validity-audit.json').exists():continue
  config=folder/'config.json'
  if not config.exists():write(config,read(OUT/'C2/config.json'),immutable=True)
  r=dict(read(OUT/'C2/recipe.json'));r.update(arm=arm,stage=arm,recipe_path=str(path.resolve()),config=str(config.resolve()),config_sha256=sha(config),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),world_size=1 if arm.startswith('C4') else 4,global_batch=96 if arm.startswith('C4') else 384,flow_only=arm=='C3',source_probabilities={str(s):n/6000 for s,n in COUNTS.items()} if arm in ('C1','C2') else {str(s):1/9 for s in COUNTS})
  identity=dict(manifest_sha256=r['manifest_sha256'])
  if arm.startswith('C4'):r['source_probabilities']={arm.split('source')[1]:1.}
  if arm=='C1':r.update(curated_manifest=str((OUT/'curated-rebound.json').resolve()),curated_sha256=sha(OUT/'curated-rebound.json'));identity['curated_sha256']=r['curated_sha256']
  if arm=='C5':
   gate=OUT/'c5-valid-windows.npy';audit=read(OUT/'C5-validity-audit.json');r.update(validity_gate=str(gate.resolve()),validity_gate_sha256=sha(gate),validity_definition=audit['gate'],validity_counts=audit['counts'],small_sources_under200=[s for s,v in audit['counts'].items() if v['valid24']<200]);identity['validity_gate_sha256']=sha(gate);identity['validity_definition']=audit['gate']
  r['data_identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  write(path,r,immutable=True);write(folder/'progress.json',dict(status='PREREGISTERED',step=0,automatic_restart=False,data_identity_sha256=r['data_identity_sha256']))
 print('version2 recipes prepared')
if __name__=='__main__':main()
