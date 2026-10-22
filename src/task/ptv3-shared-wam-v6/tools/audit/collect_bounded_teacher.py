"""Run the existing official collector with a registered hard control cap.

Only two source edits are allowed: cap captured controls, and replace a Git
query with explicit working-tree identity. Frozen instrumented source is kept.
No model learning or changed teacher action logic.
"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
SOURCE=ROOT/'src/task/ptv3-shared-wam/tools/run/collect_rl660_batch.py'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);a=p.parse_args()
 if a.gpu!=4:raise ValueError('GPU4 only')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 names=json.loads(a.panel.read_text())['sequences'][:6]
 selection=out/'selection.json';selection.write_text(json.dumps(names,indent=2)+'\n')
 raw=SOURCE.read_text();patches={
 'lengths = task.max_episode_length.clone()':'source_lengths = task.max_episode_length.clone()\n            lengths = source_lengths.clamp(max=601)\n            manifest[\'source_frames_by_env\'] = source_lengths.tolist()\n            manifest[\'capture_frames_by_env\'] = lengths.tolist()\n            manifest[\'registered_maximum_controls\'] = 600',
 "git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()":"git_commit='working-tree-no-git-owner-bounded-audit'"}
 modified=raw
 for old,new in patches.items():
  if modified.count(old)!=1:raise ValueError('collector anchor changed: '+old)
  modified=modified.replace(old,new)
 (out/'collector.original.py').write_text(raw);(out/'collector.bounded.py').write_text(modified)
 import difflib
 (out/'collector.diff').write_text(''.join(difflib.unified_diff(raw.splitlines(True),modified.splitlines(True),fromfile=str(SOURCE),tofile='collector.bounded.py')))
 index=[]
 for relative in ['src/task/ptv3-shared-wam/tools/run/collect_rl660_batch.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/inspire_reset_submission.py','src/task/consequence-evaluator/src/consequence_evaluator/native_reset.py','src/task/trajectory-policy/src/trajectory_policy/pd_retargeter.py']:
  source=ROOT/relative;target=out/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());index.append(dict(path=relative,size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns))
 (out/'source-index.json').write_text(json.dumps(index,indent=2)+'\n')
 sys.argv=[str(SOURCE),'--selection',str(selection),'--output',str(a.native_output.resolve()),'--gpu','4','--seed','195','--seconds','540','--capture-current-observation','--capture-dof-force','--terminal-reference-clamp']
 (out/'invocation.json').write_text(json.dumps(dict(executable=sys.executable,argv=sys.argv,policy_updates=0,cap_envcontrols=3600,source=str(SOURCE)),indent=2)+'\n')
 exec(compile(modified,str(out/'collector.bounded.py'),'exec'),dict(__name__='__main__',__file__=str(SOURCE)))
if __name__=='__main__':main()
