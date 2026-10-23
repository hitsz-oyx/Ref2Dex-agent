"""Summarize only completed preregistered arms; preserve progressive README history."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write
D=ROOT/'tmp/rh';OUT=D/'diagnosis';ARMS=['C1','C2','C3','C4-source1','C4-source10','C4-source0','C5']
def main():
 assert all((OUT/a/'result.json').exists() for a in ARMS),'all seven arms must finish before final synthesis'
 result={};lines=['\n## Final seven-arm comparison (measured, fixed TRAIN panel)\n','All means/deltas are mm. Positive delta means worse. Nominal recovery requires all three deltas<=0; paired4000 mother-cluster CI95 is reported separately, no equivalence margin invented.','\n|arm|updates|GPUs/global batch|elapsed s|first8 hand (delta)|contact hand (delta)|first8 object (delta)|','|---|---:|---|---:|---:|---:|---:|']
 for arm in ARMS:
  r=read(OUT/arm/'result.json');recipe=read(OUT/arm/'recipe-v2.json');g=read(OUT/arm/'grouped-panel.json');s=g['summary'];delta=s['versus_local3000']['delta_mm'];ci=s['versus_local3000']['paired_mother_cluster_ci95_mm'];stable=all(v<=0 for v in delta)
  result[arm]=dict(updates=r['updates'],elapsed_s=r['elapsed_s'],means_mm=s['means_mm'],delta_mm=delta,paired_ci95_mm=ci,all_three_stable_or_better=stable,any_significantly_worse=any(x[0]>0 for x in ci),world_size=recipe['world_size'],global_batch=recipe['global_batch'],data_identity_sha256=recipe['data_identity_sha256'])
  fmt=lambda i:'%.4f (%+.4f)'%(s['means_mm'][i],delta[i]);lines.append('|%s|%d|%d/%d|%.1f|%s|%s|%s|'%(arm,r['updates'],recipe['world_size'],recipe['global_batch'],r['elapsed_s'],fmt(0),fmt(1),fmt(2)))
 lines+=['\n|arm|first8 hand paired CI95|contact paired CI95|object paired CI95|','|---|---|---|---|']
 for arm,v in result.items():lines.append('|'+arm+'|'+'|'.join('[%+.4f,%+.4f]'%tuple(bounds) for bounds in v['paired_ci95_mm'])+'|')
 other={}
 with np.load(D/'half-training/panel-step000000/paired-panel.npz') as z:reference=z['sums'];counts=z['counts']
 rows=read(D/'fixed-panel/manifest.json')['rows']
 for arm in ['C4-source1','C4-source10','C4-source0']:
  trained=int(arm.split('source')[1]);mask=np.array([r['source_id']!=trained for r in rows]);step=read(OUT/arm/'result.json')['last_step']
  with np.load(OUT/arm/('panel-step%06d/paired-panel.npz'%step)) as z:sums=z['sums']
  baseline=reference[mask].sum(0)/counts[mask].sum(0);mean=sums[mask].sum(0)/counts[mask].sum(0);other[arm]=dict(baseline_mm=baseline.tolist(),means_mm=mean.tolist(),delta_mm=(mean-baseline).tolist(),relative_change_percent=((mean/baseline-1)*100).tolist())
 write(OUT/'C4-all-other-sources.json',other,immutable=True)
 lines+=['\nC4 pooled off-diagonal (all other panel sources), first8 hand/contact/object signed deltas mm:']
 for arm,v in other.items():lines.append('- '+arm+': '+', '.join('%+.4f'%x for x in v['delta_mm'])+'; relative '+', '.join('%+.2f%%'%x for x in v['relative_change_percent']))
 decisions=[]
 for arm in ('C1','C2','C3','C5'):
  stable=result[arm]['all_three_stable_or_better'];decisions.append(arm+': '+('all three stable/improved' if stable else 'does not recover all three; retain signed deltas and CIs'))
 if result['C2']['all_three_stable_or_better'] and not result['C5']['all_three_stable_or_better']:decisions.append('Owner joint rule C2 recovery+C5 poor: source sampling weights primary; use target-deployment source weights in recipe identity.')
 if not result['C2']['all_three_stable_or_better'] and result['C5']['all_three_stable_or_better']:decisions.append('Owner joint rule C2 poor+C5 recovery: validity gate implicated; attribution remains limited by lack of a matched unfiltered full-data uniform arm.')
 if result['C3']['all_three_stable_or_better']:decisions.append('Owner C3 rule: freezing shared/scene/backbone protects panel. Exact unchanged non-flow state is audited in C3/frozen-state-audit.json.')
 else:decisions.append('Owner C3 rule: flow-only update can still move predictions adversely; inspect target semantics and scaling, do not attribute everything to backbone parameters.')
 strong=[]
 matrix=read(OUT/'C4-interference-matrix.json')
 for arm,values in matrix.items():
  for source,v in values.items():
   if source==arm.split('source')[1]:continue
   for i,x in enumerate(v['relative_change_percent']):
    if x is not None and x>=50:strong.append(dict(trained_source=arm.split('source')[1],evaluated_source=source,metric_index=i,worsening_percent=x,rows=v['rows'],mothers=v['mothers']))
 decisions.append('Owner C4 rule: '+('strong off-diagonal interference>=50% observed; sparse-source scope and CIs still apply.' if strong else 'no off-diagonal EPE worsening>=50%; this three-source panel probe does not establish universal compatibility.'))
 lines+=['\nPreregistered decisions:']+['- '+x for x in decisions]
 lines+=['\nCausal limits: original half and new full supports differ; C3 versus half and C5 versus half are not exact single-factor comparisons. C5 removes1590 zero-hand-target windows (0.017%), not46-50% of all windows; all still have object supervision. Source4/8 whole-window invalid counts are0. Source-scale heterogeneity remains H2 screen-positive, but no scale-alignment intervention was authorized or run. Fixed TRAIN512/124 evidence only; no held-out generalization, rollout or execution claim. C4 global96 versus C1/C2/C3/C5 global384 prevents interpreting cross-arm differences as exclusively a source effect. No thresholds changed and no extra baseline/repair was run.']
 write(OUT/'final-summary.json',dict(arms=result,decisions=decisions,C4_strong_interference=strong,C4_all_other=other,scope='fixed TRAIN panel only'),immutable=True)
 with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 (OUT/'README.md').chmod(0o664)
 report=ROOT/'src/task/ptv3-shared-wam-v6/docs/audit/nine-source-degradation-20261023.md';report.write_text((OUT/'README.md').read_text())
 card=ROOT/'src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261023-nine-source-degradation.md';card.write_text(card.read_text().replace('status: RUNNING','status: COMPLETED')+'\nAll seven arms completed; exact steps/data identities and decisions in diagnosis/final-summary.json and progressive README. No push.\n')
 subprocess.run(['git','add',str(report),str(card)],cwd=ROOT,check=True)
 steps=','.join(a+'='+str(result[a]['updates']) for a in ARMS)
 subprocess.run(['git','commit','-m','Close C1-C5 diagnosis '+steps+'; full data5a89c571e072, C5 gate2c74830476af'],cwd=ROOT,check=True)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
