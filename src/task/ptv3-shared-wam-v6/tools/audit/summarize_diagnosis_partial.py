"""Finalize measured arms without inventing C3/C5 outcomes after input faults."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write
OUT=ROOT/'tmp/rh/diagnosis';DONE=['C1','C2','C4-source10','C4-source0']
def main():
 assert all((OUT/a/'grouped-panel.json').exists() for a in DONE)
 summary={};lines=['\n## Final measured synthesis with terminal blockers\n','\n|arm|status|updates|GPUs/global batch|first8 hand mm (delta)|contact mm (delta)|first8 object mm (delta)|','|---|---|---:|---|---:|---:|---:|']
 for arm in ['C1','C2','C3','C4-source1','C4-source10','C4-source0','C5']:
  if arm in DONE:
   result=read(OUT/arm/'result.json');r=read(OUT/arm/'recipe-v2.json');g=read(OUT/arm/'grouped-panel.json');s=g['summary'];d=s['versus_local3000']['delta_mm'];ci=s['versus_local3000']['paired_mother_cluster_ci95_mm']
   summary[arm]=dict(status='COMPLETED',updates=result['updates'],means_mm=s['means_mm'],delta_mm=d,paired_ci95_mm=ci,all_three_stable_or_improved=all(x<=0 for x in d),global_batch=r['global_batch'],world_size=r['world_size'],data_identity_sha256=r['data_identity_sha256'],report_commit=read(OUT/arm/'report-commit.json')['commit'])
   cell=lambda i:'%.4f (%+.4f)'%(s['means_mm'][i],d[i]);lines.append('|%s|completed|%d|%d/%d|%s|%s|%s|'%(arm,result['updates'],r['world_size'],r['global_batch'],cell(0),cell(1),cell(2)))
  elif arm=='C4-source1':
   summary[arm]=dict(status='FAILED_RENDEZVOUS_ISOLATION',last_logged_step=480,final_checkpoint=False,quality_outcome=None)
   lines.append('|C4-source1|rendezvous store failure|480 logged|1/96|NA|NA|NA|')
  elif arm=='C3':
   summary[arm]=dict(status='FAILED_FIXED_WORKSPACE',initial_run_updates=0,repaired_run='C3-r2',last_logged_step=100,last_attempted_step=104,completed_updates_inferred=103,final_checkpoint=False,quality_outcome=None)
   lines.append('|C3 / C3-r2|reader fault then input-range failure|0;100 logged/failed104|4/384|NA|NA|NA|')
  else:
   summary[arm]=dict(status='BLOCKED_FIXED_WORKSPACE',updates=0,quality_outcome=None,deterministic_fault_step=104,rank=2)
   lines.append('|C5|known deterministic input-range fault|0|4/384|NA|NA|NA|')
 lines+=['\nPaired cluster CI95 (4000 bootstrap draws, seed170):','\n|arm|first8 hand delta CI95|contact delta CI95|object delta CI95|','|---|---|---|---|']
 for arm in DONE:lines.append('|'+arm+'|'+'|'.join('[%+.4f,%+.4f]'%tuple(c) for c in summary[arm]['paired_ci95_mm'])+'|')
 matrix=read(OUT/'C4-interference-matrix.json');strong=[];self_change={}
 for arm,values in matrix.items():
  trained=arm.split('source')[1];self_change[arm]=values[trained]
  for source,v in values.items():
   if source==trained:continue
   for i,x in enumerate(v['relative_change_percent']):
    if x is not None and x>=50:strong.append(dict(trained_source=trained,evaluated_source=source,metric_index=i,EPE_worsening_percent=x,rows=v['rows'],mothers=v['mothers']))
 with np.load(ROOT/'tmp/rh/half-training/panel-step000000/paired-panel.npz') as z:reference=z['sums'];counts=z['counts']
 rows=read(ROOT/'tmp/rh/fixed-panel/manifest.json')['rows'];pooled={}
 for arm in ('C4-source10','C4-source0'):
  trained=int(arm.split('source')[1]);mask=np.array([r['source_id']!=trained for r in rows]);step=summary[arm]['updates']
  with np.load(OUT/arm/('panel-step%06d/paired-panel.npz'%step)) as z:final=z['sums']
  a=reference[mask].sum(0)/counts[mask].sum(0);b=final[mask].sum(0)/counts[mask].sum(0);pooled[arm]=dict(baseline_mm=a.tolist(),final_mm=b.tolist(),delta_mm=(b-a).tolist(),relative_change_percent=((b/a-1)*100).tolist())
 lines+=['\nC4 pooled OTHER-source signed EPE deltas (hand/contact/object):']
 for arm,v in pooled.items():lines.append('- '+arm+': '+', '.join('%+.4fmm'%x for x in v['delta_mm'])+'; relative '+', '.join('%+.2f%%'%x for x in v['relative_change_percent']))
 decisions=['C1: all three EPE improve with CI95<0 at matched global384/fresh optimizer. Generic continuation settings alone are not sufficient to explain the old half regression; source1-heavy aggregate recovery is not all-source improvement.','C2: all three EPE regress with CI95>0 despite restored curated weights. Source weighting alone is insufficient; expanded within-source support/retention/semantics/scale-optimization interactions remain. This does not isolate a toxic newly added source ID.','C3: no final quality measurement after fixed-workspace fault; shared-backbone-versus-flow mechanism remains UNRESOLVED.','C5:0 updates because its exact valid-gated sampler still hits the proven bad input; invalid-window recovery/degradation and C2+C5 joint rules are NOT EVALUABLE.','C4: '+('other-source EPE worsening>=50% occurs; see exact cells and sparse-source CIs.' if strong else 'no other-source EPE worsening>=50% in these three single-source controls; this does not establish universal compatibility.'),'H2: target-scale screen remains positive; source1 C1/C2 sample RMS ratio1.436 is below its additional2x criterion. A real valid-input workspace violation is proved for source4, but that causes hard failure and does not itself explain the earlier successfully completed500-step EPE regression.','Validity:46-50% refers to missing point-frame supervision, not missing whole windows. Whole24-step zero-hand supervision is1590/9341050=0.0170%; source4/8 invalid whole-window counts0. All those windows still have object supervision.']
 for arm,v in self_change.items():
  if all(d is not None and d>0 for d in v['delta_mm']):decisions.append(arm+' also worsens its OWN source on all three point estimates; source mixing is not necessary for that observed regression. Any>=50% off-diagonal threshold hit is reported under the owner rule, but is not a clean source-tradeoff proof when the trained source itself also deteriorates. Broader within-source support/retention/semantics/optimization remain candidates.')
 lines+=['\nPreregistered decisions and causal limits:']+['- '+x for x in decisions]+['\nNo model/loss/LR/threshold/workspace was relaxed. Original half/remaining outputs are retained. Original half support differs from new full support; C4 global96 differs from global384 controls. Fixed TRAIN512/124 evidence only; no heldout generalization/execution claim. Independent workspace repair preflight remains a proposal unless separately authorized; original C3/C5 cannot be labelled recovered/degraded from runtime failures.']
 write(OUT/'final-summary.json',dict(status='PARTIAL_BLOCKED_INPUT_CONTRACT',arms=summary,decisions=decisions,C4_strong_interference=strong,C4_self=self_change,C4_other_pooled=pooled),immutable=True)
 with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 (OUT/'README.md').chmod(0o664)
 report=ROOT/'src/task/ptv3-shared-wam-v6/docs/audit/nine-source-degradation-20261023.md';report.write_text((OUT/'README.md').read_text())
 card=ROOT/'src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261023-nine-source-degradation.md';s=card.read_text().replace('status: RUNNING','status: HOLD');card.write_text(s+'\nResult: C1 recovers; C2 regresses; C4-s10/s0 measured; C4-s1 lost; C3 input-range failure/C5 blocked.\nDecision: repair workspace admission under a separate contract before retrying uniform controls; no original-arm quality verdict for C3/C5.\n')
 state=ROOT/'src/task/ptv3-shared-wam-v6/docs/STATE.md';state.write_text('## 2026-10-23 remote C1-C5 diagnosis: partial, input-contract blocker\n\nC1 matched global384/fresh-optimizer500updates improves panel8.081/19.366/4.959mm; C2 curated weights/full support500updates regresses11.417/34.071/6.779mm, all paired CI95 positive. C4 source10/source0 completed500 updates; source1 lost at480 logged due shared rendezvous-store launcher fault; partial matrix in report. C3 initial0-update NAS permission fault repaired in C3-r2, then valid source4 hand pointsx=-2.724m violate fixed voxel origin-2m at step104; no final C3 quality panel. C5 deterministically retains/hits same valid-supervision window, blocked0updates. Do not equate missing other-hand slots with invalid windows (exact whole-window gate removes0.017%). No workspace/model/LR/threshold change or automatic retry. [Report](audit/nine-source-degradation-20261023.md).\n\n'+state.read_text())
 subprocess.run(['git','add',str(report),str(card),str(state)],cwd=ROOT,check=True)
 subprocess.run(['git','commit','-m','Summarize C1/C2/C4-s10/C4-s0 each500; C3/C4-s1 faults/C5 blocked; data5a89c571e072'],cwd=ROOT,check=True)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
