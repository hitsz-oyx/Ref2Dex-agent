"""Compare completed matched shared actor runs; never mix new/old dev panels."""
import argparse
import json
from pathlib import Path
import numpy as np

ARMS=('pred_tau','shared','shared_tau')


def summarize(root):
    runs={arm:root/('shared-actor-'+arm+'-20261015-r1') for arm in ARMS}
    packets={arm:json.loads((p/'manifest.json').read_text()) for arm,p in runs.items()}
    status={arm:p['status'] for arm,p in packets.items()}
    if any(v!='COMPLETED' for v in status.values()):
        return dict(status='INCOMPLETE',arms=status)
    configs=[];panels=[];admissions=[];parents=[]
    for arm,p in runs.items():
        packet=packets[arm]
        if packet['config']['action_interface']!=arm:raise ValueError('interface identity differs')
        configs.append({k:v for k,v in packet['config'].items() if k!='action_interface'})
        panels.append(np.load(p/'validation.npy'))
        admissions.append(json.loads((p/'rollout_admission.json').read_text()))
        parents.append({k:packet['provenance'][k] for k in ('source','parent_checkpoint','parent_step','world_size','git_commit')})
    if any(c!=configs[0] for c in configs) or any(p!=parents[0] for p in parents):
        raise ValueError('matched configuration/parent/code/world size differs')
    if any(not np.array_equal(p,panels[0]) for p in panels) or any(a!=admissions[0] for a in admissions):
        raise ValueError('matched panels/admission differ')
    rows={}
    for arm,p in runs.items():
        curve=json.loads((p/'curve.json').read_text())
        final=curve[-1]
        if final['step']!=configs[0]['updates']:raise ValueError('missing final matched update')
        rows[arm]=dict(final=final,
            selected_control=min(curve,key=lambda r:(r['physical']['PD_xyz_mm']['p95'],r['physical']['PD_xyz_mm']['mean'])),
            selected_forecast=min(curve,key=lambda r:sum(r['physical'][k]['mean'] for k in ('hand_all24_mm','scene_all24_mm'))),
            selected_total=min(curve,key=lambda r:r['objective']),
            checkpoints=json.loads((p/'checkpoint_files.json').read_text()))
    baseline=rows['pred_tau']['final']['physical']
    improvements={}
    for arm in ('shared','shared_tau'):
        metrics=rows[arm]['final']['physical']
        mean_gain=1-metrics['PD_xyz_mm']['mean']/baseline['PD_xyz_mm']['mean']
        p95_gain=1-metrics['PD_xyz_mm']['p95']/baseline['PD_xyz_mm']['p95']
        forecast_ratio=sum(metrics[k]['mean'] for k in ('hand_all24_mm','scene_all24_mm'))/sum(
            baseline[k]['mean'] for k in ('hand_all24_mm','scene_all24_mm'))
        improvements[arm]=dict(mean_gain_fraction=mean_gain,p95_gain_fraction=p95_gain,
            forecast_ratio=forecast_ratio,signal='PROMISING' if min(mean_gain,p95_gain)>=.2 and forecast_ratio<=1.2 else 'UNCLEAR')
    return dict(status='COMPLETED',matched_checks='PASS',runs={a:str(p) for a,p in runs.items()},
                rows=rows,improvements=improvements,scope='single-seed offline Probe; no grasp or formal information bottleneck claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=summarize(a.root)
    if a.output.exists():raise ValueError('fresh report required')
    a.output.mkdir(parents=True)
    (a.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','matched_checks','improvements')}))

if __name__=='__main__':main()
