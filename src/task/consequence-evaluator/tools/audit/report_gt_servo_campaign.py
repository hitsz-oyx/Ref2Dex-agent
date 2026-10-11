"""Summarize audited GT-servo probes without promoting them to validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not is_within(args.output,ROOT/'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    rows=[];runs=[]
    for run in range(1,17):
        directory=ROOT/'outputs/consequence-evaluator'/('object-relative-gt-servo-20261009-r%s'%run)
        packet_file=directory/'servo.pkl'
        with packet_file.open('rb') as stream:p=pickle.load(stream)
        audit_directory=ROOT/'outputs/consequence-evaluator'/('object-relative-gt-servo-audit-20261009-r%s'%run)
        audit=json.loads((audit_directory/'report.json').read_text())
        assert audit['status']=='PASS' and audit['packet_sha256']==hashlib.sha256(packet_file.read_bytes()).hexdigest()
        manifest=json.loads((directory/'run_manifest.json').read_text())
        summary=json.loads((directory/'servo.json').read_text())
        assert manifest['exit_code']==0 and summary['steps']==542
        layout=p.get('servo_layout','command_vs_measured')
        runs.append(dict(run_id=directory.name,layout=layout,anchor=p['object_anchor'],
            git_commit=manifest['git_commit'],packet_sha256=audit['packet_sha256'],
            worker_elapsed_s=summary['elapsed_s'],raw_gate=p['gate'],audit_status='PASS'))
        for index,name in enumerate(p['role_names']):
            m=p['metrics'][name]
            rows.append(dict(run=run,role=index,name=name,layout=layout,
                held_frames=m['maximum_held_frames'],world_coordinate_rmse_mm=m['hand_coordinate_rmse_mm'],
                object_local_coordinate_rmse_mm=m['object_local_hand_coordinate_rmse_mm'],
                full_task_success=m['full_task_success'],intermediate_loss_events=m['intermediate_loss_events'],
                teacher_behavior_valid=p['gate']['teacher_held']>=45,
                raw_role_gate=(p['gate']['role_passes'].get(name) if index else None),
                native_clips=(int(p['clipped_coordinate_counts'][:,index-1].sum()) if index else 0)))
    args.output.mkdir(parents=True)
    with (args.output/'roles.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='UNCLEAR',engineering_only=True,runs=runs,
        native_launches=16,roles_per_launch=4,total_native_worker_s=sum(r['worker_elapsed_s'] for r in runs),
        all_independent_execution_audits_pass=True,
        limitations='one scene/source/seed, correlated roles and bootstrap arms, contact-state divergence; no formal Cm or geometry sufficiency claim')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GT servo campaign: exploratory results','',
        'All16 native runs completed542 steps and passed independent execution audits. Overall: UNCLEAR.',
        'Roles are correlated; warm bootstrap arms and privileged joint/command inputs are not deployable policies.','',
        '| Run | Layout | Teacher held | World command held | Role2 held | Role3 held | Raw primary gate |',
        '|---|---|---:|---:|---:|---:|---|']
    for run in runs:
        block=[r for r in rows if r['run']==int(run['run_id'].rsplit('r',1)[1])]
        lines.append('| %s | %s | %s | %s | %s | %s | %s |'%(
            block[0]['run'],run['layout'],*[r['held_frames'] for r in block],run['raw_gate']['passed']))
    lines.extend(['','r14 full-cold held485 has a failed live-teacher behavior gate; r15 repeats it at238. Neither passes the complete geometry screen.',
        'Wrist geometry with source commanded fingers holds483 in both r14/r15; only r15 has a valid live teacher.',
        'Raw historical gates are retained, including r4 weaker-live-reference gating. roles.csv preserves all geometry/clip/outcome metrics.'])
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    groups=[('world command','#444444',lambda r:r['role']==1),
        ('candidate controls','#4277ad',lambda r:r['role']>=2 and r['layout']!='geometry_pd_inverse'),
        ('cold geometry wrist + source fingers','#2b9257',lambda r:r['role']==2 and r['layout']=='geometry_pd_inverse'),
        ('cold full11-point inverse','#c34a4a',lambda r:r['role']==3 and r['layout']=='geometry_pd_inverse')]
    for name,color,predicate in groups:
        group=[r for r in rows if predicate(r)]
        ax.scatter([r['world_coordinate_rmse_mm'] for r in group],[r['held_frames'] for r in group],s=35,color=color,label=name,alpha=.8)
        if name.startswith('cold full'):
            for r in group:ax.annotate('r%s'%r['run'],(r['world_coordinate_rmse_mm'],r['held_frames']),xytext=(5,5),textcoords='offset points')
    ax.axhline(.9*484,color='gray',ls='--',label='90%source484 (individual gates also require live control)')
    ax.set_xscale('log');ax.set_xlabel('World11-point coordinate RMSE (mm, log scale)')
    ax.set_ylabel('Maximum consecutive held frames');ax.set_ylim(-10,550)
    ax.set_title('Small world-hand error does not ensure contact maintenance')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.output/'world_error_vs_hold.png',dpi=150)
    print(json.dumps(dict(output=str(args.output),launches=16,roles=len(rows),status='UNCLEAR',native_worker_s=report['total_native_worker_s'])),flush=True)


if __name__=='__main__':main()
