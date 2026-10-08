"""Audit frozen outcome data and plot train-only weak-label examples."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within,K
from consequence_evaluator.value_outcomes import DATA_SCHEMA,task_trace


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();start=time.monotonic()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not 1<=len(a.source)<=3:
        p.error('fresh task-owned audit directory and <=3sources required')
    labels=a.labels.resolve();m=json.loads((labels/'manifest.json').read_text())
    if m['schema']!=DATA_SCHEMA or m['status']!='COMPLETED' or sha(labels/'windows.npz')!=m['windows_sha256']:
        raise ValueError('completed immutable labeled windows required')
    outcomes={r['episode']:r for r in m['episode_audit']};rows=[];samples={};frozen={}
    frozen[str(labels/'manifest.json')]=sha(labels/'manifest.json')
    frozen[str(labels/'windows.npz')]=m['windows_sha256']
    frozen[str(Path(__file__).resolve())]=sha(__file__)
    for source in a.source:
        source=source.resolve();path=source/'manifest.json';raw=json.loads(path.read_text())
        if m['source_inputs'].get(str(path))!=sha(path):raise ValueError('labels belong to another raw manifest')
        frozen[str(path)]=sha(path)
        bank=json.loads(Path(raw['perturbation_bank']).read_text()) if 'perturbation_bank' in raw else None
        for r in raw['episodes']:
            if time.monotonic()-start>120:raise TimeoutError('fixed120s audit deadline')
            files=[source/r['path'],source/r['diagnostics']]
            for file,key in zip(files,('sha256','diagnostics_sha256')):
                if not is_within(file.resolve(),source) or sha(file)!=r[key]:raise ValueError('source episode drift')
                frozen[str(file)]=r[key]
            with np.load(files[0],allow_pickle=False) as f:packet={k:f[k] for k in f.files}
            with np.load(files[1],allow_pickle=False) as f:d={k:f[k] for k in f.files}
            trace=task_trace(packet,d);expected=outcomes[r['episode']]
            if int(trace['task_success'])!=expected['success']:raise ValueError('physical label recomputation disagrees')
            t=r['perturbation_tick'];row=dict(episode=r['episode'],split=r['split'],
                phase=r['assigned_phase'],success=int(trace['task_success']),max_held_frames=int(trace['held_run'].max()),
                final_support_gap_m=float(trace['support_gap'][-1]),final_settled=bool(trace['settled'][-1]),
                trigger=t,bank_member=r.get('bank_member',-1))
            if t>=0:
                plan=packet['residual_plan'][t];delta=d['actual_residual'][t:t+K]
                base=packet['action'][t:t+K]-delta
                projected=np.clip(base+plan,-1,1)
                row.update(gain=bank['members'][r['bank_member']]['gain'],requested_l2=float(np.linalg.norm(plan)),
                    actual_l2=float(np.linalg.norm(delta)),request_peak=float(np.abs(plan).max()),
                    clip_fraction=float(np.mean(np.abs(base+plan-projected)>1e-7)),
                    measured_command_vs_clipped_request_max=float(np.abs(projected-packet['action'][t:t+K]).max()))
            rows.append(row)
            # Only train examples are selected for later human label inspection.
            category=('clean-success' if t<0 and row['success'] else
                      'perturbed-success' if t>=0 and row['success'] else
                      'failed-no-stable-hold' if t>=0 and row['max_held_frames']<45 else
                      'failed-after-hold' if t>=0 and not row['success'] else None)
            if r['split']=='train' and category and category not in samples:samples[category]=(packet,d,trace,row)
    if len(rows)!=len(outcomes) or len({r['episode'] for r in rows})!=len(rows):
        raise ValueError('audit source coverage mismatch')
    result=dict(status='COMPLETED',scope='numeric weak-label audit; train-only examples, not manually confirmed collisions',
        training_allowed=m['training_allowed'],counts=m['counts'],force_proxy_used=False,
        phase_outcomes={split:{phase:dict(Counter(str(r['success']) for r in rows if r['split']==split and r['phase']==phase))
                              for phase in sorted({r['phase'] for r in rows})} for split in ('train','val','test')},
        gain_outcomes={split:{str(gain):dict(Counter(str(r['success']) for r in rows if r['split']==split and r.get('gain')==gain))
                             for gain in sorted({r['gain'] for r in rows if 'gain' in r})} for split in ('train','val','test')},
        episode_audit=rows,source_hashes=frozen)
    if any(sha(path)!=value for path,value in frozen.items()):raise ValueError('audit source drift')
    out.mkdir(parents=True)
    scratch=ROOT/'tmp/consequence-value-plots';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(MPLCONFIGDIR=str(scratch),TMPDIR=str(ROOT/'tmp'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(samples),figsize=(4*len(samples),6),squeeze=False,sharex='col')
    for column,(category,(packet,d,t,row)) in enumerate(samples.items()):
        seconds=packet['timestamps'];ref=d['reference_object_pose'][:,2,3]-d['reference_object_pose'][0,2,3]
        axes[0,column].plot(seconds,t['height'],label='measured height')
        axes[0,column].plot(seconds,ref,alpha=.6,label='reference height')
        axes[0,column].plot(seconds,t['support_gap'],ls=':',label='bottom/table gap')
        axes[0,column].axhline(.03,color='gray',ls='--',lw=.7)
        axes[0,column].set_title(category+'\nS='+str(row['success'])+' | '+row['episode'],fontsize=8)
        axes[1,column].plot(seconds,t['gap'],label='sampled hand/object gap')
        axes[1,column].axhline(.01,color='gray',ls='--',lw=.7);axes[1,column].set_yscale('log')
        for ax in axes[:,column]:
            ax.axvline(t['place_start']/30,color='green',ls='--',lw=.7)
            if row['trigger']>=0:ax.axvspan(row['trigger']/30,(row['trigger']+K)/30,color='orange',alpha=.2)
            ax.grid(alpha=.2);ax.legend(fontsize=6)
        axes[1,column].set_xlabel('time (s)');axes[0,column].set_ylabel('height/gap (m)')
        axes[1,column].set_ylabel('geometric gap (m)')
    fig.tight_layout();fig.savefig(out/'train-weak-label-examples.png',dpi=150);plt.close(fig)
    result['elapsed_s']=time.monotonic()-start
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('training_allowed','phase_outcomes','gain_outcomes','elapsed_s')},indent=2))


if __name__=='__main__':main()
