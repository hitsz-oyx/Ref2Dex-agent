"""CPU analysis of immutable, GPU-extracted conditional-neighborhood caches."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v3.multimodality import choose_neighbors, neighborhood_metrics, motion_metrics

SOURCES = {0:'OakInk2', 1:'GRAB', 2:'ARCTIC', 4:'UniDex', 5:'TACO', 6:'HOT', 9:'H2O', 10:'HOI4D'}


def load(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k:f[k] for k in f.files if k != 'metadata'}
        meta = [json.loads(x) for x in f['metadata']]
    return data, meta


def bind_recordings(metadata, manifest):
    """A gap-split file is not an independent recording (notably H2O)."""
    for m in metadata:
        m['group']=m['capture']
        if m['source_id']<4: continue
        if manifest is None: raise ValueError('Native manifest required for physical recording identities')
        row=manifest['sequences'][m['row']]
        assert row['source_id']==m['source_id']
        assert str(row['objects'][row['target_slot']])==m['geometry_key']
        m['capture_file']=m['capture']
        # UniDex episode names repeat across independent rollout batches.
        # Only H2O is known to split a single take into several files.
        if m['source_id']==9:
            m['capture']='9:'+row['source_sequence']
        m['group']=row['group_id']


def summary(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(a): return dict(n=0, median=None, mean=None, p25=None, p75=None)
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                p25=float(np.quantile(a,.25)), p75=float(np.quantile(a,.75)))


def capture_bootstrap(rows, value, seed=214, unit='capture'):
    """Equal capture weighting; windows within a capture are not independent."""
    groups = defaultdict(list)
    for r in rows:
        if r.get(value) is not None: groups[(r['source_id'],r[unit])].append(float(r[value]))
    a = np.asarray([np.mean(v) for v in groups.values()])
    if not len(a): return dict(captures=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    draws = np.mean(a[rng.integers(len(a),size=(2000,len(a)))], axis=1)
    return dict(captures=len(a), mean=float(a.mean()), ci95=np.quantile(draws,[.025,.975]).tolist())


def reduce_rows(rows):
    keys = ['ade_m','ade8_m','fde_m','zero_motion_ade_m','constant_velocity_ade_m','first8_motion_ratio','all24_motion_ratio','h24_motion_ratio',
            'speed_ratio','acceleration_ratio','scene_error8_m','scene_error24_m',
            'conditional_spread_m','pred_mean_m','pred_real_m','pred_mode_m','pred_medoid_m','pred_nn3_m','gt_nn3_m','nn3_ratio']
    return dict(windows=len(rows), captures=len({r['capture'] for r in rows}),
                metrics={k:summary([r.get(k) for r in rows]) for k in keys},
                capture_weighted={k:capture_bootstrap(rows,k) for k in ['ade_m','all24_motion_ratio','speed_ratio']},
                group_weighted={k:capture_bootstrap(rows,k,unit='group') for k in ['ade_m','all24_motion_ratio','speed_ratio']},
                contact_proxy={k:summary([r.get(k) for r in rows]) for k in
                               ['gt_contact8_10mm','pred_contact8_10mm','gt_contact24_10mm','pred_contact24_10mm',
                                'gt_contact8_20mm','pred_contact8_20mm','gt_contact24_20mm','pred_contact24_20mm']})


def analyze(cache, out, native_manifest=None):
    progress = json.loads((cache/'progress.json').read_text())
    assert progress['status']=='COMPLETED', 'Use a complete frozen cache'
    out.mkdir(parents=True, exist_ok=False)
    manifest=json.loads(native_manifest.read_text()) if native_manifest else None
    rows=[]; cases=[]; conditional=[]; control_groups=defaultdict(list)
    for path in sorted(cache.glob('dev-*.npz')):
        sid=int(path.stem.split('-')[1]); q,mq=load(path)
        bank_path=cache/f'train-{sid}.npz'
        bank,mb=load(bank_path) if bank_path.exists() else (None,[])
        bind_recordings(mq,manifest);bind_recordings(mb,manifest)
        if bank is not None:
            # Fit all embedding transforms on TRAIN only, within each source.
            center=bank['z'].mean(0); scale=np.maximum(bank['z'].std(0),.05)
            bz=(bank['z']-center)/scale; bz/=np.maximum(np.linalg.norm(bz,axis=1,keepdims=True),1e-8)
        source_rows=[]
        for i,m in enumerate(mq):
            row=dict(source_id=sid,query=i,capture=m['capture'],group=m['group'],row=m['row'],tick=m['tick'],
                     **motion_metrics(q['pred_delta'][i],q['delta'][i]))
            # Avoid interpreting ratios for nearly static GT as attenuation.
            if row['all24_gt_motion_m']<.005 or row['gt_speed']<.01:
                row['speed_ratio']=None;row['acceleration_ratio']=None
            velocity=q['history'][i,-1,m['side']]-q['history'][i,-2,m['side']]
            cv=np.arange(1,25)[:,None,None]*velocity[None]
            row.update(zero_motion_ade_m=float(np.linalg.norm(q['delta'][i],axis=-1).mean()),
                       constant_velocity_ade_m=float(np.linalg.norm(cv-q['delta'][i],axis=-1).mean()))
            row.update(scene_error8_m=float(q['scene_error'][i,7]),scene_error24_m=float(q['scene_error'][i,23]))
            for hi,h in enumerate((8,24)):
                for epsilon,mm in ((.01,10),(.02,20)):
                    for pi,name in enumerate(('gt','pred')):
                        row[f'{name}_contact{h}_{mm}mm']=float(np.mean(q['tip_proximity'][i,hi,pi]<epsilon))
            rows.append(row); source_rows.append(row)
            if bank is None:
                conditional.append(dict(source_id=sid,query=i,capture=m['capture'],unsupported='no cross-split geometry'))
                continue
            z=(q['z'][i]-center)/scale; z/=max(np.linalg.norm(z),1e-8); zd=1-bz@z
            query={k:v[i] for k,v in q.items()}
            by_control={}
            for gate in (.02,.05):
                for retrieval in ('latent','raw'):
                    selected=choose_neighbors(query,bank,m,mb,zd if retrieval=='latent' else None,gate)
                    record=dict(source_id=sid,query=i,capture=m['capture'],gate_m=gate,retrieval=retrieval,
                                neighbors=len(selected), supported=len(selected)>=8)
                    if len(selected)>=8:
                        ids=np.array([x[0] for x in selected]); absolute=neighborhood_metrics(bank['future'][ids],q['prediction'][i],q['future'][i])
                        displacement=neighborhood_metrics(bank['delta'][ids],q['pred_delta'][i],q['delta'][i])
                        record.update(absolute=absolute,displacement=displacement,
                                      condition_max_m=np.max(np.array(selected)[:,1:],axis=0).tolist(),
                                      neighbor_indices=ids.tolist())
                        record['robust_multimodal']=bool(absolute['multimodal'] and displacement['multimodal'])
                        record['averaging']=bool(record['robust_multimodal'] and
                            all(x.get('between_modes',False) and x.get('off_empirical_support',False) for x in (absolute,displacement)))
                        if gate==.05 and retrieval=='latent':
                            # Low/high spread is defined within source after all queries are collected.
                            row.update(conditional_spread_m=absolute['spread_m'],
                                pred_nn3_m=absolute['pred_nn3_m'],gt_nn3_m=absolute['gt_nn3_m'],
                                nn3_ratio=absolute['pred_nn3_m']/max(absolute['gt_nn3_m'],.001))
                            for key in ('pred_mean_m','pred_real_m','pred_mode_m','pred_medoid_m'):
                                row[key]=absolute.get(key)
                            if absolute['multimodal'] or len(cases)<3:
                                cases.append((record,query,bank['future'][ids]))
                    conditional.append(record); by_control[(gate,retrieval)]=record
                    control_groups[(sid,gate,retrieval)].append(record)
            # Primary decision requires the tight gate AND both retrieval controls and coordinate spaces.
            tight=[by_control[(.02,r)] for r in ('latent','raw')]
            row['robust_multimodal']=all(r.get('robust_multimodal',False) for r in tight)
            row['robust_averaging']=all(r.get('averaging',False) for r in tight)
        eligible=[r for r in source_rows if 'conditional_spread_m' in r]
        if eligible:
            cuts=np.quantile([r['conditional_spread_m'] for r in eligible],[1/3,2/3])
            for r in eligible:r['spread_group']=('low','middle','high')[int(np.searchsorted(cuts,r['conditional_spread_m'],side='right'))]
        print(json.dumps(dict(source=SOURCES.get(sid,str(sid)),dev=len(mq),bank=len(mb),supported50=sum('conditional_spread_m' in r for r in source_rows))),flush=True)

    groups={}
    for (sid,gate,retrieval),rs in control_groups.items():
        supported=[r for r in rs if r['supported']]
        multi=[r for r in supported if r['robust_multimodal']]
        group=dict(queries=len(rs),supported=len(supported),supported_captures=len({r['capture'] for r in supported}),
                   supported16=sum(r['neighbors']>=16 for r in supported),multimodal=len(multi),
                   multimodal_captures=len({r['capture'] for r in multi}), averaging=sum(r['averaging'] for r in multi),
                   averaging_capture_rate=capture_bootstrap(multi,'averaging'))
        for space in ('absolute','displacement'):
            group[space]={key:summary([r[space].get(key) for r in supported]) for key in
                          ('spread_m','pred_mean_m','pred_real_m','pred_mode_m','pred_medoid_m','pred_nn3_m','gt_nn3_m','mean_nn3_m')}
        groups[f'{sid}-{gate}-{retrieval}']=group
    robust=[r for r in rows if r.get('robust_multimodal')]
    decision=capture_bootstrap(robust,'robust_averaging')
    label='PROMISING' if decision['captures']>=30 and decision['mean']>=.2 else 'UNCLEAR'
    result=dict(schema='ref2dex.mse-multimodality.v2',checkpoint_step=progress['checkpoint_step'],cache_progress=progress,
                native_manifest=str(native_manifest),
                conclusion=label,decision=decision,counts=dict(dev=len(rows),captures=len({(r['source_id'],r['capture']) for r in rows})),
                sources={str(sid):reduce_rows([r for r in rows if r['source_id']==sid]) for sid in sorted({r['source_id'] for r in rows})},
                conditional=groups,spread_groups={f'{sid}-{name}':reduce_rows([r for r in rows if r['source_id']==sid and r.get('spread_group')==name])
                    for sid in SOURCES for name in ('low','middle','high') if any(r['source_id']==sid and r.get('spread_group')==name for r in rows)})
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'queries.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (out/'conditional.json').write_text(json.dumps(conditional,indent=2,allow_nan=False))
    make_plots(rows,cases,out)
    write_report(result,out)
    print(json.dumps(dict(conclusion=label,decision=decision,output=str(out))))


def make_plots(rows,cases,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    sids=sorted({r['source_id'] for r in rows});fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in zip(axes,['ade_m','all24_motion_ratio','speed_ratio'],['Hand ADE (mm)','Motion amplitude / GT','Speed / GT']):
        values=[[r[key]*(1000 if key=='ade_m' else 1) for r in rows if r['source_id']==sid and r.get(key) is not None] for sid in sids]
        ax.boxplot(values,showfliers=False);ax.set_xticks(range(1,len(sids)+1));ax.set_xticklabels([SOURCES[x] for x in sids],rotation=55,ha='right');ax.set_ylabel(label)
        if key!='ade_m':ax.axhline(1,color='gray',linestyle='--')
    fig.suptitle('Frozen MSE checkpoint: per-source DEV diagnostics');fig.tight_layout();fig.savefig(out/'prediction_diagnostics.png',dpi=180);plt.close(fig)
    supported=[r for r in rows if 'conditional_spread_m' in r]
    if supported:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for sid in sids:
            rs=[r for r in supported if r['source_id']==sid]
            if not rs:continue
            axes[0].scatter([r['conditional_spread_m']*1000 for r in rs],[r['nn3_ratio'] for r in rs],s=12,alpha=.6,label=SOURCES[sid])
            axes[1].scatter([r['gt_nn3_m']*1000 for r in rs],[r['pred_nn3_m']*1000 for r in rs],s=12,alpha=.6)
        axes[0].axhline(1,color='gray',linestyle='--');axes[0].set(xlabel='Conditional future spread (mm)',ylabel='Prediction / GT empirical 3-NN distance');axes[0].legend(fontsize=7)
        lim=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,lim],[0,lim],'k--');axes[1].set(xlabel='GT empirical 3-NN distance (mm)',ylabel='Prediction empirical 3-NN distance (mm)')
        fig.tight_layout();fig.savefig(out/'conditional_support.png',dpi=180);plt.close(fig)
    cases.sort(key=lambda c:(c[0].get('averaging',False),c[0]['absolute'].get('silhouette',-1)),reverse=True)
    for number,(r,q,f) in enumerate(cases[:6]):
        prefix=out/f'case-{number:02d}'
        np.savez_compressed(str(prefix)+'.npz',neighbors=f,prediction=q['prediction'],truth=q['future'],start=q['start'])
        prefix.with_suffix('.json').write_text(json.dumps(r,indent=2))
        fig=plt.figure(figsize=(10,4));ax=fig.add_subplot(121,projection='3d')
        for path in f:ax.plot(*path[:,0].T,color='gray',alpha=.3,lw=.7)
        for path,label,color in [(q['prediction'],'MSE','red'),(q['future'],'DEV GT','green'),(f.mean(0),'TRAIN mean','blue')]:ax.plot(*path[:,0].T,label=label,color=color,lw=2)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Wrist trajectories');ax.legend(fontsize=7)
        pca=PCA(n_components=2).fit(f.reshape(len(f),-1));xy=pca.transform(f.reshape(len(f),-1));ax2=fig.add_subplot(122);ax2.scatter(*xy.T,c='gray',s=20,label='TRAIN futures')
        for path,label,color in [(q['prediction'],'MSE','red'),(q['future'],'DEV GT','green'),(f.mean(0),'TRAIN mean','blue')]:
            point=pca.transform(path.reshape(1,-1))[0];ax2.scatter(*point,color=color,label=label,marker='x',s=70)
        ax2.legend(fontsize=7);ax2.set(xlabel='Future PCA 1',ylabel='Future PCA 2',title=f'Source {r["source_id"]}, DEV {r["query"]}, gate 50 mm')
        fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)


def write_report(result,out):
    text=['# Existing-weight MSE conditional multimodality audit','',f'Checkpoint step: {result["checkpoint_step"]}. Decision label: **{result["conclusion"]}**. No FM model was trained in this audit.','',
          '| Source | DEV windows / captures | ADE mm (median) | Motion / GT (median) | Speed / GT (median) |', '|---|---:|---:|---:|---:|']
    def fmt(v,scale=1):return '—' if v is None else f'{v*scale:.3f}'
    for sid,r in result['sources'].items():
        m=r['metrics'];text.append(f'| {SOURCES[int(sid)]} | {r["windows"]} / {r["captures"]} | {fmt(m["ade_m"]["median"],1000)} | {fmt(m["all24_motion_ratio"]["median"])} | {fmt(m["speed_ratio"]["median"])} |')
    text+=['','| Source / gate / retrieval | Supported ≥8 | ≥16 | Multimodal in both spaces | Averaging + off support |','|---|---:|---:|---:|---:|']
    for name,r in result['conditional'].items():text.append(f'| {name} | {r["supported"]}/{r["queries"]} | {r["supported16"]} | {r["multimodal"]} | {r["averaging"]} |')
    text+=['',f'Tight-gate/raw-and-latent robust multimodal captures and averaging rate: `{json.dumps(result["decision"])}`. Bootstrap resamples independent DEV captures; window medians in the first table are descriptive.','',
           'Limits: finite-radius neighbors are not identical continuous H,G conditions. Multimodality requires both absolute and displacement-space separation; primary decision also requires both retrieval methods at 20 mm. Undertraining, missing observations, residual bad data, and finite bank support can explain errors. UniDex/HOI4D have no TRAIN/DEV target geometry overlap and cannot be diagnosed conditionally here. Complete future hand coverage is a selection filter. One neighbor per physical recording; H2O split segments are rebound to manifest source_sequence, other sources retain unique file identities (UniDex episode names repeat across independent rollout batches). A maximum eight windows per file was sampled before this correction; capture/group weighting handles unequal query counts. Session/group bootstrap is also reported in JSON. Embedding whitening is fit on TRAIN only. No future hand label enters retrieval.','',
           'Contact is an unsigned distance to a sparse cached object cloud after a GT rigid transform, not mesh SDF, penetration, or grasp success. Motion ratios exclude GT amplitude below 5 mm; speed/acceleration ratios also exclude all24 GT amplitude below 5 mm or mean GT speed below 0.01 m/s. Sources are reported separately. Correlation with spread alone is not evidence of conditional averaging. Bootstrap intervals are conditional on the fixed TRAIN bank and omit bank sampling/clustering uncertainty.','',
           'A PROMISING audit only justifies a matched frozen-backbone FM versus refitted MSE head probe, with equal data and compute, single-sample accuracy, minADE/FDE@M, mode precision/coverage and sampling cost. UNCLEAR means the existing weights/neighborhoods cannot select the architecture.','',
           '![Per-source prediction diagnostics](prediction_diagnostics.png)','']
    if (out/'conditional_support.png').exists():text+=['![Empirical conditional support](conditional_support.png)','']
    (out/'report.md').write_text('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-manifest',type=Path,help='Exact bound native manifest, for original recording/group IDs')
    a=p.parse_args();analyze(a.cache,a.output,a.native_manifest)
