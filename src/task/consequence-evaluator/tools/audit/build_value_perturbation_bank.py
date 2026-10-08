"""Bounded train-only nominal failure audit and empirical control-error bank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within,K
from consequence_evaluator.value_outcomes import RAW_SCHEMA,DATA_SCHEMA
from consequence_evaluator.value_perturbations import BANK_SCHEMA,error_chunk


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();raw=a.source.resolve();labels=a.labels.resolve()
    if (out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator')
            or not is_within(raw,ROOT/'outputs/consequence-evaluator')
            or not is_within(labels,ROOT/'outputs/consequence-evaluator')):
        p.error('task-owned inputs and fresh bank file required')
    m=json.loads((raw/'manifest.json').read_text());y=json.loads((labels/'manifest.json').read_text())
    if (m.get('schema')!=RAW_SCHEMA or m.get('status')!='COMPLETED' or m.get('clean_only') is not True
            or m.get('split')!='train' or not 4<=len(m['episodes'])<=128
            or y.get('schema')!=DATA_SCHEMA or y.get('status')!='COMPLETED'):
        raise ValueError('completed bounded nominal train episodes and value labels required')
    frozen={str(raw/'manifest.json'):digest(raw/'manifest.json'),
            str(labels/'manifest.json'):digest(labels/'manifest.json'),str(Path(__file__).resolve()):digest(__file__)}
    if y['source_inputs'].get(str(raw/'manifest.json'))!=frozen[str(raw/'manifest.json')]:
        raise ValueError('labels do not belong to the supplied raw source')
    route_path=Path(next(path for path in m['sources'] if path.endswith('route.json')))
    if digest(route_path)!=m['sources'][str(route_path)]:raise ValueError('nominal route drift')
    frozen[str(route_path)]=digest(route_path)
    actor_sha=json.loads(route_path.read_text())['experts']['official_inspire']['sha256']
    audit={r['episode']:r for r in y['episode_audit']};packets=[];records=[]
    for r in m['episodes']:
        if r['split']!='train' or r['episode'] not in audit or r['assigned_phase']!='clean':
            raise ValueError('held-out or intervened source is forbidden')
        files=[raw/r['path'],raw/r['diagnostics']]
        for path,key in zip(files,('sha256','diagnostics_sha256')):
            if not is_within(path.resolve(),raw) or digest(path)!=r[key]:
                raise ValueError('raw source identity mismatch')
            frozen[str(path)]=r[key]
        with np.load(files[0],allow_pickle=False) as f:packet={k:f[k] for k in f.files}
        with np.load(files[1],allow_pickle=False) as f:d={k:f[k] for k in f.files}
        # Remove domain noise/clipping differences from executed controls to
        # recover the logged frozen-policy base; residual_plan is nominal zero.
        base=packet['action']-d['actual_residual']
        if base.shape!=(r['steps'],18) or not np.isfinite(base).all() or np.any(packet['residual_plan']):
            raise ValueError('nominal normalized policy control clock required')
        packets.append((base,d));records.append(r)
    positive=[base for r,(base,d) in zip(records,packets) if audit[r['episode']]['success']]
    if len(positive)<4 or len({len(base) for base in positive})!=1:
        raise ValueError('at least4clock-aligned successful nominal train episodes required')
    reference=np.median(np.stack(positive),axis=0);chunks=[];members=[]
    for r,(base,d) in zip(records,packets):
        row=audit[r['episode']]
        if row['success']:continue
        place=row['place_start'];near=d['surface_gap']<=.01
        support=d['table_footprint'].astype(bool)&(np.abs(d['support_gap'])<=.02)
        bad=np.flatnonzero((np.arange(len(near))>=place)&~near&~support)
        if not len(bad):continue
        onset=int(bad[0]);stop=min(onset,len(base));begin=max(place,stop-K)
        if stop-begin<4:continue
        errors=base[begin:stop]-reference[begin:stop]
        for gain in (.5,1.,1.5):
            chunks.append(error_chunk(errors,gain).tolist())
            members.append(dict(episode=r['episode'],phase='place',failure_onset=onset,
                                source_control_begin=begin,source_control_end=stop,gain=gain))
    if not chunks:raise ValueError('no supported pre-failure placing deviations; do not fabricate a distribution')
    for name in ('value_perturbations.py','value_outcomes.py','collection.py'):
        path=TASK/'src/consequence_evaluator'/name;frozen[str(path)]=digest(path)
    if any(digest(path)!=expected for path,expected in frozen.items()):raise ValueError('source drift during bank audit')
    result=dict(schema=BANK_SCHEMA,source_split='train',source_seed=m['seed'],phase='place',
        source_actor_sha256=actor_sha,
        sources=frozen,chunks=chunks,members=members,reference_success_count=len(positive),
        source_failure_count=sum(not row['success'] for row in audit.values()),
        semantics='empirical pre-failure policy control deviation from same-tick successful median; replay candidate only',
        covariance_fitted=False,causal_failure_direction_established=False,limits=dict(residual=.2,wrist_translation=.05),
        smoothing='four median temporal bins, linear interpolation, sin-squared taper, zero endpoints',
        gains=[.5,1.,1.5],sampling='uniform member with replacement, train bank frozen for held-out groups')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(chunks=len(chunks),members=members,output=str(out)),indent=2))


if __name__=='__main__':main()
