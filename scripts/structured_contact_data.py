"""Read-only union of audited HF19 random and HF20 generated H10 source."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from qualify_contact_geometry_source import load as old_load

BASE=ROOT/'src/task/CmResidual/research/contact_consequence/output'
HF19=BASE/'P-20261002-contact-geometry-source-r1'
HF20=BASE/'P-20261002-optimized-contact-opportunity-source-r1'


def load():
    import torch
    old_audit=ROOT/'docs/experiments/probes/P-20261002-contact-geometry-source-audit-r1.json'
    records,_,adequate,hashes=old_load(HF19,old_audit)
    if not adequate:
        raise ValueError('original audited random source inadequate')
    origin=[19]*len(records)
    path=HF20/'run_manifest.json';source=json.loads(path.read_text())
    primary=ROOT/'docs/experiments/probes/P-20261002-optimized-contact-opportunity-results-r1.json'
    audit_path=ROOT/'docs/experiments/probes/P-20261002-optimized-contact-opportunity-audit-r1.json'
    audit=json.loads(audit_path.read_text())
    if source['run_status']!='COMPLETED' or source['child_exit_code']!=0 or [p['seed'] for p in source['phases']]!=list(range(591,603)):
        raise ValueError('complete fixed generated panel required')
    if not audit['audit_passed'] or audit['result_sha256']!=sha(primary):
        raise ValueError('generated primary independent audit required')
    hashes.update(source['input_sha256'])
    for p in (path,primary,audit_path):hashes[str(p.resolve())]=sha(p)
    for phase in source['phases']:
        p=Path(phase['directory'])/'records.pt';planning=Path(phase['directory'])/'planning.pt'
        a=Path(phase['audit']);control=HF20/'phase-controls'/f"seed{phase['seed']}"/'run_manifest.json'
        report=json.loads(a.read_text())
        if (phase['run_status']!='COMPLETED' or phase['native_exit_code']!=0 or phase['audit_exit_code']!=0
                or sha(p)!=phase['result']['record_sha256'] or sha(planning)!=phase['result']['planning_sha256']
                or sha(a)!=phase['audit_sha256'] or report['run_status']!='COMPLETED'
                or report['run_manifest_sha256']!=sha(control)):
            raise ValueError('generated source/audit/hash drift')
        b=torch.load(p,map_location='cpu',weights_only=False)
        if b['schema']!='ref2dex.optimized_contact_source.v1' or b['future_done'].any() or b['model_training']:
            raise ValueError('complete generated-source schema required')
        records.append(b);origin.append(20)
        for f in (p,planning,a,control):hashes[str(f.resolve())]=sha(f)
    identities=set()
    for b in records:
        for env,tick,motion,start,bucket in zip(b['env_id'],b['trigger'],b['motion_id'],b['start_frame'],b['split_group_bucket']):
            identity=(b['seed'],int(env),int(tick))
            expected=int(hashlib.sha256(f'12651/{int(motion)}/{int(start)}'.encode()).hexdigest()[:8],16)%100
            if identity in identities or int(bucket)!=expected:
                raise ValueError('source identity/group split drift')
            identities.add(identity)
    for p in (Path(__file__),ROOT/'scripts/qualify_contact_geometry_source.py',ROOT/'scripts/run_paired_evaluator_resolution.py'):
        hashes[str(p.resolve())]=sha(p)
    if any(sha(Path(k))!=v for k,v in hashes.items()):
        raise ValueError('read-only source input drift')
    return records,origin,hashes
