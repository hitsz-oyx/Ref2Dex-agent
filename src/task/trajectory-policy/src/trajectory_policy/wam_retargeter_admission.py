"""Authenticate an optional R coverage adaptation without transferring calibration."""
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[5]
BASE_R_SHA='414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f'
NORMALIZATION=('state_mean','state_scale','tau_mean','tau_scale','target_mean','target_scale')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            value.update(block)
    return value.hexdigest()


def validate_r_packet(packet,base):
    """CPU tensor contract: same architecture, finite weights, exact frozen norms."""
    import torch
    if (packet.get('schema')!=base.get('schema') or packet.get('schema')!='ref2dex.competent-pd-retargeter.v1'
            or packet.get('width')!=base.get('width') or packet.get('use_tau') is not True
            or base.get('use_tau') is not True or packet.get('base_R_sha256')!=BASE_R_SHA
            or not isinstance(packet.get('model'),dict) or set(packet['model'])!=set(base['model'])):
        raise ValueError('R architecture/schema/base/keys differ')
    for name,value in packet['model'].items():
        original=base['model'][name]
        if (not isinstance(value,torch.Tensor) or value.shape!=original.shape
                or value.dtype!=original.dtype or not torch.isfinite(value).all()):
            raise ValueError('R state shape/dtype/finite differs: '+name)
    for name in NORMALIZATION:
        if name not in packet['model']:
            raise ValueError('R normalization missing: '+name)
        value=packet['model'][name].detach().contiguous().reshape(-1).view(torch.uint8)
        original=base['model'][name].detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(value,original):
            raise ValueError('R normalization changed: '+name)


def retargeter_admission(r_dir=None,head_manifest=None):
    """Return selected path, JSON identity and all consumed evidence hashes."""
    owner=ROOT/'outputs/trajectory-policy'
    base=owner/'competent-pd-fit-20261011-r1/tau.pt'
    physical=owner/'competent-pd-execution-20261011-r1'
    pm=json.loads((physical/'manifest.json').read_text())
    if pm['input_sha256'].get(str(base))!=BASE_R_SHA or sha(base)!=BASE_R_SHA:
        raise ValueError('original physically calibrated R identity drift')
    if head_manifest is not None and head_manifest['input_sha256'].get(str(base))!=BASE_R_SHA:
        raise ValueError('head training R evidence differs from qualified base')
    hashes={str(base):BASE_R_SHA,str(physical/'manifest.json'):sha(physical/'manifest.json')}
    identity=dict(R_override=False,base_R_sha256=BASE_R_SHA,head_training_R_sha256=BASE_R_SHA,
        execution_R_sha256=BASE_R_SHA,R_identity_changed_since_head=False,
        scope='original physically calibrated R; unchanged default selection')
    from .wam_joint import JOINT_SCHEMA, paired_manifest_admission, policy_patch_admission, state_hash
    if head_manifest is not None and head_manifest.get('schema')==JOINT_SCHEMA and r_dir is None:
        raise ValueError('jointly learned WAM requires its exact jointly learned R directory')
    if r_dir is None:
        return base,identity,hashes
    directory=Path(r_dir).resolve()
    if owner not in directory.parents:
        raise ValueError('task-owned adapted R directory required')
    checkpoint=directory/'tau.pt'
    manifest=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    if manifest.get('schema')==JOINT_SCHEMA:
        if (manifest.get('result_sha256')!=sha(directory/'result.json')
                or manifest.get('base_R_sha256')!=BASE_R_SHA):
            raise ValueError('joint result/base R binding differs')
        paired_manifest_admission(manifest,result,head_manifest,sha(directory/'head.pt'),sha(checkpoint))
        inputs=manifest.get('input_sha256',{})
        if not inputs or any(sha(path)!=digest for path,digest in inputs.items()):
            raise ValueError('joint actual input identity drift')
        import torch
        packet=torch.load(checkpoint,map_location='cpu')
        base_packet=torch.load(base,map_location='cpu')
        validate_r_packet(packet,base_packet)
        net_state={k[4:]:v for k,v in packet['model'].items() if k.startswith('net.')}
        if state_hash(net_state)!=result['train_final_hash']['R']:
            raise ValueError('joint learned R weights differ from final training identity')
        if (packet.get('joint_training_schema')!=JOINT_SCHEMA or packet.get('run_id')!=manifest['run_id']
                or packet.get('input_sha256')!=inputs):
            raise ValueError('joint R packet provenance differs')
        head=torch.load(directory/'head.pt',map_location='cpu')
        policy_patch_admission(head,manifest,result)
        hashes.update(inputs)
        hashes.update({str(directory/k):sha(directory/k) for k in ('head.pt','tau.pt','manifest.json','result.json')})
        identity.update(R_override=True,head_training_R_sha256=sha(checkpoint),execution_R_sha256=sha(checkpoint),
            R_identity_changed_since_head=False,jointly_trained_pair=True,head_sha256=sha(directory/'head.pt'),
            scope='exact jointly trained WAM/R simulation diagnostic; original R physical qualification does not transfer')
        return checkpoint,identity,hashes
    if (manifest.get('schema')!='ref2dex.wam-recovery-R-fit.v1'
            or manifest.get('status')!='COMPLETED' or result.get('status')!='COMPLETED'
            or sha(directory/'result.json')!=manifest.get('result_sha256')
            or result.get('offline_screen_pass') is not True
            or manifest.get('base_R_sha256')!=BASE_R_SHA or result.get('base_R_sha256')!=BASE_R_SHA
            or sha(checkpoint)!=manifest.get('checkpoint_sha256')
            or sha(checkpoint)!=result.get('checkpoint_sha256')):
        raise ValueError('completed screened adapted R identity required')
    pr=json.loads((physical/'result.json').read_text())
    summary=pr.get('summary',{}).get('gt_tau_R',{})
    if (pr.get('teacher_readiness') is not True or summary.get('episodes',0)<1
            or summary.get('success')!=summary['episodes'] or summary.get('hold45')!=summary['episodes']):
        raise ValueError('base R original GT physical qualification required')
    hashes[str(physical/'result.json')]=sha(physical/'result.json')
    inputs=manifest.get('input_sha256',{})
    if not inputs or any(sha(path)!=digest for path,digest in inputs.items()):
        raise ValueError('adapted R actual consumption input drift')
    hashes.update(inputs)
    admission_path=Path(manifest.get('admission_path','')).resolve()
    admission_sha=manifest.get('admission_sha256')
    if (directory not in admission_path.parents or admission_path!=Path(result.get('admission_path','')).resolve()
            or sha(admission_path)!=admission_sha or admission_sha!=result.get('admission_sha256')):
        raise ValueError('adapted R scoped admission identity differs')
    admission=json.loads(admission_path.read_text())
    if (admission.get('schema')!='ref2dex.unmixed-recovery-R-admission.v1'
            or admission.get('admitted') is not True or admission.get('R_training_allowed') is not True
            or admission.get('actor_future_hand_training_allowed') is not False
            or admission.get('mixed_policy_future_hand_training_allowed') is not False
            or tuple(admission.get(k) for k in ('role','first_tick','last_tick','horizon','windows'))!=(3,32,168,24,137)):
        raise ValueError('exact R-only unmixed expert recovery admission required')
    paths={}
    for kind in ('native_source','native_manifest','root_audit','native_result','nominal_audit','nominal_trajectory'):
        path=Path(admission.get(kind+'_path','')).resolve()
        digest=admission.get(kind+'_sha256')
        if owner not in path.parents or sha(path)!=digest or inputs.get(str(path))!=digest:
            raise ValueError('scoped R source/audit binding differs: '+kind)
        paths[kind]=path
        hashes[str(path)]=digest
    source=json.loads(paths['native_manifest'].read_text())
    audit=json.loads(paths['root_audit'].read_text())
    sr=json.loads(paths['native_result'].read_text())
    source_sha=sha(paths['native_source'])
    if (source.get('status')!='COMPLETED' or source.get('controls')!=192 or source.get('takeover_tick')!=32
            or source.get('actor_future_hand_training_allowed') is not False
            or source.get('mixed_policy_future_hand_training_allowed') is not False
            or source.get('trajectory_sha256')!=source_sha
            or admission.get('source_trajectory_sha256')!=source_sha
            or audit.get('pass') is not True or audit.get('native_data_sha256')!=source_sha
            or audit.get('controls')!=192 or audit.get('takeover_tick')!=32
            or audit.get('checks_max')!=0 or audit.get('check_count')!=3468
            or sr.get('local_grasp_calibration') is not True or sr.get('local_grasp_recovery') is not True):
        raise ValueError('unmixed recovery source/root audit/local grasp chain differs')
    import torch
    packet=torch.load(checkpoint,map_location='cpu')
    base_packet=torch.load(base,map_location='cpu')
    validate_r_packet(packet,base_packet)
    nominal_audit=json.loads(paths['nominal_audit'].read_text())
    if (nominal_audit.get('pass') is not True
            or admission.get('nominal_audit_bound_by_base_R_sha256')!=BASE_R_SHA
            or any(base_packet.get('input_sha256',{}).get(str(paths[k]))!=sha(paths[k])
                for k in ('nominal_audit','nominal_trajectory'))):
        raise ValueError('original nominal audit/data not bound by qualified base R')
    if (packet.get('scoped_admission_sha256')!=admission_sha
            or ('admission_path' in packet and packet['admission_path']!=str(admission_path))
            or ('admission_sha256' in packet and packet['admission_sha256']!=admission_sha)
            or packet.get('input_sha256')!=inputs):
        raise ValueError('R packet provenance differs from completed manifest')
    hashes.update({str(directory/'manifest.json'):sha(directory/'manifest.json'),
        str(directory/'result.json'):sha(directory/'result.json'),str(checkpoint):sha(checkpoint),
        str(admission_path):admission_sha})
    identity.update(R_override=True,execution_R_sha256=sha(checkpoint),
        R_identity_changed_since_head=sha(checkpoint)!=BASE_R_SHA,admission_path=str(admission_path),
        admission_sha256=admission_sha,packet_admission_binding='scoped_admission_sha256',
        scope='fixed WAM head + adapted R simulated diagnosis; base physical qualification does not transfer')
    return checkpoint,identity,hashes
