"""Audit captured current observations against raw same-execution state/controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def audit_sequence(data, record, contract, controller, *, include_reference=True):
    length = record['frames']
    bodies = len(contract['body_names'])
    shapes = dict(q=(length,18),dq=(length,18),hand=(length,11,3),obj=(length,4,4),
        velocity=(length,6),reference_obj=(length,4,4),frame_tick=(length,),
        native_current_obs=(length,721),body_state=(length,bodies,13),
        body_contact_forces=(length,bodies,3),object_contact_force=(length,3),
        reference_current=(length,contract['reference_dim']),
        action=(length-1,18),applied=(length-1,18),pd_targets=(length-1,18))
    if not include_reference:
        for key in ('reference_obj','reference_current'):shapes.pop(key)
    for key,shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError('invalid shape/value: '+key)
    np.testing.assert_array_equal(data['frame_tick'],np.arange(length))
    np.testing.assert_array_equal(data['action'],data['applied'])
    if np.abs(data['applied']).max()>1.000001:
        raise ValueError('unclipped applied command')
    contact = (abs(data['body_contact_forces'][:,contract['contact_body_ids']])>.1).any(-1).astype(np.float32)
    np.testing.assert_array_equal(data['native_current_obs'][:,252:257],contact)
    np.testing.assert_array_equal(data['hand'],data['body_state'][:,contract['measured_hand_body_ids'],:3])
    if include_reference:
        reference = data['reference_current']
        ref_contact = reference[:,168:184][:,[3,6,9,12,15]]
        error = ref_contact*((ref_contact+1)/2-contact)
        np.testing.assert_array_equal(data['native_current_obs'][:,401:406],error)
        np.testing.assert_allclose(data['native_current_obs'][:,:3],reference[:,4:7]-data['body_state'][:,6,:3],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(data['reference_obj'][:,:3,3],reference[:,106:109])
    applied = data['applied'].copy()
    applied[:,6:] = (1+applied[:,6:])/2
    pd = np.asarray(controller['offset'],np.float32)+np.asarray(controller['scale'],np.float32)*applied
    pd[:,:6] += data['q'][:-1,:6]
    for distal,parent,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        pd[:,distal] = pd[:,parent]*ratio
    max_pd_error = float(abs(pd-data['pd_targets']).max())
    if max_pd_error>1e-6:
        raise ValueError('same-tick action/q_PD roundtrip failed')
    return dict(frames=length,commands=length-1,max_pd_error=max_pd_error,
                frames_with_measured_contact=int(contact.any(-1).sum()))


def audit_corpus(root,admit=False):
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'COMPLETED' or not manifest.get('capture_current_observation'):
        raise ValueError('completed native-current capture required')
    contract = manifest['observation_contract']
    if (contract.get('dimension') != 721 or contract.get('reference_delta') != 0
            or not contract.get('live_future_invariance_passed')):
        raise ValueError('missing live native-current gates')
    records = manifest['records']
    if len(records) != len({r['sequence'] for r in records}):
        raise ValueError('duplicate sequence identity')
    if 'inherited_split' in manifest:
        if len(records) != 660 or set(manifest['inherited_split']) != {r['sequence'] for r in records}:
            raise ValueError('incomplete full membership')
        for record in records:
            if record['split'] != manifest['inherited_split'][record['sequence']]:
                raise ValueError('changed original TRAIN/dev split')
    rows = {}
    for record in records:
        path = (root/record['path']/'trajectory.npz').resolve()
        if root.resolve() not in path.parents:
            raise ValueError('capture path escapes run')
        with np.load(path,allow_pickle=False) as data:
            rows[record['sequence']] = audit_sequence(data,record,contract,manifest['native_controller'])
    report = dict(status='PASS',sequences=len(rows),frames=sum(r['frames'] for r in rows.values()),
        commands=sum(r['commands'] for r in rows.values()),
        frames_with_measured_contact=sum(r['frames_with_measured_contact'] for r in rows.values()),
        maximum_pd_error=max(r['max_pd_error'] for r in rows.values()),records=rows,
        scope='engineering schema, same-tick raw contact/root/hand/reference and independent action-to-PD checks; no input-benefit/grasp claim')
    (root/'current_observation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if admit:
        if len(records) != 660 or not manifest.get('capture_complete'):
            raise ValueError('only full capture may be admitted')
        snapshot = root/'pre-audit-manifest.json'
        if not snapshot.exists():
            snapshot.write_text(manifest_path.read_text())
        manifest.update(training_allowed=True,native_observation_audit='current_observation_audit.json')
        temporary = root/'manifest.json.part'
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(manifest_path)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=0)
    p.add_argument('--admit',action='store_true')
    args = p.parse_args()
    deadline = time.monotonic()+args.wait_seconds
    while True:
        state = json.loads((args.corpus/'manifest.json').read_text())['status']
        if state == 'COMPLETED':break
        if state == 'FAILED' or time.monotonic()>=deadline:
            raise RuntimeError('capture failed or audit wait budget expired')
        time.sleep(30)
    audit_corpus(args.corpus,args.admit)
